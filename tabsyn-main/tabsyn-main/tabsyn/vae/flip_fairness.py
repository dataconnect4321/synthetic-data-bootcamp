"""FLIP fairness objectives (Zhang et al. 2025, arXiv:2508.21815).

Implements the Hilbert-Schmidt-independence-based disentanglement used by
FLIP (Fair Latent Intervention under Privacy guarantees) for TabSyn's VAE:

  - CKA^T: transposed centered kernel alignment, a normalized adaptation of
    HSIC (Eqs. 5-7 of the paper). With a linear kernel and centered inputs
    it reduces to a ratio of Frobenius norms. Used as the disentanglement
    divergence D' between representations of different protected groups.
  - Sliced Wasserstein distance (SWD): used as the divergence penalty D
    between the reference (Phase-1) encoder distribution and the current
    encoder distribution (Eq. 10).
  - Uniform attribute loss L_S (Eq. 3): pushes the mean softmax of the
    reconstructed protected-attribute logits toward the uniform
    distribution, so every protected group is predicted with equal
    probability regardless of its training frequency.
  - BalancedGroupSampler: mini-batch sampling with equal group
    representation (Section 4.2.1 / Eq. 11).
"""

import torch
import numpy as np
from torch.utils.data import Sampler


def register_tabsyn_grad_samplers():
    """Register native Opacus grad samplers for TabSyn's custom modules.

    Without these, Opacus falls back to a functorch (vmap) re-forward for
    the custom Tokenizer/Transformer modules, which fails on nn.Embedding
    indexing under vmap. The native samplers compute per-sample gradients
    directly from the captured activations/backprops:

      Tokenizer:
        - weight (numerical tokens): grad = sum over tokens of
          grad_output * x_num  (elementwise product, Linear-like)
        - category_embeddings: Embedding-like scatter of grad_output rows
          onto the looked-up category indices
        - bias: sum of grad_output over the batch dimension per token slot
    """
    try:
        from opacus.grad_sample.utils import register_grad_sampler
    except ImportError:
        return  # opacus not installed; DP-SGD unavailable

    import torch.nn as nn
    from tabsyn.vae.model import Tokenizer, Reconstructor

    @register_grad_sampler(Reconstructor)
    def compute_reconstructor_grad_sample(layer, activations, backprops):
        """Per-sample gradients for the Reconstructor's custom `weight`
        parameter: recon_x_num = (h_num * weight).sum(-1).

        The cat_recons Linear submodules are standard nn.Linear layers and
        receive their own native Opacus hooks, so only the custom elementwise
        weight needs handling here.

        activations: (h,) - the decoder output
        backprops: per-sample grads w.r.t. recon_x_num, delivered either as
                   n tensors of shape (d_num,) or one (n, d_num) tensor
        """
        (h,) = activations
        n = h.shape[0]
        ret = {}

        # Normalize backprops to (n, d_num)
        if torch.is_tensor(backprops[0]) and backprops[0].dim() == 2 \
                and backprops[0].shape[0] == n:
            go_num = backprops[0]
        else:
            go_num = torch.stack(list(backprops[:n]), dim=0)

        h_num = h[:, : layer.d_numerical]            # (n, d_num, d_token)

        # recon_x_num[n, j] = sum_d h_num[n, j, d] * weight[j, d]
        # => gs[n, j, d] = go_num[n, j] * h_num[n, j, d]
        if layer.weight.requires_grad:
            ret[layer.weight] = go_num[:, :, None] * h_num

        return ret

    @register_grad_sampler(Tokenizer)
    def compute_tokenizer_grad_sample(layer, activations, backprops):
        """Per-sample gradients for the Tokenizer's custom `weight` and
        `bias` parameters.

        forward: x = cat([weight[None] * x_num_full[:,:,None],
                          category_embeddings(x_cat + offsets)], dim=1)
                  then x = x + cat([zeros(1), bias])[None]

        The category_embeddings is a standard nn.Embedding with its own
        native Opacus hook, so only the elementwise weight/bias need
        handling here.

        activations: (x_num_raw_or_None, x_cat) as passed to forward
        backprops: per-sample grads w.r.t. the output, either one
                   (n, n_tokens, d) tensor or n tensors of (n_tokens, d)
        """
        x_num_raw, x_cat = activations

        # Normalize backprops to (n, n_tokens, d)
        if torch.is_tensor(backprops[0]) and backprops[0].dim() == 3:
            bp = backprops[0]
        else:
            bp = torch.stack(list(backprops), dim=0)
        n = bp.shape[0]
        ret = {}

        # --- numerical weight: token j of the output is
        # weight[j] * x_num_full[:, j] (weight has d_num+1 rows, one per
        # token including CLS; the CLS column of x_num_full is 1).
        if layer.weight.requires_grad:
            if x_num_raw is not None:
                d_num = x_num_raw.shape[1]
                x_num_full = torch.cat(
                    [torch.ones(n, 1, device=x_num_raw.device,
                                dtype=x_num_raw.dtype), x_num_raw],
                    dim=1,
                )
                go = bp[:, : d_num + 1, :]          # (n, d_num+1, d)
                # output[n, j, d] = weight[j, d] * x_num_full[n, j]
                # => gs[n, j, d] = go[n, j, d] * x_num_full[n, j]
                gs = go * x_num_full[:, :, None]     # (n, d_num+1, d)
                ret[layer.weight] = gs
            else:
                ret[layer.weight] = torch.zeros(
                    n, *layer.weight.shape, device=bp.device, dtype=bp.dtype)

        # --- bias: in forward, bias (with a zero row prepended for CLS) is
        # added to every sample's tokens: x[n, t, :] += bias_cat[t-1, :]
        # for t >= 1. Per-sample bias grad = bp[:, 1:, :] (no sum - each
        # bias row maps 1:1 to one token position).
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = bp[:, 1:, :]        # (n, n_tokens-1, d)

        return ret


def get_dp_trainable_parameters(model):
    """Parameters that participate in the forward graph and thus receive
    per-sample gradients under DP-SGD.

    TabSyn's Transformer defines `head` and `last_normalization`
    submodules that are never applied in forward(), so they never
    receive gradients. DPOptimizer raises if any param in its groups
    lacks a grad sample, so these dead params must be excluded.
    """
    dead_parts = ('.head.', '.last_normalization.')
    return [p for name, p in model.named_parameters()
            if not any(part in name for part in dead_parts)]


class RDPAccountant:
    """Renyi Differential Privacy accountant for DP-SGD training.

    Delegates to Opacus's audited RDPAccountant (Wang et al. 2019
    subsampled-Gaussian RDP bound) when Opacus is installed, and falls
    back to the *unamplified* pure-Gaussian RDP bound (no subsampling
    credit) otherwise - conservative but valid in either case.

    NOTE: the previous hand-rolled bound was incorrect. For alpha*q > 1
    it decayed in alpha, so the reported epsilon shrank as the RDP order
    grew, producing implausibly small epsilon values (e.g. ~0.5 where
    the true value is orders of magnitude larger). Privacy accounting
    is subtle; do not hand-roll it.

    NOTE on sample_rate: DP-SGD's subsampled-Gaussian bound assumes
    Poisson sampling at rate q. The BalancedGroupSampler draws
    fixed-size group-balanced batches in which every minority-group
    sample appears exactly once per epoch, so a minority record's
    per-step inclusion probability is per_group_batch / m_minority -
    LARGER than batch/N. Pass that rate here (gamma_max in the paper,
    Prop. 1 / Eq. 13), not batch/N, or epsilon is understated for
    exactly the group the fairness intervention targets.
    """

    def __init__(self, noise_multiplier: float, sample_rate: float,
                 delta: float = 1e-5, alphas=None):
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = float(sample_rate)
        self.delta = float(delta)
        self.steps = 0
        # Alpha grid used only by the no-Opacus fallback bound
        self.alphas = alphas if alphas is not None else [
            1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505))
        self._opacus = None
        try:
            from opacus.accountants.rdp import RDPAccountant as _OpacusRDP
            self._opacus = _OpacusRDP()
        except ImportError:
            self._opacus = None  # fall back to pure-Gaussian bound

    def step(self):
        """Record one DP-SGD iteration."""
        self.steps += 1
        if self._opacus is not None:
            self._opacus.step(noise_multiplier=self.noise_multiplier,
                              sample_rate=self.sample_rate)

    def get_rdp_curve(self):
        """Return {alpha: rdp_epsilon} of the accumulated guarantee.

        Needed for composing this budget with others (see
        compose_rdp_budgets): valid composition sums the RDP at each
        order alpha and converts to (eps, delta) once, rather than
        adding per-mechanism epsilons (which is not a valid rule).

        Uses Opacus's audited subsampled-Gaussian RDP bound
        (compute_rdp, Wang et al. 2019) when available, so the curve
        retains the full subsampling amplification. An earlier version
        fell back to the unamplified pure-Gaussian bound (alpha/2*sigma^2
        per step), which is valid but discards all amplification and
        inflated the composed total by 2-5x. Without Opacus, the
        unamplified bound is used (conservative upper bound).
        """
        if self.steps == 0:
            return {a: 0.0 for a in self.alphas}
        try:
            from opacus.accountants.analysis.rdp import compute_rdp
            curve = compute_rdp(q=self.sample_rate,
                                noise_multiplier=self.noise_multiplier,
                                steps=self.steps,
                                orders=self.alphas)
            return {a: float(e) for a, e in zip(self.alphas, curve)}
        except ImportError:
            return {a: (a / (2.0 * self.noise_multiplier ** 2)) * self.steps
                    for a in self.alphas}

    def get_privacy_spent(self):
        """Return (epsilon, optimal_alpha) of the DP guarantee so far.

        alpha is None under the Opacus accountant (it minimizes
        internally and does not expose the optimal order)."""
        if self.steps == 0:
            return 0.0, None
        if self._opacus is not None:
            return self._opacus.get_epsilon(self.delta), None
        # Fallback: pure Gaussian mechanism RDP composed over steps,
        # with NO subsampling amplification. Always valid, merely
        # conservative (larger epsilon than the subsampled bound).
        best_eps, best_alpha = float('inf'), None
        for alpha in self.alphas:
            if alpha <= 1:
                continue
            eps = (alpha / (2.0 * self.noise_multiplier ** 2)) * self.steps \
                + np.log(1.0 / self.delta) / (alpha - 1)
            if eps < best_eps:
                best_eps, best_alpha = eps, alpha
        return best_eps, best_alpha


def linear_cka_t(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Transposed linear CKA between two representation matrices.

    FLIP compares feature covariance patterns across protected groups:
    CKA^T(A, B) := CKA(A^T, B^T)  (Eq. 7 of the paper). With centered
    inputs and a linear kernel this reduces to the cosine similarity of
    the two groups' feature covariance matrices:

        CKA^T(A, B) = <A^T A, B^T B>_F / (||A^T A||_F * ||B^T B||_F)

    NOTE: the numerator is the Frobenius inner product of the two
    within-group covariances, NOT ||A^T B||^2_F. The latter pairs row i
    of group A with row i of group B - an arbitrary pairing (batch
    order) that measures nothing meaningful - and forced the old
    implementation to truncate groups to a common size. The correct
    form needs no equal group sizes.

    Args:
        A: (n_A, p) activations of group A (rows = samples).
        B: (n_B, p) activations of group B (n_B may differ from n_A).

    Returns:
        Scalar CKA^T value in [0, 1]. Higher = more similar covariance
        structure; disentanglement *maximizes* this (equivalently
        minimizes its negative).
    """
    A = A - A.mean(dim=0, keepdim=True)
    B = B - B.mean(dim=0, keepdim=True)

    AA = A.t() @ A
    BB = B.t() @ B
    denom = (torch.norm(AA, p="fro") * torch.norm(BB, p="fro"))
    if denom == 0:
        return torch.tensor(0.0, device=A.device)
    return (AA * BB).sum() / denom


def _group_slices(rep: torch.Tensor, group: torch.Tensor):
    """Per-group row blocks with at least 2 samples (covariance needs 2)."""
    blocks = []
    for g in torch.unique(group):
        idx = torch.where(group == g)[0]
        if idx.numel() >= 2:
            blocks.append(rep[idx])
    return blocks


def group_mean_shift(A: torch.Tensor, B: torch.Tensor,
                     eps: float = 1e-8) -> torch.Tensor:
    """Normalized group mean gap ||mu_A - mu_B||^2 / pooled variance.

    CKA^T centers each group on its own mean (dim=0), so it is blind to
    a pure mean offset between groups. This term restores that signal:
    the squared distance between the group means, divided by the total
    pooled within-group variance (sum over features), so it is
    dimensionless and independent of the representation's scale.
    """
    nA, nB = A.shape[0], B.shape[0]
    pooled = ((nA - 1) * A.var(dim=0) + (nB - 1) * B.var(dim=0)) / (nA + nB - 2)
    gap = (A.mean(dim=0) - B.mean(dim=0)).pow(2).sum()
    return gap / (pooled.sum() + eps)


def disentanglement_loss(rep: torch.Tensor, group: torch.Tensor,
                         mean_weight: float = 0.0) -> torch.Tensor:
    """Negative mean pairwise CKA^T between protected groups (D' in Eq. 10),
    plus an optional group mean-shift penalty.

    Args:
        rep: (n, p) representation matrix (latent means, detokenized
             logits, or decoder output) for the current batch.
        group: (n,) integer protected-group labels for the batch.
        mean_weight: weight of the normalized mean-difference term
             (group_mean_shift). 0 gives the paper's pure -CKA^T.

    Returns:
        Scalar loss; minimizing it makes the feature covariance patterns
        (and, with mean_weight > 0, the means) of all protected groups
        converge.
    """
    group_reps = _group_slices(rep, group)
    if len(group_reps) < 2:
        return torch.tensor(0.0, device=rep.device)

    total = 0.0
    n_pairs = 0
    for i in range(len(group_reps)):
        for j in range(i + 1, len(group_reps)):
            total = total - linear_cka_t(group_reps[i], group_reps[j])
            if mean_weight:
                total = total + mean_weight * group_mean_shift(
                    group_reps[i], group_reps[j])
            n_pairs += 1

    return total / n_pairs


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Differentiable 1-D Wasserstein-1 distance between two empirical
    samples of possibly different sizes (quantile-function matching)."""
    m = max(a.numel(), b.numel())

    def quantiles(x):
        xs = torch.sort(x).values
        n = xs.numel()
        pos = ((torch.arange(m, device=x.device, dtype=x.dtype) + 0.5)
               / m * n - 0.5).clamp(0, n - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=n - 1)
        w = pos - lo.to(x.dtype)
        return xs[lo] * (1 - w) + xs[hi] * w

    return (quantiles(a) - quantiles(b)).abs().mean()


def binary_logit_gap_w1(logits: torch.Tensor, group: torch.Tensor,
                        eps: float = 1e-4) -> torch.Tensor:
    """Detokenizer-stage group gap for a binary categorical column.

    Two logits per row give CKA^T nothing to compare, so the column is
    reduced to the logit gap d = l1 - l2 and the groups' distributions of
    d are compared with W1, divided by the pooled within-group std.
    Synthetic categories are chosen by argmax (d > 0), so the approval
    rate is the share of d above zero; two groups with equal mean d can
    still have very different rates, which W1 detects and a mean gap does
    not. Mean over group pairs; >= 0.
    """
    d = logits[:, 0] - logits[:, 1]
    blocks = _group_slices(d.unsqueeze(1), group)
    blocks = [b.squeeze(1) for b in blocks]
    if len(blocks) < 2:
        return torch.tensor(0.0, device=logits.device)

    total, n_pairs = 0.0, 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            nA, nB = blocks[i].numel(), blocks[j].numel()
            pooled_var = ((nA - 1) * blocks[i].var() + (nB - 1) * blocks[j].var()) \
                / (nA + nB - 2)
            total = total + wasserstein_1d(blocks[i], blocks[j]) \
                / torch.sqrt(pooled_var + eps)
            n_pairs += 1
    return total / n_pairs


def shared_mean_ratio(rep: torch.Tensor) -> float:
    """||pooled mean||^2 / total variance of a representation.

    Diagnostic for the paper-literal row-centering (dim=1) CKA^T: a large
    offset shared by all groups drowns out group signal under it. A small
    ratio means that alternative is safe."""
    var = rep.var(dim=0).sum()
    return float(rep.mean(dim=0).pow(2).sum() / (var + 1e-12))


class GradientReversal(torch.autograd.Function):
    """Identity forward, negated backward: the classic GRL layer."""

    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class AdversaryHead(torch.nn.Module):
    """Per-record protected-attribute adversary on the decoded non-protected
    logits. A gradient-reversal layer makes minimizing its CE loss MAXIMIZE
    the adversary's error, scrubbing S from the other columns' logits.

    The loss is a per-record cross-entropy (each row's logits vs its own
    group label), so under DP it routes through Opacus's per-sample pass
    (pass 1), NOT the batch-coupled group-level mechanism.
    """

    def __init__(self, d_in, n_groups, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_groups))

    def forward(self, x, lambd=1.0):
        return self.net(gradient_reversal(x, lambd))


# --------------------------------------------------------------------------
# In-model proxy decoupling
# --------------------------------------------------------------------------
# A merged column (Adult: `relationship`) carries a sex-specific label
# (Husband vs Wife) that is a near-copy of S. Instead of re-drawing that
# label after generation (tabsyn.vae.temper.decouple_curve, strength tau),
# these terms make it independent of the OTHER tokens inside the VAE
# (strength lambda), and CKA^T is applied to the non-proxy tokens only.

def cat_token_index(d_num, j, stage='mu'):
    """Token position of categorical column j. mu_z is laid out
    [CLS, num_1..num_d, cat_1..cat_k]; the decoder output h drops CLS."""
    return (1 if stage == 'mu' else 0) + d_num + j


def drop_tokens(rep, tokens):
    """rep (n, T, d) with the given token positions removed."""
    keep = [t for t in range(rep.shape[1]) if t not in set(tokens)]
    return rep[:, keep]


def proxy_sublabel(cat, proxy_idx, pos_code, neg_code):
    """Rows holding one of the merged column's two sex-specific labels, and
    the binary sub-label on those rows (1 = pos_code, e.g. Husband).

    Returns (mask (n,) bool, label (n,) long; label is 0 off the mask)."""
    col = cat[:, proxy_idx]
    mask = (col == pos_code) | (col == neg_code)
    return mask, (col == pos_code).long()


def hsic_normalized(X, y):
    """Normalized biased HSIC between a representation X (n, d) and a binary
    label y (n,): RBF kernel on X (median-distance bandwidth, detached),
    linear kernel on the centered one-hot of y. In [0, 1]; 0 = X carries no
    (kernel-detectable) information about y. Batch-coupled, so under DP it
    belongs in the group-level pass."""
    n = X.shape[0]
    if n < 4 or y.min() == y.max():
        return torch.tensor(0.0, device=X.device)
    X = X.float()
    d2 = torch.cdist(X, X).pow(2)
    with torch.no_grad():
        sigma2 = d2[d2 > 0].median().clamp_min(1e-8)
    K = torch.exp(-d2 / (2 * sigma2))
    Kc = K - K.mean(0, keepdim=True) - K.mean(1, keepdim=True) + K.mean()
    yc = y.float() - y.float().mean()
    # HSIC(X, y) ∝ yc^T Kc yc ; HSIC(y, y) ∝ (yc^T yc)^2 ; HSIC(X, X) ∝ ||Kc||_F^2
    hxy = yc @ Kc @ yc
    hxx = (Kc * Kc).sum()
    hyy = (yc @ yc).pow(2)
    return hxy / torch.sqrt(hxx * hyy).clamp_min(1e-12)


def proxy_independence_loss(stages, sub_label, mask, proxy_idx, s_idx):
    """HSIC between the merged column's sex-specific label and the remaining
    representation (latent mu_z and decoder output h with the proxy and S
    tokens removed), on the rows that carry the label. Mean of the two
    stages. Minimizing it leaves the label recoverable only from the proxy
    token itself."""
    if mask.sum() < 4:
        return torch.tensor(0.0, device=stages['mu_z'].device)
    d_num = stages['recon_x_num'].shape[1]
    losses = []
    for key, stage in (('mu_z', 'mu'), ('h', 'h')):
        rep = drop_tokens(stages[key],
                          [cat_token_index(d_num, proxy_idx, stage),
                           cat_token_index(d_num, s_idx, stage)])
        rep = rep[mask].reshape(int(mask.sum()), -1)
        losses.append(hsic_normalized(rep, sub_label[mask]))
    return torch.stack(losses).mean()


def proxy_adversary_input_dim(n_tokens_mu, d_token):
    """Input width of the proxy adversary: mu_z minus the proxy and S tokens."""
    return (n_tokens_mu - 2) * d_token


def proxy_adversary_loss(adv_head, stages, sub_label, mask, proxy_idx, s_idx,
                         lambd=1.0):
    """Per-record alternative to proxy_independence_loss: a gradient-reversal
    adversary predicts the sex-specific label from mu_z without the proxy
    and S tokens. Sum of per-row CE terms (masked rows only), so under DP
    it rides the per-sample pass."""
    d_num = stages['recon_x_num'].shape[1]
    rep = drop_tokens(stages['mu_z'],
                      [cat_token_index(d_num, proxy_idx, 'mu'),
                       cat_token_index(d_num, s_idx, 'mu')])
    rep = rep.reshape(rep.shape[0], -1)
    ce = torch.nn.functional.cross_entropy(adv_head(rep, lambd), sub_label,
                                           reduction='none')
    # batch MEAN of the masked per-row terms (not a sum): scale-invariant
    # to batch size, matching the other Phase-2 losses
    return (ce * mask.float()).mean()


def marginal_matching_penalty(stages, X_num, X_cat, s_idx,
                              n_quantiles=8):
    """Penalty pulling each column's decoded marginal toward the batch's real
    marginal: category shares (TV) for categorical columns, quantile
    matching (mean absolute quantile gap) for numeric columns. Zero when the
    decoded batch equals the real batch. Per-batch (not per-record); under
    DP it belongs in the batch-coupled pass."""
    pen = 0.0
    n = X_num.shape[0]
    # numeric: quantile matching
    qs = torch.linspace(0, 1, n_quantiles, device=X_num.device)[1:-1]
    real_q = torch.quantile(X_num, qs, dim=0)          # (Q, d_num)
    syn_q = torch.quantile(stages['recon_x_num'], qs, dim=0)
    pen = pen + (real_q - syn_q).abs().mean()
    # categorical: share TV per column
    tv = 0.0
    for i, logits in enumerate(stages['recon_x_cat']):
        k = logits.shape[1]
        real_share = torch.bincount(X_cat[:, i], minlength=k).float() / n
        syn_share = torch.softmax(logits, dim=1).mean(dim=0)
        tv = tv + 0.5 * (real_share - syn_share).abs().sum()
    pen = pen + tv / len(stages['recon_x_cat'])
    return pen


def multi_stage_disentanglement_loss(stages: dict, group: torch.Tensor,
                                     s_idx=None,
                                     mean_weight: float = 1.0,
                                     binary_w1: bool = True,
                                     exclude_cat=(),
                                     t_idx=None,
                                     target_in_fairness=False) -> torch.Tensor:
    """FLIP multi-stage disentanglement (Sec. 4.3.2 of the paper).

    Applies the negative-CKA^T disentanglement at three intervention
    stages and returns the mean:

      1. Latent space:   mu_z (flattened over tokens) - primary objective.
      2. Detokenizer:   feature-wise on the pre-aggregation logits
                        (numerical logits + each categorical logit),
                        mean-aggregated so features with more unique
                        values are not disproportionately weighted.
                        Binary categorical columns (2 logits) use the
                        W1 logit-gap term instead of CKA^T.
      3. Decoder:       decoder output h (flattened over tokens), whose
                        dimensionality matches the latent space.

    Deviation from the paper: CKA^T keeps per-group centering (dim=0),
    which cannot see group mean offsets, so each stage also adds
    mean_weight * the normalized group mean gap (group_mean_shift).

    Args:
        stages: dict from Model_VAE.forward_with_stages() containing
                'mu_z', 'h', 'recon_x_num', 'recon_x_cat'.
        group: (n,) integer protected-group labels.
        s_idx: index of the protected attribute among the categorical
               columns (its logits are skipped at the detokenizer stage
               since they are supervised by the uniform attribute loss).
        exclude_cat: categorical columns treated as proxies: their tokens are
               dropped from mu_z and h and their logits skipped, so CKA^T
               act on the non-proxy representation only (the proxy is
               handled by proxy_independence_loss instead).

    Returns:
        Scalar mean disentanglement loss across the three stages.
    """
    device = stages['mu_z'].device
    losses = []

    # Stage 1: latent space
    mu, h = stages['mu_z'], stages['h']
    if exclude_cat:
        d_num = stages['recon_x_num'].shape[1]
        mu = drop_tokens(mu, [cat_token_index(d_num, j, 'mu')
                              for j in exclude_cat])
        h = drop_tokens(h, [cat_token_index(d_num, j, 'h')
                            for j in exclude_cat])
    mu_flat = mu.reshape(mu.shape[0], -1)
    h_flat = h.reshape(h.shape[0], -1)
    losses.append(disentanglement_loss(mu_flat, group, mean_weight))

    # Stage 2: detokenizer - feature-wise, mean-aggregated
    feature_losses = []
    num_logits = stages['recon_x_num']
    if num_logits is not None:
        feature_losses.append(disentanglement_loss(num_logits, group,
                                                   mean_weight))
    for i, cat_logits in enumerate(stages['recon_x_cat']):
        if cat_logits is None:
            continue
        if s_idx is not None and i == s_idx:
            continue  # protected attribute handled by L_S
        # TARGET_IN_FAIRNESS (default off): the target's logits stay out of
        # the detokenizer-stage terms unless outcome parity is intended.
        if t_idx is not None and i == t_idx and not target_in_fairness:
            continue
        if i in exclude_cat:
            continue  # proxy handled by proxy_independence_loss
        if binary_w1 and cat_logits.shape[1] == 2:
            feature_losses.append(binary_logit_gap_w1(cat_logits, group))
        else:
            feature_losses.append(disentanglement_loss(cat_logits, group,
                                                       mean_weight))
    if feature_losses:
        losses.append(torch.stack(feature_losses).mean())

    # Stage 3: decoder output. The decoder is nonlinear and can put group
    # differences back in, so it gets the same CKA^T + mean treatment.
    losses.append(disentanglement_loss(h_flat, group, mean_weight))

    return torch.stack(losses).mean()


def sliced_wasserstein_distance(X: torch.Tensor, Y: torch.Tensor,
                                n_projections: int = 50,
                                theta: torch.Tensor = None) -> torch.Tensor:
    """Sliced Wasserstein distance between two point sets (D in Eq. 10).

    Projects both sets onto random 1-D directions and averages the
    1-D Wasserstein-1 distances. Used as the divergence penalty keeping
    the Phase-2 encoder close to the frozen Phase-1 reference encoder.

    Args:
        X: (n, d) samples from distribution 1 (e.g. reference latents).
        Y: (n, d) samples from distribution 2 (e.g. current latents).
        n_projections: number of random projections to average over.
        theta: optional (n_projections, d) fixed projection directions
               (for testing symmetry with identical projections).
    """
    n, d = X.shape
    if theta is None:
        # Random unit-norm projection directions
        theta = torch.randn(n_projections, d, device=X.device, dtype=X.dtype)
        theta = theta / torch.norm(theta, dim=1, keepdim=True)

    X_proj = X @ theta.t()  # (n, n_projections)
    Y_proj = Y @ theta.t()

    X_sorted = torch.sort(X_proj, dim=0).values
    Y_sorted = torch.sort(Y_proj, dim=0).values

    # Mean absolute difference between sorted projections = 1-D W1 distance
    w1 = torch.mean(torch.abs(X_sorted - Y_sorted))
    return w1


def uniform_attribute_loss(s_logits: torch.Tensor) -> torch.Tensor:
    """Uniform attribute loss L_S (Eq. 3 of the paper).

    l2-norm between the mean group-wise softmax of the reconstructed
    protected-attribute logits and the uniform distribution.

    Args:
        s_logits: (n, |S|) reconstructed logits for the protected
                  attribute (one row per sample in the batch).

    Returns:
        Scalar loss; zero when all groups are predicted uniformly.
    """
    n, n_groups = s_logits.shape
    mean_probs = torch.softmax(s_logits, dim=1).mean(dim=0)
    uniform = torch.full_like(mean_probs, 1.0 / n_groups)
    return torch.norm(mean_probs - uniform, p=2)


def uniform_attribute_loss_per_sample(s_logits: torch.Tensor) -> torch.Tensor:
    """Per-sample (DP-SGD-decomposable) version of L_S.

    The batch-mean L_S above couples every sample in the batch (each
    sample's gradient depends on all others through the mean softmax),
    which breaks DP-SGD's per-sample clipping. This variant assigns each
    sample its own loss:

        l_i = || softmax(s_i) - uniform ||_2

    summed over the batch. Minimizing the sum pushes every individual
    reconstruction toward uniform protected-attribute logits, which is a
    strictly stronger condition than a uniform batch mean (the mean can
    be uniform while individual rows are confidently wrong). Because it
    is a sum of per-sample terms, per-sample gradients are isolated and
    clipping bounds each record's influence - the DP guarantee covers it.
    """
    n, n_groups = s_logits.shape
    probs = torch.softmax(s_logits, dim=1)
    uniform = torch.full_like(probs, 1.0 / n_groups)
    return torch.norm(probs - uniform, p=2, dim=1).sum()


class GroupLevelDPMechanism:
    """Gaussian mechanism for the batch-coupled fairness losses
    (sliced-Wasserstein anchor + CKA^T disentanglement).

    WHY THIS EXISTS: DP-SGD's record-level guarantee requires each
    sample's gradient to be clipped *in isolation*. The SWD and CKA^T terms
    are batch-level functionals - one record changes the gradient
    attributed to every other record in the batch - so per-sample
    clipping cannot bound any individual's influence for these terms.
    Pretending otherwise (routing them through Opacus) silently voids
    the guarantee.

    WHAT THIS DOES: computes the batch-coupled gradient on the full
    batch, clips the ENTIRE gradient vector to a fixed L2 norm C, and
    adds Gaussian noise with std sigma_g * C. Because the whole vector
    is clipped, the mechanism's output depends on the data only through
    a vector of norm <= C: under substitution adjacency (neighboring
    datasets differ in one record) the sensitivity is 2C, so the
    effective noise-to-sensitivity ratio is sigma_g/2 and each step's
    RDP is
        eps_rdp(alpha) = alpha / (2*(sigma_g/2)^2) = 2*alpha/sigma_g^2
    (NOT alpha/(2*sigma_g^2) - that assumed sensitivity C and
    under-accounted by 4x).

    GUARANTEE: record-level (epsilon, delta)-DP for the fairness-gradient
    updates, under either substitution or add/remove adjacency (the 2C
    bound holds for both), so this mechanism's RDP composes additively
    with the DP-SGD budget (see compose_rdp_budgets).

    SUBSAMPLING AMPLIFICATION: when the batch is drawn by Poisson
    sampling at rate q (PoissonGroupSampler), the amplification theorem
    applies to ANY mechanism run on the sampled batch - including this
    clipped-and-noised gradient release. Pass sample_rate=q (gamma_max,
    the largest per-record inclusion probability across groups) and
    the per-step RDP is accounted with Opacus's subsampled-Gaussian
    bound at effective noise multiplier sigma_g/2 (noise std sigma_g*C
    over substitution sensitivity 2C). Without a sample rate (or with
    sample_rate=None), the unamplified 2*alpha/sigma_g^2 per step is
    used - valid but far looser (the fairness term was the largest
    part of the composed total before this credit was added).
    """

    def __init__(self, max_group_grad_norm: float, noise_multiplier: float,
                 sample_rate: float = None):
        self.max_group_grad_norm = float(max_group_grad_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.sample_rate = sample_rate  # q for Poisson amplification
        self.steps = 0

    def add_noised(self, params, grads):
        """Clip, noise, and ACCUMULATE the batch-coupled gradient into
        p.grad, to be called after the record-level DP-SGD aggregation
        has written p.grad and before the optimizer step.

        Args:
            params: list of parameters (p.grad already holds the noisy
                record-level aggregate; the noised group-level gradient
                is ADDED to it so one optimizer step applies both).
            grads: dict {param: batch-coupled gradient tensor} captured
                from the fairness-loss backward pass.
        """
        self.steps += 1
        items = [(p, grads[p]) for p in params if p in grads]
        if not items:
            return
        # Global L2 clip across all parameters (one vector)
        total_norm = torch.norm(torch.stack(
            [g.norm() for _, g in items]))
        scale = self.max_group_grad_norm / (total_norm + 1e-12)
        for p, g in items:
            g = g * scale if scale < 1.0 else g
            noise = torch.normal(
                mean=0.0,
                std=self.noise_multiplier * self.max_group_grad_norm,
                size=g.shape, device=g.device, dtype=g.dtype)
            if p.grad is None:
                p.grad = g + noise
            else:
                p.grad = p.grad + g + noise

    def get_rdp_epsilon(self, alpha: float) -> float:
        """Record-level RDP of order alpha accumulated so far.

        Gaussian mechanism with noise std sigma_g*C and substitution
        sensitivity 2C: effective sigma = sigma_g/2, so each step adds
        alpha / (2*(sigma_g/2)^2) = 2*alpha/sigma_g^2. With Poisson
        sampling at rate q, the subsampled-Gaussian bound applies at
        effective noise multiplier sigma_g/2 (see class docstring).
        """
        if self.steps == 0:
            return 0.0
        if self.sample_rate is not None and self.sample_rate > 0:
            # Subsampled bound at ONE step (compute_rdp composes over
            # steps internally, so call it with steps=1 and multiply).
            try:
                from opacus.accountants.analysis.rdp import compute_rdp
                one = compute_rdp(q=self.sample_rate,
                                  noise_multiplier=self.noise_multiplier / 2.0,
                                  steps=1, orders=[alpha])
                return float(one[0]) * self.steps
            except ImportError:
                pass
        return (2.0 * alpha / self.noise_multiplier ** 2) * self.steps

    def get_rdp_curve(self, alphas=None):
        """Return {alpha: rdp_epsilon} for composition (see
        compose_rdp_budgets). Uses the subsampled-Gaussian bound when a
        sample_rate is set (Poisson sampling), else the unamplified
        bound."""
        if alphas is None:
            alphas = [1 + x / 10.0 for x in range(1, 100)] + \
                list(range(11, 505))
        if self.steps == 0:
            return {a: 0.0 for a in alphas}
        if self.sample_rate is not None and self.sample_rate > 0:
            try:
                from opacus.accountants.analysis.rdp import compute_rdp
                curve = compute_rdp(q=self.sample_rate,
                                    noise_multiplier=self.noise_multiplier / 2.0,
                                    steps=self.steps, orders=alphas)
                return {a: float(e) for a, e in zip(alphas, curve)}
            except ImportError:
                pass
        return {a: self.get_rdp_epsilon(a) for a in alphas}

    def get_privacy_spent(self, delta: float, alphas=None):
        """Return (epsilon, optimal_alpha) for the record-level
        guarantee (substitution adjacency)."""
        if self.steps == 0:
            return 0.0, None
        if alphas is None:
            alphas = [1 + x / 10.0 for x in range(1, 100)] + \
                list(range(11, 505))
        best_eps, best_alpha = float('inf'), None
        for alpha in alphas:
            if alpha <= 1:
                continue
            eps = self.get_rdp_epsilon(alpha) + np.log(1.0 / delta) / (alpha - 1)
            if eps < best_eps:
                best_eps, best_alpha = eps, alpha
        return best_eps, best_alpha


def dp_release_mean_std(values: torch.Tensor, noise_multiplier: float,
                        max_norm: float, n_rows: int):
    """Release per-column mean/std of a tensor under the Gaussian mechanism.

    Used for the latent statistics (and numeric clamp bounds) needed at
    generation time. Computing them directly on the training data would
    be a fresh non-private computation on raw records - min/max in
    particular are notoriously sensitive (a single extreme record moves
    them arbitrarily). Instead:

      1. Each ROW's contribution is clipped to L2 norm max_norm
         (bounding any single record's influence on the statistics).
      2. The summed statistic is noised with Gaussian noise scaled by
         noise_multiplier * max_norm (sensitivity of a clipped sum).
      3. The release is accounted as one Gaussian mechanism application.

    For per-column min/max bounds we release a clipped MEAN plus a
    multiple of the released std as a robust range proxy - order
    statistics cannot be released with bounded sensitivity, but a
    mean +/- k*std range is a standard DP surrogate.

    Returns (mean, std, rdp_orders) where rdp_orders is a list of
    (alpha, eps_rdp) pairs for accounting.
    """
    # values: (n, d) - clip each row to max_norm
    row_norms = values.norm(dim=1, keepdim=True)
    scale = max_norm / (row_norms + 1e-12)
    scale = torch.where(scale < 1.0, scale, torch.ones_like(scale))
    clipped = values * scale

    # Sensitivity of a sum of row-clipped values: 2*max_norm for
    # add/remove adjacency (each row contributes at most max_norm).
    sens = 2.0 * max_norm
    noise_std = noise_multiplier * sens

    n, d = clipped.shape
    noise = torch.normal(mean=0.0, std=noise_std, size=(d,),
                         device=values.device, dtype=values.dtype)
    sum_released = clipped.sum(dim=0) + noise
    mean_released = sum_released / max(1, n)

    # Std via clipped second moment (same mechanism, one more release)
    sq = clipped.pow(2)
    # Clip squared contributions so a single row's squared contribution
    # is also bounded: each element of sq is at most max_norm^2/d * d =
    # max_norm^2 (row norm bound); sensitivity of the sq-sum is
    # max_norm^2 per column.
    noise2 = torch.normal(mean=0.0, std=noise_multiplier * (max_norm ** 2),
                          size=(d,), device=values.device,
                          dtype=values.dtype)
    sumsq_released = sq.sum(dim=0) + noise2
    var_released = sumsq_released / max(1, n) - mean_released.pow(2)
    std_released = var_released.clamp_min(1e-6).sqrt()

    # RDP for two Gaussian releases: 2 * alpha / (2 sigma^2) each with
    # its own sigma; report per-release orders for the caller to compose
    rdp_orders = []
    for alpha in [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505)):
        eps1 = alpha / (2.0 * noise_multiplier ** 2)
        # Second release uses sigma2 = noise_multiplier * max_norm^2 / (2*max_norm^2)
        # = noise_multiplier/2 relative to its sensitivity -> sigma_rel = nm/2
        eps2 = alpha / (2.0 * (noise_multiplier / 2.0) ** 2)
        rdp_orders.append((alpha, eps1 + eps2))
    return mean_released, std_released, rdp_orders


def compose_rdp_budgets(budgets, delta, alphas=None):
    """Compose multiple privacy budgets into one (epsilon, alpha).

    Every mechanism in the pipeline touches the same records, so the
    true end-to-end guarantee is the composition of ALL of them:
    DP-SGD (record-level), the fairness-gradient Gaussian mechanism
    (record-level, substitution adjacency), the statistics releases,
    and the diffusion stage's DP-SGD.

    Valid RDP composition sums the Renyi divergence at each order
    alpha and converts to (epsilon, delta) ONCE:
        eps = min_alpha [ sum_i rdp_i(alpha) + log(1/delta)/(alpha-1) ]
    Adding per-mechanism epsilons directly is NOT a valid composition
    rule (each mechanism minimizes over a different alpha).

    Args:
        budgets: list of either
            - {alpha: rdp} dicts (from RDPAccountant.get_rdp_curve(),
              GroupLevelDPMechanism.get_rdp_curve(), or
              dp_release_mean_std's rdp_orders converted with
              dict(rdp_orders)), or
            - (alpha, rdp) pair lists (converted internally).
        delta: target delta for the combined guarantee.
        alphas: optional alpha grid; defaults to the standard grid.

    Returns:
        (epsilon, optimal_alpha) of the composed guarantee.

    NOTE: multiple releases passed as ONE pair list with duplicate
    alpha keys would silently drop all but the last entry per alpha
    (dict() keeps the last). Either pass each release as its own budget
    in the list, or pre-sum by alpha - this function sums correctly
    across the budgets it is given, but cannot detect duplicates
    within a single pair list.
    """
    if alphas is None:
        alphas = [1 + x / 10.0 for x in range(1, 100)] + \
            list(range(11, 505))
    # Normalize every budget to {alpha: rdp}, SUMMING duplicate alpha
    # entries within a pair list (two dp_release_mean_std releases
    # extended into one list share the same alpha grid; dict() would
    # keep only the last and silently drop the other release).
    curves = []
    for b in budgets:
        if not b:
            continue
        if isinstance(b, dict):
            curves.append(dict(b))
        else:
            summed = {}
            for alpha, rdp in b:
                summed[alpha] = summed.get(alpha, 0.0) + rdp
            curves.append(summed)

    if not curves:
        return 0.0, None

    # Union of alphas present in all curves (compose only where every
    # budget has a value; missing orders are skipped conservatively by
    # using the intersection)
    common = set(curves[0].keys())
    for c in curves[1:]:
        common &= set(c.keys())
    if not common:
        # No overlapping alphas: fall back to each curve's own grid by
        # interpolating is unsafe - instead evaluate each curve's min
        # alpha available and compose at the union with 0 for missing
        # (NOT conservative). Safer: raise.
        raise ValueError("Budgets have no common alpha orders; "
                         "use the same alpha grid for all mechanisms.")

    best_eps, best_alpha = float('inf'), None
    for alpha in sorted(common):
        if alpha <= 1:
            continue
        total_rdp = sum(c[alpha] for c in curves)
        eps = total_rdp + np.log(1.0 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps, best_alpha = eps, alpha
    return best_eps, best_alpha


class BalancedGroupSampler(Sampler):
    """Mini-batch sampler with equal protected-group representation.

    Implements the balanced sampling strategy of Section 4.2.1: each
    batch contains the same number of samples from every protected
    group. Iterations per epoch follow Eq. 11 with CEIL division,
        L = ceil(m / b * |G|)
    where m is the smallest group size, b the per-group batch share and
    |G| the number of groups, so every minority-group sample is seen
    exactly once per epoch (a final partial batch is included).
    """

    def __init__(self, group_labels, batch_size: int, seed: int = 0):
        """Args:
            group_labels: (n,) array/Series of protected-group labels.
            batch_size: total batch size (split evenly across groups).
            seed: RNG seed for shuffling.
        """
        self.labels = np.asarray(group_labels)
        self.groups = np.unique(self.labels)
        self.n_groups = len(self.groups)
        if self.n_groups < 2:
            raise ValueError("Balanced sampling needs >= 2 protected groups.")
        if batch_size < self.n_groups:
            raise ValueError("Batch size must be >= number of groups.")

        self.per_group = batch_size // self.n_groups
        self.batch_size = self.per_group * self.n_groups
        self.seed = seed
        self.epoch = 0

        self.group_indices = {g: np.where(self.labels == g)[0]
                              for g in self.groups}
        # Eq. 11: iterations per epoch (floor(m / b_per_group)); the
        # final partial batch is included so every minority-group sample
        # is seen exactly once per epoch.
        m = min(len(v) for v in self.group_indices.values())
        self.num_batches = max(1, -(-m // self.per_group))  # ceil division

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        # Shuffle each group's indices once per epoch
        shuffled = {g: rng.permutation(v)
                    for g, v in self.group_indices.items()}
        for b in range(self.num_batches):
            batch = []
            start = b * self.per_group
            for g in self.groups:
                # Final batch may be partial for some groups (wrap-around
                # is avoided; partial batches are simply smaller)
                idx = shuffled[g][start:start + self.per_group]
                batch.extend(idx.tolist())
            if batch:
                rng.shuffle(batch)
                yield batch

    def __len__(self):
        return self.num_batches


class PoissonGroupSampler(Sampler):
    """Per-group Poisson (independent) sampling.

    WHY: Opacus's subsampled-Gaussian RDP bound formally assumes each
    record is included in each step independently with probability q
    (Poisson sampling). The shuffle-based BalancedGroupSampler draws
    fixed-size batches in which every minority sample appears exactly
    once per epoch - a different sampling process, and Chua et al.
    (ICML 2024, arXiv:2403.17673) showed shuffle-based DP-SGD can leak
    MORE than Poisson accounting reports. Using this sampler makes the
    Poisson accounting assumption exactly true.

    Each step, every record is included independently with probability
    q = per_group_batch / group_size computed PER GROUP, so every group
    has the same expected representation (the balanced-sampling goal of
    Sec. 4.2.1) while each record's inclusion is independent. The
    accountant's sampling rate is gamma_max = max over groups of
    q_g = per_group_batch / m_g (the minority group's rate, the largest).

    Batch sizes are random (Binomial); pass expected_batch_size to the
    DPOptimizer accordingly. Empty batches are skipped.

    Args:
        group_labels: (n,) protected-group labels.
        per_group_batch: expected number of samples per group per step.
        steps_per_epoch: number of Poisson steps per epoch.
        seed: RNG seed.
    """

    def __init__(self, group_labels, per_group_batch: int,
                 steps_per_epoch: int, seed: int = 0):
        self.labels = np.asarray(group_labels)
        self.groups = np.unique(self.labels)
        self.n_groups = len(self.groups)
        if self.n_groups < 2:
            raise ValueError("Poisson sampling needs >= 2 protected groups.")
        self.per_group_batch = int(per_group_batch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = seed
        self.epoch = 0

        self.group_indices = {g: np.where(self.labels == g)[0]
                               for g in self.groups}
        # Per-group inclusion probability: same expected representation
        # per group regardless of its size.
        self.q = {g: min(1.0, self.per_group_batch / len(idx))
                  for g, idx in self.group_indices.items()}
        # gamma_max: the largest per-record inclusion probability (the
        # minority group's) - the rate the accountant must use.
        self.gamma_max = max(self.q.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps_per_epoch):
            batch = []
            for g in self.groups:
                idx = self.group_indices[g]
                mask = rng.random(len(idx)) < self.q[g]
                batch.extend(idx[mask].tolist())
            if batch:
                rng.shuffle(batch)
                yield batch

    def __len__(self):
        return self.steps_per_epoch