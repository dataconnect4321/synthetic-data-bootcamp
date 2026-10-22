"""Stage 2 of FLIP: TabSyn's latent diffusion model, trained on the VAE's
latents, sampled, and decoded back to records (used by run_adult_flip.py).

  train_diffusion        TabSyn's MLPDiffusion denoiser (EDM) on the latents
  sample_from_diffusion  new latents from the diffusion prior
  decode_latents         latents -> numeric and categorical columns
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

from run_flip_demo import clip_nonnegative  # noqa: E402
from tabsyn.model import MLPDiffusion, Model  # noqa: E402
from tabsyn.diffusion_utils import sample as edm_sample  # noqa: E402
from tabsyn.vae.flip_fairness import (dp_release_mean_std,  # noqa: E402
                                      RDPAccountant)

# ==========================================================================
# DIFFUSION PARAMETERS - Stage 2 knobs
# ==========================================================================
DIFFUSION_PARAMS = {
    # Training epochs for the latent diffusion denoiser. TabSyn's default
    # pipeline uses 10000+; on a 12k-row sample a few thousand suffices.
    'EPOCHS': 3000,

    # Batch size for diffusion training. MUST be much smaller under
    # DP-SGD than under plain Adam: Opacus stores a per-sample gradient
    # copy of every parameter, so memory scales as batch_size *
    # n_params * 4 bytes. With DIM_T=1024 the largest layer is
    # Linear(1024, 2048) (~2.1M params): batch 4096 needs ~34 GB of
    # grad_sample alone (OOM on an 8 GB GPU); batch 256 needs ~2 GB.
    # Lower batch also LOWERS the sampling rate q, improving the
    # privacy budget - the only cost is more steps per epoch.
    'BATCH_SIZE': 256,

    'LR': 1e-3,

    # Hidden width of the MLPDiffusion denoiser (TabSyn default: 1024).
    'DIM_T': 1024,

    # Number of Euler/Heun function evaluations when sampling latents
    # from the trained diffusion prior (TabSyn default: 50).
    'SAMPLE_STEPS': 50,

    # How many synthetic rows to generate (defaults to the training-set
    # size, matching the demo).
    'N_GEN': None,  # None -> len(training data)

    # ---- DP-SGD (Stage 2) ---------------------------------------------
    # Train the latent diffusion prior with DP-SGD so the FULL pipeline
    # carries an end-to-end guarantee. The latents are outputs of the
    # DP-trained VAE encoder (post-processing of a private model), so
    # DP-SGD on them composes with Stage 1's budget into a guarantee
    # w.r.t. the raw training data.
    #
    # The EDM loss is per-sample decomposable (each row draws its own
    # sigma and noise; the loss is a per-row squared error), and
    # MLPDiffusion is standard Linear+SiLU (no BatchNorm), so Opacus's
    # per-sample clipping covers every parameter of this stage.
    'USE_DP': True,          # falls back to plain Adam when False

    # DP-SGD noise multiplier for Stage 2. With BATCH_SIZE=256 on a
    # 12k-row sample the sampling rate q ~ 0.02, which gives a
    # meaningful epsilon at moderate noise; raise NOISE_MULTIPLIER
    # and/or lower EPOCHS to tighten the budget further.
    'NOISE_MULTIPLIER': 1.0,

    # Per-sample gradient L2 clipping norm for Stage 2.
    'MAX_GRAD_NORM': 1.0,

    # Target delta for Stage 2's (epsilon, delta)-DP guarantee.
    'DP_DELTA': 1e-5,

    # Stop after this many epochs without a lower training loss, as
    # TabSyn does (patience 500). None trains for all EPOCHS. Only valid
    # without DP: the decision reads the training loss, which is not
    # covered by the accounting.
    'EARLY_STOP_PATIENCE': None,
}


def train_diffusion(mu, std=None, params=None, device=None):
    """Train TabSyn's MLPDiffusion denoiser (EDM preconditioning) on the
    VAE latents.

    With std given, trains on reparameterized latents z = mu + std * eps
    with fresh eps per batch (Sec. 4.1.2 of the paper). With std=None it
    trains on mu alone (TabSyn's default; the final Adult runs). mu may be a
    tensor or an array (n, tokens*d).

    Follows tabsyn/main.py's normalization convention: latents are shifted
    by their mean and halved before training, and sampled latents are
    doubled and un-shifted (sample() in tabsyn/sample.py). The mean shift
    is a function of the DP-trained encoder's outputs (post-processing of
    a private model), so it is covered by Stage 1's guarantee.

    PRIVACY: when params['USE_DP'] is True the denoiser is trained with
    DP-SGD (Opacus GradSampleModule + DPOptimizer) and its own RDP
    accountant. The EDM loss is per-sample decomposable and MLPDiffusion
    is standard Linear+SiLU, so per-sample clipping covers the whole
    stage - no group-level mechanism is needed here (unlike Stage 1's
    batch-coupled fairness terms). The Stage-2 epsilon composes with
    Stage 1's budget into the pipeline's end-to-end guarantee.

    Returns (model, latent_mean, accountant) where accountant is None
    when DP is disabled.
    """
    mu = torch.as_tensor(mu).float().cpu()
    mean = mu.mean(dim=0, keepdim=True)
    mu_norm = (mu - mean) / 2                 # (z - mean) / 2 with
    std_norm = (torch.as_tensor(std).float().cpu() / 2   # z = mu + std * eps
                if std is not None else None)

    in_dim = mu_norm.shape[1]
    denoise_fn = MLPDiffusion(in_dim, params['DIM_T']).to(device)
    model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

    accountant = None
    if params.get('USE_DP'):
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule

        # MLPDiffusion is standard Linear+SiLU: Opacus's native grad
        # samplers handle every parameter (strict=False tolerates the
        # non-trainable PositionalEmbedding/FourierEmbedding submodules).
        wrapped = GradSampleModule(model, strict=False)
        dp_params = [p for p in wrapped.parameters() if p.requires_grad]
        optimizer = DPOptimizer(
            optimizer=torch.optim.Adam(dp_params, lr=params['LR'],
                                       weight_decay=0),
            noise_multiplier=params['NOISE_MULTIPLIER'],
            max_grad_norm=params['MAX_GRAD_NORM'],
            expected_batch_size=params['BATCH_SIZE'])
        # Poisson sampling via DPDataLoader: each record included
        # independently per step at rate batch/N - matches the
        # accountant's sample_rate exactly.
        accountant = RDPAccountant(
            noise_multiplier=params['NOISE_MULTIPLIER'],
            sample_rate=params['BATCH_SIZE'] / mu_norm.shape[0],
            delta=params['DP_DELTA'])
        train_model = wrapped
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'],
                                     weight_decay=0)
        train_model = model

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20)

    dataset = (torch.utils.data.TensorDataset(mu_norm, std_norm)
               if std_norm is not None
               else torch.utils.data.TensorDataset(mu_norm))
    if params.get('USE_DP'):
        # DPDataLoader performs true Poisson sampling (each record
        # included independently per step at rate batch/N), which is
        # exactly what the subsampled-Gaussian accounting assumes. A
        # plain shuffle=True loader violates that assumption - the same
        # issue Chua et al. (ICML 2024) raise for DP-SGD generally.
        from opacus.data_loader import DPDataLoader
        loader = DPDataLoader.from_data_loader(
            torch.utils.data.DataLoader(
                dataset, batch_size=params['BATCH_SIZE'],
                num_workers=0, drop_last=False))
    else:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=params['BATCH_SIZE'], shuffle=True,
            num_workers=0, drop_last=False)

    patience = params.get('EARLY_STOP_PATIENCE')
    if patience and params.get('USE_DP'):
        raise ValueError('EARLY_STOP_PATIENCE reads the training loss, which '
                         'the DP accounting does not cover; use it only '
                         'with USE_DP=False')
    best_loss, since_best = float('inf'), 0

    model.train()
    # Diffusion length in OPTIMIZER STEPS when DIFF_STEPS is set: a smaller
    # training sample must not silently undertrain the prior (an undertrained
    # prior generates latents near the mean, which decode to near-identical
    # low-income rows). Falls back to EPOCHS otherwise.
    max_steps = params.get('DIFF_STEPS')
    steps_done = 0
    epoch = 0
    while True:
        if max_steps is not None and steps_done >= max_steps:
            break
        if max_steps is None and epoch >= params['EPOCHS']:
            break
        epoch_loss, n_seen = 0.0, 0
        for batch in loader:
            if std_norm is not None:
                batch_mu, batch_std = batch[0].to(device), batch[1].to(device)
                inputs = batch_mu + batch_std * torch.randn_like(batch_std)
            else:
                inputs = batch[0].to(device)   # no extra RNG draw
            loss = train_model(inputs).mean()

            optimizer.zero_grad()
            loss.backward()
            if params.get('USE_DP'):
                if optimizer.pre_step():
                    optimizer.original_optimizer.step()
                accountant.step()
            else:
                optimizer.step()

            epoch_loss += loss.item() * len(inputs)
            n_seen += len(inputs)
            steps_done += 1

        scheduler.step(epoch_loss / max(1, n_seen))
        if epoch % 100 == 0 or (max_steps is None and epoch == params['EPOCHS'] - 1) \
                or (max_steps is not None and steps_done >= max_steps):
            msg = (f'  diffusion epoch {epoch:5d}  '
                   f'loss={epoch_loss / max(1, n_seen):.6f}  '
                   f'steps={steps_done}')
            if accountant is not None and accountant.steps > 0:
                eps_d, _ = accountant.get_privacy_spent()
                msg += f'  eps_diffusion={eps_d:.2f}'
            print(msg)
        epoch += 1

        if patience:
            if epoch_loss / max(1, n_seen) < best_loss:
                best_loss, since_best = epoch_loss / max(1, n_seen), 0
            else:
                since_best += 1
                if since_best >= patience:
                    print(f'  early stop at diffusion epoch {epoch} '
                          f'(no improvement for {patience} epochs, '
                          f'best loss={best_loss:.6f})')
                    break

    return model, mean, accountant


@torch.no_grad()
def sample_from_diffusion(diffusion_model, mean, n_gen, latent_shape,
                          params, device):
    """Sample new latents from the trained diffusion prior and undo the
    (x - mean)/2 normalization."""
    in_dim = int(np.prod(latent_shape))
    x = edm_sample(diffusion_model.denoise_fn_D, n_gen, in_dim,
                   num_steps=params['SAMPLE_STEPS'], device=device)
    x = x * 2 + mean.to(device)
    return x.reshape(n_gen, *latent_shape)


@torch.no_grad()
def decode_latents(vae_model, latents, device, X_num=None, params=None,
                   num_floor=None, sample_idx=None, num_means=None,
                   num_stds=None, bounds=None, sample_all=False):
    """Decode sampled latents back to data space through the frozen VAE
    decoder + Reconstructor, with a DP-released range-clip backstop.

    PRIVACY: the original version clipped synthetic numerics to the
    real data's per-column min/max - a fresh non-private computation on
    raw records (min/max are especially sensitive: one extreme record
    moves them arbitrarily). Order statistics cannot be released with
    bounded sensitivity, so the clip bounds are instead a DP range
    proxy: mean +/- k*std released under the Gaussian mechanism
    (dp_release_mean_std), spent once into the privacy budget.
    """
    vae_model.eval()
    h = vae_model.VAE.decode_tokens(latents[:, 1:])
    recon_num, recon_cat = vae_model.Reconstructor(h)

    syn_num = recon_num.cpu().numpy()
    syn_cat = np.stack([c.argmax(dim=-1).cpu().numpy() for c in recon_cat],
                       axis=1)
    if sample_all:
        # every categorical drawn from its softmax instead of argmax
        syn_cat = np.stack([torch.multinomial(torch.softmax(c, dim=-1), 1)
                            .squeeze(1).cpu().numpy() for c in recon_cat],
                           axis=1)
    elif sample_idx is not None:
        # L_S drives this column's logits to uniform, so argmax would emit one
        # constant class; drawing from the softmax gives the random assignment.
        probs = torch.softmax(recon_cat[sample_idx], dim=-1)
        syn_cat[:, sample_idx] = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()

    if bounds is not None:
        # range-clip bounds released earlier in the run: re-used, so a
        # re-decode (e.g. rejection sampling) spends no extra budget
        syn_num = np.clip(syn_num, bounds[0][None, :], bounds[1][None, :])
    elif X_num is not None and params is not None:
        Xn = X_num.to(device)
        r_mean, r_std, stats_budget = dp_release_mean_std(
            Xn, params['STATS_NOISE_MULTIPLIER'],
            params['STATS_MAX_NORM'], Xn.shape[0])
        k = params['STATS_RANGE_K']
        lo = (r_mean - k * r_std).cpu().numpy()
        hi = (r_mean + k * r_std).cpu().numpy()
        syn_num = np.clip(syn_num, lo[None, :], hi[None, :])
        decode_latents.last_bounds = (lo, hi)
    # Domain-knowledge clamp: all numeric columns in this dataset (loan
    # amount, income, LTV, property value, tract population) are
    # non-negative by definition. Public knowledge, no privacy cost.
    # The standardized image of real-unit 0 is -mean/std (clipping
    # standardized values at 0 would flatten every below-average value).
    # num_floor passes it directly; num_means/num_stds compute it.
    if num_floor is not None:
        syn_num = np.maximum(syn_num, num_floor[None, :])
    else:
        syn_num = clip_nonnegative(syn_num, num_means, num_stds)
    if bounds is None and X_num is not None and params is not None:
        return syn_num, syn_cat, [stats_budget]
    return syn_num, syn_cat, []


# ==========================================================================
# Main
# ==========================================================================
