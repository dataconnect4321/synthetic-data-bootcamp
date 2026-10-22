"""Stage 1 of FLIP: the fair (and optionally private) TabSyn VAE, plus the
decoding and fidelity helpers used by run_adult_flip.py.

  train_flip   Phase 1 (quality): beta-VAE loss + uniform attribute loss L_S
               Phase 2 (fairness): + SWD divergence anchor + lambda * CKA^T
               at the latent / detokenizer / decoder stages (+ the joint
               decoded-output stage in the 'final' preset), with balanced
               group sampling and optional DP-SGD (Eq. 13 per-group noise)
  decode_synthetic_data / infer_numeric_schema / apply_numeric_schema
               turn decoder outputs into released records
  fidelity_metrics
               per-column W1 (numeric) and TV distance (categorical)

TRADEOFF_PARAMS holds the defaults; run_adult_flip.py sets the Adult
columns and overrides the method settings from its presets.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.model import Model_VAE
from tabsyn.vae.flip_fairness import (
    linear_cka_t,
    multi_stage_disentanglement_loss,
    sliced_wasserstein_distance,
    uniform_attribute_loss,
    decoded_joint_loss,
    marginal_matching_penalty,
    BalancedGroupSampler,
    PoissonGroupSampler,
    RDPAccountant,
    GroupwiseRDPAccountant,
    eq13_sigma_global,
    GroupLevelDPMechanism,
    register_tabsyn_grad_samplers,
    get_dp_trainable_parameters,
)
from tabsyn.vae.main import compute_loss

# ==========================================================================
# TRADEOFF PARAMETERS - the fidelity / privacy / fairness knob panel
# ==========================================================================
TRADEOFF_PARAMS = {
    # ---- FAIRNESS knobs -------------------------------------------------
    # Weight of the CKA^T disentanglement term (Phase 2). Higher lambda ->
    # stronger bias removal in the latent space, but pushing too far
    # degrades reconstruction fidelity (paper Fig. 3: fairness-quality
    # trade-off). 0 disables explicit disentanglement.
    'LAMBDA_FAIR': 2.0,

    # Weight of the normalized group mean-difference term added to CKA^T
    # at the latent, detokenizer and decoder stages (CKA^T's per-group
    # centering cannot see mean offsets). 0 = paper's pure CKA^T.
    'MEAN_SHIFT_WEIGHT': 1.0,
    # Score binary non-protected columns at the detokenizer stage with the
    # W1 logit-gap term instead of CKA^T (which has nothing to compare for
    # 2 logits).
    'BINARY_W1': True,
    # Lambda ramp: LAMBDA_FAIR ramps linearly from 0 over the first
    # LAMBDA_RAMP_FRAC of Phase 2, so the fairness pressure grows after the
    # Phase-2 representation has stabilized (guards against collapse).
    'LAMBDA_RAMP_FRAC': 0.25,
    # Marginal guard: KL(real || synthetic) category shares (+ numeric
    # quantiles), batch-coupled (pass 2 under DP). Stops the fairness terms
    # from deleting sex-skewed categories ('levelling down').
    'MARGINAL_WEIGHT': 0.0,
    # Joint decoded-output stage: CKA^T + mean gap on the whole decoded row
    # (RFF-lifted), weighted by the same ramped lambda. Extends the paper's
    # per-column detokenizer stage, which cannot see joint distributions.
    'DECODED_JOINT': False,
    # How the disentanglement stages combine: 'mean' or 'sum' (full lambda
    # per stage, a per-stage reading of Eq. 10).
    'STAGE_REDUCE': 'mean',
    # CKA^T centering: 'group' (per-group, dim=0; paired with
    # MEAN_SHIFT_WEIGHT - the final Adult setup) or 'row' (across
    # activations, dim=1 - paper-literal Eqs. 5-6). See linear_cka_t.
    'CKA_CENTER': 'group',
    # Detokenizer-stage numerical input: 'summed' (the numerical outputs as
    # one block) or 'pre' (each feature's pre-aggregation logits
    # h_num * weight, as Sec. 4.3.2 describes).
    'NUM_STAGE': 'summed',
    # SWD divergence penalty: 'latent' (vs a frozen Phase-1 encoder) or
    # 'latent+decoder' (vs a frozen Phase-1 VAE, Sec. 4.3.2).
    'DIV_STAGES': 'latent',

    # L_S variant: 'per_sample' (DP-safe) or 'batch_mean' (paper Eq. 3).
    # LS_VARIANT below is the same switch (batch_mean if either says so).
    'LS_MODE': 'per_sample',
    # Keep the protected attribute in the reconstruction CE (paper-literal).
    # Default False: this repo's random-assignment design.
    'RECON_S': False,

    # Epochs of quality-only training before disentanglement starts.
    # More Phase-1 epochs -> better representation first, gentler
    # fairness intervention afterwards.
    'PHASE1_EPOCHS': 100,
    'PHASE2_EPOCHS': 60,

    # ---- PRIVACY knobs --------------------------------------------------
    # DP-SGD Gaussian noise multiplier (sigma). Higher sigma -> stronger
    # (epsilon, delta) privacy guarantee, but noisier gradients -> lower
    # fidelity, and per the paper, noise also weakens the fairness
    # intervention's effectiveness (Sec. 6.3).
    'NOISE_MULTIPLIER': 0.8,

    # Per-sample gradient L2 clipping norm. Smaller -> tighter privacy
    # accounting per step but slower/harder optimization.
    'MAX_GRAD_NORM': 1.0,

    # Target delta for the (epsilon, delta)-DP guarantee.
    'DP_DELTA': 1e-5,

    # Master switch for DP-SGD. Turning this off isolates the fairness
    # intervention's effect without privacy noise.
    'USE_DP': True,

    # ---- FIDELITY knobs -------------------------------------------------
    # KL-divergence weight (beta-VAE). Lower beta -> better reconstruction
    # (fidelity) at the cost of a less regular latent space.
    'BETA': 1e-3,
    # 'fixed' uses BETA throughout. 'decay' follows the paper (Sec. 4.3.1,
    # after TabSyn): start at BETA_MAX and multiply by BETA_DECAY every
    # BETA_DECAY_EVERY epochs, floored at BETA_MIN. TabSyn triggers the
    # decay on a training-loss plateau; that reads raw training data
    # outside the DP accounting, so this schedule is fixed in advance.
    # 'plateau' is TabSyn's version (start at BETA_MAX, multiply by
    # BETA_LAMBDA after BETA_PATIENCE epochs without a lower Phase-1 epoch
    # loss) - used by run_adult_flip.py, non-DP only.
    'BETA_SCHEDULE': 'fixed',
    'BETA_MAX': 1e-2,
    'BETA_MIN': 1e-5,
    'BETA_DECAY': 0.7,
    'BETA_DECAY_EVERY': 100,
    'BETA_LAMBDA': 0.7,
    'BETA_PATIENCE': 10,
    # Uniform attribute loss L_S. 'batch_mean' is the paper's Eq. 3: only
    # the batch-average softmax of S must be uniform, so rows keep a
    # definite group. It couples the batch, so under DP it goes through
    # the GroupLevelDPMechanism in BOTH phases. 'per_sample' pushes every
    # row to uniform (DP-decomposable, but argmax decoding then collapses
    # the synthetic S column to one group).
    'LS_VARIANT': 'per_sample',
    # Path of a Phase-1 checkpoint (non-DP only): written at the phase
    # boundary; if it exists, Phase 1 is skipped. None disables it.
    'PHASE1_CHECKPOINT': None,

    # ---- STABILITY knobs -------------------------------------------------
    # Cap on the SWD divergence penalty. Without this, the penalty grows
    # unboundedly as the Phase-2 encoder drifts from the frozen reference
    # (compounded by DP-noise gradient perturbations), which can explode
    # the loss and destabilize long runs.
    'MAX_DIV_PENALTY': 10.0,
    # Group-level DP mechanism knobs (Phase-2 batch-coupled fairness
    # terms). The SWD anchor and CKA^T disentanglement are batch-level
    # functionals: one record changes every other record's gradient, so
    # record-level DP-SGD cannot cover them. Instead their gradient is
    # clipped to MAX_GROUP_GRAD_NORM (bounding sensitivity to any one
    # protected group) and noised at GROUP_NOISE_MULTIPLIER, giving a
    # GROUP-level (epsilon, delta) guarantee, accounted separately from
    # the record-level DP-SGD budget and reported alongside it.
    'MAX_GROUP_GRAD_NORM': 0.5,
    'GROUP_NOISE_MULTIPLIER': 3.0,

    # DP release of generation-time statistics (latent mean/std and the
    # numeric clamp bounds). Computing these directly on training data
    # would be a fresh non-private computation on raw records (min/max
    # are especially sensitive). Each release is a Gaussian mechanism
    # with row contributions clipped to STATS_MAX_NORM and noise
    # STATS_NOISE_MULTIPLIER, spent once (not per step) into the budget.
    'STATS_MAX_NORM': 10.0,
    'STATS_NOISE_MULTIPLIER': 1.0,
    # Multiple of released std for the DP range proxy (mean +/- k*std);
    # order statistics (true min/max) cannot be released privately with
    # bounded sensitivity.
    'STATS_RANGE_K': 4.0,
    # Learning-rate scale applied when Phase 2 begins. Smaller Phase-2
    # steps keep the fairness intervention from destroying the Phase-1
    # representation (the paper's 'controlled bias mitigation').
    'PHASE2_LR_SCALE': 0.2,

    # Adam's epsilon. DP-SGD noise gives every parameter a nonzero
    # gradient variance even when the true signal is ~0; Adam's default
    # eps=1e-8 lets 1/sqrt(v) blow up on that noise. A larger eps damps
    # this and is the main fix for exploding loss under DP-SGD.
    'ADAM_EPS': 1e-4,

    # Hard cap on the aggregated (post-clip, post-noise) gradient norm
    # right before the optimizer step - a safety net beyond Opacus's
    # per-sample clipping, which only bounds each sample's contribution,
    # not the noisy sum actually applied to the weights.
    'MAX_OPTIMIZER_GRAD_NORM': 5.0,

    # L2 weight decay. DP-SGD injects Gaussian noise into every gradient
    # step regardless of the true signal, which makes weights undergo an
    # unbounded random walk over many epochs. Weight decay adds a
    # restoring force pulling weights back toward zero, which is the
    # standard fix for this drift (gradient clipping alone does not help,
    # since large logits saturate softmax and keep the CE gradient small
    # even while the loss *value* explodes).
    'WEIGHT_DECAY': 1e-2,

    # Clamp applied to reconstructed categorical logits before the
    # cross-entropy loss, as a hard backstop against the same failure
    # mode: even one drifted weight can produce a huge logit whose CE
    # loss value is enormous despite a small, saturated gradient.
    'LOGIT_CLAMP': 15.0,

    # Clamp applied to the reconstructed numeric features before the MSE
    # loss. Unlike CE, squared error does NOT saturate - its gradient
    # grows linearly with the error - so a drifted weight here keeps
    # feeding a larger and larger gradient back in. This closes that loop.
    'NUM_RECON_CLAMP': 8.0,

    # ---- Data / compute knobs -------------------------------------------
    'SAMPLE_SIZE': 12000,      # rows drawn from the preprocessed CSV
    'BATCH_SIZE': 1024,        # split evenly across protected groups
    # Sampler: 'poisson' (default) makes each record's inclusion
    # independent per step, which is exactly what Opacus's subsampled-
    # Gaussian accounting assumes. 'balanced' uses the shuffle-based
    # BalancedGroupSampler (fixed-size group-balanced batches, every
    # minority sample once per epoch) - Chua et al. (ICML 2024) showed
    # shuffle-based DP-SGD can leak more than Poisson accounting
    # reports, so the Poisson accounting is NOT formally valid for it.
    'SAMPLER': 'poisson',
    # Poisson steps per epoch (only used when SAMPLER='poisson').
    # Match the balanced sampler's epoch length: ceil(m / per_group).
    'POISSON_STEPS_PER_EPOCH': None,  # None -> ceil(m / per_group)
    'D_TOKEN': 8,
    'NUM_LAYERS': 2,
    'FACTOR': 8,               # transformer FFN width factor (TabSyn: 32)
    'LR': 1e-3,
    'SEED': 42,

    # 'cuda' if available, else 'cpu' (set in main())
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
}

SENSITIVE_COL = 'derived_race'  # protected attribute - most biased in this
                               # data (White 0.61 vs minority ~0.32-0.43
                               # origination rates; DP ratio ~0.53)
TARGET_COL = 'target'          # 1 = originated, 0 = denied (from preprocess.py)

# Use the three largest race groups so balanced batches have enough
# samples per group (Joint / Not Available / tiny groups excluded)

NUMERIC_COLS = ['loan_amount', 'income', 'loan_to_value_ratio',
                'property_value', 'tract_population']
CATEGORICAL_COLS = [SENSITIVE_COL, TARGET_COL, 'conforming_loan_limit',
                    'derived_loan_product_type']


# ==========================================================================
# Data preparation
# ==========================================================================


# ==========================================================================
# FLIP training (two-phase, balanced sampling, optional DP-SGD)
# ==========================================================================
def beta_at(epoch, params):
    """KL weight for this epoch under BETA_SCHEDULE 'fixed' or 'decay'.
    ('plateau' is stateful and handled inside train_flip.)"""
    if params.get('BETA_SCHEDULE', 'fixed') != 'decay':
        return params['BETA']
    steps = epoch // params['BETA_DECAY_EVERY']
    return max(params['BETA_MAX'] * params['BETA_DECAY'] ** steps,
               params['BETA_MIN'])


def _group_clip_optimizer_cls():
    """DPOptimizer with a per-sample clipping norm (Eq. 13 / IDP-SGD):
    record i is clipped to sample_max_norms[i] instead of max_grad_norm;
    the noise stays noise_multiplier * max_grad_norm."""
    from opacus.optimizers import DPOptimizer
    from opacus.optimizers.optimizer import (_check_processed_flag,
                                             _mark_as_processed)

    class GroupClipDPOptimizer(DPOptimizer):
        sample_max_norms = None

        def clip_and_accumulate(self):
            if self.sample_max_norms is None or len(self.grad_samples[0]) == 0:
                return super().clip_and_accumulate()
            norms = torch.stack([g.reshape(len(g), -1).norm(2, dim=-1)
                                 for g in self.grad_samples], 1).norm(2, dim=1)
            factor = (self.sample_max_norms.to(norms.device)
                      / (norms + 1e-6)).clamp(max=1.0)
            for p in self.params:
                _check_processed_flag(p.grad_sample)
                gs = self._get_flat_grad_sample(p).to(p.dtype)
                g = torch.einsum('i,i...', factor.to(gs.device, p.dtype), gs)
                p.summed_grad = g if p.summed_grad is None else p.summed_grad + g
                _mark_as_processed(p.grad_sample)

    return GroupClipDPOptimizer


def train_flip(X_num, X_cat, categories, s_idx, params, device,
               phase1_checkpoint=None, factor=None):
    """Two-phase FLIP VAE training (balanced sampling, optional DP-SGD).

    Two ways to reuse Phase 1:
      * phase1_checkpoint (argument): load a banked model state_dict (from
        torch.save(vae.state_dict())) and skip the Phase-1 epochs; the
        sampler, optimizer/Opacus wrapping and frozen reference are still
        built and Phase 2 still runs - per-setting sweeps pay only for
        Phase 2.
      * params['PHASE1_CHECKPOINT'] (non-DP only): model + optimizer state,
        written at the phase boundary and loaded when the file exists.
    """
    torch.manual_seed(params['SEED'])
    n, d_numerical = X_num.shape

    factor = factor if factor is not None else params.get('FACTOR', 8)
    model = Model_VAE(params['NUM_LAYERS'], d_numerical, categories,
                      params['D_TOKEN'], n_head=1, factor=factor,
                      bias=True).to(device)
    if phase1_checkpoint and os.path.exists(phase1_checkpoint):
        model.load_state_dict(torch.load(phase1_checkpoint,
                                         map_location=device))
        print(f'loaded Phase-1 checkpoint: {phase1_checkpoint}')

    groups = X_cat[:, s_idx].long().numpy()
    per_group = params['BATCH_SIZE'] // len(np.unique(groups))
    if params.get('SAMPLER', 'poisson') == 'poisson':
        # Per-group Poisson sampling: each record included independently
        # with per-group probability q_g = per_group / m_g. This makes the
        # Poisson sampling assumption of Opacus's RDP bound exactly true
        # (the shuffle-based sampler violates it; Chua et al. ICML 2024).
        m = int(np.bincount(groups).min())
        steps_per_epoch = params.get('POISSON_STEPS_PER_EPOCH') or \
            int(np.ceil(m / per_group))
        sampler = PoissonGroupSampler(groups, per_group,
                                       steps_per_epoch,
                                       seed=params['SEED'])
        print(f'Poisson sampling: {steps_per_epoch} steps/epoch, '
              f'gamma_max={sampler.gamma_max:.4f}')
    else:
        sampler = BalancedGroupSampler(groups, params['BATCH_SIZE'],
                                       seed=params['SEED'])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_num, X_cat),
        batch_sampler=sampler)

    accountant = None
    group_mech = None
    if params['USE_DP']:
        from opacus.optimizers import DPOptimizer
        from opacus.grad_sample import GradSampleModule
        register_tabsyn_grad_samplers()
        model = GradSampleModule(model, strict=False)
        dp_params = get_dp_trainable_parameters(model)
        # Eq. 13 (GROUP_SIGMAS, one noise multiplier per group code): the
        # global multiplier and per-group clipping c_p = C*sigma_g/sigma_p
        group_sigmas = params.get('GROUP_SIGMAS')
        if group_sigmas:
            counts = np.bincount(groups, minlength=len(group_sigmas))
            params['NOISE_MULTIPLIER'] = eq13_sigma_global(group_sigmas,
                                                           counts)
            clip_scale = torch.tensor(
                [params['NOISE_MULTIPLIER'] / s_ for s_ in group_sigmas],
                dtype=torch.float32)
            print(f"Eq. 13: group sigmas {group_sigmas} -> sigma_global "
                  f"{params['NOISE_MULTIPLIER']:.3f}, clip scale "
                  f"{clip_scale.tolist()}")
            DPOptimizer = _group_clip_optimizer_cls()
        optimizer = DPOptimizer(
            optimizer=torch.optim.Adam(dp_params, lr=params['LR'],
                                       eps=params['ADAM_EPS'],
                                       weight_decay=params['WEIGHT_DECAY']),
            noise_multiplier=params['NOISE_MULTIPLIER'],
            max_grad_norm=params['MAX_GRAD_NORM'],
            expected_batch_size=params['BATCH_SIZE'])
        # Sampling rate for the accountant: gamma_max, the largest
        # per-record inclusion probability across groups. Under the
        # Poisson sampler this is exactly the minority group's q; under
        # the balanced sampler it is the conservative per-step rate
        # (every minority sample appears once per epoch). Either way it
        # is the rate that must be accounted, not batch/N.
        if isinstance(sampler, PoissonGroupSampler):
            gamma_max = sampler.gamma_max
        else:
            m_minority = int(np.bincount(groups).min())
            gamma_max = per_group / m_minority
        if group_sigmas:
            # per-group Poisson rates q_p = per_group / m_p (Eq. 12)
            rates = [per_group / c for c in counts]
            accountant = GroupwiseRDPAccountant(group_sigmas, rates,
                                                params['DP_DELTA'])
        else:
            accountant = RDPAccountant(
                noise_multiplier=params['NOISE_MULTIPLIER'],
                sample_rate=gamma_max,
                delta=params['DP_DELTA'])
        # Gaussian mechanism for the batch-coupled fairness terms
        # (SWD anchor + CKA^T): their gradients cannot be decomposed per
        # sample, so record-level DP-SGD does not cover them. Clipping
        # bounds the sensitivity (2C under substitution); the noise
        # gives a record-level guarantee. Under the Poisson sampler the
        # subsampling amplification applies to this mechanism too, so
        # gamma_max is passed for the amplified accounting (the largest
        # remaining win: the fairness term drops several-fold).
        group_mech = GroupLevelDPMechanism(
            max_group_grad_norm=params['MAX_GROUP_GRAD_NORM'],
            noise_multiplier=params['GROUP_NOISE_MULTIPLIER'],
            sample_rate=gamma_max if params.get('SAMPLER', 'poisson') == 'poisson' else None)
        # COUPLED_DP='paper': the batch-coupled terms go through DP-SGD's
        # per-sample clipping with the rest of the loss (one backward), as
        # the paper most plausibly did. Each record's own share is clipped,
        # but its influence on the OTHER records' shares (via the batch
        # mean / group matrices) is not, so epsilon is nominal for them.
        if params.get('COUPLED_DP', 'group') == 'paper':
            group_mech = None
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'],
                                     eps=params['ADAM_EPS'],
                                     weight_decay=params['WEIGHT_DECAY'])

    raw = model._module if params['USE_DP'] else model
    # A banked Phase-1 model skips its Phase-1 epochs; Phase 2 always runs.
    phase1_epochs = 0 if (phase1_checkpoint
                          and os.path.exists(phase1_checkpoint)) \
        else params['PHASE1_EPOCHS']
    total_epochs = phase1_epochs + params['PHASE2_EPOCHS']
    reference_encoder = None  # frozen Phase-1 encoder (DIV_STAGES='latent')
    reference_vae = None      # frozen Phase-1 VAE ('latent+decoder')
    # L_S: 'batch_mean' (paper Eq. 3) if either LS_MODE or LS_VARIANT says
    # so. The batch mean couples the batch, so it is routed through the
    # batch-coupled pass (GroupLevelDPMechanism under DP) in BOTH phases;
    # without DP the gradient is the same as adding it to pass 1.
    ls_batch = 'batch_mean' in (params.get('LS_MODE'),
                                params.get('LS_VARIANT'))
    div_decoder = params.get('DIV_STAGES', 'latent') == 'latent+decoder'

    # Beta: 'fixed' / 'decay' (fixed-interval, DP-safe) via beta_at, or
    # 'plateau' (TabSyn): start at BETA_MAX, multiply by BETA_LAMBDA after
    # BETA_PATIENCE epochs without a lower epoch loss, floored at BETA_MIN.
    # The plateau decision reads the training loss outside the DP accounting.
    plateau = params.get('BETA_SCHEDULE', 'fixed') == 'plateau'
    beta = params['BETA_MAX'] if plateau else beta_at(0, params)
    best_loss, stall = float('inf'), 0

    # Optional PHASE1_CHECKPOINT (model + optimizer state, non-DP only):
    # saved at the phase boundary; when the file already exists Phase 1 is
    # skipped. Skipping Phase 1 under DP would also skip its accounting.
    ckpt = params.get('PHASE1_CHECKPOINT')
    start_epoch = 0
    if ckpt:
        if params['USE_DP']:
            raise ValueError('PHASE1_CHECKPOINT is only supported with USE_DP=False')
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            raw.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = phase1_epochs
            print(f'Loaded Phase-1 checkpoint {ckpt}; starting at epoch {start_epoch}')

    for epoch in range(start_epoch, total_epochs):
        phase = 1 if epoch < phase1_epochs else 2
        if ckpt and start_epoch == 0 and epoch == phase1_epochs:
            save_phase1_checkpoint(ckpt, raw, optimizer)
        if not plateau:
            beta = beta_at(epoch, params)
        # PHASE2_BETA: the KL weight of Phase 2's ELBO, fixed (e.g. the
        # banked Phase 1's final beta); otherwise a banked run restarts the
        # schedule at BETA_MAX, since a bank stores only the weights
        if phase == 2 and params.get('PHASE2_BETA') is not None:
            beta = params['PHASE2_BETA']

        # Phase-2 LR drop: shrink the step size once disentanglement
        # starts so fairness updates perturb the quality representation
        # gently instead of destroying it.
        if phase == 2 and epoch == phase1_epochs:
            new_lr = params['LR'] * params['PHASE2_LR_SCALE']
            for g in optimizer.param_groups:
                g['lr'] = new_lr
            print(f'--- Phase 2: LR set to {new_lr:.2e} ---')

        epoch_loss = 0.0
        n_batches = 0
        cur_beta = beta

        for bnum, bcat in loader:
            if params['USE_DP'] and params.get('GROUP_SIGMAS'):
                optimizer.sample_max_norms = (
                    params['MAX_GRAD_NORM']
                    * clip_scale[bcat[:, s_idx].long().cpu()])
            bnum, bcat = bnum.to(device), bcat.to(device)
            model.train()
            optimizer.zero_grad()

            stages = raw.forward_with_stages(bnum, bcat)
            # Clamp reconstructions before the loss: a single drifted
            # weight can send a logit/numeric output to a huge value.
            # CE saturates on huge logits (small gradient despite a huge
            # loss value), but MSE does not, so both need a backstop -
            # see WEIGHT_DECAY / NUM_RECON_CLAMP notes above.
            clamp = params['LOGIT_CLAMP']
            recon_cat_clamped = [torch.clamp(c, -clamp, clamp)
                                 for c in stages['recon_x_cat']]
            recon_num_clamped = torch.clamp(
                stages['recon_x_num'], -params['NUM_RECON_CLAMP'],
                params['NUM_RECON_CLAMP'])
            mse, ce, kld, acc, loss_s = compute_loss(
                bnum, bcat, recon_num_clamped, recon_cat_clamped,
                stages['mu_z'], stages['std_z'], s_idx=s_idx,
                Recon_S_logits=recon_cat_clamped[s_idx],
                ls_mode='per_sample',
                recon_s=params.get('RECON_S', False))

            # ---- Pass 1: per-sample-decomposable loss (record-level DP) --
            # mse + ce + kld (+ the per-sample L_S) are means of per-sample
            # terms, so Opacus's per-sample clipping bounds each record's
            # influence on this gradient.
            # With the batch-mean L_S (pass 2), the per-sample one stays at
            # weight 0: S has no CE term, so without it the S head would
            # get no per-sample gradient and Opacus's pre_step() raises.
            # Phase 2 keeps L_quality (ELBO + L_S): the paper uses it as the
            # detokenizer-stage divergence penalty (Sec. 4.3.2).
            # PHASE2_RECON_WEIGHT scales its reconstruction part in Phase 2
            # (unstated in the paper; 1 = as in Phase 1).
            w_rec = (params.get('PHASE2_RECON_WEIGHT', 1.0) if phase == 2
                     else 1.0)
            loss_private = w_rec * (mse + ce) + cur_beta * kld \
                + (0.0 if ls_batch else 1.0) * loss_s
            has_coupled = ls_batch or phase == 2
            # paper-style DP: one combined backward through Opacus below
            paper_dp = (params['USE_DP'] and
                        params.get('COUPLED_DP', 'group') == 'paper')
            if not (paper_dp and has_coupled):
                loss_private.backward(retain_graph=has_coupled)

            # ---- Pass 2: batch-coupled terms (group-level DP) -----------
            # The paper's batch-mean L_S (Eq. 3), the SWD anchor and CKA^T
            # disentanglement mix samples across the batch; their gradient
            # cannot be attributed per sample. Under DP they are handled by
            # the GroupLevelDPMechanism (clip to a fixed norm, add
            # calibrated noise) instead of Opacus.
            fair_grads = None
            loss_coupled = (uniform_attribute_loss(recon_cat_clamped[s_idx])
                            if ls_batch else None)
            if phase == 2:
                mu_flat = stages['mu_z'].reshape(bnum.shape[0], -1)
                cap = params['MAX_DIV_PENALTY']
                # Divergence penalty vs. the frozen Phase-1 model (Eq. 10).
                # The snapshot is taken at the phase boundary into a fresh
                # module via state_dict (a deepcopy would also copy Opacus's
                # per-sample hooks). Each stage's penalty is clamped: without
                # a cap the SWD term grows unboundedly as the model drifts
                # from the reference under DP noise, exploding the loss.
                if div_decoder:
                    # Sec. 4.3.2: SWD at the latent AND decoder stages
                    if reference_vae is None:
                        reference_vae = Model_VAE(
                            params['NUM_LAYERS'],
                            model.Reconstructor.d_numerical,
                            categories, params['D_TOKEN'],
                            n_head=1, factor=factor, bias=True).to(device)
                        reference_vae.load_state_dict(raw.state_dict())
                        reference_vae.eval()
                        for p_ in reference_vae.parameters():
                            p_.requires_grad_(False)
                    with torch.no_grad():
                        ref_h, ref_mu, ref_lv = reference_vae.VAE(bnum, bcat)
                    h_flat = stages['h'].reshape(bnum.shape[0], -1)
                    n_b = bnum.shape[0]
                    if params.get('SWD_LATENT', 'mu') == 'z':
                        # samples of q(z|x) (Eq. 10 is written over the
                        # posterior), current and reference
                        cur_lat = (stages['mu_z'] + torch.exp(
                            0.5 * stages['std_z'])
                            * torch.randn_like(stages['mu_z'])).reshape(n_b, -1)
                        ref_lat = (ref_mu + torch.exp(0.5 * ref_lv)
                                   * torch.randn_like(ref_mu)).reshape(n_b, -1)
                    else:
                        cur_lat, ref_lat = mu_flat, ref_mu.reshape(n_b, -1)
                    ref_hf = ref_h.reshape(n_b, -1)
                    n_proj = params.get('SWD_PROJECTIONS', 50)
                    div = 0.5 * (
                        torch.clamp(sliced_wasserstein_distance(
                            ref_lat, cur_lat, n_projections=n_proj), max=cap)
                        + torch.clamp(sliced_wasserstein_distance(
                            ref_hf, h_flat,
                            n_projections=n_proj), max=cap))
                else:
                    # latent stage only (frozen encoder)
                    if reference_encoder is None:
                        from tabsyn.vae.model import Encoder_model
                        reference_encoder = Encoder_model(
                            params['NUM_LAYERS'],
                            model.Reconstructor.d_numerical,
                            categories, params['D_TOKEN'],
                            n_head=1, factor=factor).to(device)
                        reference_encoder.Tokenizer.load_state_dict(
                            raw.VAE.Tokenizer.state_dict())
                        reference_encoder.VAE_Encoder.load_state_dict(
                            raw.VAE.encoder_mu.state_dict())
                        reference_encoder.eval()
                        for p_ in reference_encoder.parameters():
                            p_.requires_grad_(False)
                    with torch.no_grad():
                        ref_mu = reference_encoder(bnum, bcat)
                    div = torch.clamp(sliced_wasserstein_distance(
                        ref_mu.reshape(bnum.shape[0], -1), mu_flat,
                        n_projections=params.get('SWD_PROJECTIONS', 50)),
                        max=cap)
                disent = multi_stage_disentanglement_loss(
                    stages, bcat[:, s_idx].long(), s_idx=s_idx,
                    mean_weight=params.get('MEAN_SHIFT_WEIGHT', 1.0),
                    binary_w1=params.get('BINARY_W1', True),
                    t_idx=params.get('TARGET_IDX'),
                    target_in_fairness=params.get('TARGET_IN_FAIRNESS',
                                                   False),
                    reduce=params.get('STAGE_REDUCE', 'mean'),
                    center=params.get('CKA_CENTER', 'group'),
                    num_stage=params.get('NUM_STAGE', 'summed'),
                    tokens=params.get('CKA_TOKENS', 'concat'),
                    kernel=params.get('CKA_KERNEL', 'linear'))
                # lambda ramp over the first LAMBDA_RAMP_FRAC of Phase 2
                # (0 = no ramp: full lambda from the first Phase-2 epoch)
                ramp = params.get('LAMBDA_RAMP_FRAC', 0.25)
                p2_frac = (epoch - phase1_epochs) / max(1, params['PHASE2_EPOCHS'])
                lam_eff = params['LAMBDA_FAIR'] * (
                    1.0 if ramp <= 0 else min(1.0, p2_frac / ramp))
                marg_pen = marginal_matching_penalty(
                    stages, bnum, bcat) \
                    if params.get('MARGINAL_WEIGHT', 0) else torch.tensor(0.0,
                        device=bnum.device)
                # DIV_WEIGHT: the divergence anchor's weight (implicit 1 in
                # Eq. 10)
                loss_fair = params.get('DIV_WEIGHT', 1.0) * div \
                    + lam_eff * disent \
                    + params.get('MARGINAL_WEIGHT', 0) * marg_pen
                if params.get('DECODED_JOINT'):
                    loss_fair = loss_fair + lam_eff * decoded_joint_loss(
                        stages, bcat[:, s_idx].long(), s_idx,
                        t_idx=None if params.get('TARGET_IN_FAIRNESS')
                        else params.get('TARGET_IDX'),
                        mean_weight=params.get(
                            'JOINT_MEAN_WEIGHT',
                            params.get('MEAN_SHIFT_WEIGHT', 1.0)))
                loss_coupled = (loss_fair if loss_coupled is None
                                else loss_coupled + loss_fair)

            if has_coupled:
                if paper_dp:
                    (loss_private + loss_coupled).backward()
                elif group_mech is not None:
                    # Capture the fairness gradient WITHOUT Opacus's hooks:
                    # they would try to pop activation buffers already
                    # consumed by pass 1 (IndexError) and misattribute the
                    # batch-coupled gradient per sample. disable_hooks() -
                    # NOT remove_hooks(), which is permanent in this Opacus
                    # version - makes the capture hooks return early while
                    # staying registered; enable_hooks() restores them for
                    # the next batch's pass 1. The captured gradient is
                    # clipped+noised by the group-level mechanism and added
                    # to the aggregate after Opacus's pre_step() below.
                    model.disable_hooks()
                    try:
                        fair_grads = torch.autograd.grad(
                            loss_coupled, dp_params, allow_unused=True)
                    finally:
                        model.enable_hooks()
                    fair_grads = {
                        p_: g for p_, g in zip(dp_params, fair_grads)
                        if g is not None}
                else:
                    # Non-DP: accumulate into the same graph
                    loss_coupled.backward()

            # Safety-net clip on the aggregated update: Opacus's
            # max_grad_norm only bounds each per-sample gradient before
            # summation/noising, not the final noisy gradient Adam sees.
            if params['USE_DP']:
                if optimizer.pre_step():
                    # Add the group-level noised fairness gradient to the
                    # record-level aggregate so one step applies both.
                    if fair_grads:
                        group_mech.add_noised(dp_params, fair_grads)
                    torch.nn.utils.clip_grad_norm_(
                        dp_params, params['MAX_OPTIMIZER_GRAD_NORM'])
                    optimizer.original_optimizer.step()
            else:
                # MAX_OPTIMIZER_GRAD_NORM None/0: no clip (TabSyn and the
                # paper clip nothing without DP)
                if params.get('MAX_OPTIMIZER_GRAD_NORM'):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), params['MAX_OPTIMIZER_GRAD_NORM'])
                optimizer.step()
            if accountant is not None:
                accountant.step()
            epoch_loss += (loss_private + (loss_coupled if has_coupled else 0)).item()
            n_batches += 1
            # Per-term Phase-2 diagnostics: is the fairness gradient drowned
            # out by the marginal/SWD terms? Logged every 100 epochs.
            if phase == 2 and epoch % 100 == 0 and bnum.shape[0] > 1:
                with torch.no_grad():
                    mu_flat_d = stages['mu_z'].reshape(bnum.shape[0], -1)
                    g = bcat[:, s_idx].long()
                    blocks = [mu_flat_d[torch.where(g == u)[0]]
                              for u in torch.unique(g)]
                    cka_val = float(np.mean([
                        linear_cka_t(blocks[i], blocks[j]).item()
                        for i in range(len(blocks))
                        for j in range(i + 1, len(blocks))])) \
                        if len(blocks) > 1 else float('nan')
                    print(f'    terms: recon={loss_private.item():.4f} '
                          f'div={div.item():.4f} disent={disent.item():.4f} '
                          f'cka_t={cka_val:.4f} marg={float(marg_pen):.4f} '
                          f'lam_eff={lam_eff:.2f}')

        # BETA_SCHEDULE='plateau' (TabSyn): after BETA_PATIENCE epochs without
        # a lower epoch loss, beta *= BETA_LAMBDA (>= BETA_MIN).
        if plateau and phase == 1:
            if epoch_loss < best_loss:
                best_loss, stall = epoch_loss, 0
            else:
                stall += 1
                if stall >= params.get('BETA_PATIENCE', 10) \
                        and beta > params.get('BETA_MIN', 1e-5):
                    beta *= params.get('BETA_LAMBDA', 0.7)
                    stall = 0
                    print(f'--- beta decayed to {beta:.2e} ---')
        if phase == 1:
            params['PHASE1_FINAL_BETA'] = float(beta)  # banked with the model

        if epoch % 10 == 0 or epoch == total_epochs - 1:
            msg = f'epoch {epoch:3d} [phase {phase}] loss={epoch_loss/max(1,n_batches):.4f}  beta={beta:.1e}'
            if accountant is not None:
                eps, alpha = accountant.get_privacy_spent()
                msg += f'  eps_record={eps:.2f}'
            if group_mech is not None and group_mech.steps > 0:
                geps, _ = group_mech.get_privacy_spent(params['DP_DELTA'])
                msg += f'  eps_group={geps:.2f}'
            print(msg)

    if ckpt and start_epoch == 0 and params['PHASE2_EPOCHS'] == 0:
        save_phase1_checkpoint(ckpt, raw, optimizer)  # loop never reached Phase 2
    return raw, accountant, group_mech


def save_phase1_checkpoint(path, model, optimizer):
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, path)
    print(f'Saved Phase-1 checkpoint {path}')


# ==========================================================================
# Generation
# ==========================================================================


def clip_nonnegative(syn_num, num_means, num_stds):
    """Clamp standardized numerics at the standardized value of 0.

    syn_num is still in standardized units ((x - mean) / std), where 0 is
    the column MEAN, not zero - clipping at 0 there floors every column at
    its mean. Real-unit 0 is -mean/std in standardized units.
    """
    if num_means is None or num_stds is None:
        return syn_num
    floor = (-np.asarray(num_means, dtype=float)
             / np.asarray(num_stds, dtype=float))
    return np.maximum(syn_num, floor[None, :])


# ==========================================================================
# Metrics
# ==========================================================================
def fidelity_metrics(real_num, syn_num, real_cat, syn_cat, cat_encoders):
    """Column-wise distributional fidelity."""
    from scipy.stats import wasserstein_distance
    # Numeric: Wasserstein-1 normalized by real std (dimensionless)
    w1 = []
    for j in range(real_num.shape[1]):
        std = real_num[:, j].std() + 1e-8
        w1.append(wasserstein_distance(real_num[:, j], syn_num[:, j]) / std)
    # Categorical: total variation distance of frequency distributions
    tv = []
    for j, c in enumerate(CATEGORICAL_COLS):
        k = len(cat_encoders[c])
        p = np.bincount(real_cat[:, j], minlength=k) / len(real_cat)
        q = np.bincount(syn_cat[:, j], minlength=k) / len(syn_cat)
        tv.append(0.5 * np.abs(p - q).sum())
    # Correlation preservation (numeric columns)
    cr = np.corrcoef(real_num.T)
    cs = np.corrcoef(syn_num.T)
    corr_delta = float(np.abs(cr - cs).mean())
    return {'num_w1_mean': float(np.mean(w1)),
            'cat_tv_mean': float(np.mean(tv)),
            'corr_delta': corr_delta}


# ==========================================================================
# Synthetic data export
# ==========================================================================


def infer_numeric_schema(real_raw, names, max_decimals=6):
    """Per-column format of the REAL numeric data, for any dataset:
    'decimals' = the fewest decimal places (0..max_decimals) every real value
    uses (0 = whole numbers), 'lo'/'hi' = the real min / max.

    PRIVACY: decimals is a schema property (a minor disclosure). lo/hi are
    order statistics of real records and are NOT DP - under DP pass
    bounds=False to apply_numeric_schema (the decoder already clips to a
    DP-released range).
    """
    real_raw = np.asarray(real_raw, dtype=np.float64)
    schema = {}
    for i, name in enumerate(names):
        x = real_raw[:, i]
        dec = max_decimals
        for k in range(max_decimals + 1):
            # tolerance absorbs float error from de-standardization
            if np.allclose(x, np.round(x, k), rtol=0, atol=1e-3 * 10.0 ** -k):
                dec = k
                break
        schema[name] = {'decimals': dec, 'lo': float(x.min()),
                        'hi': float(x.max())}
    return schema


def apply_numeric_schema(df, schema, bounds=True):
    """Make synthetic numerics look like the real column: clip to the real
    range (bounds=True) and round to the real number of decimals - so whole-
    number columns (age, counts) come out whole, never negative where the
    real data never is, etc. Returns a copy; columns not in schema are kept."""
    df = df.copy()
    for name, s in schema.items():
        if name not in df:
            continue
        x = df[name].to_numpy(dtype=np.float64)
        if bounds:
            x = np.clip(x, s['lo'], s['hi'])
        x = np.round(x, s['decimals'])
        df[name] = x.astype(np.int64) if s['decimals'] == 0 else x
    return df


def decode_synthetic_data(syn_num, syn_cat, num_means, num_stds, cat_encoders,
                          num_schema=None, schema_bounds=True):
    """Reverse the normalization/label-encoding from load_sample() so the
    synthetic array is human-readable in the same units/labels as the
    source CSV. With num_schema (infer_numeric_schema on the real training
    data), numerics are clipped and rounded to the real columns' format."""
    num_denorm = syn_num * num_stds.values[None, :] + num_means.values[None, :]
    df = pd.DataFrame(num_denorm, columns=NUMERIC_COLS)
    if num_schema is not None:
        df = apply_numeric_schema(df, num_schema, bounds=schema_bounds)
    for j, col in enumerate(CATEGORICAL_COLS):
        uniques = cat_encoders[col]
        codes = np.clip(syn_cat[:, j], 0, len(uniques) - 1)
        df[col] = uniques[codes]
    return df


# ==========================================================================
# HTML report
# ==========================================================================


# ==========================================================================
# Main
# ==========================================================================

