"""Run FLIP (VAE + latent diffusion) on UCI Adult with `sex` as the protected
attribute - the paper's Adult setup (arXiv:2508.21815, Table 2: 6 numeric +
8 categorical columns) - and score it with the paper's metrics and the
checks added in this reproduction.

One run = one 3-fold-CV fold: the generator trains on 2/3 of the rows, the
held-out 1/3 is the unseen real reference. Reuses the HMDA pipeline's
training / diffusion / decoding code by pointing run_flip_demo's column
configuration at Adult. See README Part 10.

Default (--preset paper): FLIP as published.
  Phase 1  ELBO (reconstruction + beta*KL, beta decayed TabSyn-style) + the
           batch-mean L_S (Eq. 3), balanced batches of 4096, 3000 epochs
  Phase 2  L_quality (ELBO + L_S, the detokenizer-stage divergence term)
           + SWD to the Phase-1 model at the latent and decoder stages
           + lambda * -CKA^T (linear, row-centred, Eq. 7) at the latent,
           detokenizer (per-feature mean) and decoder stages; stages
           averaged; no lambda ramp
  Stage 2  latent diffusion on reparameterized latents mu + sigma*eps
  Decode   argmax for every categorical column, sex included
  Choices the paper does not state: Phase-2 beta 1e-2 (--phase2-beta),
  100 Phase-2 epochs, stages averaged (--stage-reduce).

--preset ours: this repository's additions (the earlier final setup) -
  masked proxies, group-centred CKA^T + mean gap, binary W1, summed
  numerics, SWD at the latent only, lambda ramp, the RFF joint
  decoded-output stage, the KL marginal guard, summed stages, sampled sex.
--preset final (recommended): the paper preset plus the RFF joint
  decoded-output stage with the mean gap and the KL marginal guard; income
  stays in the fairness terms (task-agnostic).
Any individual flag overrides the preset.

Examples:
    # bank Phase 1 for a fold (tier-1 metrics of the Phase-1 model)
    python run_adult_flip.py --fold 0 --phase2-epochs 0 --tier1 \\
        --save-phase1 adult_results/ckpt/raw_f0.pt --tag raw_f0_bank
    # Phase 2 at lambda=4 from the bank, full pipeline (the paper)
    python run_adult_flip.py --fold 0 --lam 4 \\
        --phase1-checkpoint adult_results/ckpt/raw_f0.pt --tag pgrid_f0_lam4
    # this repository's additions (masked bank: --save-phase1 with --preset ours)
    python run_adult_flip.py --preset ours --fold 0 --lam 4 \\
        --phase1-checkpoint adult_results/ckpt/fin_f0.pt --tag flip_f0_lam4
    # comparison baseline: scrambling repair, no in-model fairness
    python run_adult_flip.py ... --repair-tau 1.0 --lam 0 --tag rpb_f0_lam0
    # re-decode a finished run's saved VAE + latents with the other sex
    # decoding (scoring only; no training, no diffusion)
    python run_adult_flip.py --from-run pgrid_f0_lam4 --s-decode sample \\
        --tag pgrid_f0_lam4_sample
"""

import os
# Cap BLAS/OpenMP threads per process (the sklearn/numpy/torch CPU work in
# scoring). Sweeps run several processes at once, and each defaulting to
# every core oversubscribes the CPU. Must be set before numpy/torch load.
# Override with ADULT_NUM_THREADS. The GPU is unaffected.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, os.environ.get('ADULT_NUM_THREADS', '3'))
import sys
import copy
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

import run_flip_demo as demo
import run_flip_full_pipeline as full
from tabsyn.vae.flip_fairness import (compose_rdp_budgets, RDPAccountant,
                                      GroupwiseRDPAccountant)
from tabsyn.vae.paper_metrics import (fairness_scores, epsilon_identifiability,
                                      infer_protected, group_share_tv,
                                      categorical_share_tv_all,
                                      true_group_target_gap,
                                      downstream_sp_eo,
                                      alpha_precision_beta_recall,
                                      fit_validity_rules, validity_report,
                                      invalid_rows, fit_validity_rules_dp,
                                      constrained_repair)
from tabsyn.vae.proxy_decouple import (detect_proxy_classes,
                                       describe_classes, mask_proxy_classes,
                                       repair_toward_independence,
                                       impossible_combination_rate)

ADULT_NAMES = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
               'marital-status', 'occupation', 'relationship', 'race', 'sex',
               'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'income']
# 6 numeric + 8 categorical as in the paper's Table 2 ('education' is dropped
# as a redundant label of 'education-num').
NUMERIC = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
           'hours-per-week']
CATEGORICAL = ['sex', 'income', 'workclass', 'marital-status', 'occupation',
               'relationship', 'race', 'native-country']
SENSITIVE, TARGET = 'sex', 'income'

DATA_DIR = os.path.join(HERE, 'Data')
OUT_DIR = os.path.join(HERE, 'adult_results')

# Method presets; every key is also a command-line flag that overrides it.
PRESETS = {
    # FLIP as published (Sec. 4.3); the beta value is not stated there
    'paper': dict(cka_center='row', mean_gap_weight=0.0, binary_w1=False,
                  num_stage='pre', div_stages='latent+decoder',
                  lambda_ramp=0.0, decoded_joint=False, marginal_weight=0.0,
                  stage_reduce='mean', mask_proxies=False, s_decode='argmax',
                  phase2_beta='1e-2', target_in_fairness=True),
    # this repository's additions (README Part 10, EXPERIMENTS_ADULT.md)
    'ours': dict(cka_center='group', mean_gap_weight=1.0, binary_w1=True,
                 num_stage='summed', div_stages='latent', lambda_ramp=0.25,
                 decoded_joint=True, marginal_weight=1.0, stage_reduce='sum',
                 mask_proxies=True, s_decode='sample', phase2_beta='1e-2',
                 target_in_fairness=False),
    # recommended: the paper backbone (task-agnostic, income included) plus
    # a direct dependence penalty on the decoded output - the RFF joint stage
    # with the mean gap (an MMD between the groups' decoded rows) - and the
    # KL marginal guard against levelling down (README Part 10.5)
    'final': dict(cka_center='row', mean_gap_weight=1.0, binary_w1=False,
                  num_stage='pre', div_stages='latent+decoder',
                  lambda_ramp=0.0, decoded_joint=True, marginal_weight=1.0,
                  stage_reduce='mean', mask_proxies=False, s_decode='argmax',
                  phase2_beta='1e-2', target_in_fairness=True),
}


def read_adult():
    kw = dict(names=ADULT_NAMES, skipinitialspace=True, na_values='?')
    train = pd.read_csv(os.path.join(DATA_DIR, 'adult.data'), **kw)
    test = pd.read_csv(os.path.join(DATA_DIR, 'adult.test'), skiprows=1, **kw)
    df = pd.concat([train, test], ignore_index=True)
    df['income'] = df['income'].str.rstrip('.')
    df = df.dropna().reset_index(drop=True)
    df['income'] = (df['income'] == '>50K').astype(int).astype(str)
    return df


def prepare(df, fold, n_folds, seed):
    """Fold split + encoding. Numeric stats come from the training fold only;
    category codes are shared across folds."""
    enc = {}
    codes = {}
    for c in CATEGORICAL:
        # sort so the code order does not depend on row order
        uniques = np.array(sorted(df[c].astype(str).unique()))
        enc[c] = uniques
        codes[c] = np.searchsorted(uniques, df[c].astype(str).values)
    cat = np.stack([codes[c] for c in CATEGORICAL], axis=1).astype(np.int64)
    raw_num = df[NUMERIC].values.astype(np.float64)
    tr_idx, te_idx = list(KFold(n_folds, shuffle=True,
                                random_state=seed).split(df))[fold]
    means = pd.Series(raw_num[tr_idx].mean(axis=0), index=NUMERIC)
    stds = pd.Series(raw_num[tr_idx].std(axis=0), index=NUMERIC).replace(0, 1)
    num = ((raw_num - means.values) / stds.values).astype(np.float32)
    return dict(train_num=num[tr_idx], train_cat=cat[tr_idx],
                test_num=num[te_idx], test_cat=cat[te_idx],
                encoders=enc, means=means, stds=stds)


def configure_demo():
    """Point run_flip_demo's column configuration at Adult."""
    demo.NUMERIC_COLS = NUMERIC
    demo.CATEGORICAL_COLS = CATEGORICAL
    demo.SENSITIVE_COL = SENSITIVE
    demo.TARGET_COL = TARGET


def build_params(args):
    params = copy.deepcopy(demo.TRADEOFF_PARAMS)
    params.update({
        'LAMBDA_FAIR': args.lam,
        'USE_DP': args.dp_noise is not None,
        'SEED': args.seed,
        'PHASE1_EPOCHS': args.phase1_epochs,
        'PHASE2_EPOCHS': args.phase2_epochs,
        'SAMPLE_SIZE': None,
        'WEIGHT_DECAY': args.weight_decay,
        # stability safeguards from the DP pipeline (not in TabSyn or the
        # paper); defaults keep them, the flags below turn them off
        'MAX_OPTIMIZER_GRAD_NORM': args.grad_clip,
        'ADAM_EPS': args.adam_eps,
        # the paper's training setup: full data, batch 4096 (the fairness
        # terms are batch statistics), small tokens, wide FFN, TabSyn's
        # plateau beta decay, batch-mean L_S (Eq. 3)
        'BATCH_SIZE': args.batch_size, 'D_TOKEN': 4, 'FACTOR': 32,
        'BETA_SCHEDULE': 'plateau', 'BETA_MAX': 1e-2, 'BETA_MIN': 1e-5,
        'BETA_LAMBDA': 0.7, 'BETA_PATIENCE': 10,
        'PHASE2_LR_SCALE': args.phase2_lr_scale,
        'PHASE2_RECON_WEIGHT': args.phase2_recon_weight,
        'LS_MODE': 'batch_mean',
        # disentanglement (preset + flags)
        'CKA_CENTER': args.cka_center,
        'MEAN_SHIFT_WEIGHT': args.mean_gap_weight,
        'BINARY_W1': args.binary_w1,
        'NUM_STAGE': args.num_stage,
        'DIV_STAGES': args.div_stages,
        'LAMBDA_RAMP_FRAC': args.lambda_ramp,
        'STAGE_REDUCE': args.stage_reduce,
        'DECODED_JOINT': args.decoded_joint,
        'MARGINAL_WEIGHT': args.marginal_weight,
        'TARGET_IN_FAIRNESS': args.target_in_fairness,
        'CKA_TOKENS': args.cka_tokens,
        'CKA_KERNEL': args.cka_kernel,
        'CKA_LATENT': args.cka_latent,
        'CKA_EXCLUDE_S': args.cka_exclude_s,
        'SWD_GROUP_CENTER': args.swd_group_center,
        'MASK_S_TOKEN': args.mask_s_token,
        'JOINT_MEASURE': args.joint_measure,
        'JOINT_MEAN_WEIGHT': (args.joint_mean_gap if args.joint_mean_gap
                              is not None else args.mean_gap_weight),
        'JOINT_WEIGHTING': args.joint_weighting,
        'DIV_WEIGHT': args.div_weight,
        'SWD_PROJECTIONS': args.swd_projections,
        'SWD_LATENT': args.swd_latent,
    })
    if args.dp_noise is not None:
        params['NOISE_MULTIPLIER'] = args.dp_noise
        if args.dp_group_noise is not None:
            params['GROUP_NOISE_MULTIPLIER'] = args.dp_group_noise
        if args.dp_stats_noise is not None:
            params['STATS_NOISE_MULTIPLIER'] = args.dp_stats_noise
        params['COUPLED_DP'] = args.dp_coupled
        if args.dp_group_sigmas:
            params['GROUP_SIGMAS'] = [float(x) for x in
                                      args.dp_group_sigmas.split(',')]
    dparams = copy.deepcopy(full.DIFFUSION_PARAMS)
    dparams.update({
        'USE_DP': args.dp_noise is not None and args.dp_diffusion,
        'EPOCHS': args.diff_epochs,
        'BATCH_SIZE': args.diff_batch,
        'N_GEN': None,
    })
    if args.dp_noise is not None:
        dparams['NOISE_MULTIPLIER'] = args.dp_noise
    if args.no_recon_clamp:
        params['LOGIT_CLAMP'] = params['NUM_RECON_CLAMP'] = float('inf')
    return params, dparams


def averaged_scores(num, cat, s_idx, ref=None, n_seeds=10, **cluster_kw):
    """fairness_scores (BER, A-NCB) averaged over seeds - A-NCB is a minimum
    over clusters of a GMM fit that can land on different splits per seed,
    so the median and sd are reported alongside the mean."""
    runs = [fairness_scores(num, cat, s_idx, ref=ref, seed=s, **cluster_kw)
            for s in range(n_seeds)]
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs])
        out[k], out[k + '_sd'] = float(v.mean()), float(v.std())
        out[k + '_median'] = float(np.median(v))
    return out


def real_ref_cache(args, version='famd2_v1'):
    """Cache for scores computed from the REAL data only. They depend on
    the fold, split seed and masking - not on lambda, tier-1/full or the
    repair baseline (whose generator alone sees the repaired copy) - so the
    first run on a fold computes and stores them and later runs load the
    same numbers. Returns get(name, fn): the cached value, or fn() computed
    and stored. Writes are atomic and merge with concurrent writers."""
    key = (f'real_ref_f{args.fold}of{args.folds}_ss{args.split_seed}_'
           f'{"mask" if args.mask_proxies else "raw"}_{version}')
    cache_dir = os.path.join(OUT_DIR, 'cache')
    path = os.path.join(cache_dir, key + '.json')

    def _load():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    store = _load()

    def get(name, fn):
        if name in store:
            return store[name]
        value = fn()
        store[name] = value
        os.makedirs(cache_dir, exist_ok=True)
        merged = _load()
        merged.update(store)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(merged, f, default=float)
        os.replace(tmp, path)
        return value
    return get


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lam', type=float, default=4.0, help='LAMBDA_FAIR')
    ap.add_argument('--dp-noise', type=float, default=None,
                    help='DP-SGD noise multiplier for the VAE; omit = no DP')
    ap.add_argument('--dp-group-noise', type=float, default=None,
                    help='noise multiplier of the group-level mechanism for '
                         'the batch-coupled terms (L_S batch mean, SWD, '
                         'CKA^T, joint stage); default: the pipeline value')
    ap.add_argument('--dp-stats-noise', type=float, default=None,
                    help='noise multiplier of the one DP release of numeric '
                         'range-clip bounds at decoding')
    ap.add_argument('--dp-group-sigmas', default=None,
                    help="paper Eq. 13: comma-separated noise multipliers per "
                         "sex code (Female,Male) from calibrate_dp.py --eq13; "
                         'sets the global multiplier and per-group clipping '
                         '(IDP-SGD) and turns DP on')
    ap.add_argument('--dp-coupled', choices=['group', 'paper'],
                    default='group',
                    help='DP for the batch-coupled terms: group = a separate '
                         'group-level mechanism (a valid bound, much more '
                         'noise); paper = through DP-SGD per-sample clipping '
                         'with the rest of the loss (epsilon nominal for '
                         'those terms)')
    ap.add_argument('--dp-rules-noise', type=float, default=None,
                    help='under DP, --reject-invalid learns its rules from '
                         'DP-noised 2-way category counts with this noise '
                         'multiplier (charged to epsilon); required for '
                         'rejection sampling under DP')
    ap.add_argument('--dp-clip-release', action='store_true',
                    help='under DP, release numeric range-clip bounds with '
                         'the Gaussian mechanism (charged to epsilon). Off by '
                         'default, so epsilon covers the VAE only, as in the '
                         'paper; decoding then applies only the >= 0 floor')
    ap.add_argument('--dp-diffusion', action='store_true',
                    help='also train the diffusion model with DP-SGD (same '
                         'noise as the VAE). Off by default, as in the '
                         'paper, which treats diffusion as post-processing '
                         'of the private VAE')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--split-seed', type=int, default=42,
                    help='seed for the fold split (fixed across --seed values '
                         'so seed sweeps vary only training, not data)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--phase1-epochs', type=int, default=3000)
    ap.add_argument('--phase2-epochs', type=int, default=100)
    ap.add_argument('--phase2-recon-weight', type=float, default=1.0,
                    help='weight of the reconstruction loss (MSE + CE) in '
                         "Phase 2's L_quality; below 1 lets lambda and the "
                         'SWD anchor count for more (the paper does not '
                         'weight it; 0 drops it)')
    ap.add_argument('--phase2-lr-scale', type=float, default=1.0,
                    help='Phase-2 learning rate as a multiple of the Phase-1 '
                         'rate (1e-3); the paper does not state it')
    ap.add_argument('--batch-size', type=int, default=4096,
                    help='balanced batch size (the paper does not state it for '
                         'Adult; TabSyn uses 4096). With --phase1-checkpoint '
                         'it only affects Phase 2')
    ap.add_argument('--grad-clip', type=float, default=5.0,
                    help='global gradient-norm clip on every step without DP '
                         '(a DP-pipeline safeguard; TabSyn and the paper '
                         'clip nothing). 0 = off')
    ap.add_argument('--adam-eps', type=float, default=1e-4,
                    help='Adam epsilon (DP-pipeline default 1e-4; PyTorch and '
                         'TabSyn use 1e-8)')
    ap.add_argument('--no-recon-clamp', action='store_true',
                    help='do not clamp logits (+/-15) and numeric outputs '
                         '(+/-8) before the loss (a DP-pipeline safeguard)')
    ap.add_argument('--weight-decay', type=float, default=0.0,
                    help='Adam L2 weight decay; the HMDA default (1e-2) '
                         'collapses the VAE to the mean on Adult')
    # ---- method: a preset, each piece overridable ---------------------------
    ap.add_argument('--preset', choices=sorted(PRESETS), default='paper',
                    help="'paper': FLIP as published (default); 'final': "
                         "the paper plus the joint stage, mean gap and KL "
                         "guard (recommended); 'ours': the earlier additions "
                         "(masked, group-centred). Flags below override it")
    ap.add_argument('--cka-exclude-s', action='store_true',
                    help='drop the protected attribute own token from the '
                         'latent and decoder CKA stages (target the proxies)')
    ap.add_argument('--swd-group-center', action='store_true',
                    help='centre the frozen Phase-1 reference per sex group '
                         'before the SWD anchor (it no longer pulls the '
                         'groups apart)')
    ap.add_argument('--mask-s-token', action='store_true',
                    help='Phase 2 and generation: the decoder reads the batch '
                         'mean of the sex token, not the row own token')
    ap.add_argument('--joint-measure', choices=['ckat_gap', 'dep_rbf'],
                    default='ckat_gap',
                    help='joint decoded-output stage: RFF CKA^T + mean gap, '
                         'or the loophole-free RBF CKA(Z, S)')
    ap.add_argument('--cka-center', choices=['row', 'group', 'pooled_bias',
                                             'dep_cka', 'dep_hsic',
                                             'dep_lf_lin', 'dep_lf_rbf'],
                    default=None,
                    help="CKA^T centering: 'row' (dim=1, the paper's Eqs. "
                         "5-7) or 'group' (dim=0; blind to group mean "
                         'offsets, pair it with --mean-gap-weight)')
    ap.add_argument('--mean-gap-weight', type=float, default=None,
                    help='weight of the normalized group mean-gap term at '
                         'every stage and in the joint stage (paper: 0)')
    ap.add_argument('--binary-w1', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='two-level columns at the detokenizer stage: W1 of '
                         'the logit gap instead of CKA^T (paper: off)')
    ap.add_argument('--num-stage', choices=['pre', 'summed'], default=None,
                    help="numeric detokenizer stage: 'pre' per-feature "
                         "pre-aggregation logits (paper) or 'summed'")
    ap.add_argument('--div-stages', choices=['latent+decoder', 'latent'],
                    default=None,
                    help='SWD divergence anchor at the latent and decoder '
                         '(paper) or at the latent only')
    ap.add_argument('--lambda-ramp', type=float, default=None,
                    help='fraction of Phase 2 over which lambda ramps up '
                         '(paper: 0, full lambda from the first epoch)')
    ap.add_argument('--decoded-joint', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='joint decoded-output stage: CKA^T + mean gap on the '
                         'whole decoded row, RFF-lifted, weighted by lambda')
    ap.add_argument('--marginal-weight', type=float, default=None,
                    help='weight of the KL marginal guard (stops the fairness '
                         'terms from deleting categories; paper: 0)')
    ap.add_argument('--stage-reduce', choices=['mean', 'sum'], default=None,
                    help='combine the stages by mean or sum (full lambda per '
                         'stage); the paper does not say')
    ap.add_argument('--mask-proxies', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='merge each auto-detected near-deterministic proxy '
                         'class (P(S|category) >= 0.95 pointing to different '
                         'groups, e.g. Husband/Wife) into one new category in '
                         'both folds before training (needs a bank trained '
                         'the same way)')
    ap.add_argument('--div-weight', type=float, default=1.0,
                    help='weight of the SWD divergence anchor (implicit 1 '
                         'in Eq. 10)')
    ap.add_argument('--swd-projections', type=int, default=50,
                    help='random 1-D projections per SWD estimate (the '
                         'paper does not say)')
    ap.add_argument('--swd-latent', choices=['mu', 'z'], default='mu',
                    help='latent-stage SWD on the encoder means or on '
                         'samples z = mu + sigma*eps (Eq. 10 is written over '
                         'q(z|x))')
    ap.add_argument('--joint-mean-gap', type=float, default=None,
                    help='mean-gap weight inside the joint decoded-output '
                         'stage (default: --mean-gap-weight)')
    ap.add_argument('--joint-weighting', choices=['dim', 'column'],
                    default='dim',
                    help='joint decoded-output stage: dim = unit variance per '
                         'dimension (a column weighs as much as its number of '
                         'categories); column = every column contributes '
                         'equal total variance (the paper equal-per-feature '
                         'weighting, kept joint)')
    ap.add_argument('--cka-latent', choices=['mu', 'z'], default='mu',
                    help='latent-stage CKA^T on the encoder means (mu) or on '
                         'the sampled latent z = mu + sigma*eps the decoder '
                         'sees (q(z|x), as Eq. 10 writes it)')
    ap.add_argument('--cka-kernel', choices=['linear', 'rbf', 'rff_lift'],
                    default='linear',
                    help="CKA^T kernel: 'linear' (Eq. 7), 'rbf' (Eqs. 5-6 with "
                         "an RBF kernel over the activations) or 'rff_lift' "
                         '(rows mapped by random Fourier features, then '
                         'linear CKA^T)')
    ap.add_argument('--cka-tokens', choices=['concat', 'per_token'],
                    default='concat',
                    help='latent and decoder stages: one CKA^T on all tokens '
                         'concatenated, or CKA^T per token averaged (the '
                         'paper does not say)')
    ap.add_argument('--target-in-fairness',
                    action=argparse.BooleanOptionalAction, default=None,
                    help='include the target (income) in the detokenizer-'
                         'stage and joint-stage fairness terms (paper: yes - '
                         'every non-protected column is a potential target); '
                         "off keeps the income rates' link to sex out of reach "
                         'of the fairness terms')
    ap.add_argument('--s-decode', choices=['argmax', 'sample'], default=None,
                    help='decode the generated sex by argmax (paper) or by '
                         'sampling its probabilities. Under the batch-mean '
                         'L_S every row sits near 0.5, so argmax follows a '
                         'small tilt that can track or invert the true sex; '
                         'sampling gives a random assignment')
    ap.add_argument('--phase2-beta', default=None,
                    help="KL weight of Phase 2's ELBO: a number (preset: "
                         "1e-2), or 'carry' for the bank's final Phase 1 "
                         "beta (read from BANK.json). The paper decays beta "
                         'in Phase 1 and does not state its Phase-2 value')
    ap.add_argument('--from-run', default=None,
                    help='re-decode a saved run (TAG or path to '
                         'adult_results/runs/TAG.pt) instead of training: '
                         'its Phase-2 VAE and latents, decoded with this '
                         '--s-decode and scored under this --tag. Data and '
                         'training flags are taken from the saved run')
    ap.add_argument('--score-collapsed', action='store_true',
                    help='score a run that fails the collapse guard instead '
                         'of stopping; it stays flagged collapsed, so the '
                         'tables still skip it')
    ap.add_argument('--no-save-run', action='store_true',
                    help='do not save the Phase-2 VAE and decoded latents to '
                         'adult_results/runs/TAG.pt (saved by default, for '
                         '--from-run)')
    ap.add_argument('--diff-epochs', type=int, default=2000)
    ap.add_argument('--diff-batch', type=int, default=4096)
    ap.add_argument('--tier1', action='store_true',
                    help='skip diffusion: decode z = mu + sigma*eps from the '
                         'real training rows (reconstruct-through-the-latent). '
                         'Screens fairness against each row\'s TRUE sex; '
                         'privacy metrics are meaningless here and skipped')
    ap.add_argument('--phase1-checkpoint', default=None,
                    help='load a banked Phase-1 VAE (from --save-phase1) and '
                         'skip Phase 1; Phase 2 still runs')
    ap.add_argument('--save-phase1', default=None,
                    help='save the VAE after training (with --phase2-epochs 0: '
                         'bank Phase 1)')
    ap.add_argument('--reject-invalid', type=int, default=0, metavar='N',
                    help='rejection sampling: redraw rows that break any '
                         'validity rule learned from the real training fold, '
                         'up to N rounds (tier-1: fresh eps from the same '
                         "row's posterior, keeping its source person; full "
                         'pipeline: fresh latents from the diffusion model). '
                         'Rows still invalid after N rounds are kept and '
                         'counted. 0 = off')
    ap.add_argument('--score-csv', default=None,
                    help='score an external synthetic CSV (raw values, the '
                         'same columns) on this fold with all metrics, '
                         'without training (e.g. aim_fair/ outputs)')
    ap.add_argument('--latent-source', choices=['phase2', 'phase1'],
                    default='phase2',
                    help='diagnostic: latents for decoding (tier-1) and for '
                         'diffusion training from the Phase-2 encoder (normal) '
                         'or the Phase-1 bank encoder (a pipeline that saved '
                         'its latents before Phase 2), decoded by Phase 2')
    ap.add_argument('--cat-decode', choices=['argmax', 'sample'],
                    default='argmax',
                    help='decode EVERY categorical column by argmax '
                         "(TabSyn) or by sampling from the decoder's softmax "
                         '(overrides --s-decode for sex)')
    ap.add_argument('--repair-invalid', action='store_true',
                    help='constrained decoding: fix rows that break a '
                         'categorical validity rule with the edits the '
                         "decoder's own probabilities find most likely "
                         '(keeps every row; post-processing). Uses the same '
                         'rules as --reject-invalid (DP rules under DP)')
    ap.add_argument('--repair-protect', choices=['sex', 'none'],
                    default='sex',
                    help='columns --repair-invalid may not edit: sex (so a '
                         'repair cannot re-link a proxy to it) or none')
    ap.add_argument('--repair-max-edits', type=int, default=3)
    ap.add_argument('--mask-label', default='MASK',
                    help='name of the merged category (--mask-proxies)')
    ap.add_argument('--repair-tau', type=float, default=0.0,
                    help='COMPARISON BASELINE: scramble every non-target '
                         "column's link to sex with strength tau in [0, 1] "
                         "before training, keeping each column's overall "
                         'distribution (Feldman et al. repair)')
    args = ap.parse_args()
    if args.dp_group_sigmas and args.dp_noise is None:
        args.dp_noise = 0.0   # placeholder: train_flip sets sigma_global
    # --from-run: everything except the decoding and the output tag comes
    # from the saved run, so the data preparation matches it exactly
    saved_run = None
    if args.from_run:
        path = (args.from_run if args.from_run.endswith('.pt') else
                os.path.join(OUT_DIR, 'runs', args.from_run + '.pt'))
        saved_run = torch.load(path, map_location='cpu', weights_only=True)
        for k, v in saved_run['args'].items():
            if k not in ('tag', 's_decode', 'from_run', 'no_save_run',
                         'score_collapsed', 'reject_invalid',
                         'repair_invalid', 'repair_protect',
                         'repair_max_edits', 'dp_rules_noise',
                         'cat_decode', 'latent_source'):
                setattr(args, k, v)
    # unset method flags come from the preset
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    tag = args.tag or f'lam{args.lam:g}_fold{args.fold}'
    t0 = time.time()
    configure_demo()
    params, dparams = build_params(args)
    device = torch.device(params['DEVICE'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = prepare(read_adult(), args.fold, args.folds, args.split_seed)
    enc = data['encoders']
    s_idx, t_idx = CATEGORICAL.index(SENSITIVE), CATEGORICAL.index(TARGET)
    detected = detect_proxy_classes(data['train_cat'], s_idx,
                                    exclude=(t_idx,))
    if detected:
        print('  auto-detected proxy classes: '
              + '; '.join(describe_classes(detected, CATEGORICAL, enc)))
    # Masking: each detected class becomes ONE category in BOTH folds, so
    # the generator never sees the sex-specific label and every table scored
    # against the synthetic one shares its vocabulary. The raw copies are
    # the original-data real reference and define impossible combinations.
    tr_c_raw, te_c_raw = data['train_cat'].copy(), data['test_cat'].copy()
    if args.mask_proxies:
        if not detected:
            raise SystemExit('--mask-proxies: no proxy class detected')
        mask_codes = []
        for cl in detected:
            col = CATEGORICAL[cl['col']]
            mask_codes.append(len(enc[col]))
            enc[col] = np.append(enc[col], args.mask_label)
        data['train_cat'] = mask_proxy_classes(tr_c_raw, detected, mask_codes)
        data['test_cat'] = mask_proxy_classes(te_c_raw, detected, mask_codes)
        changed = float((data['train_cat'] != tr_c_raw).any(axis=1).mean())
        print(f'  masked proxy classes -> {args.mask_label!r}: '
              f'{changed:.3f} of training rows changed')
    categories = [len(enc[c]) for c in CATEGORICAL]
    X_num = torch.tensor(data['train_num'])
    X_cat = torch.tensor(data['train_cat'])
    # Repair baseline: the generator trains on a scrambled copy;
    # data['train_*'] stays unrepaired so quality is measured against the
    # real population.
    if args.repair_tau:
        rep_n, rep_c = repair_toward_independence(
            data['train_num'], data['train_cat'], s_idx, args.repair_tau,
            exclude=(t_idx,), seed=args.split_seed)
        X_num, X_cat = torch.tensor(rep_n), torch.tensor(rep_c)
        changed = float((rep_c != data['train_cat']).any(axis=1).mean())
        print(f'  scrambling repair tau={args.repair_tau:g}: {changed:.3f} '
              f'of training rows got a new category')
    params.update(TARGET_IDX=t_idx)
    print(f'[{tag}] train={len(X_num)} test={len(data["test_num"])} '
          f'categories={categories} sex={list(enc[SENSITIVE])}')

    # ---- Stage 1: FLIP VAE ------------------------------------------------
    print('\n[Stage 1] FLIP VAE')
    bank_rdp = None   # bank Phase-1 curve when composed separately
    if args.score_csv:
        # score an external synthetic table: no model is trained
        vae = None
        accountant = group_mech = None
    elif saved_run is not None:
        print(f'  SKIPPED (--from-run): Phase-2 VAE of {saved_run["tag"]}')
        vae = demo.Model_VAE(params['NUM_LAYERS'], X_num.shape[1], categories,
                             params['D_TOKEN'], n_head=1,
                             factor=params.get('FACTOR', 8),
                             bias=True).to(device)
        vae.load_state_dict(saved_run['vae'])
        if getattr(args, 'mask_s_token', False):
            vae.VAE.s_token_mask = X_num.shape[1] + s_idx
        accountant = group_mech = None
    else:
        if args.phase2_beta != 'carry':
            params['PHASE2_BETA'] = float(args.phase2_beta)
            print(f"  Phase 2 beta fixed at {params['PHASE2_BETA']:.2e}")
        elif args.phase1_checkpoint:
            # a single Phase-1 + Phase-2 run carries beta on its own
            side = args.phase1_checkpoint + '.json'
            if not os.path.exists(side):
                raise SystemExit(f'--phase2-beta carry: {side} not found '
                                 "(the bank's final Phase 1 beta)")
            with open(side) as f:
                params['PHASE2_BETA'] = json.load(f)['phase1_final_beta']
            print(f"  Phase 2 beta carried from the bank: "
                  f"{params['PHASE2_BETA']:.2e}")
        # train_flip skips Phase 1 only when the bank file exists
        bank_loaded = bool(args.phase1_checkpoint and
                           os.path.exists(args.phase1_checkpoint))
        vae, accountant, group_mech = demo.train_flip(
            X_num, X_cat, categories, s_idx, params, device,
            phase1_checkpoint=args.phase1_checkpoint,
            factor=params.get('FACTOR', 8))
        if accountant is not None and bank_loaded:
            # a banked Phase 1 skipped its epochs here: count them from the
            # bank, which must have been trained under the same mechanism
            side = args.phase1_checkpoint + '.json'
            dp = (json.load(open(side)).get('dp')
                  if os.path.exists(side) else None)
            if dp is None:
                raise SystemExit(f'DP run on {args.phase1_checkpoint}: the '
                                 'bank has no DP accounting (trained '
                                 'without DP?)')
            same = (dp.get('coupled', 'group') ==
                    params.get('COUPLED_DP', 'group') and
                    dp.get('group_sigmas') == params.get('GROUP_SIGMAS') and
                    abs(dp['noise'] - params['NOISE_MULTIPLIER']) < 1e-9 and
                    abs(dp['group_noise'] -
                        params['GROUP_NOISE_MULTIPLIER']) < 1e-9 and
                    abs(dp['sample_rate'] - accountant.sample_rate) < 1e-9)
            if same:
                accountant.steps += dp['steps']
                if group_mech is not None:
                    group_mech.steps += dp['group_steps']
                print(f"  Phase-1 privacy from the bank: +{dp['steps']} "
                      f"DP-SGD and +{dp['group_steps']} group-mechanism steps")
            elif (dp.get('coupled', 'group') == 'paper' ==
                  params.get('COUPLED_DP', 'group')):
                # Phase 2 runs with other noise or batch than the bank (e.g.
                # --batch-size): account the bank's Phase 1 as its own
                # mechanism, composed with this run's Phase-2 accountant
                if dp.get('group_sigmas'):
                    counts = np.bincount(X_cat[:, s_idx].numpy(),
                                         minlength=len(dp['group_sigmas']))
                    rates = [dp.get('batch_size', 4096) // 2 / c
                             for c in counts]
                    bank_acc = GroupwiseRDPAccountant(
                        dp['group_sigmas'], rates, params['DP_DELTA'])
                else:
                    bank_acc = RDPAccountant(dp['noise'], dp['sample_rate'],
                                             params['DP_DELTA'])
                bank_acc.steps = dp['steps']
                bank_rdp = bank_acc.get_rdp_curve()
                print(f"  Phase-1 privacy from the bank: its own mechanism, "
                      f"{dp['steps']} steps (composed separately)")
            else:
                raise SystemExit(f'bank DP settings {dp} differ from this '
                                 'run; the accounting would not compose')
    if args.save_phase1 and saved_run is None:
        torch.save(vae.state_dict(), args.save_phase1)
        # the bank's final beta, for --phase2-beta carry
        side = {'phase1_final_beta': params.get('PHASE1_FINAL_BETA')}
        if accountant is not None:
            # Phase-1 privacy spent by the bank: a run that loads it adds
            # these steps to its own accounting (same noise and rates)
            side['dp'] = {'noise': params['NOISE_MULTIPLIER'],
                          'coupled': params.get('COUPLED_DP', 'group'),
                          'group_sigmas': params.get('GROUP_SIGMAS'),
                          'group_noise': params['GROUP_NOISE_MULTIPLIER'],
                          'sample_rate': accountant.sample_rate,
                          'batch_size': params['BATCH_SIZE'],
                          'steps': accountant.steps,
                          'group_steps': (group_mech.steps
                                          if group_mech is not None else 0)}
        with open(args.save_phase1 + '.json', 'w') as f:
            json.dump(side, f)
        print(f'  saved VAE checkpoint to {args.save_phase1}')

    if args.score_csv:
        # external synthetic table (e.g. aim_fair/): raw values -> this
        # fold's standardization and category codes; scored below exactly
        # as our own runs are
        ext = pd.read_csv(args.score_csv)
        syn_num = ((ext[NUMERIC].values.astype(np.float64)
                    - data['means'].values) / data['stds'].values)
        syn_cat = np.stack([ext[c].astype(str).map(
            {str(v): i for i, v in enumerate(enc[c])}).values
            for c in CATEGORICAL], axis=1)
        if np.isnan(syn_cat.astype(float)).any():
            raise SystemExit('--score-csv: categories not in the real data')
        syn_cat = syn_cat.astype(np.int64)
        stats_budgets, diff_acc = [], None
        reject_info = repair_info = rules_rdp = None
    else:
        # ---- Stage 2: diffusion (or tier-1 decode) ------------------------------
        vae.eval()
        with torch.no_grad():
            enc_out = vae.forward_with_stages(X_num.to(device), X_cat.to(device))
        if args.latent_source == 'phase1':
            # stage-mismatch diagnostic: latents from the Phase-1 bank encoder
            # (what diffusion would learn if its latents were saved after
            # Phase 1), decoded by this run's Phase-2 decoder
            if not (args.phase1_checkpoint and
                    os.path.exists(args.phase1_checkpoint)):
                raise SystemExit('--latent-source phase1 needs --phase1-checkpoint')
            bank_vae = demo.Model_VAE(params['NUM_LAYERS'], X_num.shape[1],
                                      categories, params['D_TOKEN'], n_head=1,
                                      factor=params.get('FACTOR', 8),
                                      bias=True).to(device)
            bank_vae.load_state_dict(torch.load(args.phase1_checkpoint,
                                                map_location=device))
            bank_vae.eval()
            with torch.no_grad():
                enc_out = bank_vae.forward_with_stages(X_num.to(device),
                                                       X_cat.to(device))
            print('  latents from the Phase-1 bank encoder, decoded by Phase 2')
        # under DP the range-clip bounds are released only with
        # --dp-clip-release (otherwise epsilon covers the VAE only)
        clip_release = args.dp_noise is None or args.dp_clip_release
        decode_kw = dict(X_num=X_num if clip_release else None,
                         params=params if clip_release else None,
                         num_floor=(-data['means'] / data['stds']).values,
                         sample_idx=s_idx if args.s_decode == 'sample' else None,
                         sample_all=args.cat_decode == 'sample')
        if saved_run is not None:
            print(f'\n[Stage 2] SKIPPED (--from-run): saved latents, '
                  f'--s-decode {args.s_decode}')
            z_dec = saved_run['z'].to(device)
            syn_num, syn_cat, stats_budgets = full.decode_latents(
                vae, z_dec, device, **decode_kw)
            diff_acc = None
        elif args.tier1:
            # z = mu + sigma*eps from REAL rows ('std_z' is the encoder's
            # log-variance). Cannot collapse; row i is decoded from training row
            # i, so its true sex is known.
            print('\n[Stage 2] SKIPPED (--tier1): decoding z = mu + sigma*eps')
            sigma = torch.exp(0.5 * enc_out['std_z'])
            z_dec = enc_out['mu_z'] + sigma * torch.randn_like(sigma)
            syn_num, syn_cat, stats_budgets = full.decode_latents(
                vae, z_dec, device, **decode_kw)
            diff_acc = None
        else:
            print('\n[Stage 2] latent diffusion')
            mu_z = enc_out['mu_z']
            latents = mu_z.reshape(mu_z.shape[0], -1).cpu().numpy()
            # trained on reparameterized latents z = mu + sigma*eps, fresh eps
            # per batch (paper Sec. 4.1.2); the encoder's 'std_z' is its
            # log-variance
            std = torch.exp(0.5 * enc_out['std_z']).reshape(
                mu_z.shape[0], -1).cpu().numpy()
            diff_model, z_mean, diff_acc = full.train_diffusion(
                latents, std=std, params=dparams, device=device)
            z_dec = full.sample_from_diffusion(diff_model, z_mean, len(X_num),
                                               mu_z.shape[1:], dparams, device)
            syn_num, syn_cat, stats_budgets = full.decode_latents(
                vae, z_dec, device, **decode_kw)

        # Rejection sampling (--reject-invalid N): every rule type, sex-linked
        # implications included, scored on the released (rounded) table as the
        # validity metric is.
        reject_info = repair_info = None
        rules_rdp = None
        if args.reject_invalid or args.repair_invalid:
            if (args.reject_invalid and saved_run is not None
                    and not args.tier1):
                raise SystemExit('--reject-invalid with --from-run needs a tier-1 '
                                 'run (the diffusion model is not saved)')
            def _raw(n):
                return n * data['stds'].values + data['means'].values
            schema_r = demo.infer_numeric_schema(_raw(data['train_num']), NUMERIC)
            if args.dp_noise is not None:
                # under DP the rules come from DP-noised counts, charged to eps
                if args.dp_rules_noise is None:
                    raise SystemExit('--reject-invalid / --repair-invalid under '
                                     'DP need --dp-rules-noise')
                rules_r, rules_rdp = fit_validity_rules_dp(
                    _raw(data['train_num']), data['train_cat'], NUMERIC,
                    CATEGORICAL, args.dp_rules_noise,
                    [enc[c] for c in CATEGORICAL], seed=args.seed)
                print(f"  DP validity rules: {len(rules_r['fd'])} fd, "
                      f"{len(rules_r['pair'])} pair, "
                      f"{len(rules_r['integer'])} integer")
            else:
                rules_r = fit_validity_rules(_raw(data['train_num']),
                                             data['train_cat'], NUMERIC,
                                             CATEGORICAL,
                                             [enc[c] for c in CATEGORICAL])

            def _invalid(n, c):
                df = demo.decode_synthetic_data(n, c, data['means'], data['stds'],
                                                enc, num_schema=schema_r,
                                                schema_bounds=args.dp_noise is None)
                codes = np.stack([df[k].astype(str).map(
                    {str(v): i for i, v in enumerate(enc[k])}).values
                    for k in CATEGORICAL], axis=1).astype(int)
                return invalid_rows(rules_r, df[NUMERIC].values.astype(float),
                                    codes)
            z_dec = torch.as_tensor(z_dec).to(device)
        if args.repair_invalid:
            # constrained decoding with the decoder's per-column probabilities
            with torch.no_grad():
                _, rc = vae.Reconstructor(vae.VAE.decoder(z_dec[:, 1:]))
                probs = [torch.softmax(c_, -1).cpu().numpy() for c_ in rc]
            before = float(_invalid(syn_num, syn_cat).mean())
            protect = (s_idx,) if args.repair_protect == 'sex' else ()
            syn_cat, rinfo = constrained_repair(syn_cat, probs, rules_r, protect,
                                                args.repair_max_edits)
            after = float(_invalid(syn_num, syn_cat).mean())
            repair_info = {'invalid_before': before, 'invalid_after': after,
                           'rows_changed': rinfo['rows_changed'],
                           'edits': {CATEGORICAL[j]: n_ for j, n_
                                     in rinfo['edits'].items()},
                           'protect': args.repair_protect}
            print(f'  constrained repair: invalid rows {before:.3f} -> '
                  f'{after:.3f}; {rinfo["rows_changed"]:.3f} of rows edited '
                  f'{repair_info["edits"]}')
        if args.reject_invalid:
            bad = _invalid(syn_num, syn_cat)
            start, rounds = float(bad.mean()), 0
            for rounds in range(1, args.reject_invalid + 1):
                idx = np.where(bad)[0]
                if len(idx) == 0:
                    rounds -= 1
                    break
                with torch.no_grad():
                    if args.tier1:
                        mu = enc_out['mu_z'][idx]
                        sd = torch.exp(0.5 * enc_out['std_z'][idx])
                        z_new = mu + sd * torch.randn_like(sd)
                    else:
                        z_new = full.sample_from_diffusion(
                            diff_model, z_mean, len(idx), z_dec.shape[1:],
                            dparams, device)
                    n_new, c_new, _ = full.decode_latents(
                        vae, z_new, device, **decode_kw,
                        bounds=getattr(full.decode_latents, 'last_bounds', None))
                ok = ~_invalid(n_new, c_new)
                fix = idx[ok]
                syn_num[fix], syn_cat[fix] = n_new[ok], c_new[ok]
                z_dec[torch.as_tensor(fix, device=device)] =                 z_new[torch.as_tensor(np.where(ok)[0], device=device)]
                bad[fix] = False
            reject_info = {'invalid_before': start, 'invalid_after':
                           float(bad.mean()), 'rounds': rounds}
            print(f'  rejection sampling: invalid rows {start:.3f} -> '
                  f'{bad.mean():.3f} after {rounds} round(s)')

        # Save the Phase-2 VAE and the decoded latents, so the run can be
        # re-decoded (--from-run, another --s-decode) without retraining or
        # re-running diffusion. Never fails the run.
        if saved_run is None and not args.no_save_run:
            try:
                os.makedirs(os.path.join(OUT_DIR, 'runs'), exist_ok=True)
                torch.save({'tag': tag, 'args': vars(args),
                            'vae': vae.state_dict(),
                            'z': torch.as_tensor(z_dec).detach().cpu()},
                           os.path.join(OUT_DIR, 'runs', f'{tag}.pt'))
            except Exception as e:  # noqa: BLE001
                print(f'  WARNING: run not saved for --from-run ({e})')

    # Released CSV: numerics clipped/rounded to the real columns' format
    # (metrics below use the unrounded decode). Under DP the real min/max is
    # not released; the decoder's DP range clip applies instead.
    def raw(n):
        return n * data['stds'].values + data['means'].values
    num_schema = demo.infer_numeric_schema(raw(data['train_num']), NUMERIC)
    syn_df = demo.decode_synthetic_data(syn_num, syn_cat, data['means'],
                                        data['stds'], enc,
                                        num_schema=num_schema,
                                        schema_bounds=args.dp_noise is None)
    os.makedirs(OUT_DIR, exist_ok=True)
    syn_df.to_csv(os.path.join(OUT_DIR, f'{tag}_synthetic.csv'), index=False)

    # ---- Metrics -----------------------------------------------------------
    print('\n[Metrics]')
    tr_n, tr_c = data['train_num'], data['train_cat']
    te_n, te_c = data['test_num'], data['test_cat']
    # BER + A-NCB (full-covariance GMM, adversary trained on the real
    # HELD-OUT fold; 10 seeds). Real references: the original data, and the
    # real data after the same masking.
    ref_cached = real_ref_cache(args)
    syn = averaged_scores(syn_num, syn_cat, s_idx, ref=(te_n, te_c))
    real_test = ref_cached('real_test', lambda: averaged_scores(
        te_n, te_c_raw, s_idx, ref=(tr_n, tr_c_raw)))
    real_test_masked = (ref_cached('real_test_masked', lambda: averaged_scores(
        te_n, te_c, s_idx, ref=(tr_n, tr_c))) if args.mask_proxies else None)
    # A-NCB with the adversary trained on the table's OWN generated sex
    # column (the paper's wording read literally). It rewards a generated
    # sex column that has drifted from the row, so it is reported next to
    # the real-trained version above, not instead of it.
    syn_own = averaged_scores(syn_num, syn_cat, s_idx, ref=None)
    real_test_own = ref_cached('real_test_own', lambda: averaged_scores(
        te_n, te_c_raw, s_idx, ref=None))
    eps_id = None if args.tier1 else epsilon_identifiability(tr_n, tr_c,
                                                            syn_num, syn_cat)
    # Tier-1 only: true-sex leakage - a real-trained classifier recovering
    # each decoded row's SOURCE sex from its non-sex columns (true-sex
    # BER = 1 - leakage). The held-out ceiling is the same classifier on the
    # real training rows.
    leak = ceiling_ho = s_agree = true_gap = None
    if args.tier1:
        true_s = tr_c[:, s_idx].astype(int)

        def _bal(pred):
            return float(np.mean([(pred[true_s == g] == g).mean()
                                  for g in np.unique(true_s)]))
        ceiling_ho = ref_cached('heldout_ceiling', lambda: _bal(
            infer_protected(te_n, te_c, tr_n, tr_c, s_idx)))
        leak = _bal(infer_protected(te_n, te_c, syn_num, syn_cat, s_idx))
        s_agree = float((syn_cat[:, s_idx] == tr_c[:, s_idx]).mean())
        true_gap = true_group_target_gap(syn_cat, tr_c, t_idx, s_idx)
        print(f'  true-sex leakage: {leak:.3f} vs held-out ceiling '
              f'{ceiling_ho:.3f} (chance 0.5); generated sex agrees with '
              f'true sex {s_agree:.3f}')
    # synthetic-trained sex classifier tested on real held-out people
    _ts = infer_protected(syn_num, syn_cat, te_n, te_c, s_idx)
    _ty = te_c[:, s_idx].astype(int)
    tstr_sex_ber = float(np.mean([(_ts[_ty == g] != g).mean()
                                  for g in np.unique(_ty)]))
    print(f'  sex BER, trained on synthetic / tested on real: '
          f'{tstr_sex_ber:.3f} (0.5 = the synthetic data teaches no sex)')
    fid = demo.fidelity_metrics(tr_n, syn_num, tr_c, syn_cat, enc)

    # Collapse guard: a collapsed run must never reach a results table.
    t_pos = list(enc[TARGET]).index('1')
    syn_income_rate = float((syn_cat[:, t_idx] == t_pos).mean())
    real_income_rate = float((tr_c[:, t_idx] == t_pos).mean())
    collapsed = (fid['num_w1_mean'] > 0.15
                 or abs(syn_income_rate - real_income_rate) > 0.05)
    if collapsed and args.score_collapsed:
        print(f'  COLLAPSED (income {syn_income_rate:.3f}, W1 '
              f'{fid["num_w1_mean"]:.3f}) - scored anyway '
              f'(--score-collapsed); flagged, kept out of tables')
    elif collapsed:
        result = {'tag': tag, 'args': vars(args), 'collapsed': True,
                  'syn_income_rate': syn_income_rate,
                  'real_income_rate': real_income_rate, 'fidelity': fid}
        with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
            json.dump(result, f, indent=2, default=float)
        raise SystemExit(f'COLLAPSED: income {syn_income_rate:.3f} vs real '
                         f'{real_income_rate:.3f}, W1 {fid["num_w1_mean"]:.3f};'
                         f' result flagged and not scored')

    # Quality: alpha-precision / beta-recall (TabSyn's naive AlphaPrecision)
    # against the training fold; the held-out fold is the real reference.
    alpha_p, beta_r = alpha_precision_beta_recall(raw(tr_n), tr_c,
                                                  raw(syn_num), syn_cat,
                                                  categories)
    alpha_p_real, beta_r_real = ref_cached(
        'alpha_beta_real', lambda: alpha_precision_beta_recall(
            raw(tr_n), tr_c, raw(te_n), te_c, categories))
    # Validity: plausibility rules learned from the real training fold,
    # scored on the released (rounded) table; held-out fold = baseline.
    rules = fit_validity_rules(raw(tr_n), tr_c, NUMERIC, CATEGORICAL,
                               [enc[c] for c in CATEGORICAL])
    syn_cat_released = np.stack([syn_df[c].astype(str).map(
        {str(v): i for i, v in enumerate(enc[c])}).values
        for c in CATEGORICAL], axis=1).astype(int)
    validity = validity_report(rules, syn_df[NUMERIC].values.astype(float),
                               syn_cat_released)
    validity_real = ref_cached('validity_real', lambda: validity_report(
        rules, raw(te_n), te_c))

    s_tv = group_share_tv(syn_cat, tr_c, s_idx)
    all_tv = categorical_share_tv_all(syn_cat, tr_c)
    ds = downstream_sp_eo(syn_num, syn_cat, tr_n, tr_c, t_idx, s_idx)
    # Utility: downstream AUC on the real test fold, model trained on the
    # synthetic data (real-model reference alongside).
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    # The downstream model uses every column but the target (sex included,
    # as a standard income model would); the version without sex is kept
    # alongside (on Adult the two agree to ~0.001).
    def _util_feats(num, cat, with_sex=True):
        keep = [j for j in range(cat.shape[1])
                if j != t_idx and (with_sex or j != s_idx)]
        return np.concatenate([num] + [np.eye(categories[j])[cat[:, j]]
                                       for j in keep], axis=1)

    def _auc(train_num, train_cat, with_sex):
        return float(roc_auc_score(
            te_c[:, t_idx] == t_pos,
            HistGradientBoostingClassifier(random_state=0).fit(
                _util_feats(train_num, train_cat, with_sex),
                train_cat[:, t_idx] == t_pos)
            .predict_proba(_util_feats(te_n, te_c, with_sex))[:, 1]))
    util_auc = _auc(syn_num, syn_cat, True)
    util_auc_no_sex = _auc(syn_num, syn_cat, False)
    util_auc_real = ref_cached('util_auc_real_with_sex',
                               lambda: _auc(tr_n, tr_c, True))
    # 'male wife' rate: (sex, category) pairs absent from the real data
    imposs = impossible_combination_rate(syn_cat, tr_c_raw, s_idx,
                                         exclude=(t_idx,))
    imposs_real = ref_cached('imposs_real', lambda: impossible_combination_rate(
        tr_c_raw, tr_c_raw, s_idx, exclude=(t_idx,)))

    # Relative-to-real (paper Table 3 form; the paper's figures are
    # 4-DATASET AVERAGES, context only)
    rel = {
        'ber_rel_pct': 100.0 * (syn['ber'] - real_test['ber']) / real_test['ber'],
        'a_ncb_rel_pct': 100.0 * (syn['a_ncb'] - real_test['a_ncb']) / real_test['a_ncb'],
        'auc_rel_pct': 100.0 * (util_auc - util_auc_real) / util_auc_real,
        'paper_targets_avg_4_datasets': {'ber_rel_pct': 67.0,
                                         'a_ncb_rel_pct': 27.0,
                                         'auc_rel_pct': -10.0},
    }
    # A 'match' needs the target and every non-sex column's shares intact.
    match_valid = (abs(syn_income_rate - real_income_rate) <= 0.05
                   and max(v for j, v in all_tv.items() if j != s_idx) <= 0.05)

    result = {
        'tag': tag, 'args': vars(args), 'collapsed': collapsed,
        'n_train': len(X_num),
        'n_test': len(te_n),
        'paper_metrics': {
            'syn': syn, 'real_test': real_test,
            'real_test_reencoded': real_test_masked,
            'syn_own_adversary': syn_own,
            'real_test_own_adversary': real_test_own,
            'eps_identifiability': eps_id,
            'decoded_leakage': leak, 'heldout_ceiling': ceiling_ho,
            # train a sex classifier on the SYNTHETIC table (X~ -> S~), test
            # on real held-out people (X -> S): does the released data teach
            # sex inference? Defined for any generator (paired or not)
            'tstr_sex_ber': tstr_sex_ber,
            # polarity-free version: an inverted label (BER > 0.5) still
            # carries sex information; 0.5 - |BER - 0.5|, 0.5 = none
            'tstr_sex_ber_folded': 0.5 - abs(tstr_sex_ber - 0.5),
            'synthetic_sex_agreement': s_agree,
            'alpha_precision': alpha_p, 'beta_recall': beta_r,
            'alpha_precision_real': alpha_p_real,
            'beta_recall_real': beta_r_real,
            'validity': validity['substantive_valid_rate'],
            'validity_real': validity_real['substantive_valid_rate'],
            'validity_detail': validity,
            's_share_tv': s_tv, 'all_cat_share_tv': all_tv,
            'true_group_target_gap': true_gap,
            'downstream_sp_eo': ds,
            'downstream_auc': util_auc, 'downstream_auc_real': util_auc_real,
            'downstream_auc_no_sex': util_auc_no_sex,
            'outcome_replication': rel, 'match_valid': match_valid,
            'impossible_combination_rate': imposs,
            'rejection_sampling': reject_info,
            'constrained_repair': repair_info,
            'dp_rules_eps': (compose_rdp_budgets([rules_rdp],
                                                 params['DP_DELTA'])[0]
                             if rules_rdp is not None else None),
            'impossible_combination_rate_real': imposs_real,
            'syn_income_rate': syn_income_rate,
            'real_income_rate': real_income_rate},
        'fidelity': fid,
    }
    if accountant is not None:
        budgets = [accountant.get_rdp_curve()]
        if group_mech is not None and group_mech.steps > 0:
            budgets.append(group_mech.get_rdp_curve())
        if diff_acc is not None and diff_acc.steps > 0:
            budgets.append(diff_acc.get_rdp_curve())
        budgets.extend(stats_budgets)
        if rules_rdp is not None:
            budgets.append(rules_rdp)
        if bank_rdp is not None:
            budgets.append(bank_rdp)
        eps_total, _ = compose_rdp_budgets(budgets, params['DP_DELTA'])
        result['eps_total'] = float(eps_total)
    result['runtime_s'] = time.time() - t0
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    print(f'  BER   syn={syn["ber"]:.3f}  real(test)={real_test["ber"]:.3f}  '
          f'(chance {syn["ber_chance"]:.2f})')
    print(f'  A-NCB syn={syn["a_ncb"]:.3f} (median {syn["a_ncb_median"]:.3f})'
          f'  real(test)={real_test["a_ncb"]:.3f}  [adversary trained on '
          f'real data]')
    print(f'  A-NCB syn={syn_own["a_ncb"]:.3f}  real(test)='
          f'{real_test_own["a_ncb"]:.3f}  [adversary trained on the '
          f'generated sex]')
    if real_test_masked is not None:
        print(f'  real reference after masking: BER '
              f'{real_test_masked["ber"]:.3f}  A-NCB '
              f'{real_test_masked["a_ncb"]:.3f}')
    print(f'  relative to real: BER {rel["ber_rel_pct"]:+.0f}%  A-NCB '
          f'{rel["a_ncb_rel_pct"]:+.0f}%  AUC {rel["auc_rel_pct"]:+.0f}%')
    print(f'  downstream AUC {util_auc:.3f} (real {util_auc_real:.3f}); '
          f'income rate {syn_income_rate:.3f} (real {real_income_rate:.3f})')
    print(f'  alpha-precision {alpha_p:.3f}  beta-recall {beta_r:.3f}  '
          f'(real held-out {alpha_p_real:.3f} / {beta_r_real:.3f})')
    print(f'  validity {validity["substantive_valid_rate"]:.3f} '
          f'(real held-out {validity_real["substantive_valid_rate"]:.3f})')
    print(f'  largest non-sex category share change '
          f'{max(v for j, v in all_tv.items() if j != s_idx):.3f}; '
          f'match-valid {match_valid}')
    print('  eps-identifiability: ' + ('n/a (tier-1)' if eps_id is None
                                       else f'{eps_id:.3f}'))
    print(f'  fidelity: {fid}')
    if 'eps_total' in result:
        print(f'  total epsilon={result["eps_total"]:.2f}')
    print(f'  runtime {result["runtime_s"]:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
    main()
