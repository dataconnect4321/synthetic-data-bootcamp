"""Run FLIP (VAE + latent diffusion) on UCI Adult with `sex` as the protected
attribute - the paper's Adult setup (arXiv:2508.21815, Table 2: 6 numeric +
8 categorical columns) - and score it with the paper's metrics and the
checks added in this reproduction.

One run = one 3-fold-CV fold: the generator trains on 2/3 of the rows, the
held-out 1/3 is the unseen real reference. Reuses the HMDA pipeline's
training / diffusion / decoding code by pointing run_flip_demo's column
configuration at Adult.

Final setup (FLIP + minimal fixes; see README):
  --paper-config --ls-mode batch_mean   the paper's training setup
  --mask-proxies                        merge auto-detected deterministic
                                        proxy classes (Husband/Wife -> MASK)
  --decoded-joint                       joint decoded-output CKA^T stage
  --marginal-weight 1.0                 KL marginal guard (no deleted
                                        categories)
  --stage-reduce sum                    full lambda per stage
  --s-decode sample                     generated sex drawn from its
                                        probabilities, not argmax

Examples:
    # bank Phase 1 for a fold (tier-1 metrics of the Phase-1 model)
    python run_adult_flip.py --paper-config --ls-mode batch_mean \\
        --mask-proxies --fold 0 --phase2-epochs 0 --tier1 \\
        --save-phase1 adult_results/ckpt/fin_f0.pt --tag fin_f0_bank
    # Phase 2 at lambda=8 from the bank, full pipeline
    python run_adult_flip.py --paper-config --ls-mode batch_mean \\
        --mask-proxies --decoded-joint --marginal-weight 1.0 \\
        --stage-reduce sum --s-decode sample --fold 0 --lam 8 \\
        --phase1-checkpoint adult_results/ckpt/fin_f0.pt --tag flip_f0_lam8
    # comparison baseline: scrambling repair, no in-model fairness
    python run_adult_flip.py ... --repair-tau 1.0 --lam 0 --tag rpb_f0_lam0
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
from tabsyn.vae.flip_fairness import compose_rdp_budgets
from tabsyn.vae.paper_metrics import (fairness_scores, epsilon_identifiability,
                                      infer_protected, group_share_tv,
                                      categorical_share_tv_all,
                                      true_group_target_gap,
                                      downstream_sp_eo,
                                      alpha_precision_beta_recall,
                                      fit_validity_rules, validity_report)
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
        'MEAN_SHIFT_WEIGHT': 1.0,
        'BINARY_W1': True,
        'USE_DP': args.dp_noise is not None,
        'SEED': args.seed,
        'PHASE1_EPOCHS': args.phase1_epochs,
        'PHASE2_EPOCHS': args.phase2_epochs,
        'SAMPLE_SIZE': None,
        'WEIGHT_DECAY': args.weight_decay,
        'LS_MODE': args.ls_mode,
        'MARGINAL_WEIGHT': args.marginal_weight,
        'STAGE_REDUCE': args.stage_reduce,
        'DECODED_JOINT': args.decoded_joint,
    })
    if args.paper_config:
        # Paper training setup: full data, large batch (the fairness terms
        # are batch statistics - small batches make them noisy), small
        # tokens, wide FFN, plateau beta decay. PHASE2_EPOCHS comes from
        # --phase2-epochs (0 to bank Phase 1) and is not overwritten.
        params.update(PHASE1_EPOCHS=3000, BATCH_SIZE=4096, D_TOKEN=4,
                      FACTOR=32, BETA_SCHEDULE='plateau',
                      BETA_MAX=1e-2, BETA_MIN=1e-5,
                      BETA_LAMBDA=0.7, BETA_PATIENCE=10, PHASE2_LR_SCALE=1.0)
    if args.dp_noise is not None:
        params['NOISE_MULTIPLIER'] = args.dp_noise
    dparams = copy.deepcopy(full.DIFFUSION_PARAMS)
    dparams.update({
        'USE_DP': args.dp_noise is not None,
        'EPOCHS': args.diff_epochs,
        'BATCH_SIZE': args.diff_batch,
        'N_GEN': None,
    })
    if args.dp_noise is not None:
        dparams['NOISE_MULTIPLIER'] = args.dp_noise
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


def real_ref_cache(args, version='clean_v1'):
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
                    help='DP-SGD noise multiplier (both stages); omit = no DP')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--split-seed', type=int, default=42,
                    help='seed for the fold split (fixed across --seed values '
                         'so seed sweeps vary only training, not data)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--phase1-epochs', type=int, default=300)
    ap.add_argument('--phase2-epochs', type=int, default=100)
    ap.add_argument('--weight-decay', type=float, default=0.0,
                    help='Adam L2 weight decay; the HMDA default (1e-2) '
                         'collapses the VAE to the mean on Adult')
    ap.add_argument('--ls-mode', choices=['per_sample', 'batch_mean'],
                    default='per_sample',
                    help='L_S variant; batch_mean is the paper-literal Eq. 3')
    ap.add_argument('--s-decode', choices=['argmax', 'sample'],
                    default='argmax',
                    help='decode the protected column by argmax or by '
                         'sampling its probabilities. Under the batch-mean '
                         'L_S every row sits within ~0.01 of uniform, so '
                         'argmax either copies the true group through that '
                         'tilt or collapses to one group; sampling gives the '
                         'balanced assignment the paper (Sec. 4.2) intends')
    ap.add_argument('--diff-epochs', type=int, default=2000)
    ap.add_argument('--diff-batch', type=int, default=4096)
    ap.add_argument('--paper-config', action='store_true',
                    help="the paper's training setup: full data, batch 4096, "
                         '3000 Phase-1 epochs, d_token 4, factor 32, beta '
                         'decay')
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
    ap.add_argument('--mask-proxies', action='store_true',
                    help='merge each auto-detected near-deterministic proxy '
                         'class (P(S|category) >= 0.95 pointing to different '
                         'groups, e.g. Husband/Wife) into one new category in '
                         'both folds before training')
    ap.add_argument('--mask-label', default='MASK',
                    help='name of the merged category')
    ap.add_argument('--decoded-joint', action='store_true',
                    help='add the joint decoded-output stage: CKA^T + mean '
                         'gap on the whole decoded row (RFF-lifted), weighted '
                         'by lambda')
    ap.add_argument('--marginal-weight', type=float, default=0.0,
                    help='weight of the KL marginal guard (stops the fairness '
                         'terms from deleting categories)')
    ap.add_argument('--stage-reduce', choices=['mean', 'sum'],
                    default='mean',
                    help='combine the latent / detokenizer / decoder stages '
                         'by mean or sum (full lambda per stage)')
    ap.add_argument('--repair-tau', type=float, default=0.0,
                    help='COMPARISON BASELINE: scramble every non-target '
                         "column's link to sex with strength tau in [0, 1] "
                         "before training, keeping each column's overall "
                         'distribution (Feldman et al. repair)')
    args = ap.parse_args()

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
    vae, accountant, group_mech = demo.train_flip(
        X_num, X_cat, categories, s_idx, params, device,
        phase1_checkpoint=args.phase1_checkpoint,
        factor=params.get('FACTOR', 8))
    if args.save_phase1:
        torch.save(vae.state_dict(), args.save_phase1)
        print(f'  saved VAE checkpoint to {args.save_phase1}')

    # ---- Stage 2: diffusion (or tier-1 decode) ------------------------------
    vae.eval()
    with torch.no_grad():
        enc_out = vae.forward_with_stages(X_num.to(device), X_cat.to(device))
    decode_kw = dict(X_num=X_num, params=params,
                     num_floor=(-data['means'] / data['stds']).values,
                     sample_idx=s_idx if args.s_decode == 'sample' else None)
    if args.tier1:
        # z = mu + sigma*eps from REAL rows ('std_z' is the encoder's
        # log-variance). Cannot collapse; row i is decoded from training row
        # i, so its true sex is known.
        print('\n[Stage 2] SKIPPED (--tier1): decoding z = mu + sigma*eps')
        sigma = torch.exp(0.5 * enc_out['std_z'])
        z = enc_out['mu_z'] + sigma * torch.randn_like(sigma)
        syn_num, syn_cat, stats_budgets = full.decode_latents(
            vae, z, device, **decode_kw)
        diff_acc = None
    else:
        print('\n[Stage 2] latent diffusion')
        mu_z = enc_out['mu_z']
        latents = mu_z.reshape(mu_z.shape[0], -1).cpu().numpy()
        # trained on mu (std=None), as in the final runs
        diff_model, z_mean, diff_acc = full.train_diffusion(
            latents, params=dparams, device=device)
        sampled = full.sample_from_diffusion(diff_model, z_mean, len(X_num),
                                             mu_z.shape[1:], dparams, device)
        syn_num, syn_cat, stats_budgets = full.decode_latents(
            vae, sampled, device, **decode_kw)

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
    fid = demo.fidelity_metrics(tr_n, syn_num, tr_c, syn_cat, enc)

    # Collapse guard: a collapsed run must never reach a results table.
    t_pos = list(enc[TARGET]).index('1')
    syn_income_rate = float((syn_cat[:, t_idx] == t_pos).mean())
    real_income_rate = float((tr_c[:, t_idx] == t_pos).mean())
    collapsed = (fid['num_w1_mean'] > 0.15
                 or abs(syn_income_rate - real_income_rate) > 0.05)
    if collapsed:
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

    def _util_feats(num, cat):
        keep = [j for j in range(cat.shape[1]) if j not in (s_idx, t_idx)]
        return np.concatenate([num] + [np.eye(categories[j])[cat[:, j]]
                                       for j in keep], axis=1)
    util_auc = float(roc_auc_score(
        te_c[:, t_idx] == t_pos,
        HistGradientBoostingClassifier(random_state=0).fit(
            _util_feats(syn_num, syn_cat), syn_cat[:, t_idx] == t_pos)
        .predict_proba(_util_feats(te_n, te_c))[:, 1]))
    util_auc_real = ref_cached('util_auc_real', lambda: float(roc_auc_score(
        te_c[:, t_idx] == t_pos,
        HistGradientBoostingClassifier(random_state=0).fit(
            _util_feats(tr_n, tr_c), tr_c[:, t_idx] == t_pos)
        .predict_proba(_util_feats(te_n, te_c))[:, 1])))
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
        'tag': tag, 'args': vars(args), 'n_train': len(X_num),
        'n_test': len(te_n),
        'paper_metrics': {
            'syn': syn, 'real_test': real_test,
            'real_test_reencoded': real_test_masked,
            'eps_identifiability': eps_id,
            'decoded_leakage': leak, 'heldout_ceiling': ceiling_ho,
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
            'outcome_replication': rel, 'match_valid': match_valid,
            'impossible_combination_rate': imposs,
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
        eps_total, _ = compose_rdp_budgets(budgets, params['DP_DELTA'])
        result['eps_total'] = float(eps_total)
    result['runtime_s'] = time.time() - t0
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    print(f'  BER   syn={syn["ber"]:.3f}  real(test)={real_test["ber"]:.3f}  '
          f'(chance {syn["ber_chance"]:.2f})')
    print(f'  A-NCB syn={syn["a_ncb"]:.3f} (median {syn["a_ncb_median"]:.3f})'
          f'  real(test)={real_test["a_ncb"]:.3f}')
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
