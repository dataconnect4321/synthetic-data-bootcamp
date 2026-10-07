"""Run the full FLIP pipeline (FLIP VAE + latent diffusion) on the UCI Adult
dataset with `sex` as the protected attribute - the paper's Adult setup
(arXiv:2508.21815, Table 2: 6 numeric + 8 categorical columns) - and score
it with the paper's metrics (BER, A-NCB, eps-identifiability).

Reuses the HMDA pipeline's training / diffusion / decoding code by pointing
run_flip_demo's column configuration at Adult. One run = one 3-fold-CV
fold: the generator trains on 2/3 of the rows, the held-out 1/3 is the
unseen real reference for BER / A-NCB.

Examples:
    python run_adult_flip.py --lam 4 --tag lam4_nodp
    python run_adult_flip.py --tier1 --decouple-curve --tag dec_posthoc
    python run_adult_flip.py --tier1 --proxy-weight 8 --proxy-exclude-disent \
        --tag dec_lam8
    python run_adult_flip.py --lam 4 --paper-literal --tag lam4_literal
    python run_adult_flip.py --diagnose-only          # shared-mean check
"""

import os
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
from tabsyn.vae.flip_fairness import (compose_rdp_budgets, shared_mean_ratio)
from tabsyn.vae.paper_metrics import (fairness_scores, epsilon_identifiability,
                                      label_validity_auc, inferred_group_gap,
                                      group_share_tv, categorical_share_tv_all,
                                      tier1_group_agreement,
                                      true_group_target_gap,
                                      downstream_sp_eo, proxy_ranking)

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


def prepare(df, fold, n_folds, seed, sample=None):
    """Fold split + encoding. Numeric stats come from the training fold only;
    category codes are shared across folds. sample=(n, [strat_cols]) draws a
    PROPORTIONAL stratified subsample of the training fold (stratifying by
    sex alone would shift the class balance and move the real baseline)."""
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
    if sample is not None:
        n_s, strat_cols = sample
        rng = np.random.default_rng(seed)
        keys = df.iloc[tr_idx][strat_cols].astype(str).agg(' | '.join, axis=1).values
        take = []
        for v in np.unique(keys):
            idx_v = np.array(tr_idx)[keys == v]
            # proportional allocation, rounded, remainder to the largest cells
            take.extend(rng.choice(idx_v, size=int(round(n_s * len(idx_v) / len(tr_idx))),
                                   replace=False))
        tr_idx = np.array(sorted(take))
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
        'MEAN_SHIFT_WEIGHT': 0.0 if args.paper_literal else 1.0,
        'BINARY_W1': not args.paper_literal,
        'USE_DP': args.dp_noise is not None,
        'SEED': args.seed,
        'PHASE1_EPOCHS': args.phase1_epochs,
        'PHASE2_EPOCHS': 0 if args.diagnose_only else args.phase2_epochs,
        'SAMPLE_SIZE': None,
        'WEIGHT_DECAY': args.weight_decay,
        'LS_MODE': args.ls_mode,
        'RECON_S': args.recon_s,
        'MARGINAL_WEIGHT': args.marginal_weight or 0.0,
        'STAGE_REDUCE': args.stage_reduce,
        'MARGINAL_KIND': args.marginal_kind,
        'PROXY_WEIGHT': args.proxy_weight,
        'DECODED_JOINT': args.decoded_joint,
        'PROXY_MODE': args.proxy_mode,
        'PROXY_EXCLUDE_DISENT': args.proxy_exclude_disent,
    })
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


def averaged_scores(num, cat, s_idx, ref=None, fast=False, n_seeds=None,
                    **cluster_kw):
    """fairness_scores averaged over seeds (classifier + GMM init): A-NCB is a
    minimum over clusters, so a single seed is noisy. Fast runs use 2 seeds.
    cluster_kw (covariance_type, n_init) selects the A-NCB clustering; the
    median is reported alongside the mean because the full-covariance fit
    can land on different splits per seed."""
    seeds = range(n_seeds or (2 if fast else 5))
    runs = [fairness_scores(num, cat, s_idx, ref=ref, seed=s, **cluster_kw)
            for s in seeds]
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs])
        out[k], out[k + '_sd'] = float(v.mean()), float(v.std())
        out[k + '_median'] = float(np.median(v))
    return out


def _probe_balanced_acc(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    clf = HistGradientBoostingClassifier(class_weight='balanced',
                                         random_state=seed)
    pred = cross_val_predict(clf, X, y,
                             cv=StratifiedKFold(3, shuffle=True,
                                                random_state=seed))
    return float(np.mean([(pred[y == g] == g).mean() for g in np.unique(y)]))


def latent_probe(vae, num, cat, s_idx, device, seed=0):
    """Balanced accuracy of a nonlinear probe predicting real sex from the
    VAE's latent, with and without the sex TOKEN. The latent is laid out
    [CLS, num_1..num_d, cat_1..cat_k], so the sex token sits at
    1 + d_num + s_idx; keeping it lets the probe read the token directly,
    which measures tokenization, not what the other columns encode. The
    dropped version is the honest 'do the other columns still encode sex'
    number; the kept version is reported alongside for comparison."""
    vae.eval()
    with torch.no_grad():
        mu = vae.forward_with_stages(torch.tensor(num).to(device),
                                     torch.tensor(cat).to(device))['mu_z']
    y = cat[:, s_idx].astype(int)
    d_num = num.shape[1]
    s_token = 1 + d_num + s_idx
    keep = [t for t in range(mu.shape[1]) if t != s_token]
    return {
        'drop_s_token': _probe_balanced_acc(
            mu[:, keep].reshape(mu.shape[0], -1).cpu().numpy(), y, seed),
        'with_s_token': _probe_balanced_acc(
            mu.reshape(mu.shape[0], -1).cpu().numpy(), y, seed),
    }


def proxy_latent_probe(vae, num, cat, s_idx, proxy_idx, codes, device,
                       seed=0):
    """Balanced accuracy of a probe predicting the merged column's
    sex-specific label (e.g. Husband vs Wife) from the latent with the proxy
    and sex tokens removed, on the rows holding that label. The direct
    in-model measure of decoupling: chance 0.5 = the other tokens no longer
    carry the label."""
    vae.eval()
    with torch.no_grad():
        mu = vae.forward_with_stages(torch.tensor(num).to(device),
                                     torch.tensor(cat).to(device))['mu_z']
    mask = np.isin(cat[:, proxy_idx], codes)
    d_num = num.shape[1]
    drop = {1 + d_num + proxy_idx, 1 + d_num + s_idx}
    keep = [t for t in range(mu.shape[1]) if t not in drop]
    X = mu[:, keep].reshape(mu.shape[0], -1).cpu().numpy()[mask]
    y = (cat[mask, proxy_idx] == codes[0]).astype(int)
    return _probe_balanced_acc(X, y, seed)


def real_feature_ceiling(num, cat, s_idx, seed=0):
    """Probe ceiling: balanced accuracy predicting sex from the REAL
    non-sex columns (numeric + one-hot categorical). How much sex the
    real features carry - the upper bound any latent trained on them can
    plausibly encode."""
    sizes = [len(np.unique(cat[:, j])) for j in range(cat.shape[1])]
    parts = [num]
    for j in range(cat.shape[1]):
        if j == s_idx:
            continue
        parts.append(np.eye(max(sizes[j], int(cat[:, j].max()) + 1))[cat[:, j]])
    return _probe_balanced_acc(np.concatenate(parts, axis=1),
                               cat[:, s_idx].astype(int), seed)


def decoded_leakage(ref_num, ref_cat, syn_num, syn_cat, true_sex, s_idx,
                    seed=0):
    """Balanced accuracy of a real-trained sex classifier predicting the
    SOURCE row's true sex from the DECODED non-sex columns. Tier-1 decodes
    real rows, so each decoded row has a known true group. Near 0.5 = the
    proxies really were removed; near the real-feature ceiling = they
    weren't, and any 'fair' gap comes from the synthetic sex column."""
    from tabsyn.vae.paper_metrics import infer_protected
    pred = infer_protected(ref_num, ref_cat, syn_num, syn_cat, s_idx,
                           seed=seed)
    y = np.asarray(true_sex).astype(int)
    return float(np.mean([(pred[y == g] == g).mean()
                          for g in np.unique(y)]))


def relationship_faithfulness(syn_cat, src_cat, rel_idx, labels):
    """Decoded `relationship` vs the source row's: exact agreement plus the
    Husband/Wife shares of BOTH the decoded and source rows. If Phase 2
    reached parity by rewriting marital proxies, it shows up here - a
    fidelity loss on exactly the column that carries sex in Adult."""
    agree = float((syn_cat[:, rel_idx] == src_cat[:, rel_idx]).mean())
    shares, src_shares = {}, {}
    for lab in labels:
        code = int(np.where(labels == lab)[0][0]) if lab in labels else None
        if code is not None:
            shares[lab] = float((syn_cat[:, rel_idx] == code).mean())
            src_shares[lab] = float((src_cat[:, rel_idx] == code).mean())
    return {'agreement': agree, 'shares': shares, 'src_shares': src_shares}


def latent_diagnostic(vae, num, cat, s_idx, device):
    """How large is the offset shared by all groups in real mu_z, relative to
    its spread, and how big is the group mean gap? Decides whether the
    paper-literal row-centering (dim=1) CKA^T would be safe."""
    vae.eval()
    with torch.no_grad():
        mu = vae.forward_with_stages(torch.tensor(num).to(device),
                                     torch.tensor(cat).to(device))['mu_z']
    mu = mu.reshape(mu.shape[0], -1)
    g = torch.tensor(cat[:, s_idx]).to(device)
    a, b = mu[g == 0], mu[g == 1]
    pooled = ((a.shape[0] - 1) * a.var(0) + (b.shape[0] - 1) * b.var(0)) \
        / (a.shape[0] + b.shape[0] - 2)
    return {
        'shared_mean_ratio': shared_mean_ratio(mu),
        'group_mean_gap_normalized': float(
            (a.mean(0) - b.mean(0)).pow(2).sum() / pooled.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--lam', type=float, default=4.0, help='LAMBDA_FAIR')
    ap.add_argument('--dp-noise', type=float, default=None,
                    help='DP-SGD noise multiplier (both stages); omit = no DP')
    ap.add_argument('--paper-literal', action='store_true',
                    help='disable the mean-shift term and binary-column W1 '
                         '(paper-literal CKA^T only)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--split-seed', type=int, default=42,
                    help='seed for the fold split and subsample (fixed '
                         'across --seed values so seed sweeps vary only '
                         'training, not data)')
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
                    help='how to decode the protected column from its '
                         'logits; argmax matches the other columns and the '
                         'paper runs, sample avoids the degenerate column '
                         'the per-sample L_S produces')
    ap.add_argument('--cat-decode', choices=['argmax', 'sample'],
                    default='argmax',
                    help='decode every categorical by argmax (default) or by '
                         'sampling its softmax, so generated shares equal the '
                         'mean probabilities the losses train on')
    ap.add_argument('--diff-epochs', type=int, default=2000)
    ap.add_argument('--diff-batch', type=int, default=4096)
    ap.add_argument('--paper-config', action='store_true',
                    help='the paper\'s training setup: full data, batch '
                         '4096, 3000+1000 epochs, d_token 4, factor 32, '
                         'beta decay. The fairness terms are batch '
                         'statistics, so a small batch makes their gradients '
                         'noisy - the suspected cause of the collapses')
    ap.add_argument('--tier1', action='store_true',
                    help='skip diffusion: decode z = mu + sigma*eps from '
                         'real rows (reconstruct-through-the-latent). '
                         'Screens fairness only; privacy metrics are '
                         'meaningless here and are not reported')
    ap.add_argument('--phase1-checkpoint', default=None,
                    help='with --tier1: load a saved Phase-1 VAE instead '
                         'of training it (path from --save-phase1)')
    ap.add_argument('--save-phase1', default=None,
                    help='save the Phase-1(+Phase-2) VAE checkpoint to this '
                         'path (use with --phase2-epochs 0 to bank Phase 1)')
    ap.add_argument('--marginal-weight', type=float, default=None,
                    help='marginal-matching penalty weight (shares + '
                         'numeric quantiles); default 0, or 1.0 under '
                         '--paper-config. Not in the paper: it resists the '
                         'output shift CKA^T needs')
    ap.add_argument('--marginal-kind', choices=['tv', 'kl'], default='tv',
                    help='marginal penalty form: tv (mean-softmax TV averaged '
                         'over columns, original) or kl (KL(real||syn) on '
                         'sharpened shares, summed over columns - stops '
                         'categories from vanishing)')
    ap.add_argument('--stage-reduce', choices=['mean', 'sum'],
                    default='mean',
                    help='combine the latent / detokenizer / decoder '
                         'disentanglement stages by mean (repo default) or '
                         'sum (full lambda per stage)')
    ap.add_argument('--recon-s', action='store_true',
                    help='keep the protected attribute in the reconstruction '
                         'CE so the synthetic S column is real (needed for '
                         'label validity, share TV, and the decoupling '
                         'baselines)')
    ap.add_argument('--proxy-col', default=None,
                    help='merged column whose sex-specific label is decoupled; '
                         'default: auto-detect via detect_proxy_classes '
                         '(P(S|category) >= 0.95, min support 1%%)')
    ap.add_argument('--proxy-labels', default=None,
                    help='the two sex-specific labels of --proxy-col; '
                         'default: auto-detected')
    ap.add_argument('--mask-proxies', action='store_true',
                    help='pre-training decoupling: merge each auto-detected '
                         'proxy class (e.g. Husband/Wife) into one new '
                         'category in both folds. The real BER/A-NCB '
                         'reference uses the original data; a masked-real '
                         'reference is reported alongside')
    ap.add_argument('--decoded-joint', choices=['off', 'linear', 'rff'],
                    default='off',
                    help='add CKA^T + mean gap on the JOINT decoded output '
                         '(what A-NCB sees), weighted by lambda; rff lifts '
                         'it with random Fourier features first')
    ap.add_argument('--repair-tau', type=float, default=0.0,
                    help='pre-training scrambling repair (Feldman et al.): '
                         'scramble every non-target column\'s link to sex '
                         'with strength tau in [0, 1], keeping each '
                         'column\'s overall distribution (nothing can '
                         'vanish). Applied after --mask-proxies')
    ap.add_argument('--mask-label', default='MASK',
                    help='name of the merged category (default MASK)')
    ap.add_argument('--decouple-curve', action='store_true',
                    help='post-hoc decoupling: re-draw the sex-specific '
                         'label from P(label | others)^tau * pi and sweep '
                         'tau (the target curve for --proxy-weight)')
    ap.add_argument('--proxy-weight', type=float, default=0.0,
                    help='in-model decoupling weight lambda: makes the '
                         'sex-specific label independent of the other '
                         'tokens (Phase 2)')
    ap.add_argument('--proxy-mode', choices=['hsic', 'adv'], default='hsic',
                    help='hsic (batch-coupled) or gradient-reversal '
                         'adversary (per-record)')
    ap.add_argument('--proxy-exclude-disent', action='store_true',
                    help='apply CKA^T to the non-proxy tokens only')
    ap.add_argument('--target-in-fairness', action='store_true',
                    help='include the target column\'s logits in the '
                         'detokenizer-stage CKA/mean/W1 terms (default: '
                         'excluded unless outcome parity is intended)')
    ap.add_argument('--diagnose-only', action='store_true',
                    help='train Phase 1 only and print the shared-mean check')
    args = ap.parse_args()

    tag = args.tag or f'lam{args.lam:g}_fold{args.fold}'
    t0 = time.time()
    configure_demo()
    params, dparams = build_params(args)
    if args.paper_config:
        # Paper training setup: full data, large batch (the fairness terms
        # are batch statistics - small batches make them noisy, which pushes
        # toward collapse), small tokens, wide FFN, and plateau beta decay.
        # NOTE: PHASE2_EPOCHS comes from --phase2-epochs (0 to bank Phase 1)
        # so this must NOT overwrite it.
        params.update(PHASE1_EPOCHS=3000,
                      BATCH_SIZE=4096, D_TOKEN=4, FACTOR=32,
                      BETA_MAX=1e-2, BETA_MIN=1e-5, BETA_LAMBDA=0.7,
                      BETA_PATIENCE=10, PHASE2_LR_SCALE=1.0,
                      MARGINAL_WEIGHT=(1.0 if args.marginal_weight is None
                                       else args.marginal_weight))
    device = torch.device(params['DEVICE'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = prepare(read_adult(), args.fold, args.folds, args.split_seed)
    X_num = torch.tensor(data['train_num'])
    X_cat = torch.tensor(data['train_cat'])
    enc = data['encoders']
    categories = [len(enc[c]) for c in CATEGORICAL]
    s_idx, t_idx = CATEGORICAL.index(SENSITIVE), CATEGORICAL.index(TARGET)
    # Automatic proxy detection (dataset-agnostic): near-deterministic
    # categories of any column, replacing the hard-coded relationship/
    # Husband,Wife. Manual --proxy-col/--proxy-labels still override.
    from tabsyn.vae.proxy_decouple import (detect_proxy_classes,
                                           describe_classes)
    detected = detect_proxy_classes(data['train_cat'], s_idx,
                                     exclude=(t_idx,))
    if detected:
        print('  auto-detected proxy classes: '
              + '; '.join(describe_classes(detected, CATEGORICAL, enc)))
    # Pre-training masking: each detected class is merged into ONE new
    # category (Husband, Wife -> MASK) in BOTH folds, so the generator never
    # sees the sex-specific label and every table scored against the
    # synthetic one shares its vocabulary. The raw copies serve only as the
    # original-data real reference (BER/A-NCB) and to define impossible
    # combinations.
    tr_c_raw, te_c_raw = data['train_cat'].copy(), data['test_cat'].copy()
    if args.mask_proxies:
        from tabsyn.vae.proxy_decouple import mask_proxy_classes
        if not detected:
            raise SystemExit('--mask-proxies: no proxy class detected')
        mask_codes = []
        for cl in detected:
            col = CATEGORICAL[cl['col']]
            mask_codes.append(len(enc[col]))
            enc[col] = np.append(enc[col], args.mask_label)
        data['train_cat'] = mask_proxy_classes(tr_c_raw, detected, mask_codes)
        data['test_cat'] = mask_proxy_classes(te_c_raw, detected, mask_codes)
        X_cat = torch.tensor(data['train_cat'])
        categories = [len(enc[c]) for c in CATEGORICAL]
        changed = float((data['train_cat'] != tr_c_raw).any(axis=1).mean())
        print(f'  masked proxy classes -> {args.mask_label!r}: '
              f'{changed:.3f} of training rows changed')
    # Pre-training scrambling repair: the GENERATOR trains on a copy of the
    # training fold whose columns' links to sex are scrambled with strength
    # tau (overall distributions kept). data['train_*'] stays unrepaired, so
    # every quality metric is measured against the real population.
    if args.repair_tau:
        from tabsyn.vae.proxy_decouple import repair_toward_independence
        rep_n, rep_c = repair_toward_independence(
            data['train_num'], data['train_cat'], s_idx, args.repair_tau,
            exclude=(t_idx,), seed=args.split_seed)
        X_num, X_cat = torch.tensor(rep_n), torch.tensor(rep_c)
        changed = float((rep_c != data['train_cat']).any(axis=1).mean())
        print(f'  scrambling repair tau={args.repair_tau:g}: {changed:.3f} '
              f'of training rows got a new category')
    # Proxy decoupling is optional: a plain baseline (no --proxy-weight,
    # no --decouple-curve) must run on datasets without a near-1:1 proxy.
    need_proxy = args.proxy_weight > 0 or args.decouple_curve
    if args.proxy_col is None:
        if not detected:
            if need_proxy:
                raise SystemExit('no proxy class detected; pass --proxy-col')
            proxy_idx = proxy_codes = None
            proxy_labels = []
        else:
            if len(detected[0]['members']) != 2:
                raise SystemExit(
                    'proxy decoupling currently supports 2-member classes; '
                    'got ' + str(detected[0]['members']))
            if len(detected) > 1:
                print(f'  NOTE: {len(detected)} proxy classes found; '
                      'decoupling only the first')
            proxy_idx = detected[0]['col']
            proxy_codes = tuple(detected[0]['members'])
            proxy_labels = [str(enc[CATEGORICAL[proxy_idx]][c])
                            for c in proxy_codes]
    else:
        proxy_idx = CATEGORICAL.index(args.proxy_col)
        manual = detect_proxy_classes(data['train_cat'], s_idx,
                                      exclude=(t_idx,))
        manual = [d for d in manual if d['col'] == proxy_idx]
        proxy_labels = (args.proxy_labels.split(',')
                        if args.proxy_labels else
                        ([str(enc[args.proxy_col][c]) for c in
                          manual[0]['members']] if manual else None))
        if proxy_labels is None or len(proxy_labels) != 2:
            raise SystemExit('proxy decoupling currently supports 2-member '
                             'classes; pass --proxy-labels with exactly two '
                             'labels')
        proxy_codes = tuple(int(np.where(enc[args.proxy_col] == lab)[0][0])
                            for lab in proxy_labels)
    if args.mask_proxies:
        # the sex-specific labels no longer exist in the masked data
        proxy_idx = proxy_codes = None
        proxy_labels = []
    params.update(PROXY_IDX=proxy_idx, PROXY_CODES=proxy_codes,
                  TARGET_IN_FAIRNESS=args.target_in_fairness,
                  TARGET_IDX=t_idx)
    print(f'[{tag}] train={len(X_num)} test={len(data["test_num"])} '
          f'categories={categories} sex={list(enc[SENSITIVE])}')

    # ---- Stage 1 -----------------------------------------------------------
    print('\n[Stage 1] FLIP VAE')
    # The checkpoint goes INTO train_flip: it skips Phase-1 epochs but still
    # builds the sampler, optimizer, frozen reference encoder, and runs
    # Phase 2 - so --lam still takes effect per setting.
    vae, accountant, group_mech = demo.train_flip(
        X_num, X_cat, categories, s_idx, params, device,
        phase1_checkpoint=args.phase1_checkpoint,
        factor=params.get('FACTOR', 8))
    if args.save_phase1:
        torch.save(vae.state_dict(), args.save_phase1)
        print(f'  saved VAE checkpoint to {args.save_phase1}')
    diag = latent_diagnostic(vae, data['train_num'], data['train_cat'],
                             s_idx, device)
    print(f'  latent diagnostic: {diag}')
    if args.diagnose_only:
        return

    # ---- Stage 2 -----------------------------------------------------------
    vae.eval()
    with torch.no_grad():
        enc_out = vae.forward_with_stages(X_num.to(device),
                                          X_cat.to(device))
    if args.tier1:
        # reconstruct-through-the-latent: z = mu + sigma*eps from REAL rows.
        # Cannot collapse (it starts from real rows), so it screens the
        # fairness effect of Phase 2 without training the diffusion prior.
        # Privacy metrics are meaningless here and are skipped.
        # NOTE: the VAE's 'std_z' field is the encoder's LOG-VARIANCE
        # (reparameterize does exp(0.5*logvar)); sigma = exp(0.5*logvar).
        print('\n[Stage 2] SKIPPED (--tier1): decoding z = mu + sigma*eps')
        sigma = torch.exp(0.5 * enc_out['std_z'])
        z = enc_out['mu_z'] + sigma * torch.randn_like(sigma)
        syn_num, syn_cat, stats_budgets = full.decode_latents(
            vae, z, device, X_num=X_num, params=params,
            num_floor=(-data['means'] / data['stds']).values,
            sample_idx=s_idx if args.s_decode == 'sample' else None,
            sample_all=args.cat_decode == 'sample')
        diff_acc = None
    else:
        print('\n[Stage 2] latent diffusion')
        mu_z = enc_out['mu_z']
        latent_shape = mu_z.shape[1:]
        latents = mu_z.reshape(mu_z.shape[0], -1).cpu().numpy()
        diff_model, z_mean, diff_acc = full.train_diffusion(
            latents, dparams, device)
        sampled = full.sample_from_diffusion(diff_model, z_mean, len(X_num),
                                             latent_shape, dparams, device)
        syn_num, syn_cat, stats_budgets = full.decode_latents(
            vae, sampled, device, X_num=X_num, params=params,
            num_floor=(-data['means'] / data['stds']).values,
            sample_idx=s_idx if args.s_decode == 'sample' else None,
            sample_all=args.cat_decode == 'sample')

    # Released CSV only: clip + round numerics to the real columns' format
    # (whole numbers stay whole, etc.). Metrics below use the unrounded
    # decode, so runs before/after this change score identically. Under DP
    # the real min/max is not released; the decoder's DP clip applies.
    num_schema = demo.infer_numeric_schema(
        data['train_num'] * data['stds'].values + data['means'].values,
        NUMERIC)
    syn_df = demo.decode_synthetic_data(syn_num, syn_cat, data['means'],
                                        data['stds'], enc,
                                        num_schema=num_schema,
                                        schema_bounds=args.dp_noise is None)
    os.makedirs(OUT_DIR, exist_ok=True)
    syn_df.to_csv(os.path.join(OUT_DIR, f'{tag}_synthetic.csv'), index=False)

    # ---- Metrics -----------------------------------------------------------
    print('\n[Metrics]')
    tr_n, tr_c = data['train_num'], data['train_cat']
    # Adversary reference = HELD-OUT fold (same in-sample-memorization fix as
    # the leakage check: training on tr_n/tr_c and predicting their decoded
    # near-copies overstates separability).
    ref_ho = (data['test_num'], data['test_cat'])
    syn = averaged_scores(syn_num, syn_cat, s_idx, ref=ref_ho)
    # A-NCB (paper-style): adversary trained out-of-fold on the SYNTHETIC
    # table's own S column (ref=None). Under the paper's design S is largely
    # invented, so this adversary learns an S that barely follows the
    # features - its labels look balanced in every cluster and A-NCB rises
    # BY CONSTRUCTION. Reported alongside the strict version (adversary
    # trained on real data, ref=ref_ho above) so the two readings of
    # 'inferred from all other features' are distinguishable.
    syn_paper = averaged_scores(syn_num, syn_cat, s_idx, ref=None)
    # A-NCB real reference: the SAME procedure as the synthetic score - an
    # adversary trained on the real TRAINING fold, applied to the real TEST
    # fold (previously the synthetic-scored adversary was reused, which is
    # a different procedure). ncb_stored is reported alongside.
    real_test = averaged_scores(data['test_num'], te_c_raw, s_idx,
                                ref=(tr_n, tr_c_raw))
    real_train = averaged_scores(tr_n, tr_c_raw, s_idx)
    # A-NCB with a FULL-covariance mixture (sklearn default, n_init=1;
    # likely the paper's setup): on Adult it recovers the married split,
    # unlike the diag fit. 10 seeds; mean and median reported.
    fc = dict(n_seeds=10, covariance_type='full', n_init=1)
    syn_fc = averaged_scores(syn_num, syn_cat, s_idx, ref=ref_ho, **fc)
    real_test_fc = averaged_scores(data['test_num'], te_c_raw, s_idx,
                                   ref=(tr_n, tr_c_raw), **fc)
    real_test_reenc_fc = (averaged_scores(data['test_num'], data['test_cat'],
                                          s_idx, ref=(tr_n, tr_c), **fc)
                          if args.mask_proxies else None)
    # secondary reference: the real data after the same masking
    real_test_reenc = (averaged_scores(data['test_num'], data['test_cat'],
                                       s_idx, ref=(tr_n, tr_c))
                       if args.mask_proxies else None)
    eps_id = None if args.tier1 else epsilon_identifiability(tr_n, tr_c,
                                                            syn_num, syn_cat)
    lv_auc = label_validity_auc(tr_n, tr_c, syn_num, syn_cat, s_idx)
    inf_gap = inferred_group_gap(syn_num, syn_cat, s_idx, t_idx, ref_ho)
    inf_gap_real = inferred_group_gap(tr_n, tr_c, s_idx, t_idx, ref_ho)
    probe = latent_probe(vae, tr_n, tr_c, s_idx, device)
    ceiling = real_feature_ceiling(tr_n, tr_c, s_idx)
    proxy_probe = (proxy_latent_probe(vae, tr_n, tr_c, s_idx, proxy_idx,
                                      proxy_codes, device)
                   if proxy_idx is not None else None)
    if proxy_probe is not None:
        print(f'  latent {"/".join(proxy_labels)} probe (no proxy/sex token): '
              f'{proxy_probe:.3f} (chance 0.5)')
    print(f'  latent sex-probe balanced acc: '
          f'{probe["drop_s_token"]:.3f} (no sex token; '
          f'{probe["with_s_token"]:.3f} with) vs real-feature ceiling '
          f'{ceiling:.3f} (chance 0.5)')
    # Tier-1 only: decoded rows come from real rows, so leakage against the
    # SOURCE row's true sex is measurable directly. The adversary trains on
    # the HELD-OUT fold for both ceiling and leakage: training on the source
    # rows themselves would let the classifier memorize, and decoded rows
    # are near-copies, so in-sample leakage reads too high.
    leak = rel_faith = ceiling_ho = s_agree = None
    te_n, te_c = data['test_num'], data['test_cat']
    if args.tier1:
        from tabsyn.vae.paper_metrics import infer_protected
        true_s = tr_c[:, s_idx].astype(int)
        def _bal(pred):
            return float(np.mean([(pred[true_s == g] == g).mean()
                                  for g in np.unique(true_s)]))
        ceiling_ho = _bal(infer_protected(te_n, te_c, tr_n, tr_c, s_idx))
        leak = _bal(infer_protected(te_n, te_c, syn_num, syn_cat, s_idx))
        s_agree = float((syn_cat[:, s_idx] == tr_c[:, s_idx]).mean())
        rel_idx = CATEGORICAL.index('relationship')
        rel_faith = relationship_faithfulness(syn_cat, tr_c, rel_idx,
                                              enc['relationship'])
        print(f'  decoded-row leakage: {leak:.3f} vs held-out ceiling '
              f'{ceiling_ho:.3f} (chance 0.5)')
        print(f'  synthetic-sex agreement with true sex: {s_agree:.3f} '
              f'(near/below 0.5 = invented column)')
        print(f'  relationship agreement: {rel_faith["agreement"]:.3f} '
              f'Husband/Wife decoded {rel_faith["shares"]} '
              f'source {rel_faith["src_shares"]}')
    fid = demo.fidelity_metrics(tr_n, syn_num, tr_c, syn_cat, enc)

    # Collapse guard: a collapsed run (undertrained prior, degenerate decode,
    # or a Phase-2 that flattens the target column) must never reach a results
    # table. Tier-1 decodes real rows, so the income-rate check compares
    # against the SOURCE rows' rate - it applies in tier-1 too.
    t_pos = list(enc[TARGET]).index('1')
    syn_income_rate = float((syn_cat[:, t_idx] == t_pos).mean())
    real_income_rate = float((tr_c[:, t_idx] == t_pos).mean())
    k_t = len(enc[TARGET])
    p_t = np.bincount(syn_cat[:, t_idx], minlength=k_t) / len(syn_cat)
    q_t = np.bincount(tr_c[:, t_idx], minlength=k_t) / len(tr_c)
    income_tv = float(0.5 * np.abs(p_t - q_t).sum())
    collapsed = (fid['num_w1_mean'] > 0.15
                 or abs(syn_income_rate - real_income_rate) > 0.05)
    if collapsed:
        result = {'tag': tag, 'args': vars(args), 'collapsed': True,
                  'syn_income_rate': syn_income_rate,
                  'real_income_rate': real_income_rate, 'fidelity': fid}
        with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
            json.dump(result, f, indent=2, default=float)
        raise SystemExit(f'COLLAPSED: income {syn_income_rate:.3f} vs real '
                          f'{real_income_rate:.3f}, W1 {fid["num_w1_mean"]:.3f}; '
                          f'result flagged and not scored')
    # alpha-precision / beta-recall (paper Sec. 5.3.1, TabSyn's naive
    # AlphaPrecision): raw-scale numerics + one-hot categoricals, against
    # the training fold the generator learned (masked when masking). The
    # held-out fold scored the same way is the real-data reference.
    from tabsyn.vae.paper_metrics import alpha_precision_beta_recall
    def _raw(n):
        return n * data['stds'].values + data['means'].values
    alpha_p, beta_r = alpha_precision_beta_recall(
        _raw(tr_n), tr_c, _raw(syn_num), syn_cat, categories)
    alpha_p_real, beta_r_real = alpha_precision_beta_recall(
        _raw(tr_n), tr_c, _raw(te_n), te_c, categories)
    print(f'  alpha-precision {alpha_p:.3f}  beta-recall {beta_r:.3f}  '
          f'(real held-out: {alpha_p_real:.3f} / {beta_r_real:.3f})')
    fair = demo.fairness_metrics(tr_c, syn_cat, t_idx, s_idx, enc,
                                 real_num=tr_n, syn_num=syn_num)

    # ---- Generic gates and metrics ---------------------------------------
    s_tv = group_share_tv(syn_cat, tr_c, s_idx)
    all_tv = categorical_share_tv_all(syn_cat, tr_c)
    t1_agree = tier1_group_agreement(syn_cat[:, s_idx], tr_c[:, s_idx]) \
        if args.tier1 else None
    true_gap = true_group_target_gap(syn_cat, tr_c, t_idx, s_idx) \
        if args.tier1 else None
    ds = downstream_sp_eo(syn_num, syn_cat, tr_n, tr_c, t_idx, s_idx)
    # Utility: downstream AUC on the REAL test fold for a model trained on
    # the synthetic data (real-model reference computed alongside).
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    def _util_feats(num, cat):
        keep = [j for j in range(cat.shape[1]) if j not in (s_idx, t_idx)]
        return np.concatenate(
            [num] + [np.eye(len(enc[CATEGORICAL[j]]))[cat[:, j]]
                      for j in keep], axis=1)
    _util_clf = HistGradientBoostingClassifier(random_state=0)
    _util_clf.fit(_util_feats(syn_num, syn_cat), syn_cat[:, t_idx] == t_pos)
    util_auc = float(roc_auc_score(
        te_c[:, t_idx] == t_pos,
        _util_clf.predict_proba(_util_feats(te_n, te_c))[:, 1]))
    _ref_clf = HistGradientBoostingClassifier(random_state=0)
    _ref_clf.fit(_util_feats(tr_n, tr_c), tr_c[:, t_idx] == t_pos)
    util_auc_real = float(roc_auc_score(
        te_c[:, t_idx] == t_pos,
        _ref_clf.predict_proba(_util_feats(te_n, te_c))[:, 1]))
    print(f'  utility: downstream AUC {util_auc:.3f} '
          f'(real-model reference {util_auc_real:.3f}); '
          f'income rate {syn_income_rate:.3f} (real {real_income_rate:.3f}), '
          f'income TV {income_tv:.3f}')
    proxies = proxy_ranking(tr_n, tr_c, s_idx, syn_num, syn_cat, ref_ho)
    print(f'  protected share TV: {s_tv:.3f}; worst cat share TV: '
          f'{max(all_tv.values()):.3f} (col {max(all_tv, key=all_tv.get)})')
    if t1_agree is not None:
        print(f'  tier-1 group agreement: {t1_agree:.3f} '
              f'(chance {1/len(enc[SENSITIVE]):.3f}); '
              f'true-group target gap: {true_gap:.3f}')
    print(f'  downstream SP diff: {ds["sp_diff"]:.3f}  '
          f'EO diff: {ds["eo_diff"]:.3f}')
    print('  top proxies (col, importance/syn dependence): '
          + ', '.join(f"c{p['col']}({'cat' if p['is_cat'] else 'num'}) "
                      f"{p['importance']:.3f}/{p['syn_dependence']:.3f}"
                      for p in proxies))

    # ---- Proxy decoupling ----------------------------------------------
    # Scored against the SOURCE rows' true sex (tier-1 only): the generated
    # S column is invented by design, so it says nothing about the proxy.
    from tabsyn.vae.temper import true_sex_joint_tv, decouple_curve
    from tabsyn.vae.proxy_decouple import (impossible_combination_rate,
                                           proxy_member_dependence)
    sp_joint_tv = decouple_results = None
    if args.tier1 and proxy_idx is not None:
        sp_joint_tv = true_sex_joint_tv(syn_cat, true_s, tr_c, s_idx,
                                        proxy_idx)
        print(f'  (true sex, {args.proxy_col}) joint TV vs real: '
              f'{sp_joint_tv:.3f}')
    if args.decouple_curve:
        decouple_results = decouple_curve(
            te_n, te_c, syn_num, syn_cat, true_s, s_idx, proxy_idx,
            *proxy_codes, [1.0, 0.75, 0.5, 0.25, 0.1, 0.0], real_cat=tr_c)
        print('  post-hoc decoupling (tau: leakage / joint TV / flipped): '
              + '; '.join(f"{r['tau']}: {r['leakage']:.3f}/"
                          f"{r['joint_tv']:.3f}/{r['flipped']:.3f}"
                          for r in decouple_results))

    # ---- Generic proxy metrics (reported on every run) -------------------
    # impossible-combination rate: share of synthetic rows carrying a
    # (protected value, category) pair essentially absent from the real
    # data (the 'male wife' rate, generalized to every column).
    imposs = impossible_combination_rate(syn_cat, tr_c_raw, s_idx,
                                          exclude=(t_idx,))
    imposs_real = impossible_combination_rate(tr_c_raw, tr_c_raw, s_idx,
                                              exclude=(t_idx,))
    print(f'  impossible-combination rate: syn {imposs["any"]:.3f} '
          f'(real reference {imposs_real["any"]:.3f}); worst col '
          f'{max(imposs["by_col"], key=imposs["by_col"].get)} '
          f'{max(imposs["by_col"].values()):.3f}')
    # proxy-member dependence: how well the other columns predict each
    # detected class's member label (balanced acc; chance 1/n_members).
    # tau=1 should drive the synthetic value to the real one's chance level.
    if detected:
        pm_dep_syn = proxy_member_dependence(syn_cat, detected,
                                             exclude=(s_idx, t_idx))
        pm_dep_real = proxy_member_dependence(tr_c_raw, detected,
                                              exclude=(s_idx, t_idx))
        print('  proxy-member dependence (syn/real): '
              + '; '.join(f'{describe_classes([d], CATEGORICAL, enc)[0]} '
                          f'{s:.3f}/{r:.3f}'
                          for d, s, r in zip(detected, pm_dep_syn,
                                             pm_dep_real)))
    else:
        pm_dep_syn = pm_dep_real = None

    # ---- Promotion gates (config thresholds) ------------------------------
    gates = {
        'label_validity_auc_min': 0.6,
        's_share_tv_max': 0.1,
        'worst_cat_share_tv_max': 0.25,
        'num_w1_max': 0.15,
        'income_rate_dev_max': 0.05,
    }
    gate_pass = {
        'label_validity': lv_auc >= gates['label_validity_auc_min'],
        's_share_tv': s_tv <= gates['s_share_tv_max'],
        'worst_cat_share_tv':
            max(all_tv.values()) <= gates['worst_cat_share_tv_max'],
        'fidelity': fid['num_w1_mean'] <= gates['num_w1_max'],
        'utility': abs(syn_income_rate - real_income_rate)
        <= gates['income_rate_dev_max'],
    }
    print(f'  promotion gates: {gate_pass} '
          f'(all passed: {all(gate_pass.values())})')

    # ---- Outcome replication: relative-to-real fields (paper Table 3 form) --
    # NOTE: the paper's Table 3 figures are 4-DATASET AVERAGES (Adult, Dutch,
    # Alabama, Utah), not Adult-specific; Figure 3 is the only Adult result
    # and has no value labels. These are context, not targets to match.
    rel = {
        'ber_rel_pct': 100.0 * (syn['ber'] - real_test['ber']) / real_test['ber'],
        'a_ncb_rel_pct': 100.0 * (syn['a_ncb'] - real_test['a_ncb']) / real_test['a_ncb'],
        'auc_rel_pct': 100.0 * (util_auc - util_auc_real) / util_auc_real,
        'paper_targets_avg_4_datasets': {'ber_rel_pct': 67.0,
                                         'a_ncb_rel_pct': 27.0,
                                         'auc_rel_pct': -10.0},
    }
    print(f'  outcome replication (vs real, per fold): '
          f'BER {rel["ber_rel_pct"]:+.0f}% '
          f'(paper 4-dataset avg +67%), '
          f'A-NCB {rel["a_ncb_rel_pct"]:+.0f}% (avg +27%), '
          f'AUC {rel["auc_rel_pct"]:+.0f}% (avg -10%)')

    # Anti-collapse gate for a 'match': a BER gain through a collapsed target
    # column is not a replication.
    non_s_tv = {j: v for j, v in all_tv.items() if j != s_idx}
    match_valid = (abs(syn_income_rate - real_income_rate) <= 0.05
                   and max(non_s_tv.values()) <= 0.05)
    print(f'  match-valid (income within 0.05, non-S share TV <= 0.05): '
          f'{match_valid}')

    result = {
        'tag': tag, 'args': vars(args), 'n_train': len(X_num),
        'n_test': len(data['test_num']),
        'paper_metrics': {
            'syn': syn, 'real_test': real_test, 'real_train': real_train,
            'real_test_reencoded': real_test_reenc,
            'alpha_precision': alpha_p, 'beta_recall': beta_r,
            'alpha_precision_real': alpha_p_real,
            'beta_recall_real': beta_r_real,
            'syn_fullcov': syn_fc, 'real_test_fullcov': real_test_fc,
            'real_test_reencoded_fullcov': real_test_reenc_fc,
            'syn_paper_style': syn_paper,
            'eps_identifiability': eps_id,
            'label_validity_auc': lv_auc,
            'inferred_group_gap': inf_gap,
            'inferred_group_gap_real': inf_gap_real,
            'latent_probe_balanced_acc': probe['drop_s_token'],
            'latent_probe_with_s_token': probe['with_s_token'],
            'real_feature_ceiling': ceiling,
            'decoded_leakage': leak,
            'heldout_ceiling': ceiling_ho,
            'synthetic_sex_agreement': s_agree,
            'relationship_faithfulness': rel_faith,
            's_share_tv': s_tv,
            'all_cat_share_tv': all_tv,
            'tier1_group_agreement': t1_agree,
            'true_group_target_gap': true_gap,
            'downstream_sp_eo': ds,
            'downstream_auc': util_auc,
            'downstream_auc_real': util_auc_real,
            'income_tv': income_tv,
            'outcome_replication': rel,
            'match_valid': match_valid,
            'proxy_ranking': proxies,
            'proxy_latent_probe': proxy_probe,
            'sex_proxy_joint_tv': sp_joint_tv,
            'decouple_curve': decouple_results,
            'impossible_combination_rate': imposs,
            'impossible_combination_rate_real': imposs_real,
            'proxy_member_dependence_syn': pm_dep_syn,
            'proxy_member_dependence_real': pm_dep_real,
            'promotion_gates': gate_pass,
            'syn_income_rate': syn_income_rate,
            'real_income_rate': real_income_rate},
        'fidelity': fid,
        'fairness_extra': {k: fair[k] for k in
                           ('real_rates', 'syn_rates', 'real_dp_diff',
                            'syn_dp_diff', 'attribute_probe_auc',
                            'downstream_fairness')},
        'latent_diagnostic': diag,
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

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, default=float)

    print(f'  BER   syn={syn["ber"]:.3f}  real(test)={real_test["ber"]:.3f}  '
          f'(chance {syn["ber_chance"]:.2f})')
    print(f'  A-NCB (strict, real-trained adversary) '
          f'syn={syn["a_ncb"]:.3f}  real(test)={real_test["a_ncb"]:.3f}')
    print(f'  A-NCB (full-cov mixture, 10 seeds) syn={syn_fc["a_ncb"]:.3f} '
          f'(median {syn_fc["a_ncb_median"]:.3f})  '
          f'real(test)={real_test_fc["a_ncb"]:.3f} '
          f'(median {real_test_fc["a_ncb_median"]:.3f})')
    if real_test_reenc is not None:
        print(f'  real reference after masking: '
              f'BER={real_test_reenc["ber"]:.3f}  '
              f'A-NCB={real_test_reenc["a_ncb"]:.3f}')
    print(f'  A-NCB (paper-style, own-S adversary) '
          f'syn={syn_paper["a_ncb"]:.3f}  real(train, own-S)='
          f'{real_train["a_ncb"]:.3f}')
    if eps_id is None:
        print('  eps-identifiability: n/a (tier-1 decode is near-copies)')
    else:
        print(f'  eps-identifiability syn={eps_id:.3f}')
    print(f'  label-validity AUC (0.5=unrelated, <0.5=inverted): {lv_auc:.3f}')
    print(f'  income gap by inferred sex: syn={inf_gap:.3f} '
          f'real={inf_gap_real:.3f}')
    print(f'  fidelity: {fid}')
    print(f'  stat. parity (income) real={fair["real_dp_diff"]:.3f} '
          f'syn={fair["syn_dp_diff"]:.3f}')
    if 'eps_total' in result:
        print(f'  total epsilon={result["eps_total"]:.2f}')
    print(f'  runtime {result["runtime_s"]:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
    main()
