"""The FLIP paper's evaluation protocol (arXiv:2508.21815, Sec. 5.3) for one
cross-validation fold: synthetic data from a generator trained on the
fold's training rows, scored against the held-out real test rows.

  - Synth AUC, statistical parity (SP), equalized odds (EO): a gradient
    boosting model trained on the synthetic data predicts the target on
    the real test fold. SP and EO are the largest gap between protected
    groups, so they also apply to more than two groups.
  - alpha-precision, beta-recall: the "naive" AlphaPrecision metrics of
    Alaa et al. (2022) as computed by synthcity and used by TabSyn's
    eval/eval_quality.py (min-max scaled numerics + one-hot categoricals,
    centre = real mean, 30 alpha levels).
  - epsilon-identifiability (Yoon et al., 2020): share of real training
    rows whose nearest synthetic row is closer than their nearest other
    real row, under Gower distance with entropy-based column weights
    (SynthEval's variant).
  - BER, A-NCB, per-feature task fairness: fairness_eval.py.

Deviations from the paper: scikit-learn HistGradientBoosting instead of
LightGBM (same algorithm family; avoids the libomp dependency on macOS).
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from fairness_eval import _clf, _features, paper_fairness_metrics

N_ALPHAS = 30
GOWER_CHUNK = 128


# ==========================================================================
# Encoding
# ==========================================================================
def encode(frames, numeric_cols, categorical_cols):
    """Encode several frames with a shared vocabulary per categorical column
    (so one-hot widths match) and standardize numerics by the FIRST
    frame's mean/std. Returns [(num, cat), ...] and the category sizes."""
    vocab = {c: sorted(set().union(*(f[c].astype(str) for f in frames)))
             for c in categorical_cols}
    ref = frames[0][numeric_cols].astype(float)
    mean, std = ref.mean(), ref.std().replace(0, 1)
    out = []
    for f in frames:
        num = ((f[numeric_cols].astype(float) - mean) / std).to_numpy()
        cat = np.stack([pd.Categorical(f[c].astype(str), categories=vocab[c]).codes
                        for c in categorical_cols], axis=1).astype(np.int64)
        out.append((num, cat))
    return out, [len(vocab[c]) for c in categorical_cols]


# ==========================================================================
# Utility and downstream fairness
# ==========================================================================
def _max_gap(values_by_group):
    vals = [v for v in values_by_group if not np.isnan(v)]
    return float(max(vals) - min(vals)) if len(vals) >= 2 else float('nan')


def downstream(syn, test, cat_sizes, s_idx, t_idx, pos_code, seed=0):
    """Train on synthetic (all features, as in the standard prediction
    task), predict the target on the real test fold."""
    (syn_num, syn_cat), (te_num, te_cat) = syn, test
    y_syn = (syn_cat[:, t_idx] == pos_code).astype(int)
    y_te = (te_cat[:, t_idx] == pos_code).astype(int)
    if len(np.unique(y_syn)) < 2:
        # The generator produced a single class: no usable model.
        return {'synth_auc': float('nan'), 'sp': float('nan'),
                'eo': float('nan'), 'syn_positive_rate': float(y_syn.mean())}
    X_syn = _features(syn_num, syn_cat, cat_sizes, skip={t_idx})
    X_te = _features(te_num, te_cat, cat_sizes, skip={t_idx})
    model = _clf(seed).fit(X_syn, y_syn)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    groups = te_cat[:, s_idx]
    present = np.unique(groups)
    sp = _max_gap([pred[groups == g].mean() for g in present])
    eo = max(_max_gap([pred[(groups == g) & (y_te == y)].mean()
                       if np.any((groups == g) & (y_te == y)) else np.nan
                       for g in present])
             for y in (0, 1))
    return {'synth_auc': float(roc_auc_score(y_te, proba)), 'sp': sp,
            'eo': float(eo), 'syn_positive_rate': float(y_syn.mean())}


# ==========================================================================
# alpha-precision / beta-recall (synthcity's naive AlphaPrecision)
# ==========================================================================
def _quality_matrix(df, numeric_cols, categorical_cols, ref):
    """Numerics min-max scaled by `ref` + one-hot categoricals with `ref`'s
    vocabulary (unseen synthetic categories become all-zero rows)."""
    num = df[numeric_cols].astype(float)
    lo, hi = ref[numeric_cols].astype(float).min(), ref[numeric_cols].astype(float).max()
    parts = [((num - lo) / (hi - lo).replace(0, 1)).to_numpy()]
    for c in categorical_cols:
        cats = sorted(ref[c].astype(str).unique())
        parts.append(np.eye(len(cats) + 1)[
            pd.Categorical(df[c].astype(str), categories=cats).codes][:, :-1])
    return np.concatenate(parts, axis=1)


def alpha_precision_beta_recall(X, X_syn):
    """Delta-precision-alpha and delta-coverage-beta, naive embedding."""
    alphas = np.linspace(0, 1, N_ALPHAS)
    center = X.mean(axis=0)
    radii = np.quantile(np.linalg.norm(X - center, axis=1), alphas)
    synth_center = X_syn.mean(axis=0)
    synth_to_center = np.linalg.norm(X_syn - center, axis=1)

    real_to_real = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    real_to_synth, idx = NearestNeighbors(n_neighbors=1).fit(X_syn).kneighbors(X)
    real_to_synth, idx = real_to_synth[:, 0], idx[:, 0]
    closest_d = np.linalg.norm(X_syn[idx] - synth_center, axis=1)
    closest_radii = np.quantile(closest_d, alphas)

    precision = [np.mean(synth_to_center <= r) for r in radii]
    coverage = [np.mean((real_to_synth <= real_to_real) & (closest_d <= r))
                for r in closest_radii]
    delta = lambda curve: 1 - np.sum(np.abs(alphas - curve)) / np.sum(alphas)
    return float(delta(np.array(precision))), float(delta(np.array(coverage)))


# ==========================================================================
# epsilon-identifiability (Gower, entropy weights)
# ==========================================================================
def _column_entropy(values):
    _, counts = np.unique(values, return_counts=True)
    return entropy(counts)


def eps_identifiability(real, syn, numeric_cols, categorical_cols):
    cols = list(numeric_cols) + list(categorical_cols)
    n_num = len(numeric_cols)
    r_num = real[numeric_cols].to_numpy(float)
    s_num = syn[numeric_cols].to_numpy(float)
    rng = r_num.max(axis=0) - r_num.min(axis=0)
    rng[rng == 0] = 1
    vocab = {c: {v: i for i, v in enumerate(sorted(real[c].astype(str).unique()))}
             for c in categorical_cols}
    enc = lambda df: np.stack([df[c].astype(str).map(vocab[c]).fillna(-1).to_numpy()
                               for c in categorical_cols], axis=1)
    r_cat, s_cat = enc(real), enc(syn)

    # Weight 1/H per column (numerics rounded, as SynthEval does); a
    # constant column carries no information and gets weight 0.
    H = np.array([_column_entropy(np.round(r_num[:, j])) for j in range(n_num)]
                 + [_column_entropy(r_cat[:, j]) for j in range(r_cat.shape[1])])
    w = np.where(H > 0, 1 / np.maximum(H, 1e-12), 0.0)
    w = (w / w.sum()).astype(np.float32)
    w_num, w_cat = w[:n_num], w[n_num:]

    def nearest(a_num, a_cat, b_num, b_cat, exclude_self):
        out = np.empty(len(a_num), dtype=np.float32)
        for i in range(0, len(a_num), GOWER_CHUNK):
            d = (np.abs(a_num[i:i + GOWER_CHUNK, None, :] - b_num[None]) / rng
                 ).astype(np.float32) @ w_num
            d += (a_cat[i:i + GOWER_CHUNK, None, :] != b_cat[None]
                  ).astype(np.float32) @ w_cat
            if exclude_self:
                d[np.arange(d.shape[0]), np.arange(i, i + d.shape[0])] = np.inf
            out[i:i + GOWER_CHUNK] = d.min(axis=1)
        return out

    to_real = nearest(r_num, r_cat, r_num, r_cat, exclude_self=True)
    to_syn = nearest(r_num, r_cat, s_num, s_cat, exclude_self=False)
    return float(np.mean(to_syn < to_real))


# ==========================================================================
# One fold
# ==========================================================================
def evaluate(train_df, test_df, syn_df, data_cfg, seed=0):
    """All paper metrics for one fold. data_cfg is the config's `data:`
    section (sensitive_col, target_col, numeric_cols, categorical_cols)."""
    s_col, t_col = data_cfg['sensitive_col'], data_cfg['target_col']
    num_cols = list(data_cfg['numeric_cols'])
    cat_cols = [s_col, t_col] + list(data_cfg['categorical_cols'])
    s_idx, t_idx = 0, 1

    (train, test, syn), cat_sizes = encode([train_df, test_df, syn_df],
                                           num_cols, cat_cols)
    vocab_t = sorted(set().union(*(f[t_col].astype(str)
                                   for f in (train_df, test_df, syn_df))))
    pos_code = vocab_t.index('1')

    out = downstream(syn, test, cat_sizes, s_idx, t_idx, pos_code, seed)
    out['real_train_auc'] = downstream(train, test, cat_sizes, s_idx, t_idx,
                                       pos_code, seed)['synth_auc']

    X = _quality_matrix(test_df, num_cols, cat_cols, ref=test_df)
    X_syn = _quality_matrix(syn_df, num_cols, cat_cols, ref=test_df)
    out['alpha_precision'], out['beta_recall'] = alpha_precision_beta_recall(X, X_syn)

    out['eps_identifiability'] = eps_identifiability(train_df, syn_df,
                                                     num_cols, cat_cols)

    fair = paper_fairness_metrics(test[0], test[1], syn[0], syn[1], cat_sizes,
                                  s_idx, num_cols, cat_cols, seed)
    out['ber'], out['a_ncb'] = fair['syn']['ber'], fair['syn']['a_ncb']
    out['ber_ideal'] = fair['syn']['ber_ideal']
    out['real_ber'], out['real_a_ncb'] = fair['real']['ber'], fair['real']['a_ncb']
    out['task_fairness'] = fair['task_fairness']
    out['syn_group_share'] = (syn_df[s_col].astype(str)
                              .value_counts(normalize=True).to_dict())
    return out
