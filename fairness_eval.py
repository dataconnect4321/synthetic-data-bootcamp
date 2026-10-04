"""Task-agnostic fairness metrics from the FLIP paper (arXiv:2508.21815).

Implements the paper's fairness evaluation (Secs. 3.2.2 and 5.3.2),
generalized from a binary protected attribute to any number of groups:

  - BER (Def. 1): balanced error rate of a classifier predicting the
    protected attribute S from all other attributes. Multi-class BER is
    1 - balanced accuracy (the mean per-group error); its ideal ("S is
    unpredictable") is 1 - 1/|S|, i.e. 0.5 for a binary S.
  - A-NCB (Def. 2, adversarial): normalized cluster balance of a
    2-component clustering of the non-protected attributes, computed
    against ADVERSARIAL group labels (S predicted from the other
    attributes, out of fold), so randomizing the S column alone cannot
    score well. 1 = every cluster is group-balanced.
  - Task fairness per feature (Eqs. 14/15): every non-protected column
    is treated as a downstream target. A model trained without S
    predicts it on the real data, and unfairness is the largest gap
    between groups - in class rates for categorical targets (Eq. 14),
    in Wasserstein-1 between the groups' prediction distributions for
    numeric targets (Eq. 15).

The paper uses LightGBM; this uses scikit-learn's histogram gradient
boosting (the same algorithm family) to avoid a new dependency.
"""

import itertools

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

N_FOLDS = 3  # the paper's 3-fold protocol
# NCB compares per-group cluster shares; a group the adversary assigns to
# only a handful of rows gives a meaningless share (1 row -> 0 or 1), so
# groups below this size are left out (the paper's binary S never hits it).
MIN_NCB_GROUP = 30


def _onehot(col, k):
    return np.eye(k)[col]


def _features(num, cat, cat_sizes, skip):
    """Numerics plus one-hot categoricals, excluding the column indices in
    `skip`. Widths come from cat_sizes so real/synthetic matrices align."""
    parts = [num]
    for j, k in enumerate(cat_sizes):
        if j not in skip:
            parts.append(_onehot(cat[:, j], k))
    return np.concatenate(parts, axis=1)


def _clf(seed):
    # early_stopping=False: above 10k rows sklearn otherwise holds out a
    # stratified validation split, which fails on a class with one row.
    return HistGradientBoostingClassifier(random_state=seed,
                                          early_stopping=False)


def ber(num, cat, cat_sizes, s_idx, seed=0):
    """BER (Def. 1) and the out-of-fold predicted groups (the adversarial
    assignment reused by A-NCB). Returns (ber, ideal, oof) or
    (nan, ideal, None) when S has too few rows per group to cross-fit."""
    y = cat[:, s_idx]
    ideal = 1.0 - 1.0 / cat_sizes[s_idx]
    counts = np.bincount(y, minlength=cat_sizes[s_idx])
    present = counts[counts > 0]
    if len(present) < 2 or present.min() < N_FOLDS:
        return float('nan'), ideal, None
    X = _features(num, cat, cat_sizes, skip={s_idx})
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = cross_val_predict(_clf(seed), X, y, cv=cv)
    return 1.0 - balanced_accuracy_score(y, oof), ideal, oof


def famd(num, cat, cat_sizes, skip):
    """Factor analysis of mixed data: standardized numerics plus
    indicator columns centered and scaled by 1/sqrt(category frequency),
    projected onto all principal components."""
    parts = []
    std = num.std(axis=0)
    keep = std > 1e-12  # constant columns carry no structure
    parts.append((num[:, keep] - num[:, keep].mean(axis=0)) / std[keep])
    for j, k in enumerate(cat_sizes):
        if j in skip:
            continue
        ind = _onehot(cat[:, j], k)
        freq = ind.mean(axis=0)
        used = freq > 0
        parts.append((ind[:, used] - freq[used]) / np.sqrt(freq[used]))
    Z = np.concatenate(parts, axis=1)
    Z = Z[:, Z.std(axis=0) > 1e-12]
    if Z.shape[1] == 0:
        return np.zeros((len(Z), 1))
    return PCA(n_components=min(Z.shape)).fit_transform(Z)


def ncb(clusters, groups, min_group=MIN_NCB_GROUP):
    """Normalized cluster balance (Def. 2) over all pairs of groups:
    min over clusters c and group pairs (a, b) of
    min(P[c | a] / P[c | b], P[c | b] / P[c | a]). Groups with fewer than
    min_group rows are left out; with fewer than two groups left, the
    groups cannot be told apart and NCB is 1."""
    ids, counts = np.unique(groups, return_counts=True)
    present = list(ids[counts >= min_group])
    if len(present) < 2:
        return 1.0
    worst = 1.0
    for c in np.unique(clusters):
        in_c = clusters == c
        shares = {g: in_c[groups == g].mean() for g in present}
        for a, b in itertools.combinations(present, 2):
            pa, pb = shares[a], shares[b]
            if pa == 0 and pb == 0:
                continue
            worst = min(worst, min(pa, pb) / max(pa, pb))
    return float(worst)


def a_ncb(num, cat, cat_sizes, s_idx, adversarial_groups, n_clusters=2,
          seed=0):
    """A-NCB: NCB of a Gaussian-mixture clustering of the FAMD components,
    against adversarially inferred groups (5.3.2)."""
    if adversarial_groups is None:
        return float('nan')
    Z = famd(num, cat, cat_sizes, skip={s_idx})
    clusters = GaussianMixture(n_components=n_clusters,
                               random_state=seed).fit_predict(Z)
    return ncb(clusters, adversarial_groups)


def _group_gap_categorical(pred, groups, n_classes):
    """Eq. 14 over all groups: max over classes of the spread of
    P[f = k | S = g] across groups."""
    present = np.unique(groups)
    rates = np.array([[np.mean(pred[groups == g] == k) for k in range(n_classes)]
                      for g in present])
    return float((rates.max(axis=0) - rates.min(axis=0)).max())


def _group_gap_numeric(pred, groups):
    """Eq. 15 over all groups: max pairwise W1 between the groups'
    prediction distributions."""
    present = np.unique(groups)
    return float(max((wasserstein_distance(pred[groups == a], pred[groups == b])
                      for a, b in itertools.combinations(present, 2)),
                     default=0.0))


def _fit_predict(kind, X_tr, y_tr, X_te, seed):
    if kind == 'cat' and len(np.unique(y_tr)) < 2:
        return np.full(len(X_te), y_tr[0])  # a single class: constant model
    model = (_clf(seed) if kind == 'cat'
             else HistGradientBoostingRegressor(random_state=seed,
                                                early_stopping=False))
    return model.fit(X_tr, y_tr).predict(X_te)


def task_fairness(real_num, real_cat, syn_num, syn_cat, cat_sizes, s_idx,
                  num_names, cat_names, seed=0):
    """Per-feature task fairness (Eqs. 14/15). For each non-protected
    column, a model that never sees S predicts it on the REAL rows and
    the gap between real groups is measured. 'syn': trained on synthetic
    data; 'real': the same model trained on real data (out of fold), as
    the bias-carrying baseline. Numeric gaps are in standardized units."""
    groups = real_cat[:, s_idx]
    columns = ([('num', j, n) for j, n in enumerate(num_names)]
               + [('cat', j, n) for j, n in enumerate(cat_names) if j != s_idx])
    out = {}
    for kind, j, name in columns:
        num_keep = [i for i in range(real_num.shape[1])
                    if not (kind == 'num' and i == j)]
        skip = {s_idx} | ({j} if kind == 'cat' else set())

        def X(num, cat):
            return _features(num[:, num_keep], cat, cat_sizes, skip)

        def y(num, cat):
            return cat[:, j] if kind == 'cat' else num[:, j]

        X_real, y_real = X(real_num, real_cat), y(real_num, real_cat)
        pred_syn = _fit_predict(kind, X(syn_num, syn_cat), y(syn_num, syn_cat),
                                X_real, seed)
        pred_real = np.empty_like(pred_syn, dtype=float)
        cv = KFold(N_FOLDS, shuffle=True, random_state=seed)
        for tr, te in cv.split(X_real):
            pred_real[te] = _fit_predict(kind, X_real[tr], y_real[tr],
                                         X_real[te], seed)

        if kind == 'cat':
            gap = lambda p: _group_gap_categorical(p.astype(int), groups,
                                                   cat_sizes[j])
        else:
            gap = lambda p: _group_gap_numeric(p, groups)
        out[name] = {'kind': kind, 'real': gap(pred_real), 'syn': gap(pred_syn)}
    return out


def paper_fairness_metrics(real_num, real_cat, syn_num, syn_cat, cat_sizes,
                           s_idx, num_names, cat_names, seed=0):
    """BER, A-NCB and per-feature task fairness for the real data (the
    biased baseline) and the synthetic data."""
    out = {}
    for which, (num, cat) in (('real', (real_num, real_cat)),
                              ('syn', (syn_num, syn_cat))):
        b, ideal, oof = ber(num, cat, cat_sizes, s_idx, seed)
        out[which] = {'ber': b, 'ber_ideal': ideal,
                      'a_ncb': a_ncb(num, cat, cat_sizes, s_idx, oof,
                                     seed=seed)}
    out['task_fairness'] = task_fairness(real_num, real_cat, syn_num, syn_cat,
                                         cat_sizes, s_idx, num_names,
                                         cat_names, seed)
    return out
