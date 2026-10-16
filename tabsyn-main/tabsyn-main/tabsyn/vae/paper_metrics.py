"""The FLIP paper's task-agnostic fairness and privacy metrics
(arXiv:2508.21815, Sec. 3.2.2 and 5.3).

  - BER:               balanced error rate of a nonlinear classifier that
                       predicts the protected attribute from all OTHER
                       columns. Ideal = chance = (K-1)/K for K groups
                       (0.5 for 2, 2/3 for 3).
  - A-NCB:             adversarial normalized cluster balance. A classifier
                       trained on the REAL data (features -> protected
                       attribute) infers the protected attribute of each
                       synthetic record from its other columns; the
                       synthetic non-protected columns are embedded with FAMD
                       and clustered with a Gaussian mixture; NCB is then
                       computed on the inferred labels. Ideal = 1. The
                       synthetic protected column is never used, so
                       randomizing it cannot change the score.
  - eps-identifiability: share of real records whose nearest synthetic record
                       is closer than their nearest other real record
                       (entropy-weighted Gower distance, after Yoon et al.
                       2020 / SynthEval). Lower is more private.

Every function takes the table as (num, cat): `num` is an (n, d_num) float
array, `cat` an (n, d_cat) array of integer category codes, and `s_idx` the
protected column's position within `cat`.
"""

import math
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def _split_features(num, cat, s_idx):
    """Feature matrix of all columns except the protected one, plus the
    protected labels and a mask of which feature columns are categorical."""
    keep = [j for j in range(cat.shape[1]) if j != s_idx]
    cat_f = cat[:, keep]
    X = np.concatenate([num.astype(np.float64), cat_f.astype(np.float64)],
                       axis=1)
    is_cat = np.array([False] * num.shape[1] + [True] * cat_f.shape[1])
    return X, cat[:, s_idx].astype(int), is_cat, num, cat_f


def _classifier(is_cat, seed):
    # Balanced class weights: a majority-class guess must not score a
    # trivially perfect BER on imbalanced groups.
    return HistGradientBoostingClassifier(
        categorical_features=is_cat, class_weight='balanced',
        max_iter=100, random_state=seed)


def predict_protected(num, cat, s_idx, n_splits=3, seed=0):
    """Out-of-fold predictions of the table's own protected column from
    its other columns with a nonlinear (gradient-boosted tree) classifier."""
    X, y, is_cat, _, _ = _split_features(num, cat, s_idx)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(_classifier(is_cat, seed), X, y, cv=cv)


def infer_protected(ref_num, ref_cat, num, cat, s_idx, seed=0):
    """Adversary: fit on the reference (real) table's non-protected columns
    -> protected column, then infer the protected attribute of every row of
    (num, cat) from its non-protected columns alone."""
    Xr, yr, is_cat, _, _ = _split_features(ref_num, ref_cat, s_idx)
    X, _, _, _, _ = _split_features(num, cat, s_idx)
    return _classifier(is_cat, seed).fit(Xr, yr).predict(X)


def balanced_error_rate(y_true, y_pred):
    """Mean per-group error rate (generalizes the paper's Eq. 1 to K
    groups). Chance level is (K-1)/K."""
    groups = np.unique(y_true)
    return float(np.mean([(y_pred[y_true == g] != g).mean() for g in groups]))


def chance_ber(n_groups):
    return (n_groups - 1) / n_groups


def famd_coordinates(num, cat_features, n_components=None):
    """Factorial analysis of mixed data: PCA over standardized numeric
    columns and centered one-hot indicators scaled by 1/sqrt(p_k).
    Returns the first n_components component scores (all if None)."""
    blocks = []
    if num.shape[1]:
        sd = num.std(axis=0)
        ok = sd > 1e-12
        blocks.append((num[:, ok] - num[:, ok].mean(axis=0)) / sd[ok])
    for j in range(cat_features.shape[1]):
        col = cat_features[:, j]
        levels, inv = np.unique(col, return_inverse=True)
        if len(levels) < 2:
            continue
        onehot = np.eye(len(levels))[inv]
        p = onehot.mean(axis=0)
        blocks.append((onehot - p) / np.sqrt(p))
    Z = np.concatenate(blocks, axis=1)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    keep = S > 1e-8
    coords = U[:, keep] * S[keep]
    return coords if n_components is None else coords[:, :n_components]


def normalized_cluster_balance(clusters, groups):
    """NCB (Def. 2): min over clusters of the min over group pairs of
    min(P[c|a]/P[c|b], P[c|b]/P[c|a]). 1 = every cluster holds the same
    share of each group; 0 = some cluster contains only one group."""
    labels = np.unique(groups)
    worst = 1.0
    for c in np.unique(clusters):
        p = [(clusters[groups == g] == c).mean() for g in labels]
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                hi, lo = max(p[i], p[j]), min(p[i], p[j])
                if hi == 0:
                    continue  # cluster empty for both groups
                worst = min(worst, lo / hi)
    return float(worst)


def cluster_features(num, cat, s_idx, n_clusters=2, seed=0,
                     covariance_type='full', n_init=1, famd_components=2):
    """GMM cluster ids of the FAMD-embedded non-protected columns.

    Full covariance with one initialization is sklearn's default and the
    likely paper setup; on Adult it recovers the married / not-married
    split. (A diagonal-covariance fit splits off a ~11% cluster unrelated to
    sex, which keeps A-NCB near 1 for real and synthetic data alike.)

    famd_components: the paper clusters "all FAMD components" without
    saying how many were computed; 2 is prince.FAMD's default. On Adult the
    GMM on the first 2 components finds the married / not-married split for
    every seed (seed sd ~0), while on all components it mostly splits off a
    tiny outlier cluster (seed sd ~0.3). None = all components."""
    _, _, _, num_f, cat_f = _split_features(num, cat, s_idx)
    coords = famd_coordinates(num_f.astype(np.float64), cat_f,
                              famd_components)
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type=covariance_type,
                          n_init=n_init, random_state=seed)
    return gmm.fit_predict(coords)


def group_share_tv(syn_cat, ref_cat, s_idx):
    """Total variation between the synthetic protected column's group shares
    and the reference (real) shares."""
    k = int(max(ref_cat[:, s_idx].max(), syn_cat[:, s_idx].max())) + 1
    p = np.bincount(syn_cat[:, s_idx], minlength=k) / len(syn_cat)
    q = np.bincount(ref_cat[:, s_idx], minlength=k) / len(ref_cat)
    return float(0.5 * np.abs(p - q).sum())


def categorical_share_tv_all(syn_cat, ref_cat):
    """Per-column share TV for EVERY categorical column. Catches any
    Wife->Husband-style collapse, not just the protected column."""
    out = {}
    for j in range(ref_cat.shape[1]):
        k = int(max(ref_cat[:, j].max(), syn_cat[:, j].max())) + 1
        p = np.bincount(syn_cat[:, j], minlength=k) / len(syn_cat)
        q = np.bincount(ref_cat[:, j], minlength=k) / len(ref_cat)
        out[j] = float(0.5 * np.abs(p - q).sum())
    return out


def true_group_target_gap(syn_cat, true_cat, t_idx, s_idx, t_pos=None):
    """Target-rate gap across the TRUE groups of the source rows (tier-1):
    immune to an invented synthetic protected column."""
    if t_pos is None:
        t_pos = int(true_cat[:, t_idx].max())
    rates = [(syn_cat[true_cat[:, s_idx] == g, t_idx] == t_pos).mean()
              for g in np.unique(true_cat[:, s_idx])]
    return float(max(rates) - min(rates))


def downstream_sp_eo(syn_num, syn_cat, real_num, real_cat, t_idx, s_idx,
                     seed=0):
    """Train a target classifier on SYNTHETIC data (protected column
    excluded), evaluate on REAL data by real group: statistical parity
    (selection-rate gap) and equalized odds (TPR gap among real positives)."""
    def feats(num, cat):
        keep = [j for j in range(cat.shape[1]) if j not in (s_idx, t_idx)]
        X = np.concatenate([num.astype(np.float64),
                             cat[:, keep].astype(np.float64)], axis=1)
        return X, np.array([False] * num.shape[1] + [True] * len(keep))
    Xs, is_cat = feats(syn_num, syn_cat)
    ys = syn_cat[:, t_idx]
    Xr, _ = feats(real_num, real_cat)
    yr = real_cat[:, t_idx]
    t_pos = int(max(ys.max(), yr.max()))
    clf = HistGradientBoostingClassifier(categorical_features=is_cat,
                                         random_state=seed)
    clf.fit(Xs, (ys == t_pos).astype(int))
    pred = clf.predict(Xr) == 1
    pos, grp = (yr == t_pos), real_cat[:, s_idx]
    sp = [pred[grp == g].mean() for g in np.unique(grp)]
    eo = [pred[pos & (grp == g)].mean() for g in np.unique(grp)]
    return {'sp_diff': float(max(sp) - min(sp)),
            'eo_diff': float(max(eo) - min(eo)),
            'sp_rates': {int(g): float(r) for g, r in zip(np.unique(grp), sp)}}


def fairness_scores(num, cat, s_idx, ref=None, n_splits=3, n_clusters=2,
                    seed=0, covariance_type='full', n_init=1):
    """BER and A-NCB of a table.

    BER uses the table's own protected column (out-of-fold classifier), as
    in the paper's Def. 1 - randomizing that column fools it. A-NCB uses an
    adversary trained on `ref` = (real_num, real_cat) and never reads the
    table's protected column. With ref=None (scoring the real data itself)
    the adversary is the same out-of-fold predictor used for BER."""
    y = cat[:, s_idx].astype(int)
    own = predict_protected(num, cat, s_idx, n_splits, seed)
    adv = own if ref is None else infer_protected(ref[0], ref[1], num, cat,
                                                  s_idx, seed)
    n_groups = len(np.unique(y))
    clusters = cluster_features(num, cat, s_idx, n_clusters, seed,
                                covariance_type, n_init)
    return {
        'ber': balanced_error_rate(y, own),
        'ber_chance': chance_ber(n_groups),
        'a_ncb': normalized_cluster_balance(clusters, adv),
        # NCB on the stored protected column, for contrast with A-NCB
        'ncb_stored': normalized_cluster_balance(clusters, y),
    }


# --------------------------------------------------------------------------
# eps-identifiability
# --------------------------------------------------------------------------

def _entropy_weights(num, cat, bins=10):
    """Yoon et al.'s per-column weights 1/H(column), normalized to sum 1.
    Numeric columns are histogrammed into `bins` equal-width bins first;
    constant columns (H = 0) get weight 0."""
    ents = []
    for j in range(num.shape[1]):
        counts, _ = np.histogram(num[:, j], bins=bins)
        ents.append(counts)
    for j in range(cat.shape[1]):
        ents.append(np.bincount(cat[:, j] - cat[:, j].min()))
    w = []
    for counts in ents:
        p = counts[counts > 0] / counts.sum()
        h = float(-(p * np.log(p)).sum())
        w.append(1.0 / h if h > 1e-12 else 0.0)
    w = np.array(w)
    return w / w.sum()


def _nearest_gower(query, ref, ranges, weights, n_num, exclude_self=False,
                   chunk=1024, device='cpu'):
    """Per-row minimum weighted Gower distance from `query` to `ref`.
    Both are (n, d) float tensors with numeric columns first. With
    exclude_self, query is ref and each row's zero self-distance is
    ignored."""
    out = torch.empty(query.shape[0], device=device)
    m = ref.shape[0]
    for start in range(0, query.shape[0], chunk):
        q = query[start:start + chunk]
        d = torch.zeros(q.shape[0], m, device=device)
        for k in range(query.shape[1]):
            if weights[k] == 0:
                continue
            diff = q[:, k:k + 1] - ref[None, :, k]
            if k < n_num:
                d += weights[k] * (diff.abs() / ranges[k]).clamp(max=1.0)
            else:
                d += weights[k] * (diff != 0).float()
        if exclude_self:
            rows = torch.arange(q.shape[0], device=device)
            d[rows, start + rows] = float('inf')
        out[start:start + chunk] = d.min(dim=1).values
    return out


def epsilon_identifiability(real_num, real_cat, syn_num, syn_cat,
                            device=None):
    """Share of real records whose nearest synthetic record is strictly
    closer than their nearest other real record (entropy-weighted Gower
    distance; numeric ranges taken from the real data). 0 = no real record
    is closer to a synthetic one than to its real neighbors."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    n_num = real_num.shape[1]
    weights = _entropy_weights(real_num, real_cat)
    ranges = torch.tensor(
        np.maximum(real_num.max(axis=0) - real_num.min(axis=0), 1e-12),
        dtype=torch.float32, device=device)

    def to_t(num, cat):
        return torch.tensor(np.concatenate([num, cat], axis=1),
                            dtype=torch.float32, device=device)

    real_t, syn_t = to_t(real_num, real_cat), to_t(syn_num, syn_cat)
    d_rr = _nearest_gower(real_t, real_t, ranges, weights, n_num,
                          exclude_self=True, device=device)
    d_rs = _nearest_gower(real_t, syn_t, ranges, weights, n_num,
                          device=device)
    return float((d_rs < d_rr).float().mean())


# --------------------------------------------------------------------------
# alpha-precision / beta-recall (Alaa et al. 2022), TabSyn's setup
# --------------------------------------------------------------------------

def _mixed_matrix(num_raw, cat, sizes):
    """Raw-scale numerics + one-hot categoricals: the representation TabSyn's
    eval_quality.py hands to synthcity's naive AlphaPrecision."""
    parts = [np.asarray(num_raw, dtype=np.float64)]
    for j, k in enumerate(sizes):
        parts.append(np.eye(k)[np.asarray(cat)[:, j]])
    return np.concatenate(parts, axis=1)


def alpha_precision_beta_recall(real_num_raw, real_cat, syn_num_raw, syn_cat,
                                sizes, n_steps=30):
    """Integrated alpha-precision and beta-recall, a port of synthcity's
    AlphaPrecision.metrics (the 'naive' variant TabSyn and the FLIP paper
    report): no learned embedding, the real data's mean as center.

    alpha-precision: is synthetic data inside the real data's alpha-support
    (fidelity). beta-recall: does synthetic data cover the real data's
    beta-support (diversity). Both in [0, 1], higher is better.
    """
    from sklearn.neighbors import NearestNeighbors
    X = _mixed_matrix(real_num_raw, real_cat, sizes)
    Y = _mixed_matrix(syn_num_raw, syn_cat, sizes)
    # synthcity's naive variant min-max scales every column on the REAL data
    # first (_normalize_covariates); without it fnlwgt / capital-gain
    # dominate every distance and alpha sits near 1 regardless of quality
    lo, span = X.min(axis=0), X.max(axis=0) - X.min(axis=0)
    span[span == 0] = 1.0
    X, Y = (X - lo) / span, (Y - lo) / span
    alphas = np.linspace(0, 1, n_steps)
    center = X.mean(axis=0)
    radii = np.quantile(np.sqrt(((X - center) ** 2).sum(1)), alphas)
    syn_center = Y.mean(axis=0)
    syn_to_center = np.sqrt(((Y - center) ** 2).sum(1))
    real_to_real = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    d_rs, idx_rs = NearestNeighbors(n_neighbors=1).fit(Y).kneighbors(X)
    d_rs, idx_rs = d_rs[:, 0], idx_rs[:, 0]
    closest_d = np.sqrt(((Y[idx_rs] - syn_center) ** 2).sum(1))
    closest_radii = np.quantile(closest_d, alphas)
    prec = np.array([(syn_to_center <= r).mean() for r in radii])
    cov = np.array([((d_rs <= real_to_real) & (closest_d <= r)).mean()
                    for r in closest_radii])
    a = 1 - np.abs(alphas - prec).sum() / alphas.sum()
    b = 1 - np.abs(alphas - cov).sum() / alphas.sum()
    return float(a), float(b)


# --------------------------------------------------------------------------
# Data validity (plausibility) rules learned from the real data
# --------------------------------------------------------------------------

def fit_validity_rules(num_raw, cat, num_names, cat_names, cat_labels=None,
                       q=0.001, min_support=200, fd_purity=0.99,
                       fd_support=0.01, pair_expected=20.0):
    """Learn dataset-agnostic plausibility rules from REAL data.

    num_raw: (n, d) numerics on their original scale; cat: (n, k) codes.
    Rules:
      range     numeric value within the real [min, max]
      integer   column whole-valued in real data -> must be whole
      cond      numeric value within the real [q, 1-q] quantiles (q=0.1%) of rows
                sharing its category (categories with >= min_support rows)
      fd        near-deterministic implication A=a -> B=b
                (P >= fd_purity, A=a on >= fd_support of rows)
      pair      category pair absent from the real data although
                independence predicts >= pair_expected rows
    """
    num_raw, cat = np.asarray(num_raw, float), np.asarray(cat)
    n = len(cat)
    lab = (lambda j, c: str(cat_labels[j][c])) if cat_labels is not None \
        else (lambda j, c: str(c))
    rules = {'range': [], 'integer': [], 'cond': [], 'fd': [], 'pair': []}
    for i, name in enumerate(num_names):
        x = num_raw[:, i]
        rules['range'].append((i, name, x.min(), x.max()))
        if np.allclose(x, np.round(x), rtol=0, atol=1e-3):
            rules['integer'].append((i, name))
    for j, cname in enumerate(cat_names):
        for c in np.unique(cat[:, j]):
            m = cat[:, j] == c
            if m.sum() < min_support:
                continue
            for i, name in enumerate(num_names):
                lo, hi = np.quantile(num_raw[m, i], [q, 1 - q])
                rules['cond'].append((j, c, i, lo, hi,
                                      f'{name} | {cname}={lab(j, c)}'))
    shares = [np.bincount(cat[:, j]) / n for j in range(cat.shape[1])]
    for a in range(cat.shape[1]):
        for b in range(cat.shape[1]):
            if a == b:
                continue
            for va in np.where(shares[a] >= fd_support)[0]:
                m = cat[:, a] == va
                cnt = np.bincount(cat[m, b], minlength=len(shares[b]))
                vb = int(cnt.argmax())
                if cnt[vb] / m.sum() >= fd_purity and shares[b][vb] < 0.99:
                    rules['fd'].append((a, int(va), b, vb,
                                        f'{cat_names[a]}={lab(a, va)} -> '
                                        f'{cat_names[b]}={lab(b, vb)}'))
            if b > a:
                tab = np.zeros((len(shares[a]), len(shares[b])), int)
                np.add.at(tab, (cat[:, a], cat[:, b]), 1)
                exp = n * np.outer(shares[a], shares[b])
                for va, vb in zip(*np.where((tab == 0) &
                                            (exp >= pair_expected))):
                    rules['pair'].append((a, int(va), b, int(vb),
                                          f'{cat_names[a]}={lab(a, va)} & '
                                          f'{cat_names[b]}={lab(b, vb)}'))
    return rules


def fit_validity_rules_dp(num_raw, cat, num_names, cat_names,
                          noise_multiplier, cat_labels=None, seed=0,
                          fd_purity=0.99, fd_support=0.01,
                          pair_expected=20.0):
    """Plausibility rules learned from DP-noised counts (for rejection
    sampling under DP).

    Released once with the Gaussian mechanism: every 2-way contingency
    table of the categorical columns. A record adds 1 to each of the
    k(k-1)/2 tables, so the L2 sensitivity is sqrt(k(k-1)/2) and the noise
    std is noise_multiplier times that; the release costs
    alpha / (2 noise_multiplier^2) of RDP at every order alpha.
    Rules (everything else is post-processing of the noisy tables):
      integer  numeric column stored as whole numbers (public schema)
      fd       A=a -> B=b when the noisy purity >= fd_purity and A=a has a
               noisy share >= fd_support
      pair     a pair whose noisy count is below 3 noise std although
               independence (noisy margins) predicts >= 15 noise std and
               >= pair_expected rows
    No range or conditional-range rules: those need order statistics of
    raw values, which have no bounded sensitivity.

    Returns (rules, rdp) with rdp = {alpha: rdp} for compose_rdp_budgets.
    """
    num_raw, cat = np.asarray(num_raw, float), np.asarray(cat)
    k = cat.shape[1]
    sizes = [int(cat[:, j].max()) + 1 if cat_labels is None
             else len(cat_labels[j]) for j in range(k)]
    lab = (lambda j, c: str(cat_labels[j][c])) if cat_labels is not None         else (lambda j, c: str(c))
    sens = math.sqrt(k * (k - 1) / 2)
    std = noise_multiplier * sens
    rng = np.random.default_rng(seed)
    rules = {'range': [], 'integer': [], 'cond': [], 'fd': [], 'pair': []}
    for i, name in enumerate(num_names):
        x = num_raw[:, i]
        if np.allclose(x, np.round(x), rtol=0, atol=1e-3):
            rules['integer'].append((i, name))
    for a in range(k):
        for b in range(a + 1, k):
            tab = np.zeros((sizes[a], sizes[b]))
            np.add.at(tab, (cat[:, a], cat[:, b]), 1)
            tab = tab + rng.normal(0.0, std, tab.shape)
            n = max(tab.sum(), 1.0)
            ra, rb = tab.sum(1).clip(0), tab.sum(0).clip(0)
            exp = np.outer(ra, rb) / n
            for va, vb in zip(*np.where((tab < np.minimum(2 * std,
                                                          0.05 * exp)) &
                                        (exp >= max(6 * std,
                                                    pair_expected)))):
                rules['pair'].append((a, int(va), b, int(vb),
                                      f'{cat_names[a]}={lab(a, va)} & '
                                      f'{cat_names[b]}={lab(b, vb)}'))
            # implications on counts clipped at 0: negative noise would
            # otherwise inflate the purity and create false rules
            tab_c = tab.clip(min=0)
            for x, y, t in ((a, b, tab_c), (b, a, tab_c.T)):
                rows = t.sum(1).clip(0)
                cols = t.sum(0).clip(0)
                for vx in np.where(rows / n >= fd_support)[0]:
                    vy = int(t[vx].argmax())
                    # upper confidence bound on the count outside vy
                    # (2 noise std per remaining cell, combined)
                    rest = rows[vx] - t[vx, vy] +                         2 * std * math.sqrt(max(t.shape[1] - 1, 1))
                    if (rest <= (1 - fd_purity) * rows[vx]
                            and cols[vy] / n < 0.99):
                        rules['fd'].append((x, int(vx), y, vy,
                                            f'{cat_names[x]}={lab(x, vx)} -> '
                                            f'{cat_names[y]}={lab(y, vy)}'))
    alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505))
    rdp = {a_: a_ / (2.0 * noise_multiplier ** 2) for a_ in alphas}
    return rules, rdp


def constrained_repair(cat, probs, rules, protect=(), max_edits=3):
    """Constrained decoding: fix rows that break a categorical rule (pair
    or fd) with the changes the generator itself finds most likely,
    instead of discarding them.

    cat: (n, k) decoded category codes; probs: list of (n, |cat_j|) per-
    column probabilities from the decoder; protect: columns never edited
    (e.g. the protected attribute, so a repair cannot re-link a proxy to
    it). Each violating row gets greedy single-column edits - the edit
    that leaves the fewest violations, ties broken by the smallest drop in
    that column's log-probability - until it breaks no categorical rule or
    max_edits edits were made. Numeric rules are not touched.

    Post-processing of the generator's output against given rules: it adds
    no privacy cost beyond the rules' own release.

    Returns (repaired codes, info dict).
    """
    cat = np.asarray(cat).copy()
    pair = list(rules.get('pair', []))
    fd = list(rules.get('fd', []))
    if not pair and not fd:
        return cat, {'rows_changed': 0.0, 'unresolved': 0.0, 'edits': {}}
    logp = [np.log(np.clip(np.asarray(p), 1e-12, None)) for p in probs]
    involved = sorted({r[0] for r in pair + fd} | {r[2] for r in pair + fd})
    editable = [j for j in involved if j not in set(protect)]

    def nviol(row):
        v = 0
        for a, va, b, vb, _ in pair:
            v += row[a] == va and row[b] == vb
        for a, va, b, vb, _ in fd:
            v += row[a] == va and row[b] != vb
        return v

    bad = np.zeros(len(cat), bool)
    for a, va, b, vb, _ in pair:
        bad |= (cat[:, a] == va) & (cat[:, b] == vb)
    for a, va, b, vb, _ in fd:
        bad |= (cat[:, a] == va) & (cat[:, b] != vb)
    edits = {j: 0 for j in editable}
    changed = unresolved = 0
    for i in np.where(bad)[0]:
        row = cat[i].copy()
        v, k = nviol(row), 0
        while v > 0 and k < max_edits:
            best = None
            for j in editable:
                cur = row[j]
                for c in range(logp[j].shape[1]):
                    if c == cur:
                        continue
                    row[j] = c
                    nv = nviol(row)
                    row[j] = cur
                    if nv >= v:
                        continue
                    key = (nv, logp[j][i, cur] - logp[j][i, c])
                    if best is None or key < best[0]:
                        best = (key, j, c)
            if best is None:
                break
            (nv, _), j, c = best
            row[j] = c
            edits[j] += 1
            k += 1
            v = nv
        changed += k > 0
        unresolved += v > 0
        cat[i] = row
    n = max(len(cat), 1)
    return cat, {'rows_changed': changed / n, 'unresolved': unresolved / n,
                 'edits': edits}


def _violations(rules, num_raw, cat, tol=1e-6):
    """Per-type row masks of rule violations, and per-rule violation rates."""
    num_raw, cat = np.asarray(num_raw, float), np.asarray(cat)
    n = len(cat)
    bad = {k: np.zeros(n, bool) for k in rules}
    worst = []
    for i, name, lo, hi in rules['range']:
        v = (num_raw[:, i] < lo - tol) | (num_raw[:, i] > hi + tol)
        bad['range'] |= v
        worst.append((v.mean(), f'range: {name} outside [{lo:g}, {hi:g}]'))
    for i, name in rules['integer']:
        v = np.abs(num_raw[:, i] - np.round(num_raw[:, i])) > 1e-3
        bad['integer'] |= v
        worst.append((v.mean(), f'integer: {name} not whole'))
    for j, c, i, lo, hi, desc in rules['cond']:
        m = cat[:, j] == c
        v = m & ((num_raw[:, i] < lo - tol) | (num_raw[:, i] > hi + tol))
        bad['cond'] |= v
        worst.append((v.mean(), f'cond: {desc} outside [{lo:g}, {hi:g}]'))
    for a, va, b, vb, desc in rules['fd']:
        v = (cat[:, a] == va) & (cat[:, b] != vb)
        bad['fd'] |= v
        worst.append((v.mean(), f'fd: {desc}'))
    for a, va, b, vb, desc in rules['pair']:
        v = (cat[:, a] == va) & (cat[:, b] == vb)
        bad['pair'] |= v
        worst.append((v.mean(), f'pair: {desc}'))
    return bad, worst


def invalid_rows(rules, num_raw, cat, tol=1e-6):
    """Boolean mask of rows that break at least one rule (any type)."""
    bad, _ = _violations(rules, num_raw, cat, tol)
    out = np.zeros(len(cat), bool)
    for v in bad.values():
        out |= v
    return out


def validity_report(rules, num_raw, cat, tol=1e-6, top=5):
    """Score a table against fitted rules. Returns the share of rows that
    break no rule, per-type violation rates (share of rows breaking at least
    one rule of that type) and the most-violated individual rules."""
    n = len(cat)
    bad, worst = _violations(rules, num_raw, cat, tol)
    any_bad = np.zeros(n, bool)
    for v in bad.values():
        any_bad |= v
    # 'substantive' ignores the whole-number format rule, which is a
    # decoding/rounding issue rather than an implausible record
    subst = np.zeros(n, bool)
    for k, v in bad.items():
        if k != 'integer':
            subst |= v
    worst = [w for w in worst if not w[1].startswith('integer')]
    worst.sort(key=lambda t: -t[0])
    return {'valid_rate': float(1 - any_bad.mean()),
            'substantive_valid_rate': float(1 - subst.mean()),
            'by_type': {k: float(v.mean()) for k, v in bad.items()},
            'n_rules': {k: len(v) for k, v in rules.items()},
            'worst': [(round(float(r), 4), d) for r, d in worst[:top] if r > 0]}
