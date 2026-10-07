"""Proxy decoupling for near-deterministic proxies of the protected attribute.

Strict disentanglement is infeasible when a category nearly determines the
protected attribute S (Adult: relationship=Husband -> male, Wife -> female):
a decoder that must reproduce that category from a sex-free latent cannot
(data-processing inequality). Instead each such proxy is factored into

  * a legitimate component  - the MERGED class ("spouse"), modelled as usual
  * a protected-specific label - WHICH member of the class ("Husband" vs
    "Wife"), made independent of the row's other features with strength tau.

Everything here is dataset-agnostic: proxy categories are found with a
pre-registered threshold on P(S | category), and only categories pointing to
DIFFERENT groups are merged (a column whose flagged categories all point to
the same group has nothing to decouple).

Uses:
  * pre-train : mask_proxy_classes() merges each class into one new
                category (e.g. MASK) before Phase 1 (--mask-proxies).
  * post-hoc  : decouple_curve() in tabsyn.vae.temper re-draws the
                member label on the decoded rows, sweeping tau
                (1 = real coupling, 0 = member independent of the row).
  * in-model  : proxy_member_labels() supplies the per-row member label that
                the gradient-reversal term in flip_fairness.proxy_decoupling_loss
                hides from the other tokens / decoded columns.

PRIVACY NOTE: detection and the P(member | merged, S) table are computed on
the real training fold. They are a handful of category-level counts, not
covered by the DP-SGD accounting - report them as an uncounted release (or
add Laplace/Gaussian noise to the counts before use).
"""

import numpy as np


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def detect_proxy_classes(cat, s_idx, threshold=0.95, min_support=0.01,
                         exclude=()):
    """Find near-deterministic proxy categories and group them into classes.

    A category c of column j is a proxy if it has at least
    min_support * n rows and max_g P(S=g | c) >= threshold. Within a column,
    the flagged categories are merged into one class only if they point to
    at least two different groups.

    Args:
        cat: (n, k) int array of categorical codes (real training data).
        s_idx: column index of the protected attribute.
        threshold: P(S | category) threshold (pre-register; default 0.95).
        min_support: minimum category share of rows (default 1%).
        exclude: column indices never treated as proxies (e.g. the target).

    Returns:
        list of dicts, one per merged class:
          {'col': j, 'members': [codes...], 'rep': representative code,
           'group_of': {code: group}, 'purity': {code: P(S=group | code)},
           'support': {code: share of rows}}
    """
    cat = np.asarray(cat)
    n = len(cat)
    s = cat[:, s_idx]
    groups = np.unique(s)
    classes = []
    for j in range(cat.shape[1]):
        if j == s_idx or j in exclude:
            continue
        flagged = {}
        for c in np.unique(cat[:, j]):
            m = cat[:, j] == c
            if m.sum() < min_support * n:
                continue
            p = np.array([(s[m] == g).mean() for g in groups])
            g_star = int(groups[p.argmax()])
            if p.max() >= threshold:
                flagged[int(c)] = (g_star, float(p.max()), float(m.mean()))
        pointed = {v[0] for v in flagged.values()}
        if len(flagged) >= 2 and len(pointed) >= 2:
            members = sorted(flagged)
            classes.append({
                'col': j,
                'members': members,
                'rep': members[0],
                'group_of': {c: flagged[c][0] for c in members},
                'purity': {c: flagged[c][1] for c in members},
                'support': {c: flagged[c][2] for c in members},
            })
    return classes


def describe_classes(classes, col_names=None, labels=None):
    """Human-readable one-line summary per class (for logs / reports)."""
    out = []
    for cl in classes:
        j = cl['col']
        name = col_names[j] if col_names is not None else f'c{j}'
        def lab(c):
            return str(labels[name][c]) if labels is not None else str(c)
        mem = ', '.join(f"{lab(c)}->S={cl['group_of'][c]} "
                        f"(purity {cl['purity'][c]:.3f}, "
                        f"support {cl['support'][c]:.3f})"
                        for c in cl['members'])
        out.append(f'{name}: [{mem}]')
    return out


def mask_proxy_classes(cat, classes, mask_codes):
    """Pre-training decoupling (the user-side fix the paper's Sec. 7.2 leaves
    to the user): every member of each detected class is replaced by ONE new
    category (e.g. Husband, Wife -> MASK). The column keeps the legitimate
    information (spouse vs not) and loses the sex-specific label, without
    producing impossible rows such as male wives.

    mask_codes: one new code per class (aligned with `classes`), outside
    the column's existing codes. Apply the same masking to every table that
    is scored against the synthetic one, since its vocabulary changes.
    """
    out = np.asarray(cat).copy()
    for cl, code in zip(classes, mask_codes):
        j = cl['col']
        out[np.isin(out[:, j], cl['members']), j] = code
    return out


def proxy_member_labels(cat, classes):
    """Per-row member label for the in-model decoupling term.

    Returns a list aligned with `classes` of (labels, mask) int arrays:
    labels[i] = index of the row's category within cl['members'] (0 for rows
    outside the class), mask[i] = 1 if the row belongs to the class.
    """
    cat = np.asarray(cat)
    out = []
    for cl in classes:
        j, mem = cl['col'], cl['members']
        lab = np.zeros(len(cat), dtype=np.int64)
        for k, c in enumerate(mem):
            lab[cat[:, j] == c] = k
        out.append((lab, np.isin(cat[:, j], mem).astype(np.int64)))
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def impossible_combination_rate(syn_cat, real_cat, s_idx, rare=0.01,
                                min_support=0.005, exclude=()):
    """Share of synthetic rows carrying a (protected value, category) pair
    that is essentially absent from the real data - the 'male wife' rate,
    generalized to every categorical column.

    A pair (g, c) in column j is 'impossible' if category c has at least
    min_support of the real rows and P_real(S=g | c) < rare.

    Returns {'any': share of rows with >= 1 impossible pair,
             'by_col': {j: share of rows impossible in column j}}.
    """
    real_cat, syn_cat = np.asarray(real_cat), np.asarray(syn_cat)
    s_r, s_y = real_cat[:, s_idx], syn_cat[:, s_idx]
    n_r = len(real_cat)
    groups = np.unique(np.concatenate([s_r, s_y]))
    any_bad = np.zeros(len(syn_cat), dtype=bool)
    by_col = {}
    for j in range(real_cat.shape[1]):
        if j == s_idx or j in exclude:
            continue
        bad = np.zeros(len(syn_cat), dtype=bool)
        for c in np.unique(real_cat[:, j]):
            m = real_cat[:, j] == c
            if m.sum() < min_support * n_r:
                continue
            for g in groups:
                if (s_r[m] == g).mean() < rare:
                    bad |= (syn_cat[:, j] == c) & (s_y == g)
        by_col[j] = float(bad.mean())
        any_bad |= bad
    return {'any': float(any_bad.mean()), 'by_col': by_col}


def proxy_member_dependence(cat, classes, exclude=(), seed=0):
    """How well the OTHER categorical columns predict each class's member
    label among the class's rows (balanced accuracy, 3-fold, gradient
    boosting; chance 1 / n_members). Pass the protected column (and anything
    else to ignore) in `exclude`. tau = 0 should drive this to chance; it is
    the direct readout of the decoupling, independent of the protected
    column."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cat = np.asarray(cat)
    res = []
    for cl, (lab, mask) in zip(classes, proxy_member_labels(cat, classes)):
        rows = mask.astype(bool)
        cols = [k for k in range(cat.shape[1]) if k != cl['col']
                and k not in exclude]
        X, y = cat[rows][:, cols], lab[rows]
        if len(np.unique(y)) < 2 or min(np.bincount(y)) < 3:
            res.append(float('nan'))
            continue
        clf = HistGradientBoostingClassifier(
            categorical_features=[True] * len(cols),
            class_weight='balanced', max_iter=100, random_state=seed)
        pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(
            3, shuffle=True, random_state=seed))
        res.append(float(np.mean([(pred[y == g] == g).mean()
                                  for g in np.unique(y)])))
    return res


# --------------------------------------------------------------------------
# Scrambling repair (Feldman et al. 2015, adapted to categories)
# --------------------------------------------------------------------------

def _target_counts(n, shares):
    """Integer counts summing to n, proportional to shares (largest
    remainders)."""
    raw = n * shares
    out = np.floor(raw).astype(int)
    out[np.argsort(-(raw - out))[:n - out.sum()]] += 1
    return out


def repair_toward_independence(num, cat, s_idx, tau, exclude=(), seed=0):
    """Scramble each column's link to the protected attribute S with
    strength tau, keeping every column's overall distribution.

    Categorical column j: each group g's category shares become
        (1 - tau) * P_g(c) + tau * P(c)
    (P = pooled shares), so the pooled share of every category is preserved
    exactly - nothing can vanish. Rows leave over-represented categories in
    order of plausibility for their new category: a classifier fit on the
    other columns (S excluded) scores log P(dest) - log P(src) and the best
    moves go first. Numeric column: quantile repair - a value keeps its rank
    within its group and moves tau of the way to the pooled distribution's
    value at that rank.

    tau=0: unchanged. tau=1: every column is marginally independent of S
    (per-column, as in Feldman et al.; joint dependence can remain).
    Returns repaired (num, cat) copies.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    num, cat = np.asarray(num, dtype=np.float64).copy(), np.asarray(cat).copy()
    s = cat[:, s_idx]
    groups = np.unique(s)
    if tau <= 0:
        return num.astype(np.float32), cat
    # numeric: quantile repair toward the pooled distribution
    for j in range(num.shape[1]):
        pooled = np.sort(num[:, j])
        for g in groups:
            m = s == g
            ranks = np.argsort(np.argsort(num[m, j])) / max(m.sum() - 1, 1)
            target = np.quantile(pooled, ranks)
            num[m, j] = (1 - tau) * num[m, j] + tau * target
    # categorical: minimal-change reassignment toward pooled shares
    cols = [j for j in range(cat.shape[1]) if j != s_idx and j not in exclude]
    for j in cols:
        k = int(cat[:, j].max()) + 1
        pooled = np.bincount(cat[:, j], minlength=k) / len(cat)
        others = [c for c in range(cat.shape[1]) if c not in (j, s_idx)]
        X = np.concatenate([num, cat[:, others]], axis=1)
        is_cat = [False] * num.shape[1] + [True] * len(others)
        clf = HistGradientBoostingClassifier(categorical_features=is_cat,
                                             max_iter=60, early_stopping=False,
                                             random_state=seed)
        clf.fit(X, cat[:, j])
        P = np.full((len(cat), k), 1e-6)
        P[:, clf.classes_] = np.clip(clf.predict_proba(X), 1e-6, None)
        logP = np.log(P)
        for g in groups:
            idx = np.where(s == g)[0]
            cur = np.bincount(cat[idx, j], minlength=k)
            own = cur / len(idx)
            want = _target_counts(len(idx), (1 - tau) * own + tau * pooled)
            surplus = cur - want                 # >0: must give rows away
            if not (surplus > 0).any():
                continue
            src_rows = idx[surplus[cat[idx, j]] > 0]
            dests = np.where(surplus < 0)[0]
            # all candidate moves, best (most plausible) first
            score = (logP[src_rows][:, dests]
                     - logP[src_rows, cat[src_rows, j]][:, None])
            order = np.argsort(-score, axis=None)
            moved = np.zeros(len(src_rows), dtype=bool)
            for flat in order:
                r, d = divmod(int(flat), len(dests))
                src, dst = cat[src_rows[r], j], dests[d]
                if moved[r] or surplus[src] <= 0 or surplus[dst] >= 0:
                    continue
                cat[src_rows[r], j] = dst
                surplus[src] -= 1
                surplus[dst] += 1
                moved[r] = True
                if not (surplus > 0).any():
                    break
    return num.astype(np.float32), cat
