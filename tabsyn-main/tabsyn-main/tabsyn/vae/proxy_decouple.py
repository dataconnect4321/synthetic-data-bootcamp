"""Proxy decoupling for near-deterministic proxies of the protected attribute.

Strict disentanglement is infeasible when a category nearly determines the
protected attribute S (Adult: relationship=Husband -> male, Wife -> female):
a decoder that must reproduce that category from a sex-free latent cannot
(data-processing inequality). Instead each such proxy class is merged into
one category ("spouse"): the legitimate information is kept, the
protected-specific label (WHICH member, "Husband" vs "Wife") is dropped.

Everything here is dataset-agnostic: proxy categories are found with a
pre-registered threshold on P(S | category), and only categories pointing to
DIFFERENT groups are merged (a column whose flagged categories all point to
the same group has nothing to decouple).

Uses:
  * mask_proxy_classes() merges each class into one new category (e.g.
    MASK); run_adult_flip.py --mask-proxies scores tables released with
    merged levels against the real data merged the same way.
  * impossible_combination_rate() reports (S, category) pairs absent from
    the real data, e.g. 'male wife'.

PRIVACY NOTE: detection is computed on the real training fold from a
handful of category-level counts, not covered by the DP-SGD accounting -
report it as an uncounted release (or add Laplace/Gaussian noise to the
counts before use). The AIM hybrid (aim_fair/) detects the classes from its
noisy measurements instead.
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
