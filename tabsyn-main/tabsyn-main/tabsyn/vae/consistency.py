"""Consistency pass for synthetic tables: enforce only STRONG learned rules.

Rules come from tabsyn.vae.paper_metrics.fit_validity_rules (learned from
the real data). Only categorical rules with strong evidence are enforced:
  * pair rules - a category pair absent from the real data although
    independence predicts >= min_expected rows (default 100); weaker ones
    (e.g. race=White & country=Haiti, ~39 expected) are regularities, not
    impossibilities, and are only reported, never enforced
  * fd rules   - near-deterministic implications (>= 99%, >= 1% support)
Rules touching the protected attribute (or any column in `exclude`) are
never enforced: a learned rule like Husband -> Male would put the bias back.

Two ways to apply them:
  reject_invalid - drop rows that break a strong rule
  edit_invalid   - change ONE field per broken rule to the most plausible
                   value (classifier fit on the real data, protected column
                   excluded) among those that make the row consistent
"""

import numpy as np


def strong_rules(rules, real_cat, min_expected=100.0, exclude=()):
    """Subset of fitted rules safe to enforce: pair rules with expected
    count >= min_expected under independence, all fd rules; nothing that
    touches a column in `exclude`."""
    real_cat = np.asarray(real_cat)
    n = len(real_cat)
    keep = {'pair': [], 'fd': []}
    for a, va, b, vb, desc in rules['pair']:
        if a in exclude or b in exclude:
            continue
        exp = n * (real_cat[:, a] == va).mean() * (real_cat[:, b] == vb).mean()
        if exp >= min_expected:
            keep['pair'].append((a, va, b, vb, desc))
    for a, va, b, vb, desc in rules['fd']:
        if a not in exclude and b not in exclude:
            keep['fd'].append((a, va, b, vb, desc))
    return keep


def violations(strong, cat):
    """(n,) bool: row breaks at least one strong rule."""
    cat = np.asarray(cat)
    bad = np.zeros(len(cat), dtype=bool)
    for a, va, b, vb, _ in strong['pair']:
        bad |= (cat[:, a] == va) & (cat[:, b] == vb)
    for a, va, b, vb, _ in strong['fd']:
        bad |= (cat[:, a] == va) & (cat[:, b] != vb)
    return bad


def reject_invalid(strong, num, cat):
    """Drop rows breaking a strong rule. Returns (num, cat, kept_mask)."""
    keep = ~violations(strong, cat)
    return np.asarray(num)[keep], np.asarray(cat)[keep], keep


def _rule_columns(strong):
    cols = set()
    for kind in ('pair', 'fd'):
        for a, _, b, _, _ in strong[kind]:
            cols.update((a, b))
    return sorted(cols)


def edit_invalid(strong, num, cat, real_num, real_cat, s_idx, sizes,
                 max_rounds=3, seed=0):
    """Repair rows breaking a strong rule by changing one field at a time.

    For every column involved in a strong rule, a classifier is fit on the
    REAL data to predict it from the other columns (protected column s_idx
    excluded). For a broken row, every alternative value of every involved
    column is tried; among candidates that leave the row consistent, the one
    with the highest log P(new value | rest of row) is applied. Repeats up
    to max_rounds; rows still inconsistent are left as they are.
    Returns (cat_edited, n_edited_rows, n_still_invalid).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    num, cat = np.asarray(num, float), np.asarray(cat).copy()
    real_num, real_cat = np.asarray(real_num, float), np.asarray(real_cat)
    cols = [j for j in _rule_columns(strong) if j != s_idx]
    models = {}
    for j in cols:
        others = [c for c in range(cat.shape[1]) if c not in (j, s_idx)]
        X = np.concatenate([real_num, real_cat[:, others]], axis=1)
        clf = HistGradientBoostingClassifier(
            categorical_features=[False] * real_num.shape[1]
            + [True] * len(others), max_iter=60, early_stopping=False,
            random_state=seed).fit(X, real_cat[:, j])
        models[j] = (clf, others)
    edited = np.zeros(len(cat), dtype=bool)
    for _ in range(max_rounds):
        bad = np.where(violations(strong, cat))[0]
        if len(bad) == 0:
            break
        best_score = np.full(len(bad), -np.inf)
        best_col = np.full(len(bad), -1)
        best_val = np.full(len(bad), -1)
        for j in cols:
            clf, others = models[j]
            X = np.concatenate([num[bad], cat[bad][:, others]], axis=1)
            P = np.full((len(bad), sizes[j]), 1e-9)
            P[:, clf.classes_] = np.clip(clf.predict_proba(X), 1e-9, None)
            for v in range(sizes[j]):
                cand = cat[bad].copy()
                cand[:, j] = v
                ok = ~violations(strong, cand) & (cat[bad, j] != v)
                score = np.where(ok, np.log(P[:, v]), -np.inf)
                better = score > best_score
                best_score[better] = score[better]
                best_col[better] = j
                best_val[better] = v
        fix = best_col >= 0
        rows = bad[fix]
        cat[rows, best_col[fix]] = best_val[fix]
        edited[rows] = True
    return cat, int(edited.sum()), int(violations(strong, cat).sum())
