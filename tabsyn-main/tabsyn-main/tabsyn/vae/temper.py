"""Post-hoc proxy decoupling on the decoded rows.

A merged column (Adult: `relationship`) carries a sex-specific label
(Husband vs Wife) that is a near-copy of the TRUE sex. Post-hoc
decoupling re-draws that label on the decoded rows that hold it, from
    Q_tau(label) ∝ P(label | other non-sex columns)^tau * pi(label)
with P a real-trained class-balanced classifier (a likelihood ratio) and
pi the real Husband/Wife share. tau=1 keeps the real coupling (sampled);
tau=0 makes the label independent of every other column. The column's
marginal is untouched. The generated S column never enters: FLIP does not
reconstruct S, so the synthetic S is invented and conditioning or scoring
on it would measure nothing.

Scored on tier-1 decodes, where each row's source (true) sex is known:
leakage = how well a real-trained classifier recovers the true sex from
the decoded non-sex columns. The leakage-vs-tau curve is the target the
in-model version (PROXY_WEIGHT, strength lambda) has to match.
"""

import numpy as np

def sublabel_conditional(ref_num, ref_cat, num, cat, s_idx, proxy_idx,
                         pos_code, neg_code, seed=0):
    """P(label = pos_code | all columns except S and the proxy) on the rows
    of (num, cat) holding pos/neg, from a classifier fit on the reference
    (real) rows holding pos/neg. Returns (mask, (m, 2) [P(pos), P(neg)])."""
    from tabsyn.vae.paper_metrics import infer_protected_scores
    keep = [j for j in range(cat.shape[1]) if j != s_idx]
    p = keep.index(proxy_idx)
    ref_mask = np.isin(ref_cat[:, proxy_idx], (pos_code, neg_code))
    mask = np.isin(cat[:, proxy_idx], (pos_code, neg_code))
    # binary column at the proxy: 1 = pos; infer_protected_scores returns
    # P(largest label) = P(pos)
    rc = ref_cat[ref_mask][:, keep].copy()
    rc[:, p] = (rc[:, p] == pos_code).astype(rc.dtype)
    c = cat[mask][:, keep].copy()
    c[:, p] = (c[:, p] == pos_code).astype(c.dtype)
    p_pos = infer_protected_scores(ref_num[ref_mask], rc, num[mask], c, p,
                                   seed=seed)
    return mask, np.stack([p_pos, 1.0 - p_pos], axis=1)


def joint_tv(cat_a, cat_b, i, j):
    """TV distance between the (column i, column j) joint distributions of
    two tables - the fidelity cost of decoupling the proxy from sex."""
    def joint(c):
        keys, counts = np.unique(c[:, [i, j]], axis=0, return_counts=True)
        return {tuple(k): v / len(c) for k, v in zip(keys, counts)}
    pa, pb = joint(cat_a), joint(cat_b)
    return float(0.5 * sum(abs(pa.get(k, 0) - pb.get(k, 0))
                           for k in set(pa) | set(pb)))


def true_sex_leakage(ref_num, ref_cat, num, cat, true_s, s_idx, seed=0):
    """Balanced accuracy of a real-trained sex classifier recovering the
    SOURCE rows' true sex from the decoded non-sex columns (0.5 = none)."""
    from tabsyn.vae.paper_metrics import infer_protected
    pred = infer_protected(ref_num, ref_cat, num, cat, s_idx, seed=seed)
    y = np.asarray(true_s).astype(int)
    return float(np.mean([(pred[y == g] == g).mean() for g in np.unique(y)]))


def true_sex_joint_tv(cat, true_s, real_cat, s_idx, proxy_idx):
    """(true sex, proxy) joint TV of the decoded rows vs the real table."""
    c = cat.copy()
    c[:, s_idx] = true_s
    return joint_tv(c, real_cat, s_idx, proxy_idx)


def decouple_curve(ref_num, ref_cat, syn_num, syn_cat, true_s, s_idx,
                   proxy_idx, pos_code, neg_code, taus, real_cat=None,
                   seed=0):
    """Leakage-vs-fidelity curve for post-hoc proxy decoupling (tier-1).

    For each tau, re-draw the sex-specific label of the decoded rows holding
    it from the tempered conditional, then score the true-sex leakage, the
    (true sex, proxy) joint TV vs the real data, and the fraction of labels
    the re-draw changed.

    ref_*: real reference (held-out fold) for both classifiers; true_s: the
    decoded rows' source sex; real_cat: real table for the joint TV
    (defaults to ref_cat).
    Returns a list of {'tau', 'leakage', 'ber_true', 'joint_tv', 'flipped'}.
    """
    rng = np.random.default_rng(seed)
    real_cat = ref_cat if real_cat is None else real_cat
    mask, P = sublabel_conditional(ref_num, ref_cat, syn_num, syn_cat, s_idx,
                                   proxy_idx, pos_code, neg_code, seed)
    ref_lab = ref_cat[np.isin(ref_cat[:, proxy_idx], (pos_code, neg_code)),
                      proxy_idx]
    pi = np.array([(ref_lab == pos_code).mean(), (ref_lab == neg_code).mean()])
    # P comes from a class-balanced classifier, i.e. it is proportional to
    # the likelihood ratio, so P^tau * pi is the calibrated posterior at
    # tau=1 and pi at tau=0 (continuous in tau - no tau=1 special case).
    logP = np.log(np.clip(P, 1e-12, None))
    results = []
    for tau in taus:
        logQ = tau * logP + np.log(np.clip(pi, 1e-12, None))
        Q = np.exp(logQ - logQ.max(axis=1, keepdims=True))
        Q /= Q.sum(axis=1, keepdims=True)
        draw_pos = rng.random(len(Q)) < Q[:, 0]
        cat = syn_cat.copy()
        old = cat[mask, proxy_idx]
        new = np.where(draw_pos, pos_code, neg_code).astype(cat.dtype)
        cat[mask, proxy_idx] = new
        leak = true_sex_leakage(ref_num, ref_cat, syn_num, cat, true_s,
                                s_idx, seed)
        results.append({'tau': float(tau), 'leakage': leak,
                        'ber_true': 1.0 - leak,
                        'joint_tv': true_sex_joint_tv(cat, true_s, real_cat,
                                                      s_idx, proxy_idx),
                        'flipped': float((old != new).mean())})
    return results
