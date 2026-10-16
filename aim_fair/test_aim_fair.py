"""Self-tests for aim_fair (run with aim_fair/.venv; no pytest needed):

    aim_fair/.venv/Scripts/python aim_fair/test_aim_fair.py
"""
import numpy as np
import pandas as pd

import adult_domain as AD
from fair_aim import (_log_conditional, coupling, detect_proxy_classes_model,
                      gibbs_fix, match_marginals, merge_categories,
                      repair_measurements, repair_plans, tested_pair_rules,
                      violates)
from mbi import Domain, LinearMeasurement, estimation


def test_encode_decode_stays_in_bin():
    df = pd.DataFrame({'age': [17, 40, 90], 'fnlwgt': [12285, 189778, 1490400],
                       'education-num': [1, 9, 16],
                       'capital-gain': [0, 2174, 99999],
                       'capital-loss': [0, 1902, 4356],
                       'hours-per-week': [1, 40, 99]})
    for c in AD.CATEGORICAL:
        df[c] = 'a'
    cats = {c: ['a'] for c in AD.CATEGORICAL}
    codes = AD.encode(df, cats)
    back = AD.decode(codes, cats, seed=1)
    for c in AD.NUMERIC:
        e = np.asarray(AD.EDGES[c])
        k = codes[c]
        assert ((back[c] >= e[k]) & (back[c] < e[k + 1])).all(), c
    # one-value bins decode exactly
    assert list(back['education-num']) == [1, 9, 16]
    assert back['capital-gain'][0] == 0 and back['hours-per-week'][1] == 40


def test_coupling_moves_p_to_q():
    p, q = np.array([.6, .3, .1, 0.]), np.array([.1, .2, .3, .4])
    for ordinal in (True, False):
        T = coupling(p, q, ordinal)
        assert np.allclose(T.sum(1), 1)
        assert np.allclose(p @ T, q), ordinal
    # 0-1 coupling keeps the shared mass in place
    T = coupling(np.array([.5, .5]), np.array([.5, .5]), False)
    assert np.allclose(T, np.eye(2))
    # monotone coupling never moves mass downwards when q is shifted up
    T = coupling(p, q, True)
    assert np.allclose(np.tril(T, -1), 0)


class _Fac:      # minimal stand-in for an mbi Factor / model projection
    def __init__(self, attrs, v):
        self.domain = Domain(attrs, list(v.shape))
        self.v = v

    def datavector(self, flatten=True):
        return self.v.reshape(-1) if flatten else self.v


class _Model:
    def __init__(self, joint):
        self.joint = joint                 # [sex, rel, inc] counts

    def project(self, attrs):
        assert tuple(attrs) == ('sex', 'rel')
        return _Fac(['sex', 'rel'], self.joint.sum(2))


def test_repair_equalizes_and_dilutes_proxy():
    # rel 0 ~ "Husband": only men, high income; rel 1 ~ "Wife"; rel 2 other
    joint = np.zeros((2, 3, 2))
    joint[0, 0] = [20, 40]; joint[0, 2] = [30, 10]
    joint[1, 1] = [10, 10]; joint[1, 2] = [70, 10]
    dom = Domain(['sex', 'rel', 'inc'], [2, 3, 2])
    m = [LinearMeasurement(joint.sum(2).reshape(-1), ('sex', 'rel'), stddev=1.),
         LinearMeasurement(joint.sum(0).reshape(-1), ('rel', 'inc'), stddev=1.),
         LinearMeasurement(joint.sum((0, 2)), ('rel',), stddev=1.)]
    plans = repair_plans(_Model(joint), dom, ['rel'], ordinal=(), tau=1.0,
                         target='average')
    r = repair_measurements(m, dom, plans)
    sr = np.asarray(r[0].noisy_measurement).reshape(2, 3)
    cond = sr / sr.sum(1, keepdims=True)
    assert np.allclose(cond[0], cond[1])                 # P(rel | sex) equal
    assert np.allclose(sr.sum(1), joint.sum((1, 2)))     # group sizes kept
    # 1-way rel marginal consistent with the repaired [sex, rel]
    assert np.allclose(r[2].noisy_measurement, sr.sum(0))
    # Husband's income rate falls towards the overall rate
    ri = np.asarray(r[1].noisy_measurement).reshape(3, 2)
    before = joint.sum(0)[0, 1] / joint.sum(0)[0].sum()
    after = ri[0, 1] / ri[0].sum()
    overall = joint[..., 1].sum() / joint.sum()
    assert overall < after < before
    assert np.isclose(ri.sum(), joint.sum())
    # pooled target keeps the overall P(rel) and still equalizes P(rel | sex)
    pp = repair_plans(_Model(joint), dom, ['rel'], ordinal=(), tau=1.0,
                      target='pooled')
    sp = np.asarray(repair_measurements(m, dom, pp)[0].noisy_measurement
                    ).reshape(2, 3)
    assert np.allclose(sp.sum(0), joint.sum((0, 2)))
    cp = sp / sp.sum(1, keepdims=True)
    assert np.allclose(cp[0], cp[1])
    # tau = 0 changes nothing
    p0 = repair_plans(_Model(joint), dom, ['rel'], ordinal=(), tau=0.0)
    r0 = repair_measurements(m, dom, p0)
    assert all(np.allclose(a.noisy_measurement, b.noisy_measurement)
               for a, b in zip(m, r0))


class _Joint:    # model stand-in holding a full joint table
    def __init__(self, attrs, joint):
        self.attrs, self.joint = list(attrs), joint

    def project(self, attrs):
        ax = tuple(i for i, a in enumerate(self.attrs) if a not in attrs)
        kept = [a for a in self.attrs if a in attrs]
        v = self.joint.sum(ax)
        v = np.transpose(v, [kept.index(a) for a in attrs])
        return _Fac(list(attrs), v)


def test_mask_detects_and_merges_sex_specific_levels():
    # rel: 0 Husband (men only), 1 Wife (women only), 2 other (mixed)
    joint = np.zeros((2, 3, 2))
    joint[0, 0] = [30, 30]; joint[1, 1] = [10, 10]
    joint[0, 2] = [40, 10]; joint[1, 2] = [50, 20]
    dom = Domain(['sex', 'rel', 'inc'], [2, 3, 2])
    merges = detect_proxy_classes_model(_Joint(dom.attributes, joint), dom)
    assert merges == [('rel', [0, 1])]
    m = [LinearMeasurement(joint.sum(0).reshape(-1), ('rel', 'inc'), stddev=1.),
         LinearMeasurement(joint.sum(2).reshape(-1), ('sex', 'rel'), stddev=1.)]
    out, nd = merge_categories(m, dom, merges)
    assert nd['rel'] == 2
    ri = np.asarray(out[0].noisy_measurement).reshape(2, 2)
    assert np.allclose(ri[1], joint.sum(0)[0] + joint.sum(0)[1])  # spouse
    assert np.allclose(ri[0], joint.sum(0)[2])                     # other
    # no merge when the flagged levels all point to the same group
    j2 = joint.copy(); j2[1, 1] = 0; j2[0, 1] = [10, 10]
    assert detect_proxy_classes_model(_Joint(dom.attributes, j2), dom) == []


def test_tested_pair_rules():
    dom = Domain(['sex', 'a', 'b'], [2, 2, 2])
    y = np.array([[500., 500.], [1000., 0.]])     # (a=1, b=1) never occurs
    m = [LinearMeasurement(y.reshape(-1), ('a', 'b'), stddev=2.),
         LinearMeasurement(np.ones(4), ('sex', 'a'), stddev=2.)]
    rules = tested_pair_rules(m, dom)
    assert rules == [('a', 'b', 1, 1)]             # sex tables ignored
    codes = {'a': np.array([1, 1, 0]), 'b': np.array([1, 0, 1])}
    assert list(violates(codes, rules)) == [True, False, False]


def _small_model():
    rng = np.random.default_rng(0)
    dom = Domain(['a', 'b', 'c'], [2, 3, 2])
    x = {'a': rng.integers(0, 2, 3000), 'c': rng.integers(0, 2, 3000)}
    x['b'] = (x['a'] + x['c'] + rng.integers(0, 2, 3000)) % 3
    def tab(cl):
        t = np.zeros([dom[c] for c in cl])
        np.add.at(t, tuple(x[c] for c in cl), 1)
        return t.reshape(-1)
    meas = [LinearMeasurement(tab(cl), cl, stddev=1.0)
            for cl in [('a', 'b'), ('b', 'c'), ('a', 'c')]]
    return estimation.MirrorDescent().estimate(dom, meas, iters=300), dom


def test_log_conditional_matches_joint():
    model, dom = _small_model()
    f = model.project(('a', 'b', 'c'))
    if tuple(f.domain.attributes) != ('a', 'b', 'c'):
        f = f.transpose(('a', 'b', 'c'))
    joint = np.asarray(f.datavector(flatten=False), np.float64)
    codes = {'a': np.array([0, 1]), 'b': np.array([0, 2]),
             'c': np.array([0, 1])}
    lp = _log_conditional(model, codes, ['a', 'b'], np.arange(2))
    for r in range(2):
        p = np.exp(lp[r] - lp[r].max()); p /= p.sum()
        q = joint[:, :, codes['c'][r]]; q = q / q.sum()
        assert np.allclose(p, q, atol=1e-3), (p, q)


def test_gibbs_fix_removes_violations_and_keeps_other_columns():
    model, dom = _small_model()
    rng = np.random.default_rng(1)
    codes = {k: np.asarray(v).astype(np.int64) for k, v in
             model.synthetic_data(rows=2000).to_dict().items()}
    rules = [('a', 'b', 1, 2), ('b', 'c', 0, 1)]
    before_c = codes['c'].copy()
    assert violates(codes, rules).any()
    gibbs_fix(model, codes, rules, rng)
    assert not violates(codes, rules).any()
    # c only changes through the (b, c) rule's rows
    moved = codes['c'] != before_c
    assert moved.mean() < 0.5
    rep = match_marginals(model, codes, rules, rng, tol=0.0)
    assert not violates(codes, rules).any()
    assert all(after <= before + 1e-9 for before, after in rep.values())


if __name__ == '__main__':
    test_log_conditional_matches_joint()
    test_gibbs_fix_removes_violations_and_keeps_other_columns()
    test_mask_detects_and_merges_sex_specific_levels()
    test_tested_pair_rules()
    test_encode_decode_stays_in_bin()
    test_coupling_moves_p_to_q()
    test_repair_equalizes_and_dilutes_proxy()
    print('aim_fair self-tests passed')
