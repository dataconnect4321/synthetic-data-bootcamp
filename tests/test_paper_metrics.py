"""Tests for the paper's metrics (tabsyn/vae/paper_metrics.py).

Run:  python -m pytest tests/test_paper_metrics.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.paper_metrics import (
    balanced_error_rate,
    fairness_scores,
    normalized_cluster_balance,
    epsilon_identifiability,
)


def _table(n, n_groups, signal, seed=0):
    """Numeric + categorical features whose distribution depends on the
    group with strength `signal`, and the protected column (last cat)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    num = rng.normal(size=(n, 3)) + signal * g[:, None]
    other = np.where(rng.random(n) < signal / (1 + signal), g,
                     rng.integers(0, n_groups, n))
    cat = np.stack([other, g], axis=1)
    return num, cat, g


def test_ber_ranges_two_and_three_groups():
    for k in (2, 3):
        num, cat, _ = _table(3000, k, signal=0.0)
        s = fairness_scores(num, cat, s_idx=1)
        assert abs(s['ber'] - (k - 1) / k) < 0.05
        assert s['ber_chance'] == (k - 1) / k


def test_ber_low_when_features_encode_group():
    num, cat, _ = _table(3000, 2, signal=3.0)
    assert fairness_scores(num, cat, s_idx=1)['ber'] < 0.1


def test_ber_not_gamed_by_majority_guess():
    """A constant majority-class guess must not look 'fair' by accident:
    with strong signal and 90/10 groups the classifier must still win."""
    rng = np.random.default_rng(1)
    g = (rng.random(4000) < 0.1).astype(int)
    num = rng.normal(size=(4000, 2)) + 4 * g[:, None]
    cat = np.stack([rng.integers(0, 3, 4000), g], axis=1)
    assert fairness_scores(num, cat, s_idx=1)['ber'] < 0.1


def test_normalized_cluster_balance_values():
    clusters = np.array([0] * 50 + [1] * 50)
    balanced = np.tile([0, 1], 50)
    assert normalized_cluster_balance(clusters, balanced) == 1.0
    skewed = np.array([0] * 50 + [1] * 50)  # cluster == group
    assert normalized_cluster_balance(clusters, skewed) == 0.0
    mixed = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
    assert abs(normalized_cluster_balance(clusters, mixed) - 0.25) < 1e-9


def test_a_ncb_low_when_group_encoded_high_when_independent():
    num, cat, _ = _table(3000, 2, signal=3.0)
    low = fairness_scores(num, cat, s_idx=1)['a_ncb']
    num, cat, _ = _table(3000, 2, signal=0.0)
    high = fairness_scores(num, cat, s_idx=1)['a_ncb']
    assert low < 0.5 < high


def test_a_ncb_cannot_be_gamed_by_scrambling_protected_column():
    real_num, real_cat, _ = _table(3000, 2, signal=3.0, seed=1)
    num, cat, _ = _table(3000, 2, signal=3.0, seed=2)  # "synthetic": group still encoded
    ref = (real_num, real_cat)
    base = fairness_scores(num, cat, s_idx=1, ref=ref)
    scrambled = cat.copy()
    scrambled[:, 1] = np.random.default_rng(5).permutation(scrambled[:, 1])
    after = fairness_scores(num, scrambled, s_idx=1, ref=ref)
    assert after['a_ncb'] == base['a_ncb']
    assert after['a_ncb'] < 0.5
    # BER and stored-label NCB are what scrambling does fool
    assert after['ber'] > 0.45 > base['ber']
    assert after['ncb_stored'] > base['ncb_stored']


def test_eps_identifiability_copy_vs_far_vs_fresh():
    rng = np.random.default_rng(0)
    num = rng.normal(size=(600, 3)).astype(np.float32)
    cat = rng.integers(0, 4, size=(600, 2))
    # exact copy with tiny jitter: every real record has a closer synthetic twin
    twin = epsilon_identifiability(num, cat, num + 1e-4, cat, device='cpu')
    assert twin > 0.95
    # synthetic data far from every real record
    far = epsilon_identifiability(num, cat, num + 50.0, (cat + 1) % 4,
                                  device='cpu')
    assert far == 0.0
    # fresh draws from the same distribution: neither side systematically closer
    fresh_num = rng.normal(size=(600, 3)).astype(np.float32)
    fresh_cat = rng.integers(0, 4, size=(600, 2))
    mid = epsilon_identifiability(num, cat, fresh_num, fresh_cat,
                                  device='cpu')
    assert 0.3 < mid < 0.7


def test_balanced_error_rate_basic():
    y = np.array([0, 0, 0, 0, 1, 1])
    assert balanced_error_rate(y, np.zeros(6, dtype=int)) == 0.5
    assert balanced_error_rate(y, y) == 0.0


def test_alpha_beta_identical_vs_shifted():
    from tabsyn.vae.paper_metrics import alpha_precision_beta_recall
    rng = np.random.default_rng(0)
    num = rng.normal(size=(1500, 3))
    cat = rng.integers(0, 4, size=(1500, 2))
    num2 = rng.normal(size=(1500, 3))
    cat2 = rng.integers(0, 4, size=(1500, 2))
    a_same, b_same = alpha_precision_beta_recall(num, cat, num2, cat2, [4, 4])
    a_far, b_far = alpha_precision_beta_recall(num, cat, num2 + 3.0, cat2,
                                               [4, 4])
    assert a_same > 0.85 and b_same > 0.3
    assert a_far < a_same - 0.3 and b_far < b_same


def test_validity_rules_catch_each_violation_type():
    from tabsyn.vae.paper_metrics import fit_validity_rules, validity_report
    rng = np.random.default_rng(0)
    n = 4000
    married = rng.integers(0, 2, n)                     # 0 never, 1 married
    spouse = np.where(married == 1, 1, rng.integers(0, 1, n))   # spouse => married
    age = np.where(married == 1, rng.integers(25, 80, n),
                   rng.integers(17, 80, n)).astype(float)
    amount = rng.integers(0, 1000, n).astype(float)
    num = np.stack([age, amount], 1)
    cat = np.stack([married, spouse], 1)
    rules = fit_validity_rules(num, cat, ['age', 'amount'],
                               ['married', 'spouse'], min_support=100)
    real = validity_report(rules, num, cat)
    assert real['valid_rate'] > 0.97
    bad_num = num[:4].copy()
    bad_num[0, 1] = -50            # negative amount: range
    bad_num[1, 0] = 40.5           # non-integer age: integer
    bad_num[2, 0] = 18             # married 18-year-old: conditional range
    bad_cat = cat[:4].copy()
    bad_cat[2] = [1, 1]
    bad_cat[3] = [0, 1]            # spouse but never married: fd + pair
    rep = validity_report(rules, bad_num, bad_cat)
    assert rep['by_type']['range'] == 0.25
    assert rep['by_type']['integer'] == 0.25
    assert rep['by_type']['cond'] >= 0.25
    assert rep['by_type']['fd'] >= 0.25 or rep['by_type']['pair'] >= 0.25
    # the per-row mask flags exactly the rows the report counts as invalid
    from tabsyn.vae.paper_metrics import invalid_rows
    mask = invalid_rows(rules, bad_num, bad_cat)
    assert mask.tolist() == [True, True, True, True]
    assert np.isclose(mask.mean(), 1 - rep['valid_rate'])
    assert np.isclose(invalid_rows(rules, num, cat).mean(), 1 - real['valid_rate'])


def test_dp_validity_rules_find_planted_impossible_pair():
    """The DP rule learner bans a combination that never occurs (male
    'wife'), keeps common combinations, and reports the Gaussian RDP."""
    from tabsyn.vae.paper_metrics import fit_validity_rules_dp
    rng = np.random.default_rng(0)
    n = 30000
    sex = rng.integers(0, 2, n)                       # 0 female, 1 male
    rel = np.where(sex == 1, rng.choice([0, 2], n), rng.choice([1, 2], n))
    other = rng.integers(0, 3, n)                     # independent column
    cat = np.stack([sex, rel, other], 1)
    num = rng.integers(0, 50, (n, 1)).astype(float)
    rules, rdp = fit_validity_rules_dp(num, cat, ['x'], ['sex', 'rel', 'o'],
                                       noise_multiplier=5.0, seed=1)
    pairs = {(a, va, b, vb) for a, va, b, vb, _ in rules['pair']}
    assert (0, 1, 1, 1) in pairs          # male & rel=1 ("wife")
    assert (0, 0, 1, 0) in pairs          # female & rel=0 ("husband")
    assert all(a != 2 and b != 2 for a, _, b, _ in pairs)
    assert rules['integer'] == [(0, 'x')]
    assert abs(rdp[2] - 2 / (2 * 25.0)) < 1e-12


def test_constrained_repair_fixes_with_model_preference_and_protects():
    """A 'female husband' is fixed by editing relationship (sex protected)
    to the most probable valid category; valid rows are untouched."""
    from tabsyn.vae.paper_metrics import constrained_repair
    # columns: 0 sex (0 F, 1 M), 1 relationship (0 husband, 1 wife, 2 other)
    rules = {'pair': [(0, 0, 1, 0, 'F & husband'), (0, 1, 1, 1, 'M & wife')],
             'fd': []}
    cat = np.array([[0, 0],     # female husband -> invalid
                    [1, 0],     # male husband   -> valid
                    [1, 1]])    # male wife      -> invalid
    p_sex = np.array([[0.4, 0.6], [0.1, 0.9], [0.2, 0.8]])
    p_rel = np.array([[0.5, 0.1, 0.4],    # row 0: 'other' beats 'wife'
                      [0.9, 0.05, 0.05],
                      [0.3, 0.3, 0.4]])
    out, info = constrained_repair(cat, [p_sex, p_rel], rules, protect=(0,))
    assert out[:, 0].tolist() == [0, 1, 1]           # sex never edited
    assert out[0, 1] == 2 and out[2, 1] == 2         # most probable valid fix
    assert out[1].tolist() == [1, 0]                 # valid row untouched
    assert info['unresolved'] == 0 and abs(info['rows_changed'] - 2 / 3) < 1e-9
    # without protection the cheaper fix for row 0 is flipping sex (0.6 > 0.4)
    out2, _ = constrained_repair(cat, [p_sex, p_rel], rules, protect=())
    assert out2[0].tolist() == [1, 0]


def test_famd_components_truncates_and_default_is_two():
    import inspect
    from tabsyn.vae.paper_metrics import famd_coordinates, cluster_features
    rng = np.random.default_rng(0)
    num = rng.normal(size=(200, 3))
    cat = rng.integers(0, 4, (200, 3))
    full = famd_coordinates(num, cat)
    two = famd_coordinates(num, cat, 2)
    assert two.shape == (200, 2) and np.allclose(two, full[:, :2])
    assert inspect.signature(cluster_features).parameters[
        'famd_components'].default == 2
