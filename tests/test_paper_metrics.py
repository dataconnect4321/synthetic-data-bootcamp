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
