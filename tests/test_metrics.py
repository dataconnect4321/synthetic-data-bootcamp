"""Tests for the evaluation code: run_flip_demo metric fixes and the paper's
fairness metrics in fairness_eval.py.

Run:  python -m pytest tests/test_metrics.py -v
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import run_flip_demo as demo
from fairness_eval import a_ncb, ber, ncb, paper_fairness_metrics

S_IDX, T_IDX = 0, 1  # positions in demo.CATEGORICAL_COLS


def _encoders(sizes):
    """cat_encoders keyed like load_sample(): target labels are '0'/'1'."""
    enc = {}
    for col, k in zip(demo.CATEGORICAL_COLS, sizes):
        labels = ['0', '1'] if col == demo.TARGET_COL else [f'{col}_{i}' for i in range(k)]
        enc[col] = pd.Index(labels)
    return enc


def _data(n, sizes, rng):
    num = rng.normal(size=(n, 3))
    cat = np.stack([rng.integers(0, k, n) for k in sizes], axis=1)
    return num, cat


def test_downstream_fairness_tolerates_missing_category():
    """Regression: synthetic data that never produces a category made the
    one-hot widths differ (X_real @ w crashed - runs/hmda/logs/demo.log)."""
    rng = np.random.default_rng(0)
    sizes = [3, 2, 3, 4]
    real_num, real_cat = _data(300, sizes, rng)
    syn_num, syn_cat = _data(300, sizes, rng)
    syn_cat[:, 3] = np.minimum(syn_cat[:, 3], 1)  # categories 2, 3 never appear
    out = demo._downstream_fairness(real_num, real_cat, syn_num, syn_cat,
                                    T_IDX, S_IDX, _encoders(sizes))
    assert 0.0 <= out['dp_diff'] <= 1.0


def test_downstream_fairness_excludes_target():
    """The target must not be a feature: with noise features, predictions
    cannot reproduce a target that is perfectly split by group."""
    rng = np.random.default_rng(1)
    sizes = [2, 2, 3, 4]
    real_num, real_cat = _data(400, sizes, rng)
    real_cat[:, T_IDX] = real_cat[:, S_IDX]  # group 0 -> '0', group 1 -> '1'
    syn_num, syn_cat = _data(400, sizes, rng)
    syn_cat[:, T_IDX] = syn_cat[:, S_IDX]
    out = demo._downstream_fairness(real_num, real_cat, syn_num, syn_cat,
                                    T_IDX, S_IDX, _encoders(sizes))
    assert out['dp_diff'] < 0.5  # reading the label back would give 1.0


def test_clip_floor_is_standardized_zero():
    """Real-unit 0 is -mean/std in standardized units; clipping at 0 there
    floored every column at its mean."""
    syn = np.array([[-5.0, 0.0, 3.0]])
    means, stds = np.array([2.0, 1.0, 0.0]), np.array([1.0, 0.5, 1.0])
    clipped = demo.clip_nonnegative(syn, means, stds)
    np.testing.assert_allclose(clipped, [[-2.0, 0.0, 3.0]])
    assert (clipped * stds + means >= 0).all()


def test_ber_near_ideal_when_independent():
    rng = np.random.default_rng(2)
    sizes = [3, 2, 3, 4]
    num, cat = _data(900, sizes, rng)
    b, ideal, _ = ber(num, cat, sizes, S_IDX, seed=0)
    assert abs(ideal - 2 / 3) < 1e-12
    assert abs(b - ideal) < 0.06


def test_ber_low_when_group_is_encoded():
    rng = np.random.default_rng(3)
    sizes = [3, 2, 3, 4]
    num, cat = _data(900, sizes, rng)
    num[:, 0] = cat[:, S_IDX] * 5.0  # a numeric column reveals the group
    b, _, _ = ber(num, cat, sizes, S_IDX, seed=0)
    assert b < 0.05


def test_ncb_balanced_and_separated():
    groups = np.array([0, 0, 1, 1])
    assert ncb(np.array([0, 1, 0, 1]), groups, min_group=1) == 1.0  # mixed
    assert ncb(np.array([0, 0, 1, 1]), groups, min_group=1) == 0.0  # = groups


def test_ncb_ignores_tiny_adversarial_groups():
    """A group the adversary assigns to a single row would force NCB to 0
    (that row sits in one cluster); such groups are left out."""
    groups = np.array([0] * 100 + [1] * 100 + [2])
    clusters = np.array([0, 1] * 100 + [0])
    assert ncb(clusters, groups) == 1.0


def test_a_ncb_detects_group_structure():
    rng = np.random.default_rng(4)
    sizes = [2, 2, 3, 4]
    num, cat = _data(600, sizes, rng)
    num[:, 0] = cat[:, S_IDX] * 8.0  # two well-separated groups
    _, _, oof = ber(num, cat, sizes, S_IDX, seed=0)
    assert a_ncb(num, cat, sizes, S_IDX, oof, seed=0) < 0.2


def test_paper_metrics_end_to_end():
    rng = np.random.default_rng(5)
    sizes = [3, 2, 3, 4]
    real_num, real_cat = _data(450, sizes, rng)
    syn_num, syn_cat = _data(450, sizes, rng)
    out = paper_fairness_metrics(real_num, real_cat, syn_num, syn_cat, sizes,
                                 S_IDX, ['a', 'b', 'c'], demo.CATEGORICAL_COLS)
    assert set(out) == {'real', 'syn', 'task_fairness'}
    # every non-protected column is scored as a task
    assert len(out['task_fairness']) == 3 + len(sizes) - 1
    assert all(v['syn'] >= 0 for v in out['task_fairness'].values())


def test_task_fairness_tolerates_single_row_class():
    """Regression: a synthetic category with one row made sklearn's
    early-stopping stratified split raise (n > 10k rows)."""
    rng = np.random.default_rng(6)
    sizes = [3, 2, 3, 4]
    real_num, real_cat = _data(10_500, sizes, rng)
    syn_num, syn_cat = _data(10_500, sizes, rng)
    syn_cat[:, 2] = 0
    syn_cat[0, 2] = 2  # category 2 appears once
    out = paper_fairness_metrics(real_num, real_cat, syn_num, syn_cat, sizes,
                                 S_IDX, ['a', 'b', 'c'], demo.CATEGORICAL_COLS)
    assert np.isfinite(out['task_fairness'][demo.CATEGORICAL_COLS[2]]['syn'])
