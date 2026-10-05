"""Tests for the paper-protocol evaluation (paper_eval.py), the fold split
in run_experiment.py, and the beta schedule in run_flip_demo.py.

Run:  python -m pytest tests/test_paper_eval.py -v
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import paper_eval
import run_experiment
from run_flip_demo import beta_at

DATA_CFG = {'sensitive_col': 'sex', 'target_col': 'target',
            'groups': ['Male', 'Female'], 'numeric_cols': ['age', 'hours'],
            'categorical_cols': ['job']}


def _frame(n, rng, bias=0.3):
    """Toy table where the target depends on sex (so SP > 0) and age."""
    sex = rng.choice(['Male', 'Female'], n, p=[0.65, 0.35])
    age = rng.normal(40, 10, n).round()
    p = 0.15 + bias * (sex == 'Male') + 0.01 * (age - 40)
    return pd.DataFrame({'sex': sex, 'target': (rng.random(n) < p).astype(int),
                         'age': age, 'hours': rng.normal(40, 8, n).round(),
                         'job': rng.choice(['a', 'b', 'c'], n)})


def test_alpha_beta_high_for_same_distribution_low_for_noise():
    rng = np.random.default_rng(0)
    X = rng.random((800, 4))
    a_same, b_same = paper_eval.alpha_precision_beta_recall(X, rng.random((800, 4)))
    a_far, b_far = paper_eval.alpha_precision_beta_recall(X, rng.random((800, 4)) + 3)
    assert a_same > 0.85 and b_same > 0.3
    assert a_far < 0.2 and b_far < 0.05


def test_eps_identifiability_copy_is_one_independent_is_about_half():
    rng = np.random.default_rng(1)
    real = _frame(600, rng)
    num, cat = ['age', 'hours'], ['sex', 'target', 'job']
    # Jitter so rows are unique: duplicate real rows would tie at distance 0.
    real['hours'] = real['hours'] + rng.random(len(real))
    # A copy puts every real row's twin at distance 0: fully identifiable.
    assert paper_eval.eps_identifiability(real, real.copy(), num, cat) > 0.95
    other = _frame(600, rng)
    other['hours'] = other['hours'] + rng.random(len(other))
    assert 0.3 < paper_eval.eps_identifiability(real, other, num, cat) < 0.7


def test_evaluate_end_to_end_and_detects_bias():
    rng = np.random.default_rng(2)
    train, test = _frame(1500, rng), _frame(800, rng)
    biased = paper_eval.evaluate(train, test, _frame(1500, rng, bias=0.3), DATA_CFG)
    fair = paper_eval.evaluate(train, test, _frame(1500, rng, bias=0.0), DATA_CFG)
    for key in ('synth_auc', 'alpha_precision', 'beta_recall', 'sp', 'eo',
                'eps_identifiability', 'ber', 'a_ncb'):
        assert np.isfinite(biased[key]), key
    assert fair['sp'] < biased['sp']


def test_evaluate_single_class_synthetic_is_nan_not_crash():
    rng = np.random.default_rng(3)
    train, test = _frame(600, rng), _frame(300, rng)
    syn = _frame(600, rng)
    syn['target'] = 0
    out = paper_eval.evaluate(train, test, syn, DATA_CFG)
    assert np.isnan(out['synth_auc']) and out['syn_positive_rate'] == 0.0


def test_folds_partition_rows_and_keep_ratios(tmp_path):
    rng = np.random.default_rng(4)
    df = _frame(3000, rng)
    src = tmp_path / 'data.csv'
    df.to_csv(src, index=False)
    cfg = {'paths': {'preprocessed_data': str(src)}, 'data': DATA_CFG}
    seen = []
    for k in range(run_experiment.N_FOLDS):
        tr, te = (pd.read_csv(p) for p in
                  run_experiment.make_fold(cfg, k, str(tmp_path / f'fold{k}')))
        assert len(tr) + len(te) == len(df)
        assert abs(te['target'].mean() - df['target'].mean()) < 0.02
        assert abs((te['sex'] == 'Female').mean() - (df['sex'] == 'Female').mean()) < 0.02
        seen.append(te)
    # The three test folds cover every row exactly once.
    assert len(pd.concat(seen).drop_duplicates()) == len(df.drop_duplicates())


def _tiny_training(use_dp, ls_variant, phase1=2, phase2=1, ckpt=None):
    import copy
    import torch
    import run_flip_demo as demo
    rng = np.random.default_rng(5)
    n = 256
    X_num = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float32)
    X_cat = torch.tensor(np.stack([rng.integers(0, 2, n), rng.integers(0, 2, n),
                                   rng.integers(0, 3, n)], axis=1))
    p = copy.deepcopy(demo.TRADEOFF_PARAMS)
    p.update(USE_DP=use_dp, LS_VARIANT=ls_variant, PHASE1_EPOCHS=phase1,
             PHASE2_EPOCHS=phase2, BATCH_SIZE=64, D_TOKEN=4, DEVICE='cpu',
             NOISE_MULTIPLIER=1.0, GROUP_NOISE_MULTIPLIER=1.0,
             PHASE1_CHECKPOINT=ckpt)
    model, accountant, group_mech = demo.train_flip(X_num, X_cat, [2, 2, 3], 0, p,
                                                    torch.device('cpu'))
    return accountant, group_mech, model


def test_phase1_checkpoint_saved_then_reused(tmp_path, capsys):
    import torch
    ckpt = str(tmp_path / 'phase1.pt')
    _, _, m1 = _tiny_training(False, 'batch_mean', phase2=0, ckpt=ckpt)
    assert os.path.exists(ckpt)
    capsys.readouterr()
    _, _, m2 = _tiny_training(False, 'batch_mean', phase2=0, ckpt=ckpt)
    assert 'Loaded Phase-1 checkpoint' in capsys.readouterr().out
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)  # no Phase-1 training happened again


def test_phase1_checkpoint_refused_under_dp(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        _tiny_training(True, 'batch_mean', ckpt=str(tmp_path / 'p.pt'))


def test_batch_mean_ls_trains_without_dp():
    _tiny_training(use_dp=False, ls_variant='batch_mean')


def test_batch_mean_ls_uses_group_mechanism_in_both_phases():
    acc, group, _ = _tiny_training(use_dp=True, ls_variant="batch_mean")
    assert group.steps == acc.steps > 0          # every step, Phase 1 too
    acc, group, _ = _tiny_training(use_dp=True, ls_variant="per_sample")
    assert 0 < group.steps < acc.steps           # Phase 2 only


def test_beta_schedule():
    fixed = {'BETA': 1e-3, 'BETA_SCHEDULE': 'fixed'}
    assert beta_at(0, fixed) == beta_at(5000, fixed) == 1e-3
    decay = {'BETA': 1e-3, 'BETA_SCHEDULE': 'decay', 'BETA_MAX': 1e-2,
             'BETA_MIN': 1e-5, 'BETA_DECAY': 0.7, 'BETA_DECAY_EVERY': 100}
    assert beta_at(0, decay) == 1e-2
    assert beta_at(99, decay) == 1e-2
    assert np.isclose(beta_at(100, decay), 7e-3)
    assert beta_at(100000, decay) == 1e-5
