"""Consistency pass: strong-rule selection, reject, edit.

Run:  python -m pytest tests/test_consistency.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.paper_metrics import fit_validity_rules
from tabsyn.vae.consistency import (strong_rules, violations, reject_invalid,
                                    edit_invalid)


def _real(n=6000, seed=0):
    """col0 = S; col1 marital (0 never, 1 married, 2 divorced);
    col2 relationship (0 spouse, 1 not-in-family, 2 own-child): spouse only
    when married; col3 rare category pair that is merely uncommon."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    marital = rng.choice(3, n, p=[0.4, 0.45, 0.15])
    rel = np.where(marital == 1, 0, rng.choice([1, 2], n))
    other = rng.integers(0, 3, n)
    num = rng.normal(size=(n, 1))
    return num, np.stack([s, marital, rel, other], 1)


def test_strong_rules_drop_weak_pairs_and_protected():
    num, cat = _real()
    rules = fit_validity_rules(num, cat, ['x'], ['s', 'm', 'r', 'o'],
                               min_support=100, pair_expected=5)
    strong = strong_rules(rules, cat, min_expected=100, exclude=(0,))
    for kind in ('pair', 'fd'):
        for a, _, b, _, _ in strong[kind]:
            assert 0 not in (a, b)
    # never-married & spouse (expected ~ 0.4*0.45*n) is enforced
    assert any(a == 1 and va == 0 and b == 2 and vb == 0
               for a, va, b, vb, _ in strong['pair'])
    assert not violations(strong, cat).any()


def test_reject_and_edit_restore_consistency():
    num, cat = _real()
    rules = fit_validity_rules(num, cat, ['x'], ['s', 'm', 'r', 'o'],
                               min_support=100)
    strong = strong_rules(rules, cat, exclude=(0,))
    syn = cat.copy()
    bad_rows = np.arange(0, 600)
    syn[bad_rows, 1] = 0          # never married ...
    syn[bad_rows, 2] = 0          # ... but spouse
    assert violations(strong, syn).sum() >= 600
    _, rcat, kept = reject_invalid(strong, num, syn)
    assert not violations(strong, rcat).any() and kept.sum() <= len(syn) - 600
    ecat, n_edit, n_left = edit_invalid(strong, num, syn, num, cat, 0,
                                        [2, 3, 3, 3])
    assert n_left == 0 and n_edit >= 600
    assert not violations(strong, ecat).any()
    assert (ecat[:, 0] == syn[:, 0]).all()          # S never edited
    # only one field changed per fixed row
    assert ((ecat[bad_rows] != syn[bad_rows]).sum(1) == 1).all()
