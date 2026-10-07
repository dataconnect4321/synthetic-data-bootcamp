"""Tests for tabsyn.vae.proxy_decouple (automatic proxy detection,
masking, the scrambling repair baseline, and the
impossible-combination metric). Standalone: no dependency on flip_fairness.

Run:  python -m pytest tests/test_proxy_decouple.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.proxy_decouple import (detect_proxy_classes,
                                       impossible_combination_rate)


def toy_table(n=4000, seed=0):
    """col0 = S (0/1), col1 = relationship-like proxy:
    0=husband (S=1), 1=wife (S=0), 2=other (mixed); col2 = a feature that
    depends on the proxy member (husband -> 1 more often)."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    spouse = rng.random(n) < 0.5
    rel = np.where(spouse, np.where(s == 1, 0, 1), 2)
    feat = np.where(rel == 0, rng.random(n) < 0.8,
                    rng.random(n) < 0.2).astype(int)
    return np.stack([s, rel, feat], axis=1)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------

def test_detect_finds_cross_group_class_only():
    cat = toy_table()
    cls = detect_proxy_classes(cat, s_idx=0)
    assert len(cls) == 1
    assert cls[0]['col'] == 1 and cls[0]['members'] == [0, 1]
    assert cls[0]['group_of'] == {0: 1, 1: 0}


def test_detect_skips_same_group_and_excluded():
    cat = toy_table()
    # an excluded column is never a proxy
    assert detect_proxy_classes(cat, s_idx=0, exclude=(1,)) == []
    # only ONE flagged category (cat 1 is pure S=0; cat 0 is mixed): nothing
    # to decouple, so no class
    s = cat[:, 0]
    one_sided = np.where(s == 1, 0,
                         np.random.default_rng(1).integers(0, 2, len(s)))
    cat2 = np.stack([s, one_sided], axis=1)
    cls = detect_proxy_classes(cat2, s_idx=0)
    assert cls == []


# --------------------------------------------------------------------------
# Pre-training masking (--mask-proxies)
# --------------------------------------------------------------------------

def test_mask_merges_class_into_new_code_only():
    from tabsyn.vae.proxy_decouple import mask_proxy_classes
    cat = toy_table()
    cls = detect_proxy_classes(cat, s_idx=0)
    out = mask_proxy_classes(cat, cls, [3])
    spouse = cat[:, 1] != 2
    assert (out[spouse, 1] == 3).all()            # husband, wife -> MASK
    assert (out[~spouse, 1] == cat[~spouse, 1]).all()
    assert (out[:, [0, 2]] == cat[:, [0, 2]]).all()


def test_mask_removes_deterministic_proxy_without_impossible_rows():
    from tabsyn.vae.proxy_decouple import (mask_proxy_classes,
                                           impossible_combination_rate)
    cat = toy_table()
    cls = detect_proxy_classes(cat, s_idx=0)
    out = mask_proxy_classes(cat, cls, [3])
    # nothing left to detect, and no (S, category) pair absent from the real
    # data appears (a re-draw would create male wives)
    assert detect_proxy_classes(out, s_idx=0) == []
    assert impossible_combination_rate(out, cat, s_idx=0)['any'] == 0.0


# --------------------------------------------------------------------------
# Scrambling repair (--repair-tau)
# --------------------------------------------------------------------------

def _skewed(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    occ = np.where(s == 1, rng.choice(3, n, p=[0.7, 0.2, 0.1]),
                   rng.choice(3, n, p=[0.1, 0.3, 0.6]))
    other = rng.integers(0, 4, n)
    num = (rng.normal(size=(n, 1)) + 1.5 * s[:, None]).astype(np.float32)
    return num, np.stack([s, occ, other], axis=1)


def test_repair_tau_zero_is_identity():
    from tabsyn.vae.proxy_decouple import repair_toward_independence
    num, cat = _skewed()
    n2, c2 = repair_toward_independence(num, cat, 0, tau=0.0)
    assert (c2 == cat).all() and np.allclose(n2, num)


def test_repair_full_removes_dependence_and_keeps_shares():
    from tabsyn.vae.proxy_decouple import repair_toward_independence
    num, cat = _skewed()
    n2, c2 = repair_toward_independence(num, cat, 0, tau=1.0)
    s = cat[:, 0]
    # pooled shares preserved exactly, S untouched
    assert (np.bincount(c2[:, 1], minlength=3)
            == np.bincount(cat[:, 1], minlength=3)).all()
    assert (c2[:, 0] == s).all()
    # per-group shares now equal (up to rounding)
    p1 = np.bincount(c2[s == 1, 1], minlength=3) / (s == 1).sum()
    p0 = np.bincount(c2[s == 0, 1], minlength=3) / (s == 0).sum()
    assert np.abs(p1 - p0).max() < 0.01
    # numeric group means equalized
    assert abs(n2[s == 1, 0].mean() - n2[s == 0, 0].mean()) < 0.05


def test_repair_partial_tau_is_in_between():
    from tabsyn.vae.proxy_decouple import repair_toward_independence
    num, cat = _skewed()
    s = cat[:, 0]
    def gap(c):
        return abs((c[s == 1, 1] == 0).mean() - (c[s == 0, 1] == 0).mean())
    g0 = gap(cat)
    g5 = gap(repair_toward_independence(num, cat, 0, tau=0.5)[1])
    assert 0.4 * g0 < g5 < 0.6 * g0
