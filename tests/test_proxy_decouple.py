"""Tests for tabsyn.vae.proxy_decouple (automatic proxy detection,
merge/expand with tau, and the impossible-combination / member-dependence
metrics). Standalone: no dependency on flip_fairness.

Run:  python -m pytest tests/test_proxy_decouple.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.proxy_decouple import (detect_proxy_classes, merge_proxies,
                                       fit_member_tables, expand_merged,
                                       proxy_member_labels,
                                       impossible_combination_rate,
                                       proxy_member_dependence)


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
# Detection / merge / expand
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


def test_merge_and_expand_tau0_is_consistent():
    cat = toy_table()
    cls = detect_proxy_classes(cat, s_idx=0)
    tb = fit_member_tables(cat, 0, cls)
    merged = merge_proxies(cat, cls)
    assert set(np.unique(merged[:, 1])) == {0, 2}
    out = expand_merged(merged, cat[:, 0], cls, tb, tau=0.0)
    # tau=0: member follows S -> reproduces the real column almost exactly
    assert (out[:, 1] == cat[:, 1]).mean() > 0.99
    assert impossible_combination_rate(out, cat, 0)['any'] < 0.01


def test_expand_tau1_decouples_member_from_features():
    cat = toy_table()
    cls = detect_proxy_classes(cat, s_idx=0)
    tb = fit_member_tables(cat, 0, cls)
    out = expand_merged(merge_proxies(cat, cls), cat[:, 0], cls, tb, tau=1.0)
    # rows outside the class are untouched
    other = cat[:, 1] == 2
    assert (out[other, 1] == 2).all()
    # member shares within the class match the real marginal
    in_cls = out[:, 1] != 2
    real_share = (cat[cat[:, 1] != 2, 1] == 0).mean()
    assert abs((out[in_cls, 1] == 0).mean() - real_share) < 0.04
    # 'male wives' appear (about half the spouse rows)
    assert impossible_combination_rate(out, cat, 0)['any'] > 0.15
    # and the member is no longer predictable from the other feature
    dep = proxy_member_dependence(out, cls, exclude=(0,))[0]
    dep_real = proxy_member_dependence(cat, cls, exclude=(0,))[0]
    assert dep_real > 0.7 and dep < 0.56


def test_member_labels():
    cat = toy_table(200)
    cls = detect_proxy_classes(cat, s_idx=0, min_support=0.0)
    (lab, mask), = proxy_member_labels(cat, cls)
    assert (mask == (cat[:, 1] != 2)).all()
    assert (lab[cat[:, 1] == 1] == 1).all() and (lab[cat[:, 1] == 0] == 0).all()
