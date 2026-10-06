"""Tests for the tempered-sampling baseline and the in-model adversary.

Run:  python -m pytest tests/test_temper_adversary.py -v
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.temper import temper_protected
from tabsyn.vae.flip_fairness import (gradient_reversal, AdversaryHead,
                                      adversary_loss,
                                      marginal_matching_penalty)


# --------------------------------------------------------------------------
# temper_protected
# --------------------------------------------------------------------------

def test_temper_tau_one_is_identity():
    P = np.array([[0.7, 0.3], [0.2, 0.8]])
    pi = np.array([0.6, 0.4])
    out = temper_protected(P, pi, tau=1.0)
    assert np.allclose(out, P, atol=1e-9)


def test_temper_tau_zero_is_reference_shares():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    pi = np.array([0.25, 0.75])
    out = temper_protected(P, pi, tau=0.0)
    assert np.allclose(out, np.tile(pi, (2, 1)), atol=1e-9)


def test_temper_interpolates_monotonically():
    """As tau drops, the tempered distribution moves from P toward pi."""
    P = np.array([[0.9, 0.1]])
    pi = np.array([0.5, 0.5])
    # distance from pi of the tempered first-group probability
    d_pi = [abs(temper_protected(P, pi, tau)[0, 0] - 0.5)
            for tau in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert all(d_pi[i] >= d_pi[i + 1] - 1e-9 for i in range(len(d_pi) - 1))
    # and from P
    d_p = [abs(temper_protected(P, pi, tau)[0, 0] - 0.9)
           for tau in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert all(d_p[i] <= d_p[i + 1] + 1e-9 for i in range(len(d_p) - 1))


def test_temper_rows_normalized():
    P = np.random.default_rng(0).dirichlet([1, 1, 1], size=50)
    pi = np.array([0.2, 0.3, 0.5])
    for tau in (1.0, 0.5, 0.1, 0.0):
        assert np.allclose(temper_protected(P, pi, tau).sum(axis=1), 1.0)


# --------------------------------------------------------------------------
# gradient reversal
# --------------------------------------------------------------------------

def test_grl_flips_gradient_sign():
    x = torch.randn(8, 4, requires_grad=True)
    y = gradient_reversal(x, lambd=2.0)
    y.sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


def test_grl_identity_forward():
    x = torch.randn(8, 4)
    assert torch.allclose(gradient_reversal(x, 1.0), x)


# --------------------------------------------------------------------------
# adversary head / loss
# --------------------------------------------------------------------------

def _stages(n=32, d_num=4, categories=(3, 2, 5)):
    return {
        'mu_z': torch.randn(n, 5, 8),
        'h': torch.randn(n, 5, 8),
        'recon_x_num': torch.randn(n, d_num),
        'recon_x_cat': [torch.randn(n, c) for c in categories],
        'std_z': torch.randn(n, 5, 8).abs(),
    }


def test_adversary_loss_is_per_record():
    """The adversary CE is a batch MEAN of per-row terms (scale-invariant,
    matching the other Phase-2 losses) and still per-record decomposable:
    changing one row's group changes only that row's term."""
    stages = _stages()
    group = torch.randint(0, 2, (32,))
    head = AdversaryHead(4 + 3 + 5, 2)
    total = adversary_loss(head, stages, group, s_idx=1)
    # recompute with row 0's group changed: only row 0's term may differ
    group2 = group.clone(); group2[0] = 1 - group2[0]
    total2 = adversary_loss(head, stages, group2, s_idx=1)
    assert not torch.isclose(total, total2)
    # per-row decomposition: mean of single-row losses equals the total
    per_row = torch.nn.functional.cross_entropy(
        head(torch.cat([stages['recon_x_num']] +
                       [c for i, c in enumerate(stages['recon_x_cat']) if i != 1],
                      dim=1)), group, reduction='none')
    assert torch.isclose(total, per_row.mean(), atol=1e-5)


def test_adversary_excludes_protected_column():
    """The adversary's input width must exclude the protected column."""
    stages = _stages()
    head = AdversaryHead(4 + 3 + 5, 2)  # d_num + cat0 + cat2 (cat1 = S)
    group = torch.randint(0, 2, (32,))
    adversary_loss(head, stages, group, s_idx=1)  # must not raise


def test_adversary_gradient_reversal_direction():
    """With the GRL, minimizing the loss moves the features to MAXIMIZE the
    adversary's error: the gradient on the features is the NEGATIVE of the
    plain-CE gradient."""
    torch.manual_seed(0)
    stages = _stages()
    group = torch.randint(0, 2, (32,))
    head = AdversaryHead(4 + 3 + 5, 2)
    for p in head.parameters():
        p.requires_grad_(False)
    x = torch.cat([stages['recon_x_num']] +
                  [c for i, c in enumerate(stages['recon_x_cat']) if i != 1],
                  dim=1).detach().requires_grad_(True)
    logits = head(x, lambd=1.0)
    torch.nn.functional.cross_entropy(logits, group, reduction='sum').backward()
    g_rev = x.grad.clone()
    x2 = x.detach().requires_grad_(True)
    logits2 = head.net(x2)
    torch.nn.functional.cross_entropy(logits2, group, reduction='sum').backward()
    assert torch.allclose(g_rev, -x2.grad, atol=1e-6)


# --------------------------------------------------------------------------
# marginal matching
# --------------------------------------------------------------------------

def test_marginal_penalty_zero_on_identical_batches():
    stages = _stages(n=64)
    X_num = stages['recon_x_num'].clone()
    X_cat = torch.stack([c.argmax(-1) for c in stages['recon_x_cat']], dim=1)
    # make the categorical logits a one-hot of the argmax so the softmax
    # mean equals the true share
    onehot = [torch.nn.functional.one_hot(c.argmax(-1), c.shape[1]).float()
              for c in stages['recon_x_cat']]
    stages2 = dict(stages, recon_x_num=X_num, recon_x_cat=onehot)
    pen = marginal_matching_penalty(stages2, X_num, X_cat, s_idx=1)
    # numeric part is exactly 0 (identical quantiles); the categorical part
    # is ~0 up to one-hot rounding, so the total is small but not exactly 0
    assert pen.item() < 0.05


def test_marginal_penalty_positive_for_shifted_batch():
    stages = _stages(n=64)
    X_num = stages['recon_x_num'] + 1.0  # shifted numerics
    X_cat = torch.stack([c.argmax(-1) for c in stages['recon_x_cat']], dim=1)
    pen = marginal_matching_penalty(stages, X_num, X_cat, s_idx=1)
    assert pen.item() > 0.05
