"""Tests for the shared gradient-reversal machinery and the marginal penalty.

Run:  python -m pytest tests/test_temper_adversary.py -v
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.flip_fairness import (gradient_reversal, AdversaryHead,
                                      marginal_matching_penalty)


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


def _stages(n=32, d_num=4, categories=(3, 2, 5)):
    return {
        'mu_z': torch.randn(n, 5, 8),
        'h': torch.randn(n, 5, 8),
        'recon_x_num': torch.randn(n, d_num),
        'recon_x_cat': [torch.randn(n, c) for c in categories],
        'std_z': torch.randn(n, 5, 8).abs(),
    }


# --------------------------------------------------------------------------
# marginal matching
# --------------------------------------------------------------------------

def test_marginal_penalty_zero_on_identical_batches():
    torch.manual_seed(0)
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
    assert pen.item() < 0.07


def test_marginal_penalty_positive_for_shifted_batch():
    torch.manual_seed(0)
    stages = _stages(n=64)
    X_num = stages['recon_x_num'] + 1.0  # shifted numerics
    X_cat = torch.stack([c.argmax(-1) for c in stages['recon_x_cat']], dim=1)
    pen = marginal_matching_penalty(stages, X_num, X_cat, s_idx=1)
    assert pen.item() > 0.05
