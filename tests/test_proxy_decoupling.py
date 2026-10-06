"""Tests for proxy decoupling: the post-hoc tau version and the in-model
lambda version (HSIC / adversary on the non-proxy tokens).

Run:  python -m pytest tests/test_proxy_decoupling.py -v
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tabsyn-main', 'tabsyn-main'))

from tabsyn.vae.flip_fairness import (cat_token_index, drop_tokens,
                                      proxy_sublabel, hsic_normalized,
                                      proxy_independence_loss,
                                      proxy_adversary_loss,
                                      proxy_adversary_input_dim,
                                      AdversaryHead,
                                      multi_stage_disentanglement_loss)
from tabsyn.vae.temper import joint_tv, decouple_curve


D_NUM, CATS, D_TOK = 2, (2, 3, 4), 4      # cat 0 = S, cat 2 = proxy
S_IDX, P_IDX, POS, NEG = 0, 2, 0, 1


def _stages(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    T = 1 + D_NUM + len(CATS)
    return {
        'mu_z': torch.randn(n, T, D_TOK, generator=g),
        'h': torch.randn(n, T - 1, D_TOK, generator=g),
        'recon_x_num': torch.randn(n, D_NUM, generator=g),
        'recon_x_cat': [torch.randn(n, c, generator=g) for c in CATS],
        'std_z': torch.zeros(n, T, D_TOK),
    }


# --------------------------------------------------------------------------
# token bookkeeping
# --------------------------------------------------------------------------

def test_token_index_layout():
    # mu_z = [CLS, num0, num1, cat0, cat1, cat2]; h drops CLS
    assert cat_token_index(D_NUM, 2, 'mu') == 5
    assert cat_token_index(D_NUM, 2, 'h') == 4


def test_drop_tokens():
    rep = torch.arange(6).float().view(1, 6, 1)
    out = drop_tokens(rep, [3, 5])
    assert out.flatten().tolist() == [0, 1, 2, 4]


def test_proxy_sublabel():
    cat = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 3]])
    mask, lab = proxy_sublabel(cat, P_IDX, POS, NEG)
    assert mask.tolist() == [True, True, False]
    assert lab[mask].tolist() == [1, 0]


# --------------------------------------------------------------------------
# HSIC
# --------------------------------------------------------------------------

def test_hsic_high_when_dependent_low_when_independent():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (200,))
    X_dep = torch.randn(200, 5) + 3.0 * y[:, None].float()
    X_ind = torch.randn(200, 5)
    assert hsic_normalized(X_dep, y) > 5 * hsic_normalized(X_ind, y)


def test_hsic_zero_for_constant_label():
    assert hsic_normalized(torch.randn(10, 3),
                           torch.zeros(10, dtype=torch.long)).item() == 0.0


def test_proxy_loss_ignores_proxy_and_s_tokens():
    """Encoding the label ONLY in the proxy and S tokens must not raise the
    loss: those tokens are excluded from the 'remaining' representation."""
    st = _stages()
    cat = torch.stack([torch.randint(0, c, (64,)) for c in CATS], 1)
    cat[:, P_IDX] = torch.randint(0, 2, (64,))   # every row holds a label
    mask, lab = proxy_sublabel(cat, P_IDX, POS, NEG)
    base = proxy_independence_loss(st, lab, mask, P_IDX, S_IDX)
    st2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    for key, stage in (('mu_z', 'mu'), ('h', 'h')):
        for j in (P_IDX, S_IDX):
            st2[key][:, cat_token_index(D_NUM, j, stage)] += 10 * lab[:, None]
    same = proxy_independence_loss(st2, lab, mask, P_IDX, S_IDX)
    assert torch.isclose(base, same, atol=1e-6)
    # ...but leaking it into a numeric token does
    st3 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in st.items()}
    st3['mu_z'][:, 1] += 10 * lab[:, None]
    st3['h'][:, 0] += 10 * lab[:, None]
    assert proxy_independence_loss(st3, lab, mask, P_IDX, S_IDX) > base + 0.1


def test_proxy_loss_gradient_reduces_dependence():
    """A few gradient steps on the HSIC term shrink the label's footprint in
    the remaining latent."""
    torch.manual_seed(0)
    st = _stages(n=128)
    lab = torch.randint(0, 2, (128,))
    mask = torch.ones(128, dtype=torch.bool)
    mu = st['mu_z'].clone()
    mu[:, 1] += 3 * lab[:, None]
    mu.requires_grad_(True)
    opt = torch.optim.SGD([mu], lr=50.0)
    start = None
    for _ in range(30):
        opt.zero_grad()
        loss = proxy_independence_loss(dict(st, mu_z=mu), lab, mask,
                                       P_IDX, S_IDX)
        start = loss.item() if start is None else start
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * start


# --------------------------------------------------------------------------
# adversary variant
# --------------------------------------------------------------------------

def test_proxy_adversary_masked_and_per_record():
    st = _stages()
    T = st['mu_z'].shape[1]
    head = AdversaryHead(proxy_adversary_input_dim(T, D_TOK), 2)
    lab = torch.randint(0, 2, (64,))
    mask = torch.zeros(64, dtype=torch.bool)
    mask[:10] = True
    total = proxy_adversary_loss(head, st, lab, mask, P_IDX, S_IDX)
    # off-mask rows contribute nothing
    lab2 = lab.clone(); lab2[20] = 1 - lab2[20]
    assert torch.isclose(total, proxy_adversary_loss(head, st, lab2, mask,
                                                     P_IDX, S_IDX))
    lab3 = lab.clone(); lab3[0] = 1 - lab3[0]
    assert not torch.isclose(total, proxy_adversary_loss(head, st, lab3, mask,
                                                         P_IDX, S_IDX))


# --------------------------------------------------------------------------
# CKA^T / MMD on the non-proxy representation
# --------------------------------------------------------------------------

def test_disentanglement_exclude_cat_ignores_proxy():
    st = _stages(n=80)
    group = torch.randint(0, 2, (80,))
    base = multi_stage_disentanglement_loss(st, group, s_idx=S_IDX,
                                            exclude_cat=(P_IDX,))
    st2 = {k: (v.clone() if torch.is_tensor(v) else list(v))
           for k, v in st.items()}
    st2['mu_z'][:, cat_token_index(D_NUM, P_IDX, 'mu')] += 5 * group[:, None]
    st2['h'][:, cat_token_index(D_NUM, P_IDX, 'h')] += 5 * group[:, None]
    st2['recon_x_cat'][P_IDX] = st2['recon_x_cat'][P_IDX] + 5 * group[:, None]
    excl = multi_stage_disentanglement_loss(st2, group, s_idx=S_IDX,
                                            exclude_cat=(P_IDX,))
    assert torch.isclose(base, excl, atol=1e-5)
    # without the exclusion the same change is visible
    incl0 = multi_stage_disentanglement_loss(st, group, s_idx=S_IDX)
    incl = multi_stage_disentanglement_loss(st2, group, s_idx=S_IDX)
    assert not torch.isclose(incl0, incl, atol=1e-4)


# --------------------------------------------------------------------------
# post-hoc decoupling
# --------------------------------------------------------------------------

def _proxy_table(n, seed):
    """S in {0,1}; proxy column: 0 (pos, 'Husband') iff S=1 among spouse
    rows, 1 (neg) iff S=0; code 2 = non-spouse. One numeric noise column."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    spouse = rng.random(n) < 0.5
    proxy = np.where(spouse, np.where(s == 1, POS, NEG), 2)
    other = rng.integers(0, 3, n)
    cat = np.stack([s, other, proxy], axis=1).astype(np.int64)
    num = rng.normal(size=(n, 1)).astype(np.float32)
    return num, cat


def test_joint_tv():
    _, a = _proxy_table(500, 0)
    assert joint_tv(a, a, S_IDX, P_IDX) == 0.0
    b = a.copy(); b[:, P_IDX] = 2
    assert joint_tv(a, b, S_IDX, P_IDX) > 0.4


def test_decouple_curve_ignores_generated_s_column():
    """FLIP's generated S column is invented by design: the curve must score
    against the TRUE sex and never read the generated column."""
    ref_n, ref_c = _proxy_table(1500, 1)
    syn_n, syn_c = _proxy_table(1500, 2)
    true_s = syn_c[:, S_IDX].copy()
    a, b = syn_c.copy(), syn_c.copy()
    a[:, S_IDX] = np.random.default_rng(8).integers(0, 2, len(a))
    b[:, S_IDX] = np.random.default_rng(9).integers(0, 2, len(b))
    ra = decouple_curve(ref_n, ref_c, syn_n, a, true_s, S_IDX, P_IDX,
                        POS, NEG, [1.0, 0.5, 0.0], seed=0)
    rb = decouple_curve(ref_n, ref_c, syn_n, b, true_s, S_IDX, P_IDX,
                        POS, NEG, [1.0, 0.5, 0.0], seed=0)
    assert ra == rb


def test_decouple_curve_slope_when_other_columns_carry_label():
    rng = np.random.default_rng(3)

    def table(n):
        s = rng.integers(0, 2, n)
        spouse = rng.random(n) < 0.6
        proxy = np.where(spouse, np.where(s == 1, POS, NEG), 2)
        # 'other' copies the sex-specific label 90% of the time (e.g. an
        # occupation proxy), so P(label | non-sex columns) is informative
        other = np.where(rng.random(n) < 0.9, s, 1 - s)
        cat = np.stack([s, other, proxy], 1).astype(np.int64)
        return rng.normal(size=(n, 1)).astype(np.float32), cat

    ref_n, ref_c = table(3000)
    syn_n, syn_c = table(3000)
    true_s = syn_c[:, S_IDX].copy()
    syn_c[:, S_IDX] = rng.integers(0, 2, len(syn_c))   # invented S
    r1, r0 = decouple_curve(ref_n, ref_c, syn_n, syn_c, true_s, S_IDX, P_IDX,
                            POS, NEG, [1.0, 0.0], real_cat=ref_c, seed=0)
    # tau=0 makes the label independent: leakage drops, joint fidelity drops
    assert r0['leakage'] < r1['leakage'] - 0.05
    assert r0['joint_tv'] > r1['joint_tv'] + 0.05
    assert r0['flipped'] > r1['flipped']
