"""Find the noise multipliers that give a target composed epsilon for an
Adult run (run_adult_flip.py with --dp-noise), from the same accountants and
step counts the training loop uses.

The three mechanisms are scaled together: DP-SGD on the VAE (sigma),
the group-level mechanism for the batch-coupled terms (GROUP_RATIO * sigma)
and the one release of numeric clip bounds at decoding (STATS_RATIO * sigma).
Diffusion is not private (the paper's setting), so it is not counted.

    python calibrate_dp.py --eps 3 2 1 [--fold 0]
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))
sys.path.insert(0, HERE)
import run_adult_flip as R  # noqa: E402
import run_flip_demo as demo  # noqa: E402
from tabsyn.vae.flip_fairness import (RDPAccountant, GroupLevelDPMechanism,  # noqa: E402
                                      dp_release_mean_std, compose_rdp_budgets,
                                      eq13_sigma_global)

GROUP_RATIO, STATS_RATIO = 4.0, 10.0
RULES_SHARE = 0.1   # share of the target epsilon for the DP rejection rules


def composed_eps(sigma, q, steps, p, coupled='group'):
    acc = RDPAccountant(sigma, q, p['DP_DELTA'])
    acc.steps = steps
    grp = GroupLevelDPMechanism(p['MAX_GROUP_GRAD_NORM'], GROUP_RATIO * sigma,
                                sample_rate=q)
    grp.steps = steps          # batch-mean L_S runs it in both phases
    _, _, stats = dp_release_mean_std(torch.zeros(2, 2), STATS_RATIO * sigma,
                                      p['STATS_MAX_NORM'], 2)
    curves = [acc.get_rdp_curve(), stats]
    if coupled == 'group':   # 'paper': coupled terms ride on DP-SGD
        curves.append(grp.get_rdp_curve())
    eps, _ = compose_rdp_budgets(curves, p['DP_DELTA'])
    return eps


def search(eps_fn, target, lo=0.3, hi=2000.0):
    """Smallest noise multiplier with eps_fn(noise) <= target."""
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if eps_fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, nargs='+', default=[3, 2, 1])
    ap.add_argument('--fold', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--phase1-epochs', type=int, default=3000)
    ap.add_argument('--phase2-epochs', type=int, default=100)
    ap.add_argument('--coupled', choices=['group', 'paper'], default='group')
    ap.add_argument('--phase2-batch', type=int, default=None,
                    help='with --eq13: noise for a Phase 2 run at this batch '
                         'size on a bank trained at batch 4096 (Phase 1 '
                         'accounted at the bank noise, Phase 2 separately)')
    ap.add_argument('--eq13', action='store_true',
                    help="per-group noise (paper Eq. 13): each sex group's "
                         'sigma_p meets the target at its own sampling rate; '
                         'paper-style coupling only')
    a = ap.parse_args()
    p = demo.TRADEOFF_PARAMS
    df = R.read_adult()
    s_idx = R.CATEGORICAL.index(R.SENSITIVE)
    per_group = 4096 // 2
    for f in a.fold:
        data = R.prepare(df, f, 3, 42)
        m = int(np.bincount(data['train_cat'][:, s_idx]).min())
        q = per_group / m
        spe = math.ceil(m / per_group)
        steps = (a.phase1_epochs + a.phase2_epochs) * spe
        if a.eq13:
            # epsilon covers the VAE (per-group sigma_p, Eq. 13) plus the
            # one release of 2-way counts for DP rejection rules, which gets
            # RULES_SHARE of the target on its own (no clip-bounds release)
            counts = np.bincount(data['train_cat'][:, s_idx])
            rates = per_group / counts
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505))
            for target in a.eps:
                def rules_curve(s_):
                    return {al: al / (2.0 * s_ ** 2) for al in alphas}
                rules_sigma = search(lambda s_: compose_rdp_budgets(
                    [rules_curve(s_)], p['DP_DELTA'])[0], RULES_SHARE * target)
                sig = []
                for q_p in rates:
                    def eps_p(s_, q_p=q_p):
                        acc = RDPAccountant(s_, q_p, p['DP_DELTA'])
                        acc.steps = steps
                        return compose_rdp_budgets(
                            [acc.get_rdp_curve(), rules_curve(rules_sigma)],
                            p['DP_DELTA'])[0]
                    sig.append(search(eps_p, target))
                if a.phase2_batch:
                    # bank: Phase 1 only, at the bank noise; Phase 2 at the
                    # smaller batch gets the rest of the budget per group
                    t1 = a.phase1_epochs * spe
                    m2 = a.phase2_batch // 2
                    t2 = a.phase2_epochs * math.ceil(counts.min() / m2)
                    sig2 = []
                    for q_p, s_bank in zip(rates, sig):
                        bank = RDPAccountant(s_bank, q_p, p['DP_DELTA'])
                        bank.steps = t1
                        bank_c = bank.get_rdp_curve()
                        q2 = m2 / counts[len(sig2)]
                        def eps2(s_, q2=q2, bank_c=bank_c):
                            acc = RDPAccountant(s_, q2, p['DP_DELTA'])
                            acc.steps = t2
                            return compose_rdp_budgets(
                                [bank_c, acc.get_rdp_curve(),
                                 rules_curve(rules_sigma)], p['DP_DELTA'])[0]
                        sig2.append(search(eps2, target))
                    print(f'fold {f} eps={target:g} phase-2 batch '
                          f'{a.phase2_batch} ({t2} steps): bank '
                          f'--dp-group-sigmas {sig[0]:.4f},{sig[1]:.4f}; '
                          f'phase 2 --dp-group-sigmas '
                          f'{sig2[0]:.4f},{sig2[1]:.4f}')
                    continue
                print(f'fold {f}: counts={counts.tolist()} eps={target:g}: '
                      f'--dp-group-sigmas {sig[0]:.4f},{sig[1]:.4f} '
                      f'--dp-rules-noise {rules_sigma:.4f}  (sigma_global '
                      f'{eq13_sigma_global(sig, counts):.4f})')
            continue
        for target in a.eps:
            lo, hi = 0.3, 500.0
            for _ in range(60):
                mid = math.sqrt(lo * hi)
                if composed_eps(mid, q, steps, p, a.coupled) > target:
                    lo = mid
                else:
                    hi = mid
            print(f'fold {f}: m={m} q={q:.4f} steps={steps}  eps={target:g}: '
                  f'--dp-noise {hi:.4f} --dp-group-noise {GROUP_RATIO * hi:.4f} '
                  f'--dp-stats-noise {STATS_RATIO * hi:.4f}  '
                  f'(check eps={composed_eps(hi, q, steps, p, a.coupled):.3f}, '
                  f'delta={p["DP_DELTA"]:g})')


if __name__ == '__main__':
    main()
