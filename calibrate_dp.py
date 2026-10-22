"""Per-group noise multipliers (the paper's Eq. 13) for a target epsilon on
an Adult fold, from the same RDP accountant and step counts the training
loop uses (run_adult_flip.py --dp-group-sigmas).

Each sex group's multiplier sigma_p meets the target at that group's own
sampling rate (balanced batches of 4096: 2048 rows per group per step) over
Phase 1 + Phase 2. The batch-coupled fairness terms go through the same
noisy step (the paper's nominal accounting); diffusion is not counted, as in
the paper. --reserve keeps a share of epsilon for a separate Gaussian release
(0.1 reproduces the multipliers used in this repository's runs, which
reserved 10% for an optional rule release).

    python calibrate_dp.py --eps 3 --fold 0
"""
import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))
sys.path.insert(0, HERE)
import run_adult_flip as R  # noqa: E402
import run_flip_demo as demo  # noqa: E402
from tabsyn.vae.flip_fairness import (RDPAccountant, compose_rdp_budgets,  # noqa: E402
                                      eq13_sigma_global)

ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 505))


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
    ap.add_argument('--reserve', type=float, default=0.1,
                    help='share of epsilon kept for a separate release '
                         '(0.1 = the runs in this repository)')
    a = ap.parse_args()
    delta = demo.TRADEOFF_PARAMS['DP_DELTA']
    df = R.read_adult()
    s_idx = R.CATEGORICAL.index(R.SENSITIVE)
    per_group = 4096 // 2
    for f in a.fold:
        data = R.prepare(df, f, 3, 42)
        counts = np.bincount(data['train_cat'][:, s_idx])
        rates = per_group / counts
        steps = (a.phase1_epochs + a.phase2_epochs) * math.ceil(
            counts.min() / per_group)
        for target in a.eps:
            reserved = []
            if a.reserve > 0:
                def gauss(s_):
                    return {al: al / (2.0 * s_ ** 2) for al in ALPHAS}
                s_res = search(lambda s_: compose_rdp_budgets(
                    [gauss(s_)], delta)[0], a.reserve * target)
                reserved = [gauss(s_res)]
            sig = []
            for q_p in rates:
                def eps_p(s_, q_p=q_p):
                    acc = RDPAccountant(s_, q_p, delta)
                    acc.steps = steps
                    return compose_rdp_budgets([acc.get_rdp_curve()]
                                               + reserved, delta)[0]
                sig.append(search(eps_p, target))
            print(f'fold {f}: counts={counts.tolist()} eps={target:g} '
                  f'(delta {delta:g}): --dp-group-sigmas '
                  f'{sig[0]:.4f},{sig[1]:.4f}  (sigma_global '
                  f'{eq13_sigma_global(sig, counts):.4f})')


if __name__ == '__main__':
    main()
