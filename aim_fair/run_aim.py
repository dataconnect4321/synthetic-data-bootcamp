"""Generate Adult synthetic data with the AIM hybrid for one CV fold.

Uses the aim_fair/.venv environment (mbi + JAX). Reads the fold written by
export_folds.py, discretizes with public bin edges, runs AIM over all 2-way
marginals at (epsilon, delta) and saves the noisy measurements, then writes
one synthetic table per refit (--refits seeds) from them: proxy masking,
repair (tau), consistency rules and pair redraw (fair_aim.FairAIM.synthesize).
Refits and variants are post-processing of the same private measurements.

    aim_fair/.venv/Scripts/python aim_fair/run_aim.py --fold 0 --epsilon 3
    # plain AIM from the same measurements:
    aim_fair/.venv/Scripts/python aim_fair/run_aim.py --fold 0 --epsilon 3 \\
        --load aim_fair/out/aim_e3_f0_measurements.pkl --tau 0 --no-mask \\
        --no-fix --tag aimplain_e3_f0

Score a table with the main environment (merged levels need --mask-proxies):

    python run_adult_flip.py --score-csv aim_fair/out/TAG_r0_synthetic.csv \\
        --fold F --score-collapsed --mask-proxies --no-save-run --tag TAG_r0
"""
import argparse
import itertools
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import adult_domain as AD
from fair_aim import FairAIM
from mbi import Dataset, Domain, LinearMeasurement

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--epsilon', type=float, default=3.0)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--refits', type=int, nargs='+', default=[0, 1, 2],
                    help='seeds of the refits written from the measurements')
    ap.add_argument('--tau', type=float, default=1.0,
                    help='repair amount (0 = no repair)')
    ap.add_argument('--no-mask', action='store_true',
                    help='keep sex-specific category levels')
    ap.add_argument('--no-fix', action='store_true',
                    help='no consistency rules / pair redraw')
    ap.add_argument('--load', default=None,
                    help='reuse saved noisy measurements instead of running '
                         'AIM (post-processing only)')
    ap.add_argument('--mask-label', default='MASK')
    ap.add_argument('--max-model-size', type=float, default=80)
    ap.add_argument('--max-iters', type=int, default=1000)
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0, help='AIM measuring seed')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    tag = a.tag or f'aim_e{a.epsilon:g}_f{a.fold}'
    out_dir = os.path.join(HERE, 'out')
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    train = pd.read_csv(os.path.join(HERE, 'data', f'fold{a.fold}_train.csv'))
    # public schema: the category labels (sorted, as in run_adult_flip.py)
    categories = {c: sorted(train[c].astype(str).unique())
                  for c in AD.CATEGORICAL}
    sizes = AD.domain_sizes(categories)
    domain = Domain(list(sizes), list(sizes.values()))
    mech = FairAIM(a.epsilon, a.delta, prng=np.random.RandomState(a.seed),
                   max_model_size=a.max_model_size, max_iters=a.max_iters,
                   rounds=a.rounds)
    if a.load:
        with open(a.load, 'rb') as f:
            mech.measurements = [LinearMeasurement(np.asarray(y), tuple(cl),
                                                   stddev=sd)
                                 for cl, y, sd in pickle.load(f)]
        mech.potentials = None
    else:
        data = Dataset(AD.encode(train, categories), domain)
        workload = [(cl, 1.0) for cl in itertools.combinations(
            domain.attributes, 2) if domain.size(cl) <= 10000]
        print(f'[{tag}] {len(train)} rows, {len(workload)} workload '
              f'marginals, rho={mech.rho:.4f}', flush=True)
        mech.run(data, workload)
        with open(os.path.join(out_dir, f'{tag}_measurements.pkl'), 'wb') as f:
            pickle.dump([(tuple(m.clique), np.asarray(m.noisy_measurement),
                          float(m.stddev)) for m in mech.measurements], f)

    repair_cols = [c for c in domain.attributes if c != 'sex']
    info = {}
    for r in a.refits:
        mech.prng = np.random.RandomState(1000 + r)
        np.random.seed(r)        # mbi's sampler draws from numpy's global RNG
        _, codes = mech.synthesize(domain, len(train), repair_cols,
                                   ordinal=AD.NUMERIC, tau=a.tau,
                                   mask=not a.no_mask, fix=not a.no_fix,
                                   exclude=AD.NUMERIC)
        li = mech.last_info
        cats = dict(categories)
        for c, members in li['merges']:   # merged level = last code
            cats[c] = [v for k, v in enumerate(categories[c])
                       if k not in members] + [a.mask_label]
        AD.decode(codes, cats, seed=r).to_csv(
            os.path.join(out_dir, f'{tag}_r{r}_synthetic.csv'), index=False)
        info[f'r{r}'] = {
            'merged': {c: [categories[c][k] for k in m]
                       for c, m in li['merges']},
            'n_rules': len(li['rules']),
            **{k: li[k] for k in ('broken_before_fix', 'rows_redrawn',
                                  'broken_after_fix', 'marginal_fix')
               if k in li}}
        print(f'[{tag}] refit {r} written {info[f"r{r}"]}', flush=True)
    with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
        json.dump({'tag': tag, 'args': vars(a), 'rho': mech.rho,
                   'refits': info, 'runtime_s': time.time() - t0}, f,
                  indent=1, default=str)
    print(f'[{tag}] done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
