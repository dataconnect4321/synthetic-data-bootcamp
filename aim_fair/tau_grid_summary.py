"""Scorecard table for the leakage explorer (scripts/run_tau_grid.sh): the
AIM hybrid refitted at each repair amount tau from each fold's saved private
measurements (Adult, eps 3), plain AIM from the same measurements, and the
real training data.

Each source has one row per fold (`fold`: the mean over its refits) and one
summary row (`fold` empty): the mean over folds, with <metric>_sd the sd
across the fold means (ddof 1), as in final_chart.aim_vs_vae and RESULTS.md.
With one fold, the sd is across that fold's refits instead; with one run, it
is empty. The per-fold rows let the page pair the hybrid with plain AIM from
the same measurements. `runs` is the scorer tag pattern of the row (refits
r = 0 .. refits - 1 by default), so the numbers can be traced to
adult_results/.

The real-data row takes the scorer's own references (BER, A-NCB, AUC,
validity, impossible rows) from any scored json of each fold. The scorer has
no real reference for income parity and EO, so they are computed here with
the same function (paper_metrics.downstream_sp_eo), trained on the real
training fold and evaluated on it, as the synthetic tables are.

Writes results/utility_tau_grid.csv. Main environment (torch).

    python aim_fair/tau_grid_summary.py [--tag aimfinal] [--plain aimfinalplain]
        [--folds 0 1 2] [--refits 0 1 2]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from final_chart import METRICS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
KEYS = ['ber', 'ancb', 'auc', 'income_parity', 'eo_gap', 'validity',
        'impossible']
GET = {k: g for k, _, _, g in METRICS}
REAL = {   # the scorer's real-data references
    'ber': lambda d: d['paper_metrics']['real_test']['ber'],
    'ancb': lambda d: d['paper_metrics']['real_test_own_adversary']['a_ncb'],
    'auc': lambda d: d['paper_metrics']['downstream_auc_real'],
    'validity': lambda d: d['paper_metrics']['validity_real'],
    'impossible': lambda d: d['paper_metrics'][
        'impossible_combination_rate_real']['any'],
}
RES = os.path.join(ROOT, 'adult_results')


def real_parity_eo(fold):
    sys.path.insert(0, os.path.join(ROOT, 'tabsyn-main', 'tabsyn-main'))
    import adult_domain as AD
    from tabsyn.vae.paper_metrics import downstream_sp_eo
    df = pd.read_csv(os.path.join(HERE, 'data', f'fold{fold}_train.csv'),
                     dtype={'income': str})
    cat = np.stack([df[c].astype(str).map(
        {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))})
        .to_numpy() for c in AD.CATEGORICAL], 1)
    num = df[AD.NUMERIC].to_numpy(np.float64)
    r = downstream_sp_eo(num, cat, num, cat, AD.CATEGORICAL.index('income'),
                         AD.CATEGORICAL.index('sex'))
    return r['sp_diff'], r['eo_diff']


def rows_of(source, tau, runs, per_fold):
    """{fold: [[value per KEYS] per refit]} -> one row per fold, then the
    summary row: the mean of the fold means and the sd across them (across
    refits with one fold; empty with one run)."""
    nan = np.full(len(KEYS), np.nan)

    def row(fold, mean, sd, folds, refits):
        r = {'source': source, 'tau': '' if tau is None else tau,
             'fold': fold, 'folds': folds, 'refits': refits, 'runs': runs}
        r.update({k: round(float(m), 4) for k, m in zip(KEYS, mean)})
        r.update({k + '_sd': '' if np.isnan(s) else round(float(s), 4)
                  for k, s in zip(KEYS, sd)})
        return r

    means = {f: np.mean(v, axis=0) for f, v in sorted(per_fold.items())}
    out = [row(f, m, nan, 1, len(per_fold[f])) for f, m in means.items()]
    m = np.array(list(means.values()))
    if len(m) > 1:
        sd = m.std(0, ddof=1)
    else:
        runs_ = np.array(next(iter(per_fold.values())))
        sd = runs_.std(0, ddof=1) if len(runs_) > 1 else nan
    return out + [row('', m.mean(0), sd, len(m),
                      min(len(v) for v in per_fold.values()))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='aimfinal')
    ap.add_argument('--plain', default=None,
                    help='plain AIM tag (default: TAG + "plain")')
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--refits', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    plain = a.plain or a.tag + 'plain'

    def scored(pattern):
        """{fold: [[KEYS values] per refit]} of the scorer jsons that exist,
        and one json per fold for the real references."""
        vals, any_js = {}, {}
        for f in a.folds:
            for r in a.refits:
                path = os.path.join(RES, pattern.format(f=f, r=r) + '.json')
                if os.path.exists(path):
                    with open(path) as fh:
                        js = json.load(fh)
                    vals.setdefault(f, []).append(
                        [float(GET[k](js)) for k in KEYS])
                    any_js.setdefault(f, js)
        return vals, any_js

    taus = sorted({float(m.group(1)) for p in glob.glob(os.path.join(
        RES, f'{a.tag}_tau*_e3_f*_r*.json')) if (m := re.search(
            r'_tau([0-9.]+)_e3_f\d+_r\d+\.json$', p))})
    rows, real_js = [], {}
    for tau, source, pattern in (
            [(t, 'AIM hybrid', f'{a.tag}_tau{t:g}_e3_f{{f}}_r{{r}}')
             for t in taus] +
            [(None, 'Plain AIM', f'{plain}_e3_f{{f}}_r{{r}}')]):
        vals, any_js = scored(pattern)
        if vals:
            rows += rows_of(source, tau, pattern, vals)
            for f, js in any_js.items():
                real_js.setdefault(f, js)
    if not real_js:
        sys.exit(f'no scored {a.tag}_tau*_e3_f*_r*.json in adult_results/: '
                 'run scripts/run_tau_grid.sh')
    real = {}
    for f, js in sorted(real_js.items()):
        sp, eo = real_parity_eo(f)
        v = {**{k: float(REAL[k](js)) for k in REAL},
             'income_parity': sp, 'eo_gap': eo}
        real[f] = [[v[k] for k in KEYS]]
    rows += rows_of('Real data', None, 'fold{f}_train', real)
    out = os.path.join(ROOT, 'results', 'utility_tau_grid.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['source', 'tau'] + KEYS + [k + '_sd' for k in KEYS]
                           + ['fold', 'folds', 'refits', 'runs'])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r['fold'] != '':
            continue
        print(f"{r['source']:10s} tau={r['tau']!s:5s} "
              f"{r['folds']}x{r['refits']} "
              + ' '.join(f'{k}={r[k]:.3f}' for k in KEYS))
    print('wrote', os.path.normpath(out))


if __name__ == '__main__':
    main()
