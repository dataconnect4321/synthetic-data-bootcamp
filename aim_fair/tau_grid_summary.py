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

BER, A-NCB, AUC, validity and impossible rows are the scorer's
(run_adult_flip.py --score-csv jsons). The downstream fairness metrics are
FLIP's own protocol (flip_protocol.py: LightGBM trained on the table,
predicting real test people), as in the FLIP-faithful comparison: income
parity and EO gap (TPR gap) at threshold 0.5, and the means over targets of
Eq. 14 (sp_cat) and W1 / sd (w1_num). They are computed here for every run
and cached in adult_results/flipproto/. So the fold-0 rows at tau 1 equal
the AIM hybrid of results/flip_faithful_comparison.csv (the same tables).

The real-data row takes the scorer's own references (BER, A-NCB, AUC,
validity, impossible rows) from any scored json of each fold, and the
protocol's scores of the real training fold.

Writes results/utility_tau_grid.csv. Main environment (torch, lightgbm).

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

import flip_protocol as FP
from final_chart import METRICS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
KEYS = ['ber', 'ancb', 'auc', 'income_parity', 'eo_gap', 'sp_cat',
        'w1_num', 'validity', 'impossible']
GET = {k: g for k, _, _, g in METRICS}
PROTO = {'income_parity': 'parity', 'eo_gap': 'eo_gap', 'sp_cat': 'sp_cat',
         'w1_num': 'w1_num'}   # KEYS from flip_protocol.run_scores
REAL = {   # the scorer's real-data references
    'ber': lambda d: d['paper_metrics']['real_test']['ber'],
    'ancb': lambda d: d['paper_metrics']['real_test_own_adversary']['a_ncb'],
    'auc': lambda d: d['paper_metrics']['downstream_auc_real'],
    'validity': lambda d: d['paper_metrics']['validity_real'],
    'impossible': lambda d: d['paper_metrics'][
        'impossible_combination_rate_real']['any'],
}
RES = os.path.join(ROOT, 'adult_results')


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
        """{fold: [(tag, scorer json) per refit]} of the jsons that exist."""
        out = {}
        for f in a.folds:
            for r in a.refits:
                tag = pattern.format(f=f, r=r)
                path = os.path.join(RES, tag + '.json')
                if os.path.exists(path):
                    with open(path) as fh:
                        out.setdefault(f, []).append((tag, json.load(fh)))
        return out

    taus = sorted({float(m.group(1)) for p in glob.glob(os.path.join(
        RES, f'{a.tag}_tau*_e3_f*_r*.json')) if (m := re.search(
            r'_tau([0-9.]+)_e3_f\d+_r\d+\.json$', p))})
    sources = [(source, tau, pattern, js) for tau, source, pattern in (
        [(t, 'AIM hybrid', f'{a.tag}_tau{t:g}_e3_f{{f}}_r{{r}}') for t in taus]
        + [(None, 'Plain AIM', f'{plain}_e3_f{{f}}_r{{r}}')])
        if (js := scored(pattern))]
    if not sources:
        sys.exit(f'no scored {a.tag}_tau*_e3_f*_r*.json in adult_results/: '
                 'run scripts/run_tau_grid.sh')
    # one json per fold for the scorer's real references
    real_js = {}
    for *_, js in sources:
        for f, runs in js.items():
            real_js.setdefault(f, runs[0][1])
    # FLIP's protocol for every table, and the real training folds
    proto = FP.run_scores(
        [(tag, os.path.join('aim_fair', 'out', tag + '_synthetic.csv'), f)
         for *_, js in sources for f, runs in js.items() for tag, _ in runs]
        + [(f'fold{f}_train', os.path.join('aim_fair', 'data',
                                           f'fold{f}_train.csv'), f)
           for f in real_js])

    def value(k, tag, js, get):
        return float(proto[tag][PROTO[k]] if k in PROTO else get[k](js))

    rows = []
    for source, tau, pattern, js in sources:
        rows += rows_of(source, tau, pattern, {
            f: [[value(k, tag, d, GET) for k in KEYS] for tag, d in runs]
            for f, runs in js.items()})
    real = {f: [[value(k, f'fold{f}_train', d, REAL) for k in KEYS]]
            for f, d in sorted(real_js.items())}
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
