"""Scorecard table for the leakage explorer (scripts/run_tau_grid.sh): the
AIM hybrid refitted at each repair amount tau from one set of private
measurements (Adult, fold 0, eps 3), plain AIM from the same measurements,
and the real training data. One refit per tau, so no sd.

The real-data row takes the scorer's own references (BER, A-NCB, AUC,
validity, impossible rows) from any scored json of the fold. The scorer has
no real reference for income parity and EO, so they are computed here with
the same function (paper_metrics.downstream_sp_eo), trained on the real
training fold and evaluated on it, as the synthetic tables are.

Writes results/utility_tau_grid.csv. Main environment (torch).

    python aim_fair/tau_grid_summary.py [--tag lk]
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


def real_parity_eo():
    sys.path.insert(0, os.path.join(ROOT, 'tabsyn-main', 'tabsyn-main'))
    import adult_domain as AD
    from tabsyn.vae.paper_metrics import downstream_sp_eo
    df = pd.read_csv(os.path.join(HERE, 'data', 'fold0_train.csv'),
                     dtype={'income': str})
    cat = np.stack([df[c].astype(str).map(
        {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))})
        .to_numpy() for c in AD.CATEGORICAL], 1)
    num = df[AD.NUMERIC].to_numpy(np.float64)
    r = downstream_sp_eo(num, cat, num, cat, AD.CATEGORICAL.index('income'),
                         AD.CATEGORICAL.index('sex'))
    return r['sp_diff'], r['eo_diff']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='lk')
    a = ap.parse_args()
    res = os.path.join(ROOT, 'adult_results')
    rows, any_js = [], None
    grid = []
    for path in glob.glob(os.path.join(res, f'{a.tag}_tau*_e3_f0_r0.json')):
        m = re.search(r'_tau([0-9.]+)_e3_f0_r0\.json$', path)
        grid.append((float(m.group(1)), path))
    for tau, path in sorted(grid) + [
            (None, os.path.join(res, f'{a.tag}_plain_e3_f0_r0.json'))]:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            js = json.load(fh)
        any_js = any_js or js
        rows.append({'source': 'AIM hybrid' if tau is not None else
                     'Plain AIM', 'tau': '' if tau is None else tau,
                     **{k: round(float(GET[k](js)), 4) for k in KEYS}})
    if any_js is None:
        sys.exit(f'no scored {a.tag}_tau*_e3_f0_r0.json in adult_results/: '
                 'run scripts/run_tau_grid.sh')
    sp, eo = real_parity_eo()
    rows.append({'source': 'Real data', 'tau': '',
                 **{k: round(float(REAL[k](any_js)), 4) for k in REAL},
                 'income_parity': round(sp, 4), 'eo_gap': round(eo, 4)})
    out = os.path.join(ROOT, 'results', 'utility_tau_grid.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, ['source', 'tau'] + KEYS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['source']:10s} tau={r['tau']!s:5s} "
              + ' '.join(f'{k}={r[k]:.3f}' for k in KEYS))
    print('wrote', os.path.normpath(out))


if __name__ == '__main__':
    main()
