"""Aggregate paper-config runs across the 3 folds: mean +/- spread per setting.

    python aggregate_folds.py [pattern]
    default pattern: adult_results/pc_f*_*.json
"""

import json
import glob
import re
import sys
import numpy as np
from collections import defaultdict


def get(d, *paths):
    for p in paths:
        x = d
        try:
            for k in p.split('.'):
                x = x[k]
            return float(x)
        except (KeyError, TypeError, ValueError):
            pass
    return np.nan


METRICS = {
    'BER syn':       ['paper_metrics.syn.ber'],
    'BER real':      ['paper_metrics.real_test.ber'],
    'BER rel %':     ['paper_metrics.outcome_replication.ber_rel_pct'],
    'A-NCB syn':     ['paper_metrics.syn.a_ncb'],
    'A-NCB real':    ['paper_metrics.real_test.a_ncb'],
    'A-NCB rel %':   ['paper_metrics.outcome_replication.a_ncb_rel_pct'],
    'AUC syn':       ['paper_metrics.downstream_auc'],
    'AUC real':      ['paper_metrics.downstream_auc_real'],
    'AUC rel %':     ['paper_metrics.outcome_replication.auc_rel_pct'],
    'leakage':       ['paper_metrics.decoded_leakage'],
    'ceiling':       ['paper_metrics.heldout_ceiling'],
    'true gap':      ['paper_metrics.true_group_target_gap'],
    'income rate':   ['paper_metrics.syn_income_rate'],
    'match-valid':   ['paper_metrics.match_valid'],
}


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'adult_results/pc_f*_*.json'
    rows = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(pattern):
        m = re.search(r'pc_f(\d)_(.+)\.json$', path.replace('\\', '/'))
        if not m:
            continue
        d = json.load(open(path))
        for name, paths in METRICS.items():
            rows[m.group(2)][name].append(get(d, *paths))
        # true-sex BER = 1 - balanced accuracy of predicting true sex
        l = rows[m.group(2)]['leakage'][-1]
        c = rows[m.group(2)]['ceiling'][-1]
        rows[m.group(2)]['BER true syn'].append(1 - l)
        rows[m.group(2)]['BER true real'].append(1 - c)

    for setting in sorted(rows):
        n = len(rows[setting]['BER syn'])
        print(f'\n== {setting}  (n folds = {n})')
        for name, vals in rows[setting].items():
            v = np.array(vals, float)
            v = v[~np.isnan(v)]
            if len(v):
                sd = v.std(ddof=1) if len(v) > 1 else 0.0
                print(f'  {name:14s} {v.mean():8.3f} +/- {sd:.3f}   '
                      f'folds: {np.round(v, 3).tolist()}')


if __name__ == '__main__':
    main()
