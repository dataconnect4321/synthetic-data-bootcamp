"""Table of the AIM hybrid on the FLIP paper's other datasets
(run_datasets.sh): real training data (reference), plain AIM and the AIM
hybrid per dataset. Refits are averaged within a fold, then mean and sd
across folds. The target is income for ACS-I and occupation for Dutch.

Writes results/datasets_summary.csv.

    python aim_fair/datasets_summary.py
"""
import csv
import json
import os

import numpy as np

from final_chart import METRICS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
EXTRA = [('tstr_folded', 'Release teaches real sex, folded tstr', '0.5 best',
          lambda d: d['paper_metrics']['tstr_sex_ber_folded']),
         ('num_w1', 'Numeric distortion (mean W1/sd)', 'lower',
          lambda d: d['fidelity']['num_w1_mean'])]
DATASETS = ['acs_al', 'acs_ut', 'dutch']
SOURCES = [   # name, tag pattern, refits
    ('Real data (reference)', '{d}_real_f{f}', [None]),
    ('Plain AIM, eps 3', '{d}_aimplain_e3_f{f}_r{r}', [0, 1, 2]),
    ('AIM hybrid, eps 3', '{d}_aim_e3_f{f}_r{r}', [0, 1, 2]),
]


def main():
    metrics = [(k, g) for k, _, _, g in METRICS + EXTRA]
    rows = []
    for d in DATASETS:
        for name, pattern, refits in SOURCES:
            per_fold = []
            for f in range(3):
                v = []
                for r in refits:
                    path = os.path.join(ROOT, 'adult_results',
                                        pattern.format(d=d, f=f, r=r) + '.json')
                    if os.path.exists(path):
                        with open(path) as fh:
                            js = json.load(fh)
                        v.append([float(g(js)) if g(js) is not None
                                  else np.nan for _, g in metrics])
                if v:
                    per_fold.append(np.nanmean(v, axis=0))
            if not per_fold:
                continue
            a = np.array(per_fold)
            row = {'dataset': d, 'source': name, 'folds': len(a)}
            for (k, _), m, s in zip(metrics, np.nanmean(a, 0),
                                    np.nanstd(a, 0, ddof=1) if len(a) > 1
                                    else np.zeros(a.shape[1])):
                row[k], row[k + '_sd'] = round(float(m), 4), round(float(s), 4)
            rows.append(row)
            print(f'{d:7s} {name:22s} folds={len(a)} '
                  + ' '.join(f'{k}={row[k]:.3f}' for k, _ in metrics))
    if rows:
        out = os.path.join(ROOT, 'results', 'datasets_summary.csv')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('wrote', os.path.normpath(out))


if __name__ == '__main__':
    main()
