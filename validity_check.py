"""Data-validity (plausibility) check of synthetic tables against rules
learned from the real training fold - no hand-written domain rules.

    python validity_check.py --fold 1 [--mask] [--sample 5000] TAG [TAG ...]

Rules (tabsyn.vae.paper_metrics.fit_validity_rules): numeric range, whole-
number format, numeric range within each category (0.5-99.5%),
near-deterministic category implications (>= 99%), and category pairs that
never occur in real data although independence predicts >= 20 rows. The
real held-out fold is scored too: its rate is the false-alarm baseline.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))

import run_adult_flip as ra
from tabsyn.vae.paper_metrics import fit_validity_rules, validity_report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--mask', action='store_true',
                    help='runs used --mask-proxies (Husband/Wife -> MASK)')
    ap.add_argument('--postprocess', action='store_true',
                    help='apply the real-data numeric schema (clip + round, '
                         'as new runs do when writing the CSV) before '
                         'scoring - for CSVs written before that change')
    ap.add_argument('--sample', type=int, default=5000,
                    help='rows sampled from each synthetic table')
    args = ap.parse_args()

    data = ra.prepare(ra.read_adult(), args.fold, 3, 42)
    enc = data['encoders']
    tr_c, te_c = data['train_cat'], data['test_cat']
    if args.mask:
        from tabsyn.vae.proxy_decouple import (detect_proxy_classes,
                                               mask_proxy_classes)
        s_idx = ra.CATEGORICAL.index(ra.SENSITIVE)
        t_idx = ra.CATEGORICAL.index(ra.TARGET)
        det = detect_proxy_classes(tr_c, s_idx, exclude=(t_idx,))
        codes = []
        for cl in det:
            col = ra.CATEGORICAL[cl['col']]
            codes.append(len(enc[col]))
            enc[col] = np.append(enc[col], 'MASK')
        tr_c = mask_proxy_classes(tr_c, det, codes)
        te_c = mask_proxy_classes(te_c, det, codes)

    def raw(n):
        return n * data['stds'].values + data['means'].values

    labels = [enc[c] for c in ra.CATEGORICAL]
    import run_flip_demo as demo
    schema = demo.infer_numeric_schema(raw(data['train_num']), ra.NUMERIC)
    rules = fit_validity_rules(raw(data['train_num']), tr_c, ra.NUMERIC,
                               ra.CATEGORICAL, labels)
    print('rules learned: ' + ', '.join(f'{k} {len(v)}'
                                        for k, v in rules.items()))

    def show(name, rep):
        bt = rep['by_type']
        print(f'\n{name:22s} valid rows {rep["substantive_valid_rate"]:.3f} '
              f'(with whole-number format: {rep["valid_rate"]:.3f})\n'
              f'{"":22s} violation rate by rule type: '
              + '  '.join(f'{k} {v:.3f}' for k, v in bt.items()))
        for r, d in rep['worst']:
            print(f'    {r:.3f}  {d}')

    show('REAL held-out', validity_report(rules, raw(data['test_num']), te_c))
    rng = np.random.default_rng(0)
    for tag in args.tags:
        df = pd.read_csv(os.path.join(ra.OUT_DIR, f'{tag}_synthetic.csv'))
        df = df.iloc[rng.choice(len(df), min(args.sample, len(df)),
                                replace=False)]
        if args.postprocess:
            df = demo.apply_numeric_schema(df, schema)
        cat = np.stack([df[c].astype(str).map(
            {str(v): i for i, v in enumerate(enc[c])}).values
            for c in ra.CATEGORICAL], axis=1)
        if np.isnan(cat.astype(float)).any():
            print(f'\n{tag}: categories outside the vocabulary (wrong --mask?)')
            continue
        show(tag, validity_report(rules, df[ra.NUMERIC].values,
                                  cat.astype(int)))


if __name__ == '__main__':
    main()
