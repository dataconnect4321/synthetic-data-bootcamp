"""Top metrics for synthetic tables as generated, after EDIT and after REJECT
consistency passes (tabsyn.vae.consistency: strong learned rules only,
protected attribute never enforced).

    python consistency_eval.py --fold 1 --mask [--min-expected 100] TAG ...

Numerics are first clipped/rounded to the real format (as new CSVs are).
Tier-1 tables (tag ending in _t1, or --tier1) also get true-sex leakage /
true-sex BER: tier-1 row i is decoded from training row i, so its real sex
is known (rejected rows are dropped from that comparison too).
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
import run_flip_demo as demo
from tabsyn.vae.paper_metrics import (fit_validity_rules, validity_report,
                                      alpha_precision_beta_recall,
                                      infer_protected,
                                      categorical_share_tv_all)
from tabsyn.vae.consistency import (strong_rules, violations, reject_invalid,
                                    edit_invalid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--mask', action='store_true')
    ap.add_argument('--tier1', action='store_true')
    ap.add_argument('--min-expected', type=float, default=100.0)
    ap.add_argument('--seeds', type=int, default=10,
                    help='A-NCB scoring seeds (full-covariance GMM)')
    args = ap.parse_args()

    data = ra.prepare(ra.read_adult(), args.fold, 3, 42)
    enc = data['encoders']
    s_idx = ra.CATEGORICAL.index(ra.SENSITIVE)
    t_idx = ra.CATEGORICAL.index(ra.TARGET)
    tr_c, te_c = data['train_cat'], data['test_cat']
    if args.mask:
        from tabsyn.vae.proxy_decouple import (detect_proxy_classes,
                                               mask_proxy_classes)
        det = detect_proxy_classes(tr_c, s_idx, exclude=(t_idx,))
        codes = []
        for cl in det:
            col = ra.CATEGORICAL[cl['col']]
            codes.append(len(enc[col]))
            enc[col] = np.append(enc[col], 'MASK')
        tr_c = mask_proxy_classes(tr_c, det, codes)
        te_c = mask_proxy_classes(te_c, det, codes)
    tr_n, te_n = data['train_num'], data['test_num']
    sizes = [len(enc[c]) for c in ra.CATEGORICAL]
    means, stds = data['means'].values, data['stds'].values
    raw = lambda n: n * stds + means
    std = lambda r: ((r - means) / stds).astype(np.float32)

    rules = fit_validity_rules(raw(tr_n), tr_c, ra.NUMERIC, ra.CATEGORICAL,
                               [enc[c] for c in ra.CATEGORICAL])
    strong = strong_rules(rules, tr_c, args.min_expected, exclude=(s_idx,))
    print(f'enforced rules ({len(strong["pair"])} pair, {len(strong["fd"])} '
          f'implication; expected >= {args.min_expected:g}, sex excluded):')
    for kind in ('pair', 'fd'):
        for *_, d in strong[kind]:
            print(f'  {kind}: {d}')
    schema = demo.infer_numeric_schema(raw(tr_n), ra.NUMERIC)
    t_pos = list(enc[ra.TARGET]).index('1')

    def util_feats(n, c):
        keep = [j for j in range(c.shape[1]) if j not in (s_idx, t_idx)]
        return np.concatenate([n] + [np.eye(sizes[j])[c[:, j]]
                                     for j in keep], axis=1)

    def metrics(n_std, c, true_s=None):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        fc = ra.averaged_scores(n_std, c, s_idx, ref=(te_n, te_c),
                                n_seeds=args.seeds, covariance_type='full',
                                n_init=1)
        clf = HistGradientBoostingClassifier(random_state=0).fit(
            util_feats(n_std, c), c[:, t_idx] == t_pos)
        auc = roc_auc_score(te_c[:, t_idx] == t_pos,
                            clf.predict_proba(util_feats(te_n, te_c))[:, 1])
        a, b = alpha_precision_beta_recall(raw(tr_n), tr_c, raw(n_std), c,
                                           sizes)
        tv = categorical_share_tv_all(c, tr_c)
        out = {'BER': fc['ber'], 'A-NCB': fc['a_ncb'],
               'A-NCB med': fc['a_ncb_median'], 'AUC': auc, 'alpha': a,
               'beta': b,
               'max share chg': max(v for k, v in tv.items() if k != s_idx),
               'valid': validity_report(rules, raw(n_std), c)[
                   'substantive_valid_rate'],
               'rows': len(c)}
        if true_s is not None:
            pred = infer_protected(te_n, te_c, n_std, c, s_idx)
            leak = np.mean([(pred[true_s == g] == g).mean()
                            for g in np.unique(true_s)])
            out['leakage'] = leak
            out['true-sex BER'] = 1 - leak
        return out

    def table(rows):
        keys = [k for k in rows[0][1]]
        print(f'  {"version":12s} ' + ' '.join(f'{k:>13s}' for k in keys))
        for name, m in rows:
            print(f'  {name:12s} ' + ' '.join(
                f'{m[k]:13.3f}' if isinstance(m[k], float) else f'{m[k]:13d}'
                for k in keys))

    print('\n== REAL (held-out fold, as scored for synthetic)')
    table([('real', metrics(te_n, te_c, None))])

    for tag in args.tags:
        df = pd.read_csv(os.path.join(ra.OUT_DIR, f'{tag}_synthetic.csv'))
        df = demo.apply_numeric_schema(df, schema)
        cat = np.stack([df[c].astype(str).map(
            {str(v): i for i, v in enumerate(enc[c])}).values
            for c in ra.CATEGORICAL], axis=1).astype(int)
        n_std = std(df[ra.NUMERIC].values.astype(float))
        tier1 = args.tier1 or tag.endswith('_t1')
        true_s = tr_c[:, s_idx].astype(int) if tier1 else None
        n_bad = int(violations(strong, cat).sum())
        e_cat, n_edit, n_left = edit_invalid(strong, n_std, cat, tr_n, tr_c,
                                             s_idx, sizes)
        r_num, r_cat, kept = reject_invalid(strong, n_std, cat)
        print(f'\n== {tag}: {n_bad} rows ({n_bad / len(cat):.1%}) break an '
              f'enforced rule; edit changed {n_edit} rows ({n_left} left); '
              f'reject kept {kept.sum()}')
        table([('generated', metrics(n_std, cat, true_s)),
               ('edited', metrics(n_std, e_cat, true_s)),
               ('rejected', metrics(r_num, r_cat,
                                    true_s[kept] if true_s is not None
                                    else None))])


if __name__ == '__main__':
    main()
