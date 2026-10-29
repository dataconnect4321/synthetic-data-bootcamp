"""Downstream fairness under the FLIP paper's own protocol (arXiv
2508.21815, Sections 5.3 and 6.5), for its faithful reproduction and the AIM
tables, so every source is scored the same way.

FLIP's protocol as stated: LightGBM, the 3-fold splits' unseen test sets,
every non-protected attribute as a target in turn (all other columns but
sex as features, trained on the source, predicted on real test people):
  categorical target  statistical parity, Eq. 14: the largest gap over
                      classes in P(predicted class = c) between the sexes
                      (argmax predictions)
  numeric target      Eq. 15: Wasserstein-1 distance between the sexes'
                      predicted values (also divided by the real sd, so
                      targets can be averaged)
FLIP does not define its equalized-odds number or say which target its
Fig. 3 uses, so for income the candidates are all reported, at 0.5 and at
matched selection (the share the real-data model selects): the TPR gap
(equal opportunity), the FPR gap, their max (fairlearn's equalized odds
difference) and their mean (average odds).

Sources: real training fold; plain AIM and AIM hybrid at tau 1 (3 folds x
3 refits: the tau grid's tables, scripts/run_tau_grid.sh, so the dashboard's
scorecard scores the same tables); FLIP faithful reproduction (lambda 4,
nominal eps 3; fold 0 only, the table that exists). Rows marked fold0
restrict every source to fold 0 for the like-for-like FLIP row. A source
whose tables are not on disk keeps its rows from the existing csvs (printed),
so the FLIP reproduction survives on a machine without its table.

    python aim_fair/flip_protocol.py      # main environment (lightgbm)
Writes results/flip_protocol_targets.csv, results/flip_protocol_income.csv.
"""
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import wasserstein_distance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
CAT = ['income', 'workclass', 'marital-status', 'occupation', 'relationship',
       'race', 'native-country']
SEX, MASK = 'sex', 'MASK'
SOURCES = [   # name, path pattern, refits, folds
    ('Real data', 'aim_fair/data/fold{f}_train.csv', [None], [0, 1, 2]),
    ('Plain AIM', 'aim_fair/out/aimfinalplain_e3_f{f}_r{r}_synthetic.csv',
     [0, 1, 2], [0, 1, 2]),
    ('AIM hybrid', 'aim_fair/out/aimfinal_tau1_e3_f{f}_r{r}_synthetic.csv',
     [0, 1, 2], [0, 1, 2]),
    ('FLIP reproduction (lambda 4, nominal eps 3)',
     'adult_results/flipfaithful_e3_f0_lam4_synthetic.csv', [None], [0]),
]


def read(p):
    d = pd.read_csv(os.path.join(ROOT, p))
    for c in CAT + [SEX]:
        d[c] = d[c].astype(str)
    for c in NUM:
        d[c] = d[c].astype(float)
    return d


def design(train, test, target):
    feats = [c for c in NUM + CAT if c != target]
    test = test.copy()
    for c in feats:          # merged proxy levels: same masking of real rows
        if c in CAT and MASK in set(train[c]):
            test.loc[~test[c].isin(set(train[c])), c] = MASK
    out = []
    for d in (train, test):
        x = d[feats].copy()
        for c in feats:
            if c in CAT:
                x[c] = pd.Categorical(x[c], categories=sorted(
                    set(train[c]) | set(test[c])))
        out.append(x)
    return out


def one(train, test, target, male, sd):
    xtr, xte = design(train, test, target)
    if target in NUM:
        m = LGBMRegressor(verbose=-1, n_jobs=2, random_state=0).fit(
            xtr, train[target])
        p = m.predict(xte)
        w = wasserstein_distance(p[male], p[~male])
        return {'metric': 'W1', 'value': w, 'value_sd': w / sd}
    y = train[target]
    m = LGBMClassifier(verbose=-1, n_jobs=2, random_state=0).fit(xtr, y)
    pred = m.predict(xte)
    classes = sorted(set(pred) | set(test[target]))
    sp = max(abs((pred[male] == c).mean() - (pred[~male] == c).mean())
             for c in classes)
    return {'metric': 'SP (Eq. 14)', 'value': sp, 'value_sd': sp}


def income(train, test, male, k):
    xtr, xte = design(train, test, 'income')
    m = LGBMClassifier(verbose=-1, n_jobs=2, random_state=0).fit(
        xtr, (train['income'] == '1').astype(int))
    p = m.predict_proba(xte)[:, 1]
    y = (test['income'] == '1').to_numpy()
    if k is None:            # the real-data model: matched = its own share
        k = (p > 0.5).mean()
    out = {}
    for mode, thr in (('0.5', 0.5), ('matched', None)):
        if thr is None:
            thr = np.quantile(p, 1 - k)
        s = p > thr
        tpr = s[male & y].mean() - s[~male & y].mean()
        fpr = s[male & ~y].mean() - s[~male & ~y].mean()
        out[mode] = {'sel': s.mean(),
                     'parity': s[male].mean() - s[~male].mean(),
                     'ratio': s[~male].mean() / s[male].mean(),
                     'EO = TPR gap': abs(tpr), 'FPR gap': abs(fpr),
                     'max(TPR, FPR) gap': max(abs(tpr), abs(fpr)),
                     'mean(TPR, FPR) gap': (abs(tpr) + abs(fpr)) / 2}
    return out, (p > 0.5).mean()


def table_scores(path, f):
    """One table (path relative to the repo root) under the protocol, against
    fold f's real test people: income parity and EO (TPR gap) at threshold
    0.5, and the means over targets of Eq. 14 (categorical) and W1 / sd
    (numeric), as flip_faithful_chart.py reports them."""
    test = read(f'aim_fair/data/fold{f}_test.csv')
    real = read(f'aim_fair/data/fold{f}_train.csv')
    male = (test[SEX] == 'Male').to_numpy()
    tr = read(path)[test.columns]
    t = [one(tr, test, c, male, real[c].std() if c in NUM else None)
         for c in NUM + CAT]
    res, _ = income(tr, test, male, None)
    return {'parity': res['0.5']['parity'], 'eo_gap': res['0.5']['EO = TPR gap'],
            'sp_cat': np.mean([r['value_sd'] for r in t if r['metric'] != 'W1']),
            'w1_num': np.mean([r['value_sd'] for r in t if r['metric'] == 'W1'])}


def run_scores(runs, n_jobs=4):
    """{tag: table_scores} for runs [(tag, table path, fold)], cached per tag
    in adult_results/flipproto/<tag>.json, as the scorer caches its jsons (a
    cached tag is not rescored). Used by tau_grid_summary.py for the
    scorecard."""
    import json

    def cache(tag):
        return os.path.join(ROOT, 'adult_results', 'flipproto', f'{tag}.json')

    os.makedirs(os.path.dirname(cache('x')), exist_ok=True)
    todo = [r for r in runs if not os.path.exists(cache(r[0]))]
    done = Parallel(n_jobs=n_jobs)(delayed(table_scores)(p, f)
                                   for _, p, f in todo)
    for (tag, p, f), s in zip(todo, done):
        with open(cache(tag), 'w') as fh:
            json.dump({'table': p, 'fold': f,
                       **{k: float(v) for k, v in s.items()}}, fh, indent=1)
    out = {}
    for tag, _, _ in runs:
        with open(cache(tag)) as fh:
            out[tag] = json.load(fh)
    return out


def on_disk(pat, refits, folds):
    return all(os.path.exists(os.path.join(ROOT, pat.format(f=f, r=r)))
               for f in folds for r in refits)


def fold_job(f, skip=()):
    test = read(f'aim_fair/data/fold{f}_test.csv')
    real = read(f'aim_fair/data/fold{f}_train.csv')
    male = (test[SEX] == 'Male').to_numpy()
    sd = {c: real[c].std() for c in NUM}
    t_rows, i_rows, k = [], [], None
    for name, pat, refits, folds in SOURCES:
        if f not in folds or name in skip:
            continue
        for r in refits:
            tr = read(pat.format(f=f, r=r))[test.columns]
            for target in NUM + CAT:
                t_rows.append({'fold': f, 'refit': r, 'source': name,
                               'target': target,
                               **one(tr, test, target, male, sd.get(target))})
            res, share = income(tr, test, male, k)
            if k is None:
                k = share
            for mode, m in res.items():
                i_rows.append({'fold': f, 'refit': r, 'source': name,
                               'threshold': mode, **m})
    print(f'fold {f} done', flush=True)
    return pd.DataFrame(t_rows), pd.DataFrame(i_rows)


def summarize(df, keys, scope):
    """Refits averaged within a fold, then folds; scope 'all' or 'fold0'."""
    if scope == 'fold0':
        df = df[df.fold == 0]
    g = df.groupby(['fold'] + keys, sort=False).mean(numeric_only=True)
    g = g.drop(columns=[c for c in ('refit',) if c in g]).reset_index()
    return g.groupby(keys, sort=False).mean(numeric_only=True).drop(
        columns='fold').reset_index()


def main():
    skip = [s for s, p, r, f in SOURCES if not on_disk(p, r, f)]
    paths = {k: os.path.join(ROOT, 'results', f'flip_protocol_{k}.csv')
             for k in ('targets', 'income')}
    kept = {}
    for k, p in paths.items():   # rows of the missing sources, as committed
        old = pd.read_csv(p) if skip and os.path.exists(p) else None
        kept[k] = None if old is None else old[old.source.isin(skip)]
    for s in skip:
        if kept['targets'] is None or s not in set(kept['targets'].source):
            raise SystemExit(f'{s}: tables missing and no rows in {paths["targets"]}')
        print(f'{s}: tables not on disk, rows kept from the existing csvs')
    parts = Parallel(n_jobs=3)(delayed(fold_job)(f, skip) for f in range(3))
    targets = pd.concat([p[0] for p in parts])
    inc = pd.concat([p[1] for p in parts])
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 50)
    out_t, out_i = [], []
    for scope in ('fold0', 'all'):
        t = summarize(targets, ['source', 'target', 'metric'], scope)
        t['scope'] = scope
        if kept['targets'] is not None:
            t = pd.concat([t, kept['targets'][kept['targets'].scope == scope]])
        out_t.append(t)
        piv = t.pivot(index='target', columns='source', values='value_sd')
        piv = piv[[s for s, *_ in SOURCES if s in piv]]
        piv.loc['MEAN, categorical (SP)'] = piv.loc[CAT].mean()
        piv.loc['WORST, categorical (SP)'] = piv.loc[CAT].max()
        piv.loc['MEAN, numeric (W1 / sd)'] = piv.loc[NUM].mean()
        piv.loc['WORST, numeric (W1 / sd)'] = piv.loc[NUM].max()
        print(f'\n== FLIP Eq. 14 / 15 per target, scope {scope}')
        print(piv.round(3).to_string())
        i = summarize(inc, ['source', 'threshold'], scope)
        i['scope'] = scope
        if kept['income'] is not None:
            i = pd.concat([i, kept['income'][kept['income'].scope == scope]])
        out_i.append(i)
        print(f'\n== income: parity and EO candidates, scope {scope}')
        print(i.round(3).to_string(index=False))
    pd.concat(out_t).round(4).to_csv(paths['targets'], index=False)
    pd.concat(out_i).round(4).to_csv(paths['income'], index=False)


if __name__ == '__main__':
    main()
