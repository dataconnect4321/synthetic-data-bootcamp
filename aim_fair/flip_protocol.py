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

Sources: real training fold; plain AIM and AIM hybrid (3 folds x 3 refits,
run_adult_refits.sh); FLIP faithful reproduction (lambda 4, nominal eps 3;
fold 0 only, the table that exists). Rows marked fold0 restrict every source
to fold 0 for the like-for-like FLIP row.

    aim_fair/.venv-aif360/Scripts/python aim_fair/flip_protocol.py   # lightgbm
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
    ('Plain AIM', 'aim_fair/out/aimreplain_e3_f{f}_r{r}_synthetic.csv',
     [0, 1, 2], [0, 1, 2]),
    ('AIM hybrid', 'aim_fair/out/aimre_e3_f{f}_r{r}_synthetic.csv',
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


def fold_job(f):
    test = read(f'aim_fair/data/fold{f}_test.csv')
    real = read(f'aim_fair/data/fold{f}_train.csv')
    male = (test[SEX] == 'Male').to_numpy()
    sd = {c: real[c].std() for c in NUM}
    t_rows, i_rows, k = [], [], None
    for name, pat, refits, folds in SOURCES:
        if f not in folds:
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
    parts = Parallel(n_jobs=3)(delayed(fold_job)(f) for f in range(3))
    targets = pd.concat([p[0] for p in parts])
    inc = pd.concat([p[1] for p in parts])
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 50)
    out_t, out_i = [], []
    for scope in ('fold0', 'all'):
        t = summarize(targets, ['source', 'target', 'metric'], scope)
        t['scope'] = scope
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
        out_i.append(i)
        print(f'\n== income: parity and EO candidates, scope {scope}')
        print(i.round(3).to_string(index=False))
    pd.concat(out_t).round(4).to_csv(
        os.path.join(ROOT, 'results', 'flip_protocol_targets.csv'), index=False)
    pd.concat(out_i).round(4).to_csv(
        os.path.join(ROOT, 'results', 'flip_protocol_income.csv'), index=False)


if __name__ == '__main__':
    main()
