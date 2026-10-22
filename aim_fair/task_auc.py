"""Task independence (RESULTS.md): train on synthetic, test on real, for
several Adult targets. Sources per fold: real training data (ceiling), plain
AIM and the AIM hybrid (run_final.sh tables, same measurements, first 3
refits). Features:
every other column (relationship Husband/Wife merged to MASK everywhere so
all sources share one vocabulary); for 'married' the relationship column is
dropped (it encodes marriage directly). Metric: ROC AUC on the real test
fold (macro one-vs-rest for occupation). Writes results/task_auc.csv.

    python aim_fair/task_auc.py [--hybrid aimfinal --plain aimfinalplain]
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NUM = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
       'hours-per-week']
CAT = ['sex', 'income', 'workclass', 'marital-status', 'occupation',
       'relationship', 'race', 'native-country']
TARGETS = {   # name: (label function, columns to drop besides the source)
    'income >50K': (lambda d: d['income'].astype(str) == '1', ['income']),
    'married': (lambda d: d['marital-status'] == 'Married-civ-spouse',
                ['marital-status', 'relationship']),
    'college degree (edu >= 13)': (lambda d: d['education-num'] >= 13,
                                   ['education-num']),
    'age >= 40': (lambda d: d['age'] >= 40, ['age']),
    'long hours (>= 45)': (lambda d: d['hours-per-week'] >= 45,
                           ['hours-per-week']),
    'capital gain > 0': (lambda d: d['capital-gain'] > 0, ['capital-gain']),
    'private sector': (lambda d: d['workclass'] == 'Private', ['workclass']),
    'race White': (lambda d: d['race'] == 'White', ['race']),
    'occupation (14 classes)': (lambda d: d['occupation'], ['occupation']),
    'sex (removed by design)': (lambda d: d['sex'] == 'Male', ['sex']),
}


def prep(d):
    d = d.copy()
    d['relationship'] = d['relationship'].replace(
        {'Husband': 'MASK', 'Wife': 'MASK'})
    d['income'] = d['income'].astype(str)
    return d


def auc(train, test, name):
    lab, drop = TARGETS[name]
    feats = [c for c in NUM + CAT if c not in drop]
    cats = [c for c in feats if c in CAT]
    vocab = {c: sorted(set(train[c].astype(str)) | set(test[c].astype(str)))
             for c in cats}

    def X(d):
        out = d[feats].copy()
        for c in cats:
            out[c] = pd.Categorical(out[c].astype(str), categories=vocab[c]).codes
        return out
    ytr, yte = lab(train), lab(test)
    clf = HistGradientBoostingClassifier(categorical_features=[
        feats.index(c) for c in cats], random_state=0)
    clf.fit(X(train), ytr)
    p = clf.predict_proba(X(test))
    if p.shape[1] == 2:
        return roc_auc_score(yte, p[:, 1])
    classes = list(clf.classes_)
    keep = yte.isin(classes)
    return roc_auc_score(yte[keep], p[keep.to_numpy()], multi_class='ovr',
                         average='macro', labels=classes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hybrid', default='aimfinal')
    ap.add_argument('--plain', default='aimfinalplain')
    a = ap.parse_args()
    rows = []
    for f in range(3):
        real_tr = prep(pd.read_csv(f'{P}/aim_fair/data/fold{f}_train.csv'))
        real_te = prep(pd.read_csv(f'{P}/aim_fair/data/fold{f}_test.csv'))
        sources = {'real': [real_tr],
                   'plain AIM': [prep(pd.read_csv(
                       f'{P}/aim_fair/out/{a.plain}_e3_f{f}_r{r}_synthetic.csv'))
                       for r in range(3)],
                   'AIM hybrid': [prep(pd.read_csv(
                       f'{P}/aim_fair/out/{a.hybrid}_e3_f{f}_r{r}_synthetic.csv'))
                       for r in range(3)]}
        for name in TARGETS:
            for src, tables in sources.items():
                rows.append(dict(fold=f, target=name, source=src,
                                 auc=np.mean([auc(t, real_te, name)
                                              for t in tables])))
        print(f'fold {f} done', flush=True)

    res = pd.DataFrame(rows)
    fold_mean = res.groupby(['target', 'source']).auc.mean().unstack()
    fold_sd = res.groupby(['target', 'source']).auc.std().unstack()
    tab = fold_mean[['real', 'plain AIM', 'AIM hybrid']].copy()
    tab['privacy cost'] = tab['plain AIM'] - tab['real']
    tab['fairness cost'] = tab['AIM hybrid'] - tab['plain AIM']
    tab['hybrid / real'] = (tab['AIM hybrid'] - 0.5) / (tab['real'] - 0.5)
    tab['hybrid sd'] = fold_sd['AIM hybrid']
    pd.set_option('display.width', 200)
    print(tab.round(3).sort_values('fairness cost').to_string())
    os.makedirs(f'{P}/results', exist_ok=True)
    tab.round(4).to_csv(f'{P}/results/task_auc.csv')


if __name__ == '__main__':
    main()
