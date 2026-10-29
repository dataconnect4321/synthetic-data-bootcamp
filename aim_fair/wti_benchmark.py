"""Task-specific fairness as benchmarked in "Where to Intervene?" (Angelozzi
& Arcolezi, PoPETs 2026, arXiv 2607.07471; code: github.com/vinicius-verona/
dp-fair-intervention-benchmark) vs the task-agnostic AIM hybrid, on ACS-I
Utah preprocessed as in the paper (datasets.py: acs_ut_wti).

Faithful to the paper's code:
  splits     per seed, train_test_split(test_size=0.4, random_state=seed),
             then the 40% split 50 / 50 into calibration / test (same call);
             AIM synthesizes the 60% training split only
  encoding   every column an integer code (OCCP group, POBP raw code,
             RAC1P White alone = 1, SEX Male = 1 = privileged); features
             are ALL columns but the target, sex included, MinMax-scaled
             on the training table
  methods    AIF360, their settings:
               pre   Reweighing, LFR (defaults), DisparateImpactRemover
                     (repair level 1, fit_transform on train, calibration
                     and test separately)
               in    ExponentiatedGradientReduction and GridSearchReduction,
                     'EqualizedOdds', drop_prot_attr=False
               post  EqOddsPostprocessing, RejectOptionClassification
                     (defaults), CalibratedEqOddsPostprocessing (fnr),
                     fitted on the real calibration split
  models     XGBoost (binary:logistic), random forest (defaults), logistic
             regression (saga, elasticnet, l1_ratio 0.5, C 0.8,
             max_iter 10000)
  metrics    AIF360 ClassificationMetric, unprivileged = female: accuracy,
             F1, SPD, EOD (TPR), MAD (accuracy), signed female - male

Sources: Real (the paper's Fair-only baseline), plain AIM (DP-only, then
DP+Fair with each method) and the AIM hybrid (task-agnostic, also with each
method), all AIM tables from ONE set of measurements per seed (epsilon 3).
Deviations: AIM from mbi instead of SmartNoise (AGEP / WKHP privately
binned, not ordinal / categorical codes); 5 seeds and epsilon 3 only. AIF360
0.6.1's LFR passes `disp` to scipy's fmin_l_bfgs_b, which scipy >= 1.15 no
longer accepts; a shim drops that argument (the optimization is unchanged).
(seed, source) jobs run in parallel (WTI_JOBS, default 6), XGBoost and
random forest with 2 threads each.

    python aim_fair/wti_benchmark.py export                    # main env
    bash aim_fair/run_wti_benchmark.sh                         # AIM + eval
    aim_fair/.venv-aif360/Scripts/python aim_fair/wti_benchmark.py evaluate
    python aim_fair/wti_benchmark.py combine     # after scoring (below)
Writes results/wti_benchmark_{runs,summary}.csv.

combine: one apples-to-apples table on the same seed splits, the paper's
metrics under our names (parity = |SPD|, EO = |EOD|, accuracy gap = |MAD|;
absolute, averaged over the 3 classifiers and 5 seeds) next to ours, which
describe the RELEASE and so exist only for the tables (real, plain AIM,
hybrid), not for a classifier-level fix:
  run_adult_flip.py --dataset acs_ut_wti --split-csvs <train> <test>
      --score-csv <table>: BER, A-NCB (paper's form), folded tstr, outcome
      AUC, validity, impossible rows, alpha / beta, epsilon-identifiability
  flip_protocol.py --dataset acs_ut_wti: mean parity over all categorical
      targets (Eq. 14), mean W1 / sd over all numeric targets (Eq. 15)
LFR is left out: with AIF360's defaults it maps every label to one class
(an error in 39 of 45 runs, as in the paper's code; degenerate otherwise).
Writes results/wti_benchmark_comparison.csv.
"""
import argparse
import os
import sys
import time
import types
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
DS, SEX, TARGET = 'acs_ut_wti', 'SEX', 'income'
SEEDS = [0, 1, 2, 3, 4]
COLS = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX',
        'RAC1P', 'income']


def path(kind, s):
    return os.path.join(HERE, 'data', f'{DS}_fold{s}_{kind}.csv')


def synth(tag, s):
    return os.path.join(HERE, 'out', f'{DS}_{tag}_e3_s{s}_r0_synthetic.csv')


def export():
    from sklearn.model_selection import train_test_split
    sys.path.insert(0, ROOT)
    import datasets as D
    df = D.load(DS)[COLS]
    for s in SEEDS:
        train, rest = train_test_split(df, test_size=0.4, random_state=s)
        calib, test = train_test_split(rest, test_size=0.5, random_state=s)
        for kind, part in (('train', train), ('calib', calib),
                           ('test', test)):
            part.to_csv(path(kind, s), index=False)
        print(f'seed {s}: {len(train)} train / {len(calib)} calibration / '
              f'{len(test)} test')


def encode(d):
    """The paper's integer coding; SEX Male = 1 (privileged)."""
    out = pd.DataFrame(index=d.index)
    for c in COLS:
        v = d[c].astype(str)
        if c == SEX:
            out[c] = (v == 'Male').astype(int)
        elif c == 'POBP' or c == 'OCCP':     # codes; MASK (if any) = -1
            out[c] = pd.to_numeric(v, errors='coerce').fillna(-1).astype(int)
        else:
            out[c] = pd.to_numeric(v, errors='coerce').fillna(-1).round()
    return out.astype(float)


def classifiers():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    return {
        'XGB': lambda s: XGBClassifier(objective='binary:logistic',
                                       random_state=s, n_jobs=2),
        'RF': lambda s: RandomForestClassifier(random_state=s, n_jobs=2),
        'LR': lambda s: LogisticRegression(
            random_state=s, max_iter=10000, solver='saga',
            penalty='elasticnet', l1_ratio=0.5, C=0.8)}


def bld(x, y):
    from aif360.datasets import BinaryLabelDataset
    d = pd.concat([x.reset_index(drop=True),
                   pd.Series(np.asarray(y), name=TARGET)], axis=1)
    return BinaryLabelDataset(df=d, label_names=[TARGET],
                              protected_attribute_names=[SEX])


def metrics(true_ds, labels, scores=None):
    from aif360.metrics import ClassificationMetric
    pred = true_ds.copy(deepcopy=True)
    pred.labels = np.asarray(labels).reshape(-1, 1).astype(float)
    if scores is not None:
        pred.scores = np.asarray(scores).reshape(-1, 1)
    m = ClassificationMetric(true_ds, pred, unprivileged_groups=[{SEX: 0}],
                             privileged_groups=[{SEX: 1}])
    prec, rec = m.precision(), m.recall()
    return {'acc': m.accuracy(), 'f1': 2 * prec * rec / max(prec + rec, 1e-12),
            'sel': float(np.mean(labels)),
            'spd': m.statistical_parity_difference(),
            'eod': m.equal_opportunity_difference(),
            'mad': m.accuracy(privileged=False) - m.accuracy(privileged=True)}


def run_source(tr, cal, te, make, seed, lfr_cache):
    """Original model + the eight AIF360 mitigators, the paper's code paths;
    returns {method: metrics}."""
    from aif360.algorithms.inprocessing import (ExponentiatedGradientReduction,
                                                GridSearchReduction)
    from aif360.algorithms.postprocessing import (
        CalibratedEqOddsPostprocessing, EqOddsPostprocessing,
        RejectOptionClassification)
    import scipy.optimize as so
    from aif360.algorithms.preprocessing import lfr as lfr_mod
    from aif360.algorithms.preprocessing import (LFR, DisparateImpactRemover,
                                                 Reweighing)
    lfr_mod.optim = types.SimpleNamespace(    # scipy >= 1.15: no `disp`
        fmin_l_bfgs_b=lambda *a, disp=None, **k: so.fmin_l_bfgs_b(*a, **k))
    from sklearn.preprocessing import MinMaxScaler
    priv, unpriv = [{SEX: 1}], [{SEX: 0}]
    feats = [c for c in COLS if c != TARGET]
    xtr, ytr = tr[feats], tr[TARGET].astype(int)
    xca, yca = cal[feats], cal[TARGET].astype(int)
    xte, yte = te[feats], te[TARGET].astype(int)
    sc = MinMaxScaler().fit(xtr)
    S = lambda x: pd.DataFrame(sc.transform(x), columns=feats)   # noqa: E731
    xtr_s, xca_s, xte_s = S(xtr), S(xca), S(xte)
    dtr, dca, dte = bld(xtr_s, ytr), bld(xca_s, yca), bld(xte_s, yte)
    out, t0 = {}, time.time()

    def done(name, m):
        nonlocal t0
        out[name] = {**m, 'secs': time.time() - t0}
        t0 = time.time()

    base = make(seed).fit(xtr_s, ytr.to_numpy())
    p_te = base.predict_proba(xte_s)[:, 1]
    p_ca = base.predict_proba(xca_s)[:, 1]
    done('none', metrics(dte, p_te >= 0.5, p_te))

    # pre-processing
    for name in ('reweigh', 'dir', 'lfr'):
        try:
            if name == 'reweigh':
                rw = Reweighing(unprivileged_groups=unpriv,
                                privileged_groups=priv).fit(dtr)
                mtr, mte = rw.transform(dtr), dte
            elif name == 'dir':
                mtr = DisparateImpactRemover(
                    repair_level=1, sensitive_attribute=SEX).fit_transform(dtr)
                mte = DisparateImpactRemover(
                    repair_level=1, sensitive_attribute=SEX).fit_transform(dte)
            else:
                if 'lfr' not in lfr_cache:
                    lf = LFR(unprivileged_groups=unpriv,
                             privileged_groups=priv, seed=seed,
                             verbose=0).fit(dtr)
                    lfr_cache['lfr'] = (lf.transform(dtr), lf.transform(dte))
                mtr, mte = lfr_cache['lfr']
            m = make(seed).fit(pd.DataFrame(mtr.features, columns=feats),
                               mtr.labels.ravel(),
                               sample_weight=mtr.instance_weights)
            p = m.predict_proba(pd.DataFrame(mte.features, columns=feats))[:, 1]
            done(name, metrics(dte, p >= 0.5, p))
        except Exception as e:                       # noqa: BLE001
            print(f'    {name} failed: {type(e).__name__}: {e}', flush=True)
            t0 = time.time()

    # in-processing
    for name, cls in (('egr', ExponentiatedGradientReduction),
                      ('gsr', GridSearchReduction)):
        try:
            kw = {'prot_attr': SEX} if name == 'gsr' else {}
            red = cls(make(seed), 'EqualizedOdds', drop_prot_attr=False,
                      **kw).fit(dtr)
            pr = red.predict(dte)
            done(name, metrics(dte, pr.labels.ravel().astype(int)))
        except Exception as e:                       # noqa: BLE001
            print(f'    {name} failed: {type(e).__name__}: {e}', flush=True)
            t0 = time.time()

    # post-processing, fitted on the real calibration split
    dca_pred, dte_pred = dca.copy(deepcopy=True), dte.copy(deepcopy=True)
    dca_pred.scores = p_ca.reshape(-1, 1)
    dca_pred.labels = (p_ca >= 0.5).astype(float).reshape(-1, 1)
    dte_pred.scores = p_te.reshape(-1, 1)
    dte_pred.labels = (p_te >= 0.5).astype(float).reshape(-1, 1)
    for name in ('eqodds', 'roc', 'ceop'):
        try:
            if name == 'eqodds':
                pp = EqOddsPostprocessing(privileged_groups=priv,
                                          unprivileged_groups=unpriv,
                                          seed=seed)
            elif name == 'roc':
                pp = RejectOptionClassification(privileged_groups=priv,
                                                unprivileged_groups=unpriv)
            else:
                pp = CalibratedEqOddsPostprocessing(
                    privileged_groups=priv, unprivileged_groups=unpriv,
                    seed=seed, cost_constraint='fnr')
            pr = pp.fit(dca, dca_pred).predict(dte_pred)
            done(name, metrics(dte, pr.labels.ravel()))
        except Exception as e:                       # noqa: BLE001
            print(f'    {name} failed: {type(e).__name__}: {e}', flush=True)
            t0 = time.time()
    return out


def job(s, src, file):
    warnings.filterwarnings('ignore')
    tr = encode(pd.read_csv(file, dtype=str)[COLS])
    cal = encode(pd.read_csv(path('calib', s), dtype=str))
    te = encode(pd.read_csv(path('test', s), dtype=str))
    rows, lfr_cache = [], {}
    for clf, make in classifiers().items():
        res = run_source(tr, cal, te, make, s, lfr_cache)
        for method, m in res.items():
            rows.append({'seed': s, 'source': src, 'classifier': clf,
                         'method': method, **m})
        print(f'seed {s} {src} {clf} done', flush=True)
    return rows


def evaluate():
    from joblib import Parallel, delayed
    jobs = []
    for s in SEEDS:
        jobs.append((s, 'Real data', path('train', s)))
        for name, tag in (('Plain AIM', 'aimplain'), ('AIM hybrid', 'aim')):
            if os.path.exists(synth(tag, s)):
                jobs.append((s, name, synth(tag, s)))
    parts = Parallel(n_jobs=int(os.environ.get('WTI_JOBS', 6)), verbose=5)(
        delayed(job)(*j) for j in jobs)
    df = pd.DataFrame([r for p in parts for r in p])
    order = {'Real data': 0, 'Plain AIM': 1, 'AIM hybrid': 2}
    df = df.sort_values(['source', 'seed'], key=lambda c: c.map(order)
                        if c.name == 'source' else c, kind='stable')
    df.to_csv(os.path.join(ROOT, 'results', 'wti_benchmark_runs.csv'),
              index=False)
    keys = ['classifier', 'source', 'method']
    g = df.groupby(keys, sort=False)
    summ = g.mean(numeric_only=True).drop(columns='seed').join(
        g.std(numeric_only=True).drop(columns='seed').add_suffix('_sd'))
    summ = summ.reset_index()
    summ.round(4).to_csv(os.path.join(ROOT, 'results',
                                      'wti_benchmark_summary.csv'),
                         index=False)
    pd.set_option('display.width', 250)
    print(summ[keys + ['acc', 'f1', 'sel', 'spd', 'eod', 'mad']].round(3)
          .to_string(index=False))


METHODS = [('none', 'none'), ('reweigh', 'Reweighing (pre)'),
           ('dir', 'Disparate Impact Remover (pre)'),
           ('egr', 'Exponentiated Gradient (in)'),
           ('gsr', 'Grid Search (in)'),
           ('eqodds', 'Equalized Odds (post)'),
           ('roc', 'Reject Option (post)'),
           ('ceop', 'Calibrated Equalized Odds (post)')]
RELEASE = [   # key, scorer getter
    ('ber', lambda d: d['paper_metrics']['syn']['ber']),
    ('ancb', lambda d: d['paper_metrics']['syn_own_adversary']['a_ncb']),
    ('tstr', lambda d: d['paper_metrics']['tstr_sex_ber_folded']),
    ('auc', lambda d: d['paper_metrics']['downstream_auc']),
    ('validity', lambda d: d['paper_metrics']['validity']),
    ('impossible',
     lambda d: d['paper_metrics']['impossible_combination_rate']['any']),
    ('alpha', lambda d: d['paper_metrics']['alpha_precision']),
    ('beta', lambda d: d['paper_metrics']['beta_recall']),
    ('eps_id', lambda d: d['paper_metrics']['eps_identifiability'])]


def combine():
    import json
    runs = pd.read_csv(os.path.join(ROOT, 'results', 'wti_benchmark_runs.csv'))
    runs = runs[runs.method != 'lfr']
    for c in ('spd', 'eod', 'mad'):
        runs[c] = runs[c].abs()
    task = runs.groupby(['source', 'method']).agg(
        accuracy=('acc', 'mean'), f1=('f1', 'mean'), parity=('spd', 'mean'),
        eo=('eod', 'mean'), accuracy_gap=('mad', 'mean'))
    tg = pd.read_csv(os.path.join(ROOT, 'results',
                                  'flip_protocol_targets_acs_ut_wti.csv'))
    tg = tg[tg.scope == 'all']
    tags = {'Real data': DS + '_real_s{s}', 'Plain AIM': DS + '_aimplain_e3_s{s}_r0',
            'AIM hybrid': DS + '_aim_e3_s{s}_r0'}
    rows = []
    for src, tag in tags.items():
        js = [json.load(open(os.path.join(ROOT, 'adult_results',
                                          tag.format(s=s) + '.json')))
              for s in SEEDS]
        rel = {k: float(np.mean([g(d) for d in js])) for k, g in RELEASE}
        t = tg[tg.source == src]
        rel['mean_parity_all_targets'] = t[t.metric == 'SP (Eq. 14)'].value_sd.mean()
        rel['mean_w1_all_numeric_targets'] = t[t.metric == 'W1'].value_sd.mean()
        for m, label in METHODS:
            if (src, m) not in task.index:
                continue
            rows.append({'source': src, 'method': label,
                         **task.loc[(src, m)].to_dict(),
                         **(rel if m == 'none' else {})})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(os.path.join(ROOT, 'results',
                                    'wti_benchmark_comparison.csv'), index=False)
    pd.set_option('display.width', 300)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['export', 'evaluate', 'combine'])
    {'export': export, 'evaluate': evaluate,
     'combine': combine}[ap.parse_args().step]()
