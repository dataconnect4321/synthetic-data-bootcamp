"""3-fold x lambda table for a lambda grid of run_adult_flip.py runs
(PREFIX_f{fold}_lam{lambda}{SUFFIX}), e.g. the paper grid of
scripts/run_paper_grid.sh. Run metrics come from each run JSON; every
synthetic table is also re-scored for the A-NCB sensitivity analysis:
the sex-inferring adversary trained on the table's own GENERATED sex column
(the paper's wording) or on the REAL training fold (this repository), each
with a full- and a diagonal-covariance GMM (mean of 5 seeds), plus plain NCB
on the generated sex. Prints mean +/- sd over folds per lambda next to the
real held-out fold and writes adult_results/PREFIX{SUFFIX|_full}_grid_report.csv.

    python paper_grid_report.py [PREFIX] [SUFFIX]   # defaults: pgrid _t1
    python paper_grid_report.py pgrid ""            # full-pipeline runs
    python paper_grid_report.py final _t1 0,2,4     # another prefix / lambdas
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, 'tabsyn-main/tabsyn-main')
sys.path.insert(0, '.')
import run_adult_flip as R
from tabsyn.vae.paper_metrics import fairness_scores

PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'pgrid'
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else '_t1'   # '' = full pipeline
LAMS = ([int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3
        else [0, 2, 4, 8, 12])
FOLDS = [0, 1, 2]
NUM, CAT = R.NUMERIC, R.CATEGORICAL
s_idx = CAT.index('sex')
SEEDS = range(5)


def variants(num, cat, ref):
    out = {}
    for cov in ('full', 'diag'):
        own = [fairness_scores(num, cat, s_idx, ref=None, seed=k,
                               covariance_type=cov) for k in SEEDS]
        rl = [fairness_scores(num, cat, s_idx, ref=ref, seed=k,
                              covariance_type=cov) for k in SEEDS]
        out[f'A-NCB ownS {cov}'] = np.mean([r['a_ncb'] for r in own])
        out[f'A-NCB real {cov}'] = np.mean([r['a_ncb'] for r in rl])
        if cov == 'full':
            out['NCB genS'] = np.mean([r['ncb_stored'] for r in own])
    return out


rows = []
for f in FOLDS:
    data = R.prepare(R.read_adult(), f, 3, 42)
    enc = data['encoders']
    tr = (data['train_num'], data['train_cat'])
    te = (data['test_num'], data['test_cat'])
    ref_row = {'fold': f, 'lam': 'real'}
    ref_row.update(variants(*te, ref=tr))
    ref_row['BER (paper)'] = fairness_scores(*te, s_idx, ref=None)['ber']
    rows.append(ref_row)
    for lam in LAMS:
        tag = f'{PREFIX}_f{f}_lam{lam}{SUFFIX}'
        d = json.load(open(f'adult_results/{tag}.json'))
        pm, fid = d['paper_metrics'], d['fidelity']
        s = pd.read_csv(f'adult_results/{tag}_synthetic.csv')
        num = (s[NUM].values - data['means'].values) / data['stds'].values
        cat = np.stack([s[c].astype(str).map({str(v): i for i, v in
                                              enumerate(enc[c])}).values
                        for c in CAT], axis=1).astype(int)
        r = {'fold': f, 'lam': lam,
             'BER (paper)': pm['syn']['ber'],
             'true-sex BER': (1 - pm['decoded_leakage'] if pm.get('decoded_leakage') is not None else np.nan),
             'eps-id': (pm['eps_identifiability'] if pm.get('eps_identifiability') is not None else np.nan),
             'AUC': pm['downstream_auc'],
             'alpha': pm['alpha_precision'], 'beta': pm['beta_recall'],
             'validity': pm['validity'], 'W1': fid['num_w1_mean'],
             'SP': pm['downstream_sp_eo']['sp_diff'],
             'EO': pm['downstream_sp_eo']['eo_diff'],
             'sex=true': pm['synthetic_sex_agreement'],
             'collapsed': bool(d.get('collapsed'))}
        r.update(variants(num, cat, ref=tr))
        rows.append(r)
    print(f'fold {f} scored', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'adult_results/{PREFIX}{SUFFIX or "_full"}_grid_report.csv', index=False)
cols = ['BER (paper)', 'A-NCB ownS full', 'A-NCB real full', 'NCB genS',
        'A-NCB ownS diag', 'A-NCB real diag', 'true-sex BER', 'sex=true',
        'eps-id', 'AUC', 'alpha', 'beta', 'validity', 'W1', 'SP', 'EO']
g = df.groupby(df['lam'].astype(str), sort=False)[
    [c for c in cols if c in df]].agg(['mean', 'std'])
order = ['real'] + [str(l) for l in LAMS]
g = g.loc[order]
pd.set_option('display.width', 260)
fmt = pd.DataFrame({c: [f"{g.loc[i, (c, 'mean')]:.3f} +/- "
                        f"{(g.loc[i, (c, 'std')] if not np.isnan(g.loc[i, (c, 'std')]) else 0):.3f}"
                        if not np.isnan(g.loc[i, (c, 'mean')]) else '-'
                        for i in order] for c in cols if c in df}, index=order)
print('\nmean +/- sd over 3 folds ('
      + ('tier-1' if SUFFIX else 'full pipeline') + '; real = held-out fold)')
print(fmt.T.to_string())
print('collapsed runs:', int(df['collapsed'].fillna(False).sum()))
