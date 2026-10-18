"""Final comparison table (FINAL_COMPARISON.md) from the scored runs in
adult_results/: AIM hybrid and plain AIM (run_final.sh) vs the best VAE
process (scripts/run_paper_grid.sh with PRESET=final, and
scripts/run_best_vae_dp.sh). Mean and sd across folds; AIM refits are
averaged within a fold first. Writes adult_results/final_aim_vs_vae.csv.

    python aim_fair/summarize.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'adult_results')
METRICS = {
    'income_parity': lambda p: p['downstream_sp_eo']['sp_diff'],
    'eo_gap': lambda p: p['downstream_sp_eo']['eo_diff'],
    'auc': lambda p: p['downstream_auc'],
    'validity': lambda p: p['validity'],
    'impossible_rows': lambda p: p['impossible_combination_rate']['any'],
    'income_rate': lambda p: p['syn_income_rate'],
    'ber_synthetic_label': lambda p: p['syn']['ber'],
    'tstr_folded': lambda p: p['tstr_sex_ber_folded'],
    'alpha_precision': lambda p: p['alpha_precision'],
    'beta_recall': lambda p: p['beta_recall'],
    'eps_identifiability': lambda p: p['eps_identifiability'],
    'num_w1_mean': lambda p: p['_fid']['num_w1_mean'],
    'ancb_real_trained': lambda p: p['syn']['a_ncb'],
    'ancb_own_s': lambda p: p['syn_own_adversary']['a_ncb'],
}
METHODS = [   # name, tag pattern, refits
    ('AIM hybrid, eps 3 (private bins)', 'aimfinal_e3_f{f}_r{r}', 10),
    ('AIM plain, eps 3 (private bins)', 'aimfinalplain_e3_f{f}_r{r}', 10),
    ('AIM hybrid, eps 3 (public bins)', 'aimpublic_e3_f{f}_r{r}', 10),
    ('AIM plain, eps 3 (public bins)', 'aimpublicplain_e3_f{f}_r{r}', 10),
    ('VAE final lambda 4, no DP', 'final_f{f}_lam4', 1),
    ('VAE final lambda 4, eps 3 (nominal)', 'best_vae_e3_f{f}_lam4', 1),
]


def scores(tag):
    path = os.path.join(RES, tag + '.json')
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    p = dict(d['paper_metrics'], _fid=d.get('fidelity') or {'num_w1_mean': np.nan})
    return [fn(p) for fn in METRICS.values()]


rows = []
for name, pattern, refits in METHODS:
    per_fold = []
    for f in range(3):
        v = [scores(pattern.format(f=f, r=r)) for r in range(refits)]
        v = [x for x in v if x is not None]
        if v:
            per_fold.append(np.mean(v, axis=0))
    if not per_fold:
        continue
    per_fold = np.array(per_fold)
    mean = per_fold.mean(0)
    sd = per_fold.std(0, ddof=1) if len(per_fold) > 1 else np.zeros_like(mean)
    row = {'method': name, 'folds': len(per_fold)}
    for k, m, s in zip(METRICS, mean, sd):
        row[k], row[k + '_sd'] = round(float(m), 4), round(float(s), 4)
    rows.append(row)
    print(f'{name:38s} folds={len(per_fold)}  ' + '  '.join(
        f'{k}={m:.3f}' for k, m in zip(list(METRICS)[:6], mean[:6])))

if rows:
    out = os.path.join(RES, 'final_aim_vs_vae.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('->', out)
