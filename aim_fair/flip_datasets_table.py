"""FLIP-protocol table across the FLIP paper's datasets: UCI Adult, ACS-I
Alabama, ACS-I Utah and the Dutch census, sex protected in all.

Per dataset: real training data (reference), plain AIM and the AIM hybrid,
3 folds x 3 refits (refits averaged within a fold, then folds). Adult also
has our faithful FLIP reproduction (fold 0 only, the table that exists) and
the FLIP paper's reported Fig. 3 values (Adult only).

  data level      BER, A-NCB (paper's own-adversary form), folded tstr,
                  utility and realism: run_adult_flip.py --score-csv JSONs
  downstream      flip_protocol.py (LightGBM, every non-protected attribute a
                  target; Eq. 14 parity, Eq. 15 W1 / sd), outcome at 0.5:
                  income for Adult and ACS-I, high-level occupation for Dutch

Writes results/flip_protocol_datasets.csv.

    python aim_fair/flip_datasets_table.py
"""
import json
import os

import numpy as np
import pandas as pd

from flip_faithful_chart import PAPER

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
PM = 'paper_metrics'
DATASETS = [('adult', 'UCI Adult'), ('acs_al', 'ACS-I Alabama'),
            ('acs_ut', 'ACS-I Utah'), ('dutch', 'Dutch census')]
JSON_METRICS = [   # key, label, getter
    ('ber', 'BER, synthetic sex label (0.5 best)',
     lambda d: d[PM]['syn']['ber']),
    ('ancb', "A-NCB, paper's version (higher better)",
     lambda d: d[PM]['syn_own_adversary']['a_ncb']),
    ('tstr', 'Release teaches real sex, folded tstr (0.5 best)',
     lambda d: d[PM]['tstr_sex_ber_folded']),
    ('auc', 'Outcome AUC on real people',
     lambda d: d[PM]['downstream_auc']),
    ('alpha', 'Alpha-precision', lambda d: d[PM]['alpha_precision']),
    ('beta', 'Beta-recall', lambda d: d[PM]['beta_recall']),
    ('validity', 'Validity', lambda d: d[PM]['validity']),
    ('impossible', 'Impossible rows',
     lambda d: d[PM]['impossible_combination_rate']['any']),
    ('eps_id', 'Epsilon-identifiability',
     lambda d: d[PM]['eps_identifiability']),
]
PROTO_METRICS = [
    ('parity', 'Outcome parity, threshold 0.5 (LightGBM)'),
    ('eo', 'Outcome EO gap (TPR), threshold 0.5 (LightGBM)'),
    ('sp_cat', 'Mean parity, all categorical targets (Eq. 14)'),
    ('w1_num', 'Mean W1 / sd, all numeric targets (Eq. 15)'),
]


def sources(d):
    """name, flip_protocol.py source, scorer tag pattern, refits, folds"""
    if d == 'adult':
        return [('Real data', 'Real data', 'adult_real_f{f}', [None],
                 [0, 1, 2]),
                ('Plain AIM', 'Plain AIM', 'aimreplain_e3_f{f}_r{r}',
                 [0, 1, 2], [0, 1, 2]),
                ('AIM hybrid', 'AIM hybrid', 'aimre_e3_f{f}_r{r}', [0, 1, 2],
                 [0, 1, 2]),
                ('FLIP reproduction (fold 0)',
                 'FLIP reproduction (lambda 4, nominal eps 3)',
                 'flipfaithful_e3_f0_lam4', [None], [0])]
    return [('Real data', 'Real data', d + '_real_f{f}', [None], [0, 1, 2]),
            ('Plain AIM', 'Plain AIM', d + '_aimplain_e3_f{f}_r{r}',
             [0, 1, 2], [0, 1, 2]),
            ('AIM hybrid', 'AIM hybrid', d + '_aim_e3_f{f}_r{r}', [0, 1, 2],
             [0, 1, 2])]


def json_values(pattern, refits, folds):
    per_fold = []
    for f in folds:
        runs = []
        for r in refits:
            p = os.path.join(ROOT, 'adult_results',
                             pattern.format(f=f, r=r) + '.json')
            if os.path.exists(p):
                with open(p) as fh:
                    runs.append(json.load(fh))
        if runs:
            per_fold.append({k: np.mean([g(x) for x in runs])
                             for k, _, g in JSON_METRICS})
    return {k: np.mean([v[k] for v in per_fold]) for k, *_ in JSON_METRICS} \
        if per_fold else {}


def proto_values(d, proto):
    suf = '' if d == 'adult' else '_' + d
    t = pd.read_csv(os.path.join(ROOT, 'results',
                                 f'flip_protocol_targets{suf}.csv'))
    o = pd.read_csv(os.path.join(ROOT, 'results', 'flip_protocol_income.csv'
                                 if d == 'adult' else
                                 f'flip_protocol_outcome{suf}.csv'))
    t = t[(t.scope == 'all') & (t.source == proto)]
    o = o[(o.scope == 'all') & (o.source == proto)
          & (o.threshold.astype(str) == '0.5')]
    num = t[t.metric == 'W1']
    return {'parity': o.parity.iloc[0], 'eo': o['EO = TPR gap'].iloc[0],
            'sp_cat': t[t.metric == 'SP (Eq. 14)'].value_sd.mean(),
            'w1_num': num.value_sd.mean() if len(num) else np.nan}


def main():
    rows = []
    for d, dname in DATASETS:
        for name, proto, pattern, refits, folds in sources(d):
            rows.append({'dataset': dname, 'source': name, 'folds': len(folds),
                         **json_values(pattern, refits, folds),
                         **proto_values(d, proto)})
        rows.append({'dataset': dname, 'source': 'FLIP paper (reported)',
                     'folds': np.nan,
                     **({k: PAPER.get(k, np.nan) for k, *_ in
                         JSON_METRICS + PROTO_METRICS}
                        if d == 'adult' else {})})
    df = pd.DataFrame(rows)
    df = df[~((df.source == 'FLIP paper (reported)')
              & df[[k for k, *_ in JSON_METRICS]].isna().all(axis=1))]
    df.round(4).to_csv(os.path.join(ROOT, 'results',
                                    'flip_protocol_datasets.csv'), index=False)
    pd.set_option('display.width', 250)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
