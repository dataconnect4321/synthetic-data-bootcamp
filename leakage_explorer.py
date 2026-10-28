"""Interactive dashboard of privacy (epsilon), fairness (repair amount tau)
and utility. Writes one self-contained page, results/leakage_explorer.html,
with a dataset picker (Adult and every datasets.py dataset whose fold-0
training split is on disk) and three tabs:

  Scorecard       actual hybrid releases along tau at eps 3, scored
                  (results/utility_tau_grid.csv, scripts/run_tau_grid.sh);
  Leakage         which columns leak the protected attribute, and each
                  column's per-group histogram in a release simulated in the
                  browser: Gaussian noise on each [protected x column] count
                  table (zCDP, rho split evenly), then the repair of
                  aim_fair/fair_aim.py (coupling, repair_plans; pooled
                  target). Not a Private-PGM fit; actual releases (fold 0,
                  refit 0) in aim_fair/out/ are drawn as check points;
  Before vs after the real data vs the actual hybrid release.

    python aim_fair/export_folds.py [--dataset D]   # once, writes aim_fair/data/
    bash scripts/run_tau_grid.sh                    # the scorecard (Adult)
    python leakage_explorer.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'aim_fair'))
import adult_domain as AD  # noqa: E402
import datasets as D  # noqa: E402

DATASETS = {   # name -> label shown in the picker
    'adult': 'UCI Adult',
    'acs_al': 'ACS-I Alabama',
    'acs_ut': 'ACS-I Utah',
    'dutch': 'Dutch census',
}
# page key -> (candidate release csvs in aim_fair/out/, first found wins;
# epsilon, tau). {p} is the dataset's file prefix.
CHECKPOINTS = {
    'plain': (['{p}aimfinalplain_e3_f0_r0_synthetic.csv',
               '{p}aimplain_e3_f0_r0_synthetic.csv',
               '{p}lk_plain_e3_f0_r0_synthetic.csv'], 3, 0),
    'hybrid': (['{p}aimfinal_e3_f0_r0_synthetic.csv',
                '{p}aim_e3_f0_r0_synthetic.csv',
                '{p}lk_e3_f0_r0_synthetic.csv'], 3, 1),
}
# display names of target values (the raw values are '0' / '1')
TARGET_LABELS = {'income': {'0': '≤50K', '1': '>50K'},
                 'occupation': {'0': 'low-level', '1': 'high-level'}}
TOP_K = 10        # columns with many levels: the 10 most common, then "Other"
# proxy levels: the pre-registered rule of tabsyn/vae/proxy_decouple.py
# (detect_proxy_classes), which the AIM hybrid's masking also applies
PROXY_THRESHOLD, PROXY_MIN_SUPPORT = 0.95, 0.01
BER_SEEDS = 10    # all-columns BER: seeds averaged, as the scorer does
ALL_COLUMNS = True   # False skips the classifiers (about a minute for Adult)
OUT = os.path.join(HERE, 'results', 'leakage_explorer.html')
# the scorecard: actual releases along tau (scripts/run_tau_grid.sh,
# aim_fair/tau_grid_summary.py), Adult only
SCORECARD = {'adult': (os.path.join(HERE, 'results', 'utility_tau_grid.csv'), 3)}
# the FLIP baseline: results/final_comparison.csv metric label -> key
FLIP_METRICS = {'Income parity': 'income_parity',
                'Equalized-odds gap': 'eo_gap',
                'BER, synthetic sex label': 'ber',
                "A-NCB (paper's version)*": 'ancb',
                'Income AUC on real people': 'auc',
                'Validity': 'validity', 'Impossible rows': 'impossible'}


def schema(name):
    """(numeric, categorical, protected column, target column), without
    touching adult_domain's module-level lists."""
    if name == 'adult':
        return list(AD.NUMERIC), list(AD.CATEGORICAL), 'sex', 'income'
    cfg = D.CONFIG[name]
    return (list(cfg['numeric']), list(cfg['categorical']),
            cfg['sensitive'], cfg['target'])


def prefix(name):
    return '' if name == 'adult' else name + '_'


def bin_labels(c):
    e = AD.EDGES[c]
    fmt = (lambda v: f'{v // 1000}k') if c == 'fnlwgt' else str
    return [fmt(lo) if hi - lo == 1 else f'{fmt(lo)}–{fmt(hi - 1)}'
            for lo, hi in zip(e[:-1], e[1:])]


def to_levels(df, c, numeric, target, labels=None):
    """Column of df as display levels (numeric bins as their labels)."""
    if c in numeric:
        e = np.asarray(AD.EDGES[c], float)
        x = np.clip(df[c].to_numpy(float).round(), e[0], e[-1] - 1)
        return pd.Series(np.asarray(bin_labels(c))[
            np.searchsorted(e, x, side='right') - 1], index=df.index)
    s = df[c].astype(str)
    if c == target and c in TARGET_LABELS:
        s = s.map(TARGET_LABELS[c]).fillna(s)
    if labels is not None and 'Other' in labels:
        s = s.where(s.isin(labels) | (s == 'MASK'), 'Other')
    return s


def table(df, c, labels, sens, groups, numeric, target):
    """[2, k] counts, rows in `groups` order, columns in `labels` order."""
    lv = to_levels(df, c, numeric, target, labels)
    g = df[sens].astype(str)
    return [[int(((g == s) & (lv == v)).sum()) for v in labels]
            for s in groups]


def proxy_levels(df, c, sens, groups, numeric, target):
    """Levels of categorical column c that nearly determine the group:
    >= PROXY_MIN_SUPPORT of the rows and max_g P(group | level) >=
    PROXY_THRESHOLD, kept only if they point to at least two different
    groups (detect_proxy_classes). Computed on the raw levels, before the
    top-k grouping. Returns [{level, group, purity, support}] or []."""
    lv = to_levels(df, c, numeric, target)
    ct = pd.crosstab(lv, df[sens].astype(str)).reindex(columns=groups,
                                                       fill_value=0)
    n = ct.to_numpy().sum()
    flagged = []
    for level, row in ct.iterrows():
        m = row.sum()
        if m < PROXY_MIN_SUPPORT * n:
            continue
        g = int(row.to_numpy().argmax())
        if row.iloc[g] / m >= PROXY_THRESHOLD:
            flagged.append({'level': level, 'group': groups[g],
                            'purity': round(float(row.iloc[g] / m), 4),
                            'support': round(float(m / n), 4)})
    if len(flagged) >= 2 and len({f['group'] for f in flagged}) >= 2:
        return flagged
    return []


def merged_levels(release_csv):
    """{column: [levels]} the hybrid merged into MASK, from its run_aim.py
    json (refit 0), or None if the json is missing."""
    path = os.path.join(os.path.dirname(release_csv), os.path.basename(
        release_csv).replace('_r0_synthetic.csv', '.json'))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)['refits']['r0']['merged']


def mask(df, merged):
    df = df.copy()
    for c, levels in merged.items():
        df[c] = df[c].astype(str).where(~df[c].astype(str).isin(levels), 'MASK')
    return df


def all_columns_leakage(frames, numeric, categorical, sens):
    """Leakage through all columns at once, with the scorer's classifier
    (tabsyn/vae/paper_metrics.py; run_adult_flip.py --score-csv):
      ber   the protected attribute predicted from all other columns of the
            table itself (3-fold out-of-fold), balanced error, the mean over
            BER_SEEDS classifier seeds as in run_adult_flip.averaged_scores;
      tstr  trained on the table, tested on the real held-out fold (masked
            like the table), folded: 0.5 - |BER - 0.5|.
    0.5 = no leak. frames: [(label, table, its test fold)]. Returns a list
    of rows, or None without the main (torch) environment."""
    if not ALL_COLUMNS:
        return None
    try:
        sys.path.insert(0, os.path.join(HERE, 'tabsyn-main', 'tabsyn-main'))
        from tabsyn.vae.paper_metrics import (balanced_error_rate,
                                              infer_protected,
                                              predict_protected)
    except ImportError as e:
        print(f'  all-columns leakage skipped ({e}): run with the main '
              'environment')
        return None
    tables = [t for _, df, te in frames for t in (df, te)]
    vocab = {c: sorted(set().union(*(set(t[c].astype(str)) for t in tables)))
             for c in categorical}

    def arrays(df):
        num = df[numeric].to_numpy(np.float64).reshape(len(df), len(numeric))
        cat = np.stack([df[c].astype(str).map(
            {v: i for i, v in enumerate(vocab[c])}).to_numpy()
            for c in categorical], 1).astype(np.int64)
        return num, cat

    s = categorical.index(sens)
    rows = []
    for label, df, te in frames:
        num, cat = arrays(df)
        ber = np.mean([balanced_error_rate(cat[:, s],
                                           predict_protected(num, cat, s, seed=k))
                       for k in range(BER_SEEDS)])
        te_num, te_cat = arrays(te)
        b = balanced_error_rate(te_cat[:, s],
                                infer_protected(num, cat, te_num, te_cat, s))
        rows.append({'label': label, 'ber': round(float(ber), 4),
                     'tstr': round(float(0.5 - abs(b - 0.5)), 4)})
        print(f'  {label}: BER {ber:.4f}, folded tstr {0.5 - abs(b - 0.5):.4f}')
    return rows


def flip_baseline():
    """FLIP at lambda 4, eps 3 from results/final_comparison.csv
    (aim_fair/final_chart.py), or None if missing:
      values  the FLIP paper's reported numbers (its Fig. 3, its own
              evaluation code); None where it reports nothing (validity,
              impossible rows);
      repro   FLIP as published, reproduced here (fold 0) and scored like
              the AIM tables."""
    path = os.path.join(HERE, 'results', 'final_comparison.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).set_index('metric')
    paper, repro = df['flip_paper_reported'], df['flip_faithful_repro']
    return {'label': 'FLIP paper', 'note': 'λ 4, nominal ε 3, Fig. 3',
            'values': {k: None if pd.isna(paper[m]) else float(paper[m])
                       for m, k in FLIP_METRICS.items()},
            'repro': {k: float(repro[m]) for m, k in FLIP_METRICS.items()}}


def scorecard(name):
    """{eps, rows, baseline} from the dataset's tau-grid csv, or None if
    missing."""
    if name not in SCORECARD or not os.path.exists(SCORECARD[name][0]):
        return None
    path, eps = SCORECARD[name]
    df = pd.read_csv(path)
    df['tau'] = df['tau'].astype(object).where(df['tau'].notna(), None)
    return {'eps': eps, 'rows': df.to_dict('records'),
            'baseline': flip_baseline() if name == 'adult' else None}


def build(name):
    """Page data for one dataset, or None if its fold-0 split is missing."""
    p = prefix(name)
    train = os.path.join(HERE, 'aim_fair', 'data', f'{p}fold0_train.csv')
    if not os.path.exists(train):
        return None
    numeric, categorical, sens, target = schema(name)
    read = dict(dtype={c: str for c in categorical})
    real = pd.read_csv(train, **read)
    # the two groups, the larger first
    groups = list(real[sens].astype(str).value_counts().index)
    if len(groups) != 2:
        sys.exit(f'{name}: {sens} has {len(groups)} groups; the page '
                 'assumes 2')
    releases = {}
    for key, (names, eps, tau) in CHECKPOINTS.items():
        for n in names:
            path = os.path.join(HERE, 'aim_fair', 'out', n.format(p=p))
            if os.path.exists(path):
                releases[key] = (pd.read_csv(path, **read), eps, tau,
                                 os.path.basename(path))
                break
    cols = []
    for c in numeric + categorical:
        if c == sens:
            continue
        if c in numeric:
            labels = bin_labels(c)
        else:
            counts = to_levels(real, c, numeric, target).value_counts()
            labels = list(counts.index)
            if len(labels) > TOP_K + 1:
                labels = labels[:TOP_K] + ['Other']
        # masked proxy levels exist only in the hybrid release
        if any((to_levels(d, c, numeric, target) == 'MASK').any()
               for d, *_ in releases.values()):
            labels = labels + ['MASK']
        args = (c, labels, sens, groups, numeric, target)
        cols.append({
            'name': c, 'ordinal': c in numeric, 'labels': labels,
            'proxy': [] if c in numeric else
            proxy_levels(real, c, sens, groups, numeric, target),
            'real': table(real, *args),
            'checkpoints': {k: table(d, *args)
                            for k, (d, *_) in releases.items()}})
    for c in cols:
        if c['proxy']:
            print(f"{name}: proxy levels in {c['name']}: "
                  + ', '.join(f"{p['level']} -> {p['group']} "
                              f"({p['purity']:.3f})" for p in c['proxy']))
    # before vs after: the real data against the hybrid release. The real
    # data is also shown with the levels the hybrid merged, so the effect of
    # masking is separate from the effect of the repair.
    before_after = None
    if 'hybrid' in releases:
        hyb, _, _, hyb_file = releases['hybrid']
        merged = merged_levels(os.path.join(HERE, 'aim_fair', 'out', hyb_file))
        if merged is None:   # no json: the proxy columns the release masked
            merged = {c['name']: [p['level'] for p in c['proxy']]
                      for c in cols if c['proxy'] and 'MASK' in c['labels']}
        real_m = mask(real, merged)
        for c in cols:
            if c['name'] in merged:
                c['real_masked'] = table(real_m, c['name'], c['labels'], sens,
                                         groups, numeric, target)
        print(f'{name}: before vs after, merged levels {merged}')
        test_path = train.replace('_train.csv', '_test.csv')
        leak = None
        if os.path.exists(test_path):
            test = pd.read_csv(test_path, **read)
            frames = [('Real data', real, test)]
            if merged:
                frames.append(('Real data, proxy levels merged', real_m,
                               mask(test, merged)))
            frames.append(('AIM hybrid release', hyb, mask(test, merged)))
            if 'plain' in releases:
                frames.append(('Plain AIM release (reference)',
                               releases['plain'][0], test))
            leak = all_columns_leakage(frames, numeric, categorical, sens)
        before_after = {'file': hyb_file, 'eps': releases['hybrid'][1],
                        'tau': releases['hybrid'][2], 'merged': merged,
                        'all_columns': leak}
    print(f'{name}: {len(real)} real rows, {sens} groups {groups}; '
          f'check points: '
          + (', '.join(f'{k} ({r[3]})' for k, r in releases.items())
             or 'none'))
    return {'label': DATASETS[name], 'sensitive': sens, 'target': target,
            'groups': groups, 'n': len(real), 'columns': cols,
            'checkpoints': {k: {'eps': e, 'tau': t, 'file': f}
                            for k, (_, e, t, f) in releases.items()},
            'before_after': before_after,
            'scorecard': scorecard(name)}


def main():
    data, skipped = {}, []
    for name in DATASETS:
        d = build(name)
        if d is None:
            skipped.append(name)
        else:
            data[name] = d
    if skipped:
        print('skipped (no fold-0 split; python aim_fair/export_folds.py '
              '--dataset D): ' + ', '.join(skipped))
    if not data:
        sys.exit('no dataset on disk: run python aim_fair/export_folds.py')
    with open(os.path.join(HERE, 'leakage_explorer_template.html'),
              encoding='utf-8') as f:
        page = f.read().replace('/*DATA*/null',
                                json.dumps({'datasets': data}))
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
