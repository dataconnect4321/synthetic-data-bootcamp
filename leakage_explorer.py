"""Interactive dashboard of privacy (epsilon), fairness (repair amount tau)
and utility. Writes one self-contained page, results/leakage_explorer.html,
with a dataset picker (Adult and every datasets.py dataset whose fold-0
training split is on disk, or with only the Scorecard tab when just its row
of results/datasets_summary.csv is) and five tabs:

  Scorecard       actual hybrid releases along tau at eps 3, scored, mean
                  and sd across folds or fold 0 alone (results/
                  utility_tau_grid.csv, scripts/run_tau_grid.sh); the same
                  grid feeds a strip of headline findings at the top (plain
                  AIM -> hybrid at tau 1, paired within folds);
  FLIP-faithful   the AIM hybrid vs FLIP's reproduction vs FLIP's reported
                  numbers under FLIP's own protocol (results/
                  flip_faithful_comparison.csv), with a check that its AIM
                  column equals the scorecard's fold-0 point at tau 1;
  Leakage         which columns leak the protected attribute, and each
                  column's per-group histogram in a release simulated in the
                  browser: Gaussian noise on each [protected x column] count
                  table (zCDP, rho split evenly), then the repair of
                  aim_fair/fair_aim.py (coupling, repair_plans; pooled
                  target). Not a Private-PGM fit; actual releases (fold 0,
                  refit 0) in aim_fair/out/ are drawn as check points;
  Before vs after the real data vs the actual hybrid release;
  Model graph     the marginals AIM chose to measure (fold 0's saved noisy
                  measurements): columns as nodes, measured pairs as edges,
                  in selection order. Private-PGM fits its model to these.

    python aim_fair/export_folds.py [--dataset D]   # once, writes aim_fair/data/
    bash scripts/run_tau_grid.sh                    # the scorecard (Adult)
    python leakage_explorer.py
"""
import json
import os
import pickle
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
    'hybrid': (['{p}aimfinal_tau1_e3_f0_r0_synthetic.csv',   # tau grid
                '{p}aimfinal_e3_f0_r0_synthetic.csv',
                '{p}aim_e3_f0_r0_synthetic.csv',
                '{p}lk_e3_f0_r0_synthetic.csv'], 3, 1),
}
# the saved noisy measurements behind the model graph (aim_fair/run_aim.py),
# first found wins
GRAPH_PKLS = ['{p}aimfinal_e3_f0_measurements.pkl',
              '{p}aim_e3_f0_measurements.pkl',
              '{p}lk_e3_f0_measurements.pkl']
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
SCORECARD = {'adult': (('results', 'utility_tau_grid.csv'), 3)}   # under HERE
# the other datasets: tau 1 and plain AIM only, mean and sd across folds
# (aim_fair/run_datasets.sh, aim_fair/datasets_summary.py)
SUMMARY = ('results', 'datasets_summary.csv')
SUMMARY_SOURCES = {   # its source -> (scorecard source, tau, refits per fold)
    'Real data (reference)': ('Real data', None, 1),
    'Plain AIM, eps 3': ('Plain AIM', None, 3),
    'AIM hybrid, eps 3': ('AIM hybrid', 1.0, 3)}
# the FLIP-faithful comparison (aim_fair/flip_faithful_chart.py): FLIP's own
# protocol, fold 0; its AIM hybrid is the tau grid's tau-1 tables, so it
# equals the scorecard's fold-0 row at tau 1. Metric label -> key (the
# scorecard's key where it has the metric)
FAITHFUL = ('results', 'flip_faithful_comparison.csv')   # under HERE
FAITHFUL_SOURCES = {'FLIP paper (reported)': 'paper',
                    'FLIP reproduction': 'repro', 'AIM hybrid': 'aim'}
FLIP_METRICS = {
    'Sex-inference BER (0.5 = sex unrecoverable)': 'ber',
    "A-NCB, paper's version (1 = full independence)": 'ancb',
    'Income AUC on real people (utility)': 'auc',
    'Alpha-precision (realism)': 'alpha',
    'Impossible rows (%) (e.g. male wives)': 'impossible',
    'Income parity (threshold 0.5)': 'income_parity',
    'Income EO gap* (threshold 0.5)': 'eo_gap',
    'Mean parity, all categorical targets (FLIP Eq. 14)': 'sp_cat',
    'Mean W1/sd, all numeric targets (FLIP Eq. 15)': 'w1_num',
    'Validity (rule-consistent rows)': 'validity',
    'tstr sex BER (folded; 0.5 best)': 'tstr',
    'Beta-recall': 'beta',
    'Epsilon-identifiability': 'eps_id'}
PERCENT = {'impossible'}   # the csv's percent, the scorecard's fraction
# FLIP's protocol per dataset (aim_fair/flip_protocol.py [--dataset D]): every
# attribute as a target, and the outcome at threshold 0.5; scope 'all' (3
# folds x 3 refits). {s} is '' for Adult, else '_' + the dataset.
PROTOCOL_TARGETS = 'flip_protocol_targets{s}.csv'   # in results/
PROTOCOL_OUTCOME = 'flip_protocol_outcome{s}.csv'   # Adult: ..._income.csv
PROTOCOL_SOURCES = ['Real data', 'Plain AIM', 'AIM hybrid']


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


def faithful():
    """The FLIP-faithful comparison from its csv, or None if missing:
    [{key, label, better, paper, repro, aim}] in the csv's order, in the
    scorecard's units (impossible rows as a fraction); None where a source
    reports nothing."""
    path = os.path.join(HERE, *FAITHFUL)
    if not os.path.exists(path):
        return None
    rows = []
    for r in pd.read_csv(path).to_dict('records'):
        k = FLIP_METRICS[r['metric']]
        s = 0.01 if k in PERCENT else 1
        rows.append({'key': k, 'label': r['metric'], 'better': r['better'],
                     **{v: None if pd.isna(r[c]) else round(float(r[c]) * s, 6)
                        for c, v in FAITHFUL_SOURCES.items()}})
    return rows


def flip_baseline(rows):
    """FLIP at lambda 4, eps 3 for the scorecard, from the faithful rows:
      values  the FLIP paper's reported numbers (its Fig. 3, its own
              evaluation code); None where it reports nothing (validity,
              impossible rows, the per-target means);
      repro   FLIP as published, reproduced here (fold 0) and scored like
              the AIM tables (FLIP's protocol for the downstream metrics)."""
    return {'label': 'FLIP paper', 'note': 'λ 4, nominal ε 3, Fig. 3',
            'values': {r['key']: r['paper'] for r in rows},
            'repro': {r['key']: r['repro'] for r in rows}}


def faithful_check(rows, grid):
    """The faithful AIM hybrid against the scorecard's fold-0 row at tau 1
    (the same tables): [{key, faithful, scorecard, ok}] for every metric
    both have. Both are rounded to 4 decimals, so ok allows 1e-4."""
    at = next((r for r in grid if r['source'] == 'AIM hybrid' and
               r['tau'] == 1 and r['fold'] == 0), None)
    if at is None:
        return None
    out = []
    for r in rows:
        if at.get(r['key']) is None or r['aim'] is None:
            continue
        a, b = r['aim'], float(at[r['key']])
        out.append({'key': r['key'], 'faithful': a, 'scorecard': b,
                    'ok': abs(a - b) <= 1e-4 + 1e-9})
    bad = [c['key'] for c in out if not c['ok']]
    print('FLIP-faithful AIM hybrid vs scorecard (fold 0, tau 1): '
          + (f'MISMATCH in {", ".join(bad)}' if bad else
             f'all {len(out)} shared metrics match'))
    return out


def protocol(name):
    """FLIP's protocol for the dataset's real data, plain AIM and hybrid
    (scope 'all'), or None if its csvs are missing:
      means    {source: {income_parity, eo_gap, sp_cat, w1_num}}: the
               outcome's parity and TPR gap at 0.5, and the means over
               targets of Eq. 14 and W1 / sd (as flip_datasets_table.py)
      targets  [{target, metric ('SP' | 'W1'), <source>: value_sd}] in the
               csv's order."""
    s = '' if name == 'adult' else '_' + name
    t_path = os.path.join(HERE, 'results', PROTOCOL_TARGETS.format(s=s))
    o_path = os.path.join(HERE, 'results', 'flip_protocol_income.csv'
                          if name == 'adult' else PROTOCOL_OUTCOME.format(s=s))
    if not (os.path.exists(t_path) and os.path.exists(o_path)):
        return None
    t = pd.read_csv(t_path)
    o = pd.read_csv(o_path)
    t = t[(t.scope == 'all') & t.source.isin(PROTOCOL_SOURCES)]
    o = o[(o.scope == 'all') & (o.threshold.astype(str) == '0.5')]
    means, targets = {}, {}
    for src in PROTOCOL_SOURCES:
        ts, os_ = t[t.source == src], o[o.source == src]
        if ts.empty or os_.empty:
            return None
        w1 = ts[ts.metric == 'W1'].value_sd
        means[src] = {
            'income_parity': float(os_.parity.iloc[0]),
            'eo_gap': float(os_['EO = TPR gap'].iloc[0]),
            'sp_cat': float(ts[ts.metric != 'W1'].value_sd.mean()),
            'w1_num': float(w1.mean()) if len(w1) else None}
        for r in ts.itertuples():
            targets.setdefault(r.target, {
                'target': r.target,
                'metric': 'W1' if r.metric == 'W1' else 'SP'})[src] = \
                float(r.value_sd)
    return {'means': means, 'targets': list(targets.values())}


def scorecard(name):
    """{eps, rows, baseline, targets} from the dataset's tau-grid csv, else
    its rows of results/datasets_summary.csv (tau 1 only, no per-fold rows),
    or None if neither has it. Rows carry the mean over folds, <metric>_sd
    (None when the grid has one run) and the folds x refits behind them.
    For the summary's datasets, the downstream metrics are replaced by FLIP's
    protocol, as on Adult's grid (means only: no sd)."""
    rel, eps = SCORECARD.get(name, ((), None))
    path = os.path.join(HERE, *rel)
    P = protocol(name)
    targets = P and P['targets']
    if rel and os.path.exists(path):
        df = pd.read_csv(path)
        df = df.astype(object).where(df.notna(), None)
        rows = df.to_dict('records')
        F = faithful() if name == 'adult' else None
        return {'eps': eps, 'rows': rows,
                'baseline': F and flip_baseline(F),
                'faithful': F and {'rows': F, 'check': faithful_check(F, rows)},
                'targets': targets}
    path = os.path.join(HERE, *SUMMARY)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df.dataset == name].drop(columns='dataset').rename(
        columns={'tstr_folded': 'tstr', 'tstr_folded_sd': 'tstr_sd'})
    if df.empty:
        return None
    df = df.astype(object).where(df.notna(), None)
    rows = []
    for r in df.to_dict('records'):
        source, tau, refits = SUMMARY_SOURCES[r['source']]
        if P:
            for k, v in P['means'][source].items():
                r[k], r[k + '_sd'] = v, None
        rows.append({**r, 'source': source, 'tau': tau, 'fold': None,
                     'refits': refits})
    return {'eps': 3, 'rows': rows, 'baseline': None, 'faithful': None,
            'targets': targets, 'protocol': P is not None}


def model_graph(name):
    """The marginals AIM measured, from the dataset's first saved
    measurement pickle, or None if there is none. Nodes are the columns
    (size and noise sd of their 1-way table); edges are the measured 2-way
    tables, with the rounds (1-based, among the 2-way picks, in AIM's
    selection order) that measured them."""
    for n in GRAPH_PKLS:
        path = os.path.join(HERE, 'aim_fair', 'out', n.format(p=prefix(name)))
        if os.path.exists(path):
            break
    else:
        return None
    with open(path, 'rb') as f:
        saved = pickle.load(f)
    if isinstance(saved, list):       # older files: measurements only
        saved = {'measurements': saved}
    nodes, edges, k = {}, {}, 0
    for cl, y, sd in saved['measurements']:
        cl, sd = tuple(cl), float(sd)
        if len(cl) == 1:
            nodes[cl[0]] = {'name': cl[0], 'size': int(np.size(y)), 'sigma': sd}
            continue
        k += 1
        key = tuple(sorted(cl))
        e = edges.setdefault(key, {'a': key[0], 'b': key[1], 'rounds': [],
                                   'sigma': sd, 'cells': int(np.size(y))})
        e['rounds'].append(k)
        e['sigma'] = min(e['sigma'], sd)
    for e in edges.values():          # a column only ever measured in pairs
        for c in (e['a'], e['b']):
            nodes.setdefault(c, {'name': c, 'size': None, 'sigma': None})
    numeric, categorical, _, _ = schema(name)
    order = numeric + categorical       # schema order, unknown columns last
    nodes = sorted(nodes.values(), key=lambda v: order.index(v['name'])
                   if v['name'] in order else len(order))
    rho = saved.get('rho_total')
    return {'file': os.path.basename(path), 'nodes': nodes,
            'edges': list(edges.values()), 'n_rounds': k,
            'rho_total': None if rho is None else float(rho)}


def build(name):
    """Page data for one dataset; only its scorecard if its fold-0 split is
    missing; None if it has neither."""
    p = prefix(name)
    train = os.path.join(HERE, 'aim_fair', 'data', f'{p}fold0_train.csv')
    numeric, categorical, sens, target = schema(name)
    if not os.path.exists(train):
        sc = scorecard(name)
        if sc is None:
            return None
        print(f'{name}: scorecard only (no fold-0 split)')
        return {'label': DATASETS[name], 'sensitive': sens, 'target': target,
                'scorecard_only': True, 'scorecard': sc}
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
            'scorecard': scorecard(name),
            'model_graph': model_graph(name)}


def main():
    data, skipped = {}, []
    for name in DATASETS:
        d = build(name)
        if d is None:
            skipped.append(name)
        else:
            data[name] = d
    if skipped:
        print('skipped (no fold-0 split or scorecard; python '
              'aim_fair/export_folds.py --dataset D): ' + ', '.join(skipped))
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
