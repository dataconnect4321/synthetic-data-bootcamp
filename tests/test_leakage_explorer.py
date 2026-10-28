"""The leakage explorer (leakage_explorer.py + its HTML template): the page's
JavaScript ports must match the Python they copy, and the embedded tables
must match the data.

  - coupling / repair: aim_fair/fair_aim.coupling and repair_plans (pooled
    target), on random distributions and negative noisy counts;
  - epsilon -> rho: aim_fair/vendor/cdp2adp.cdp_rho;
  - embedded counts: a direct pandas crosstab of the fold-0 data and
    releases (skipped when they are not on disk);
  - other datasets: string-coded groups, no numeric columns, no release,
    a MASK level (built from small fake folds).

The JavaScript checks need node and are skipped without it.

Run:  python -m pytest tests/test_leakage_explorer.py -v
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'aim_fair', 'vendor'))

import datasets as D  # noqa: E402
import leakage_explorer as L  # noqa: E402
from cdp2adp import cdp_rho  # noqa: E402

TEMPLATE = os.path.join(ROOT, 'leakage_explorer_template.html')
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='node not installed')

# runs the template's math (everything before the seeded noise) on cases.json
JS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8').split('<script>')[1]
  .split('// --- seeded noise')[0].replace(/const DATA = .*?;\n/s, '');
const F = new Function(src + ';return {cdpRho, coupling, release};')();
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
console.log(JSON.stringify({
  out: cases.dists.map(c => ({coup: F.coupling(c.p, c.q, c.ord),
                              rel: F.release(c.noisy, c.ord, c.tau).cond})),
  rho: cases.eps.map(e => F.cdpRho(e, 1e-5))}));
"""


def fair_aim_coupling():
    """fair_aim.coupling, taken from the source: fair_aim imports mbi, which
    the main environment does not have."""
    path = os.path.join(ROOT, 'aim_fair', 'fair_aim.py')
    tree = ast.parse(open(path, encoding='utf-8').read())
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == 'coupling')
    ns = {'np': np}
    exec(compile(ast.Module([fn], []), path, 'exec'), ns)
    return ns['coupling']


def run_js(tmp_path, cases, page=TEMPLATE):
    (tmp_path / 'h.js').write_text(JS)
    (tmp_path / 'cases.json').write_text(json.dumps(cases))
    out = subprocess.run([NODE, str(tmp_path / 'h.js'), page,
                          str(tmp_path / 'cases.json')],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def page_data(path):
    m = re.search(r'const DATA = (\{.*?\});\n', open(path, encoding='utf-8').read(), re.S)
    return json.loads(m.group(1))['datasets']


@needs_node
def test_js_coupling_and_repair_match_fair_aim(tmp_path):
    coupling = fair_aim_coupling()
    rng = np.random.default_rng(0)
    dists = []
    for i in range(200):
        k = int(rng.integers(2, 16))
        dists.append(dict(
            p=rng.dirichlet(np.ones(k) * 0.5).tolist(),
            q=rng.dirichlet(np.ones(k) * 0.5).tolist(), ord=bool(i % 2),
            # noisy counts, some negative
            noisy=(rng.integers(0, 3000, (2, k))
                   + rng.normal(0, 200, (2, k))).tolist(),
            tau=float(rng.random())))
    js = run_js(tmp_path, {'dists': dists, 'eps': []})
    for c, o in zip(dists, js['out']):
        np.testing.assert_allclose(o['coup'], coupling(c['p'], c['q'], c['ord']),
                                   atol=1e-11)
        # repair_plans with the pooled target, applied to each group
        pj = np.clip(np.array(c['noisy']), 0, None)
        pj = pj / pj.sum()
        cond = pj / np.maximum(pj.sum(1, keepdims=True), 1e-12)
        eye = np.eye(pj.shape[1])
        want = [cond[s] @ ((1 - c['tau']) * eye + c['tau']
                           * coupling(cond[s], pj.sum(0), c['ord']))
                for s in range(2)]
        np.testing.assert_allclose(o['rel'], want, atol=1e-11)


@needs_node
def test_js_rho_matches_cdp2adp(tmp_path):
    eps = [0.05, 0.1, 0.5, 1, 3, 8, 20]
    js = run_js(tmp_path, {'dists': [], 'eps': eps})
    np.testing.assert_allclose(js['rho'], [cdp_rho(e, 1e-5) for e in eps],
                               rtol=1e-9)


def test_adult_page_counts_match_crosstab(tmp_path, monkeypatch):
    train = os.path.join(ROOT, 'aim_fair', 'data', 'fold0_train.csv')
    if not os.path.exists(train):
        pytest.skip('aim_fair/data/fold0_train.csv missing')
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    A = page_data(tmp_path / 'page.html')['adult']
    real = pd.read_csv(train, dtype={'income': str})
    assert A['groups'] == ['Male', 'Female'] and A['n'] == len(real)
    frames = {'real': real}
    for k, cp in A['checkpoints'].items():
        frames[k] = pd.read_csv(os.path.join(ROOT, 'aim_fair', 'out', cp['file']),
                                dtype={'income': str})
    for c in A['columns']:
        for k, df in frames.items():
            t = np.array(c['real'] if k == 'real' else c['checkpoints'][k])
            assert t.shape == (2, len(c['labels']))
            assert t.sum(1).tolist() == [int((df.sex == g).sum())
                                         for g in A['groups']]
            # independent count: raw values -> levels -> crosstab
            if c['ordinal']:
                e = np.asarray(L.AD.EDGES[c['name']], float)
                x = np.clip(df[c['name']].to_numpy(float).round(), e[0], e[-1] - 1)
                lv = np.searchsorted(e, x, side='right') - 1
                want = [np.bincount(lv[df.sex.to_numpy() == g], minlength=len(e) - 1)
                        for g in A['groups']]
            else:
                s = df[c['name']].astype(str)
                if c['name'] == 'income':
                    s = s.map(L.TARGET_LABELS['income'])
                if 'Other' in c['labels']:
                    s = s.where(s.isin(c['labels']), 'Other')
                ct = pd.crosstab(df.sex, s).reindex(index=A['groups'],
                                                    columns=c['labels'],
                                                    fill_value=0)
                want = ct.to_numpy()
            np.testing.assert_array_equal(t, want, err_msg=f"{k} {c['name']}")


def test_build_other_datasets(tmp_path, monkeypatch):
    """Dutch-like (all categorical, groups coded '1'/'2', no release) and
    ACS-like (numeric AGEP/WKHP, a release with a MASK level) folds."""
    rng = np.random.default_rng(0)
    n = 2000
    (tmp_path / 'aim_fair' / 'data').mkdir(parents=True)
    (tmp_path / 'aim_fair' / 'out').mkdir()
    shutil.copy(TEMPLATE, tmp_path)
    cfg = D.CONFIG['dutch']
    d = pd.DataFrame({c: rng.integers(1, 6, n).astype(str)
                      for c in cfg['categorical']})
    d['sex'] = np.where(rng.random(n) < 0.6, '1', '2')
    d['occupation'] = rng.integers(0, 2, n).astype(str)
    d.to_csv(tmp_path / 'aim_fair' / 'data' / 'dutch_fold0_train.csv', index=False)
    cfg = D.CONFIG['acs_al']
    a = pd.DataFrame({c: rng.integers(1, 4, n).astype(str)
                      for c in cfg['categorical']})
    a['SEX'] = np.where(rng.random(n) < 0.4, 'Female', 'Male')
    a['income'] = rng.integers(0, 2, n).astype(str)
    a['AGEP'] = rng.integers(17, 95, n)
    a['WKHP'] = rng.integers(1, 99, n)
    cols = cfg['numeric'] + cfg['categorical']
    a[cols].to_csv(tmp_path / 'aim_fair' / 'data' / 'acs_al_fold0_train.csv',
                   index=False)
    h = a.copy()
    h.loc[h.RELP.isin(['1', '2']), 'RELP'] = 'MASK'
    h[cols].to_csv(tmp_path / 'aim_fair' / 'out'
                   / 'acs_al_aim_e3_f0_r0_synthetic.csv', index=False)
    monkeypatch.setattr(L, 'HERE', str(tmp_path))
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    P = page_data(tmp_path / 'page.html')
    assert sorted(P) == ['acs_al', 'dutch']          # adult, acs_ut skipped
    du, ac = P['dutch'], P['acs_al']
    assert du['groups'] == ['1', '2'] and du['checkpoints'] == {}
    assert len(du['columns']) == len(D.CONFIG['dutch']['categorical']) - 1
    occ = next(c for c in du['columns'] if c['name'] == 'occupation')
    assert occ['labels'] == ['low-level', 'high-level'] or \
        occ['labels'] == ['high-level', 'low-level']
    assert ac['groups'] == ['Male', 'Female'] and list(ac['checkpoints']) == ['hybrid']
    relp = next(c for c in ac['columns'] if c['name'] == 'RELP')
    assert relp['labels'][-1] == 'MASK'
    assert np.array(relp['real'])[:, -1].sum() == 0
    assert np.array(relp['checkpoints']['hybrid'])[:, -1].sum() == \
        int(a.RELP.isin(['1', '2']).sum())
    agep = next(c for c in ac['columns'] if c['name'] == 'AGEP')
    assert agep['ordinal'] and len(agep['labels']) == len(L.AD.EDGES['AGEP']) - 1
    for ds in P.values():
        for c in ds['columns']:
            assert np.array(c['real']).sum() == ds['n']


def test_proxy_levels_match_detect_proxy_classes(tmp_path, monkeypatch):
    train = os.path.join(ROOT, 'aim_fair', 'data', 'fold0_train.csv')
    if not os.path.exists(train):
        pytest.skip('aim_fair/data/fold0_train.csv missing')
    sys.path.insert(0, os.path.join(ROOT, 'tabsyn-main', 'tabsyn-main'))
    from tabsyn.vae.proxy_decouple import detect_proxy_classes
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    A = page_data(tmp_path / 'page.html')['adult']
    real = pd.read_csv(train, dtype={'income': str})
    cat = list(L.AD.CATEGORICAL)
    codes = {c: sorted(real[c].astype(str).unique()) for c in cat}
    X = np.stack([real[c].astype(str).map({v: i for i, v in enumerate(codes[c])})
                  .to_numpy() for c in cat], 1)
    want = {cat[k['col']]: sorted(codes[cat[k['col']]][m] for m in k['members'])
            for k in detect_proxy_classes(X, cat.index('sex'))}
    got = {c['name']: sorted(p['level'] for p in c['proxy'])
           for c in A['columns'] if c['proxy']}
    assert got == want == {'relationship': ['Husband', 'Wife']}


@needs_node
def test_simulation_properties(tmp_path, monkeypatch):
    """On the built Adult page: full repair (tau 1) leaves no per-column
    leakage, and the repair never changes the distortion (pooled target)."""
    if not os.path.exists(os.path.join(ROOT, 'aim_fair', 'data',
                                       'fold0_train.csv')):
        pytest.skip('aim_fair/data/fold0_train.csv missing')
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    (tmp_path / 's.js').write_text(r"""
const fs = require('fs');
let src = fs.readFileSync(process.argv[2], 'utf8').split('<script>')[1].split('// --- drawing helpers')[0];
const DATA = JSON.parse(src.match(/const DATA = (\{.*?\});\n/s)[1]);
src = src.replace(/const DATA = .*?;\n/s, '');
const T = new Function('DATA', src + ';return {S, compute, normals};')(DATA);
T.S.ds = 'adult'; T.S.z = T.normals(5);
const out = { leak1: [], dist: [] };
for (const eps of [0.1, 1, 3, 20]) {
  T.S.eps = eps;
  T.S.tau = 0; const a = T.compute();
  T.S.tau = 1; const b = T.compute();
  out.leak1.push(Math.max(...b.cols.map(c => c.tvRel)));
  out.dist.push(Math.max(...a.cols.map((c, i) => Math.abs(c.distortion - b.cols[i].distortion))));
}
console.log(JSON.stringify(out));
""")
    out = json.loads(subprocess.run(
        [NODE, str(tmp_path / 's.js'), str(tmp_path / 'page.html')],
        capture_output=True, text=True, check=True).stdout)
    assert max(out['leak1']) < 1e-12
    assert max(out['dist']) < 1e-12


def test_before_after_masked_counts(tmp_path, monkeypatch):
    """The real data with the hybrid's merged levels: read from the run's
    json, and counted like an independent crosstab."""
    hyb = os.path.join(ROOT, 'aim_fair', 'out', 'lk_e3_f0_r0_synthetic.csv')
    if not os.path.exists(hyb):
        pytest.skip('hybrid release missing')
    assert L.merged_levels(hyb) == {'relationship': ['Husband', 'Wife']}
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    A = page_data(tmp_path / 'page.html')['adult']
    assert A['before_after']['file'] == 'lk_e3_f0_r0_synthetic.csv'
    assert A['before_after']['all_columns'] is None
    real = pd.read_csv(os.path.join(ROOT, 'aim_fair', 'data', 'fold0_train.csv'))
    m = real.relationship.where(~real.relationship.isin(['Husband', 'Wife']), 'MASK')
    rel = next(c for c in A['columns'] if c['name'] == 'relationship')
    want = pd.crosstab(real.sex, m).reindex(index=A['groups'],
                                            columns=rel['labels'], fill_value=0)
    np.testing.assert_array_equal(rel['real_masked'], want.to_numpy())
    assert [c['name'] for c in A['columns'] if 'real_masked' in c] == ['relationship']


def test_all_columns_leakage_known_answers(monkeypatch):
    """The classifier helper on tables with a known answer: sex copied from
    another column leaks fully (BER 0, tstr folded 0); sex independent of
    every column does not (about 0.5)."""
    pytest.importorskip('torch')
    monkeypatch.setattr(L, 'BER_SEEDS', 1)
    rng = np.random.default_rng(0)
    n = 1500

    def frame(copy):
        df = pd.DataFrame({'x': rng.normal(size=n),
                           'a': rng.choice(['p', 'q'], n),
                           'b': rng.choice(['u', 'v', 'w'], n)})
        df['sex'] = np.where(df.a == 'p', 'M', 'F') if copy else \
            rng.choice(['M', 'F'], n)
        return df
    leaky, fair = frame(True), frame(False)
    rows = L.all_columns_leakage([('leaky', leaky, frame(True)),
                                  ('fair', fair, frame(False))],
                                 ['x'], ['sex', 'a', 'b'], 'sex')
    assert rows[0]['ber'] == 0 and rows[0]['tstr'] == 0
    assert abs(rows[1]['ber'] - 0.5) < 0.05 and abs(rows[1]['tstr'] - 0.5) < 0.05


def test_scorecard_matches_grid_and_scorer(tmp_path, monkeypatch):
    """The scorecard embedded in the page equals results/utility_tau_grid.csv,
    and the csv's hybrid and plain rows equal the scorer's json for that
    release (final_chart.METRICS: the RESULTS.md definitions)."""
    grid = os.path.join(ROOT, 'results', 'utility_tau_grid.csv')
    if not os.path.exists(grid):
        pytest.skip('results/utility_tau_grid.csv missing: scripts/run_tau_grid.sh')
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    sc = page_data(tmp_path / 'page.html')['adult']['scorecard']
    df = pd.read_csv(grid)
    assert sc['eps'] == 3 and len(sc['rows']) == len(df)
    keys = ['ber', 'ancb', 'auc', 'income_parity', 'eo_gap', 'validity',
            'impossible']
    for got, (_, want) in zip(sc['rows'], df.iterrows()):
        assert got['source'] == want['source']
        np.testing.assert_allclose([got[k] for k in keys],
                                   [want[k] for k in keys], atol=1e-12)
    assert set(df.source) == {'AIM hybrid', 'Plain AIM', 'Real data'}
    sys.path.insert(0, os.path.join(ROOT, 'aim_fair'))
    from final_chart import METRICS
    get = {k: g for k, _, _, g in METRICS}
    res = os.path.join(ROOT, 'adult_results')
    checked = 0
    for _, row in df.iterrows():
        if row.source == 'AIM hybrid':
            tag = f"lk_tau{row.tau:g}_e3_f0_r0"
        elif row.source == 'Plain AIM':
            tag = 'lk_plain_e3_f0_r0'
        else:
            continue
        path = os.path.join(res, tag + '.json')
        if not os.path.exists(path):
            continue
        js = json.load(open(path))
        for k in keys:
            assert row[k] == pytest.approx(round(float(get[k](js)), 4),
                                           abs=1e-9), (tag, k)
        checked += 1
    if not checked:
        pytest.skip('scorer jsons not in adult_results/')


def test_flip_baseline_matches_final_comparison(tmp_path, monkeypatch):
    """The scorecard's FLIP baseline equals results/final_comparison.csv's
    flip_faithful_repro (FLIP as published, lambda 4, nominal eps 3, fold 0)
    for every scorecard metric."""
    comp = os.path.join(ROOT, 'results', 'final_comparison.csv')
    grid = os.path.join(ROOT, 'results', 'utility_tau_grid.csv')
    if not (os.path.exists(comp) and os.path.exists(grid)):
        pytest.skip('results/final_comparison.csv or utility_tau_grid.csv missing')
    monkeypatch.setattr(L, 'OUT', str(tmp_path / 'page.html'))
    monkeypatch.setattr(L, 'ALL_COLUMNS', False)
    L.main()
    B = page_data(tmp_path / 'page.html')['adult']['scorecard']['baseline']
    want = pd.read_csv(comp).set_index('metric')['flip_faithful_repro']
    assert set(B['values']) == {'ber', 'ancb', 'auc', 'income_parity', 'eo_gap',
                                'validity', 'impossible'}
    for label, key in L.FLIP_METRICS.items():
        assert B['values'][key] == want[label]
