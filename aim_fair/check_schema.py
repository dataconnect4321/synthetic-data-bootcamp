"""How well do the public bin edges (adult_domain.EDGES) fit Adult's numeric
columns, and does a private, data-adaptive fallback fit better?

For each numeric column, every real value is binned and decoded back (the
same round trip a synthetic value goes through), and the decoded values are
compared with the real ones. This isolates the discretization loss; AIM's
noise comes on top. Schemes compared, with the same number of bins:

  public  - fixed public edges, uniform decoding inside a bin (current)
  dp      - private fallback: a Laplace-noised histogram on a public fine
            grid gives equal-frequency bins and within-bin decoding;
            heaped values ("spikes", spike_test) get their own bins, found
            from noisy counts (integer domain for capital gain/loss)
  naive quantile - equal-frequency bins from the real values (NOT private;
            the obvious data-adaptive alternative, without spike handling)

Metrics (real vs decoded): W1 / sd, KS, value overlap (sum over values of
min of the two shares; 1 = identical value distribution), share of the real
spike mass recovered, and bin usage (normalized entropy, largest bin).
This is an evaluation on real data: choosing the schema by these scores is
itself a data-dependent choice (see WRITEUP.md, "tuning is not covered").

    python aim_fair/check_schema.py            # main environment
"""
import os

import numpy as np
import pandas as pd

import adult_domain as AD

HERE = os.path.dirname(os.path.abspath(__file__))
RANGE = {'age': (17, 90), 'fnlwgt': (10000, 1600000), 'education-num': (1, 16),
         'capital-gain': (0, 99999), 'capital-loss': (0, 4400),
         'hours-per-week': (1, 99)}
INTEGER_GRID = {'age', 'education-num', 'hours-per-week'}
EPS_HIST, EPS_SPIKE = 0.03, 0.1
SPIKE_SHARE, SPIKE_EXCESS = 0.005, 3.0


def spike_test(cnt, eps, total):
    """Pre-specified heaping test on noisy integer counts: a value is a
    spike if its noisy count is >= SPIKE_SHARE of the rows, >= SPIKE_EXCESS
    times the mean of its two neighbours on each side, and above the level
    that Laplace noise exceeds anywhere in the domain with probability
    0.05 (union bound: ln(10 * D) / eps)."""
    d = len(cnt)
    pad = np.pad(cnt, 2, mode='edge')
    neigh = (pad[:-4] + pad[1:-3] + pad[3:-1] + pad[4:]) / 4
    noise_floor = np.log(10 * d) / eps
    return np.nonzero((cnt >= SPIKE_SHARE * total)
                      & (cnt >= SPIKE_EXCESS * np.clip(neigh, 1, None))
                      & (cnt >= noise_floor))[0]


def fine_edges(c):
    """Public fine grid: integers, or geometric edges for wide columns."""
    lo, hi = RANGE[c]
    if c in INTEGER_GRID:
        return np.arange(lo, hi + 2, dtype=float)
    if c == 'fnlwgt':
        return np.unique(np.round(np.geomspace(lo, hi + 1, 151)))
    return np.concatenate([[0, 1], np.unique(np.round(np.geomspace(1, hi + 1, 101)))[1:]])


def decode_uniform(codes, edges, rng):
    lo, hi = edges[codes], edges[codes + 1]
    return np.floor(lo + rng.random(len(codes)) * np.maximum(hi - lo, 1))


def public_scheme(c, x, rng):
    e = np.asarray(AD.EDGES[c], float)
    k = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
    return k, decode_uniform(k, e, rng), len(e) - 1


def oracle_scheme(c, x, rng, n_bins):
    q = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    e = np.concatenate([[RANGE[c][0]], q[1:-1], [RANGE[c][1] + 1]])
    k = np.clip(np.searchsorted(e, x, side='right') - 1, 0, len(e) - 2)
    return k, decode_uniform(k, e, rng), len(e) - 1


def dp_scheme(c, x, rng, n_bins):
    """Private fallback (epsilon EPS_HIST, + EPS_SPIKE for capital gain and
    loss): noisy fine histogram -> spikes, equal-frequency cuts, and
    within-bin decoding by the noisy fine shares."""
    fe = fine_edges(c)
    fine = np.clip(np.searchsorted(fe, x, side='right') - 1, 0, len(fe) - 2)
    noisy = np.bincount(fine, minlength=len(fe) - 1) + rng.laplace(
        0, 1 / EPS_HIST, len(fe) - 1)
    noisy = np.clip(noisy, 0, None)
    total = noisy.sum()
    # spikes: exact values holding >= SPIKE_SHARE of the rows
    if c in INTEGER_GRID:
        idx = spike_test(noisy, EPS_HIST, total)
        spikes, spike_cnt = fe[idx], noisy[idx]
    elif c in ('capital-gain', 'capital-loss'):
        lo, hi = RANGE[c]
        cnt = np.bincount((x - lo).astype(int), minlength=hi - lo + 1).astype(float)
        cnt += rng.laplace(0, 1 / EPS_SPIKE, len(cnt))
        idx = spike_test(cnt, EPS_SPIKE, total)
        spikes, spike_cnt = idx + lo, cnt[idx]
    else:
        spikes, spike_cnt = np.array([]), np.array([])
    order = np.argsort(spikes)
    spikes, spike_cnt = spikes[order].astype(float), spike_cnt[order]
    # the non-spike mass: noisy fine counts minus the spikes' noisy counts
    w = noisy.copy()
    for s, n_s in zip(spikes, spike_cnt):
        b = np.searchsorted(fe, s, side='right') - 1
        w[b] = max(w[b] - n_s, 0)
    n_rest = max(n_bins - len(spikes), 2)
    cdf = np.cumsum(w) / max(w.sum(), 1e-9)
    cut_idx = np.unique(np.searchsorted(cdf, np.linspace(0, 1, n_rest + 1)[1:-1]))
    edges = np.unique(np.concatenate([[fe[0]], fe[cut_idx + 1], [fe[-1]]]))
    # bins: spikes first (exact), then the cut intervals
    k = np.full(len(x), -1)
    for i, s in enumerate(spikes):
        k[x == s] = i
    rest = k < 0
    kk = np.clip(np.searchsorted(edges, x[rest], side='right') - 1, 0, len(edges) - 2)
    k[rest] = len(spikes) + kk
    # decode: spike exact; otherwise a fine bin by noisy share, then uniform
    dec = np.empty(len(x))
    for i, s in enumerate(spikes):
        dec[k == i] = s
    for b in range(len(edges) - 1):
        m = k == len(spikes) + b
        if not m.any():
            continue
        inside = np.nonzero((fe[:-1] >= edges[b]) & (fe[1:] <= edges[b + 1]))[0]
        p = w[inside].copy()
        p = p / p.sum() if p.sum() > 0 else np.full(len(inside), 1 / len(inside))
        pick = inside[rng.choice(len(inside), m.sum(), p=p)]
        dec[m] = decode_uniform(pick, fe, rng)
    return k, dec, len(spikes) + len(edges) - 1


def metrics(x, dec, k, n_bins):
    xs, ds = np.sort(x), np.sort(dec)
    w1 = np.mean(np.abs(xs - ds)) / max(x.std(), 1e-9)
    grid = np.union1d(xs, ds)
    ks = np.max(np.abs(np.searchsorted(xs, grid, side='right')
                       - np.searchsorted(ds, grid, side='right'))) / len(x)
    vx, cx = np.unique(xs, return_counts=True)
    vd, cd = np.unique(ds, return_counts=True)
    common, ix, idd = np.intersect1d(vx, vd, return_indices=True)
    overlap = np.minimum(cx[ix], cd[idd]).sum() / len(x)
    heavy = vx[cx >= SPIKE_SHARE * len(x)]      # real values with >= 0.5% of rows
    if len(heavy):
        rx = np.array([np.sum(x == v) for v in heavy])
        rd = np.array([np.sum(dec == v) for v in heavy])
        spike_rec = np.minimum(rx, rd).sum() / rx.sum()
    else:
        spike_rec = np.nan
    nz = (x > 0).mean() < 0.5      # mostly-zero columns: overlap of the non-zero part
    if nz:
        xv, xc = np.unique(xs[xs > 0], return_counts=True)
        dv, dc = np.unique(ds[ds > 0], return_counts=True)
        _, i1, i2 = np.intersect1d(xv, dv, return_indices=True)
        nz_overlap = np.minimum(xc[i1], dc[i2]).sum() / max(xc.sum(), 1)
    else:
        nz_overlap = np.nan
    share = np.bincount(k, minlength=n_bins) / len(k)
    p = share[share > 0]
    ent = -(p * np.log(p)).sum() / np.log(max(n_bins, 2))
    return dict(w1_sd=w1, ks=ks, overlap=overlap, nonzero_overlap=nz_overlap,
                spike_recovery=spike_rec,
                bins=n_bins, entropy=ent, largest_bin=share.max(),
                tiny_bins=int((share < 0.005).sum()))


def main():
    """Public edges vs the private schema actually used by run_aim.py
    --dp-schema (dp_schema.build, 5% of the epsilon-3 zCDP budget)."""
    import sys
    sys.path.insert(0, os.path.join(HERE, 'vendor'))
    from cdp2adp import cdp_rho
    import dp_schema as DS
    rho = cdp_rho(3.0, 1e-5)
    rows, picks = [], []
    for f in range(3):
        df = pd.read_csv(os.path.join(HERE, 'data', f'fold{f}_train.csv'))
        for c in AD.NUMERIC:
            x = df[c].to_numpy(float)
            k, dec, nb = public_scheme(c, x, np.random.default_rng(f))
            rows.append(dict(fold=f, column=c, scheme='public',
                             **metrics(x, dec, k, nb)))
        for s in range(5):
            rng = np.random.default_rng(100 * f + s)
            schema = DS.build(df, AD.NUMERIC, rho, AD.EDGES, rng)
            for c in AD.NUMERIC:
                x = df[c].to_numpy(float)
                e = AD.edges_of(c, schema)
                k = np.clip(np.searchsorted(e, np.clip(x, e[0], e[-1] - 1),
                                            side='right') - 1, 0, len(e) - 2)
                table = schema[c]['decode']
                dec = (DS.decode_column(k, table, e, rng) if table
                       else np.floor(e[k] + rng.random(len(k))
                                     * np.maximum(e[k + 1] - e[k], 1)))
                rows.append(dict(fold=f, column=c, scheme='dp schema',
                                 **metrics(x, dec, k, len(e) - 1)))
                picks.append(dict(fold=f, seed=s, column=c,
                                  source=schema[c]['source'],
                                  bins=len(e) - 1,
                                  spikes=' '.join(f'{v:g}' for v in
                                                  schema[c]['spikes'])))
    res = pd.DataFrame(rows)
    out = res.groupby(['column', 'scheme']).mean(numeric_only=True).drop(
        columns='fold').round(3)
    pd.set_option('display.width', 220)
    print(out.to_string())
    pk = pd.DataFrame(picks)
    print()
    print('private choice per column (share of 15 draws using dp bins):')
    print((pk.assign(dp=pk.source == 'dp').groupby('column').dp.mean())
          .round(2).to_string())
    print()
    print('example spikes (fold 0, draw 0):')
    print(pk[(pk.fold == 0) & (pk.seed == 0)][['column', 'source', 'bins',
                                                'spikes']].to_string(index=False))
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
    out.to_csv(os.path.join(HERE, 'out', 'schema_check.csv'))
    meta = schema['_meta']
    print()
    print(f"schema budget: {meta['rho_share']:.0%} of rho "
          f"({meta['releases']} Gaussian releases, sigma {meta['sigma']:.1f} rows)")


if __name__ == '__main__':
    main()
