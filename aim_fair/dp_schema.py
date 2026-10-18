"""Private, data-adaptive bins for the numeric columns (the fallback for the
public edges in adult_domain.EDGES).

One Gaussian release per column on a public fine grid (integers for age,
education-num and hours; geometric cells for the wide columns), plus an
integer-level release for capital gain / loss. Each release is a histogram
under add/remove neighbours, so its L2 sensitivity is 1. The releases share
SCHEMA_RHO_SHARE of the total zCDP budget, and AIM gets the rest
(composition in rho). Everything else here is post-processing.

From the noisy counts, with all constants fixed in advance:

- spikes (heaped values): >= SPIKE_SHARE of the rows, >= SPIKE_EXCESS times
  their neighbours, and above the Bonferroni noise floor
  sigma * Phi^-1(1 - ALPHA / cells);
- edges: every spike is a singleton bin [v, v + 1) that splits the line, so
  bins stay contiguous and ordered (the monotone coupling in fair_aim
  needs that). Each gap between spikes is one bin, subdivided into
  equal-frequency pieces only while every piece keeps >= MIN_PIECE_SHARE
  of the rows. At most MAX_BINS bins (the smallest spikes are dropped
  first);
- decoding inside a bin: cells that are significant under Benjamini-
  Hochberg (q = BH_Q, H0: count 0) get their noisy count; the rest of the
  bin's mass is spread uniformly;
- per column, the private bins replace the public ones only if spikes that
  are not already singleton public bins hold >= SWITCH_SHARE of the rows.
  Otherwise the public edges and uniform decoding are kept.
"""
import numpy as np
from scipy.stats import norm

SCHEMA_RHO_SHARE = 0.05
SPIKE_SHARE, SPIKE_EXCESS, ALPHA = 0.005, 3.0, 0.05
BH_Q, MAX_BINS, MIN_PIECE_SHARE, SWITCH_SHARE = 0.05, 20, 0.01, 0.01

RANGE = {'age': (17, 90), 'fnlwgt': (10000, 1599999), 'education-num': (1, 16),
         'capital-gain': (0, 99999), 'capital-loss': (0, 4399),
         'hours-per-week': (1, 99)}
INTEGER_FINE = ('age', 'education-num', 'hours-per-week')
INTEGER_EXTRA = ('capital-gain', 'capital-loss')


def fine_edges(c):
    lo, hi = RANGE[c]
    if c in INTEGER_FINE:
        return np.arange(lo, hi + 2, dtype=float)
    if c == 'fnlwgt':
        return np.unique(np.round(np.geomspace(lo, hi + 1, 151)))
    return np.unique(np.concatenate(
        [[0.0, 1.0], np.round(np.geomspace(1, hi + 1, 101))]))


def n_releases(columns):
    return len(columns) + sum(c in INTEGER_EXTRA for c in columns)


def sigma_for(rho_total, columns):
    """Gaussian sd of each histogram release (L2 sensitivity 1)."""
    rho_i = SCHEMA_RHO_SHARE * rho_total / n_releases(columns)
    return 1.0 / np.sqrt(2 * rho_i)


def release(df, columns, rho_total, rng):
    """The private part: one noisy fine histogram per column (+ integer
    counts for capital gain / loss). Returns {column: {...}}."""
    sigma = sigma_for(rho_total, columns)
    out = {}
    for c in columns:
        x = df[c].to_numpy(float)
        lo, hi = RANGE[c]
        x = np.clip(x, lo, hi)
        fe = fine_edges(c)
        k = np.clip(np.searchsorted(fe, x, side='right') - 1, 0, len(fe) - 2)
        r = {'fine_edges': fe,
             'fine': np.bincount(k, minlength=len(fe) - 1)
             + rng.normal(0, sigma, len(fe) - 1)}
        if c in INTEGER_EXTRA:
            r['integer'] = (np.bincount((x - lo).astype(int),
                                        minlength=hi - lo + 1)
                            + rng.normal(0, sigma, hi - lo + 1))
        out[c] = r
    return out, sigma


def _int_counts(c, r):
    """Noisy counts per integer value (integer-grid columns), or None."""
    if c in INTEGER_FINE:
        return r['fine']
    return r.get('integer')


def _mass(fe, w, a, b):
    """Noisy mass in [a, b), cells prorated by overlap length."""
    lo, hi = np.maximum(fe[:-1], a), np.minimum(fe[1:], b)
    frac = np.clip(hi - lo, 0, None) / (fe[1:] - fe[:-1])
    return float((w * frac).sum())


def spikes_from(c, r, sigma, total):
    cnt = _int_counts(c, r)
    if cnt is None:
        return np.array([]), np.array([])
    d = len(cnt)
    pad = np.pad(cnt, 2)          # outside the range: no rows
    neigh = (pad[:-4] + pad[1:-3] + pad[3:-1] + pad[4:]) / 4
    floor = sigma * norm.ppf(1 - ALPHA / d)
    idx = np.nonzero((cnt >= SPIKE_SHARE * total)
                     & (cnt >= SPIKE_EXCESS * np.clip(neigh, 1, None))
                     & (cnt >= floor))[0]
    return idx + RANGE[c][0], cnt[idx]


def build_edges(c, r, sigma, n_public_bins):
    """Ordered edges with singleton spike bins (see module docstring)."""
    lo, hi = RANGE[c]
    fe = r['fine_edges']
    total = float(np.clip(r['fine'], 0, None).sum())
    spikes, spike_cnt = spikes_from(c, r, sigma, total)
    # non-spike mass on the fine grid
    w = np.clip(r['fine'], 0, None).astype(float)
    for s, n_s in zip(spikes, spike_cnt):
        b = np.searchsorted(fe, s, side='right') - 1
        w[b] = max(w[b] - n_s, 0)

    def gaps_of(sp):
        pts = [lo] + [v for s in sp for v in (s, s + 1)] + [hi + 1]
        return [(pts[i], pts[i + 1]) for i in range(0, len(pts), 2)
                if pts[i + 1] > pts[i]]

    keep = list(np.sort(spikes))
    while keep and len(keep) + len(gaps_of(keep)) > MAX_BINS:
        smallest = min(keep, key=lambda s: spike_cnt[list(spikes).index(s)])
        keep.remove(smallest)
    gaps = gaps_of(keep)
    masses = [_mass(fe, w, a, b) for a, b in gaps]
    pieces = [1] * len(gaps)
    extra = max(n_public_bins, len(keep) + len(gaps)) - len(keep) - len(gaps)
    extra = min(extra, MAX_BINS - len(keep) - len(gaps))
    for _ in range(max(extra, 0)):
        gain = [m / (p + 1) for m, p in zip(masses, pieces)]
        g = int(np.argmax(gain))
        if gain[g] < MIN_PIECE_SHARE * total:
            break
        pieces[g] += 1
    edges = []
    for (a, b), p in zip(gaps, pieces):
        edges.append(a)
        if p > 1:      # equal-frequency cuts inside the gap, on fine cells
            inner = fe[(fe > a) & (fe < b)]
            cum = np.array([_mass(fe, w, a, e) for e in inner])
            m = _mass(fe, w, a, b)
            for j in range(1, p):
                i = np.searchsorted(cum, m * j / p)
                if i < len(inner):
                    edges.append(float(inner[i]))
    edges += [float(s) for s in keep] + [float(s) + 1 for s in keep]
    edges.append(hi + 1)
    return np.unique(np.round(edges)), np.array(keep, float)


def _bh(values, sigma):
    """Benjamini-Hochberg at BH_Q for H0 'true count is 0'."""
    p = norm.sf(values / sigma)
    order = np.argsort(p)
    m = len(p)
    ok = p[order] <= BH_Q * np.arange(1, m + 1) / m
    sig = np.zeros(m, bool)
    if ok.any():
        sig[order[:np.nonzero(ok)[0].max() + 1]] = True
    return sig


def decode_tables(c, r, sigma, edges, spikes):
    """Per bin: significant integer values with their weights and the
    residual weight spread uniformly (None: plain uniform)."""
    cnt = _int_counts(c, r)
    if cnt is None:
        return None
    lo = RANGE[c][0]
    sig = _bh(cnt, sigma)
    fe = r['fine_edges']
    w = np.clip(r['fine'], 0, None)
    tables = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a == 1:
            tables.append({'values': [a], 'weights': [1.0], 'residual': 0.0})
            continue
        vals = np.arange(a, b)
        s = sig[(vals - lo).astype(int)] & ~np.isin(vals, spikes)
        sv, sw = vals[s], np.clip(cnt[(vals[s] - lo).astype(int)], 0, None)
        resid = max(_mass(fe, w, a, b) - sw.sum(), 0.0)
        tables.append({'values': sv.tolist(), 'weights': sw.tolist(),
                       'residual': resid})
    return tables


def build(df, columns, rho_total, public_edges, rng):
    """Release + post-processing -> schema {column: {...}}."""
    rel, sigma = release(df, columns, rho_total, rng)
    schema = {'_meta': {'rho_share': SCHEMA_RHO_SHARE, 'sigma': sigma,
                        'releases': n_releases(columns)}}
    for c in columns:
        r = rel[c]
        pub = np.asarray(public_edges[c], float)
        total = float(np.clip(r['fine'], 0, None).sum())
        spikes, cnt = spikes_from(c, r, sigma, total)
        singleton_pub = {a for a, b in zip(pub[:-1], pub[1:]) if b - a == 1}
        new_mass = sum(n for s, n in zip(spikes, cnt) if s not in singleton_pub)
        if new_mass < SWITCH_SHARE * total:
            schema[c] = {'source': 'public', 'edges': pub.tolist(),
                         'spikes': [], 'decode': None}
            continue
        edges, keep = build_edges(c, r, sigma, len(pub) - 1)
        schema[c] = {'source': 'dp', 'edges': edges.tolist(),
                     'spikes': keep.tolist(),
                     'decode': decode_tables(c, r, sigma, edges, keep)}
    return schema


def decode_column(k, table, edges, rng):
    """Bin codes -> values with the private within-bin distribution."""
    edges = np.asarray(edges)
    out = np.empty(len(k))
    for b in np.unique(k):
        m = np.nonzero(k == b)[0]
        t = table[b]
        a, z = edges[b], edges[b + 1]
        vals = np.asarray(t['values'], float)
        wts = np.append(np.asarray(t['weights'], float), t['residual'])
        if wts.sum() <= 0:
            wts = np.append(np.zeros(len(vals)), 1.0)
        pick = rng.choice(len(wts), len(m), p=wts / wts.sum())
        res = pick == len(vals)
        out[m[~res]] = vals[pick[~res]]
        # residual: uniform over the bin's integers, avoiding significant values
        u = rng.integers(a, max(z, a + 1), res.sum()).astype(float)
        for _ in range(5):
            clash = np.isin(u, vals)
            if not clash.any():
                break
            u[clash] = rng.integers(a, max(z, a + 1), clash.sum())
        out[m[res]] = u
    return out
