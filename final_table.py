"""Final results table for the paper-config 3-fold runs (tier-1 and full).

Builds the comparison table (real reference | one column per lambda, with
fold spreads) from the per-fold JSONs, robust to missing or collapsed runs:
missing files and collapse-flagged runs are skipped and reported, and
every aggregate shows its fold count. A per-fold table follows, so no
result is hidden in an average.

    python final_table.py [--tier1] [--prefix pc|re] [--out FILE]

--prefix flip reads the final FLIP sweep, --prefix rpb the repair
baseline. With --mask-proxies runs the real reference is the ORIGINAL data;
the 'vs masked real' rows compare against the real data after the same
masking.
"""

import argparse
import glob
import json
import os
import sys
import numpy as np

OUT_DIR = 'adult_results'

# metric name -> (synthetic path, real-reference path, mode). Paths are
# rooted at the run JSON; a numeric path element indexes a list. Real path
# None means the real value is not defined (shown as '--'); synthetic path
# None means a real-only row. Modes: plain, invert (1 - v), headroom (the
# largest possible relative A-NCB gain, (1 - v) / v in %), rel (synthetic
# path is a (syn, real) pair; 100 * (syn - real) / real per fold).
METRICS = [
    ('BER true (1 - leakage)', 'paper_metrics.decoded_leakage',
     'paper_metrics.heldout_ceiling', 'invert'),
    ('Decoded leakage',        'paper_metrics.decoded_leakage',
     'paper_metrics.heldout_ceiling', 'plain'),
    ('BER (synthetic S col)',  'paper_metrics.syn.ber',
     'paper_metrics.real_test.ber', 'plain'),
    ('BER vs masked real',     'paper_metrics.syn.ber',
     'paper_metrics.real_test_reencoded.ber', 'plain'),
    ('BER rel. to real (%)',   'paper_metrics.outcome_replication.ber_rel_pct',
     None, 'plain'),
    # A-NCB: full-covariance GMM (sklearn default, likely the paper's
    # setup), strict real-trained adversary, 10 seeds.
    ('A-NCB',                  'paper_metrics.syn_fullcov.a_ncb|paper_metrics.syn.a_ncb',
     'paper_metrics.real_test_fullcov.a_ncb|paper_metrics.real_test.a_ncb', 'plain'),
    ('A-NCB (median of seeds)', 'paper_metrics.syn_fullcov.a_ncb_median|paper_metrics.syn.a_ncb_median',
     'paper_metrics.real_test_fullcov.a_ncb_median|paper_metrics.real_test.a_ncb_median', 'plain'),
    ('A-NCB seed sd',          'paper_metrics.syn_fullcov.a_ncb_sd|paper_metrics.syn.a_ncb_sd',
     'paper_metrics.real_test_fullcov.a_ncb_sd|paper_metrics.real_test.a_ncb_sd', 'plain'),
    ('A-NCB vs masked real',   'paper_metrics.syn_fullcov.a_ncb|paper_metrics.syn.a_ncb',
     'paper_metrics.real_test_reencoded_fullcov.a_ncb|paper_metrics.real_test_reencoded.a_ncb', 'plain'),
    ('A-NCB rel. to real (%)', ('paper_metrics.syn_fullcov.a_ncb|paper_metrics.syn.a_ncb',
                                'paper_metrics.real_test_fullcov.a_ncb|paper_metrics.real_test.a_ncb'),
     None, 'rel'),
    ('A-NCB headroom (%)',     None,
     'paper_metrics.real_test_fullcov.a_ncb|paper_metrics.real_test.a_ncb', 'headroom'),
    ('Downstream AUC',         'paper_metrics.downstream_auc',
     'paper_metrics.downstream_auc_real', 'plain'),
    ('AUC rel. to real (%)',   'paper_metrics.outcome_replication.auc_rel_pct',
     None, 'plain'),
    ('Downstream SP diff',     'paper_metrics.downstream_sp_eo.sp_diff',
     None, 'plain'),
    ('Downstream EO diff',     'paper_metrics.downstream_sp_eo.eo_diff',
     None, 'plain'),
    ('Income rate',            'paper_metrics.syn_income_rate',
     'paper_metrics.real_income_rate', 'plain'),
    ('True-group target gap',  'paper_metrics.true_group_target_gap',
     None, 'plain'),
    ('S share TV',             'paper_metrics.s_share_tv',
     None, 'plain'),
    ('Impossible-comb rate',   'paper_metrics.impossible_combination_rate.any',
     'paper_metrics.impossible_combination_rate_real.any', 'plain'),
    ('Validity (learned rules)', 'paper_metrics.validity',
     'paper_metrics.validity_real', 'plain'),
    ('alpha-precision',        'paper_metrics.alpha_precision',
     'paper_metrics.alpha_precision_real', 'plain'),
    ('beta-recall',            'paper_metrics.beta_recall',
     'paper_metrics.beta_recall_real', 'plain'),
    ('Fidelity: num W1',       'fidelity.num_w1_mean', None, 'plain'),
    ('Fidelity: cat TV',       'fidelity.cat_tv_mean', None, 'plain'),
    ('Fidelity: corr delta',   'fidelity.corr_delta', None, 'plain'),
    ('eps-identifiability',    'paper_metrics.eps_identifiability',
     None, 'plain'),
]


def get(d, path):
    """Float at a dotted path, or NaN if anything is missing. 'a|b' tries
    each path in turn (older result files stored the full-covariance A-NCB
    under syn_fullcov / real_test_fullcov)."""
    if path is None:
        return np.nan
    for p in path.split('|'):
        x = d
        try:
            for k in p.split('.'):
                x = x[int(k)] if isinstance(x, list) else x[k]
            return float(x)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return np.nan


def transform(v, mode):
    if np.isnan(v):
        return v
    if mode == 'invert':
        return 1.0 - v
    if mode == 'headroom':
        return 100.0 * (1.0 - v) / v
    return v


def load_runs(pattern):
    """Return {(lam, fold): run JSON} for valid (non-collapsed) runs
    matching the pattern. Collapsed or unreadable runs are reported."""
    runs, skipped = {}, []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        try:
            d = json.load(open(path))
        except (OSError, json.JSONDecodeError) as e:
            skipped.append(f'{name}: unreadable ({e})')
            continue
        if d.get('collapsed'):
            skipped.append(f'{name}: COLLAPSED (income '
                           f"{d.get('syn_income_rate', float('nan')):.3f})")
            continue
        if d.get('paper_metrics') is None:
            skipped.append(f'{name}: no paper_metrics (incomplete run)')
            continue
        lam = int(float(d['args']['lam']))
        fold = int(d['args']['fold'])
        runs[(lam, fold)] = d
    return runs, skipped


def syn_values(runs, lam, path, mode):
    """{fold: value} for this lambda."""
    out = {}
    for (l, f), d in sorted(runs.items()):
        if l == lam:
            if mode == 'rel':
                s_, r_ = get(d, path[0]), get(d, path[1])
                v = 100.0 * (s_ - r_) / r_ if r_ else np.nan
            else:
                v = transform(get(d, path), mode)
            if not np.isnan(v):
                out[f] = v
    return out


def real_values(runs, path, mode):
    """{fold: value} of the real reference (it depends only on the fold)."""
    out = {}
    for (l, f), d in sorted(runs.items()):
        v = transform(get(d, path), mode)
        if not np.isnan(v):
            out.setdefault(f, v)
    return out


def fmt(vals):
    if not vals:
        return '     --'
    v = np.array(list(vals.values()), float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f'{v.mean():.3f} +/- {sd:.3f} (n={len(v)})'


def fmt_folds(vals, folds):
    return ' '.join(f'{vals[f]:7.3f}' if f in vals else '     --'
                    for f in folds)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tier1', action='store_true',
                    help='aggregate the tier-1 runs instead of full diffusion')
    ap.add_argument('--prefix', default='pc',
                    help='run-tag prefix: pc (raw) or re (re-encoded)')
    ap.add_argument('--out', default=None,
                    help='also write the table to this file')
    args = ap.parse_args()

    suffix = '_t1' if args.tier1 else ''
    pattern = os.path.join(OUT_DIR, f'{args.prefix}_f?_lam?{suffix}.json')
    runs, skipped = load_runs(pattern)

    lines = []
    title = ('FINAL RESULTS - tier-1 (reconstruct-through-latent)'
             if args.tier1 else
             'FINAL RESULTS - full pipeline (VAE + latent diffusion)')
    lines.append(title)
    lines.append(f'source: {pattern}')
    lines.append('')

    if skipped:
        lines.append('Skipped / missing runs:')
        for s in skipped:
            lines.append(f'  {s}')
        lines.append('')

    folds = sorted({f for (_, f) in runs})
    lams = sorted({l for (l, _) in runs})
    lines.append(f'folds present: {folds}   lambdas present: {lams}')
    lines.append('')

    rows = []
    for name, spath, rpath, mode in METRICS:
        real = real_values(runs, rpath, mode)
        syn = {lam: (syn_values(runs, lam, spath, mode) if spath else {})
               for lam in lams}
        rows.append((name, real, syn))

    # ---- mean +/- sd ------------------------------------------------------
    header = f'{"metric":26s} | {"real ref":22s} | ' + ' | '.join(
        f'{"lam" + str(l):22s}' for l in lams)
    lines.append(header)
    lines.append('-' * len(header))
    for name, real, syn in rows:
        lines.append(f'{name:26s} | {fmt(real):22s} | ' + ' | '.join(
            f'{fmt(syn[l]):22s}' for l in lams))

    # match-valid summary
    for lam in lams:
        mv = [d['paper_metrics']['match_valid'] for (l, f), d in runs.items()
              if l == lam and 'match_valid' in d['paper_metrics']]
        if mv:
            lines.append(f'match-valid lam{lam}: {sum(mv)}/{len(mv)} folds')

    # ---- per fold ---------------------------------------------------------
    lines.append('')
    lines.append('PER FOLD (folds ' + ', '.join(map(str, folds)) + ')')
    block = 8 * len(folds) - 1
    header = f'{"metric":26s} | {"real ref":{block}s} | ' + ' | '.join(
        f'{"lam" + str(l):{block}s}' for l in lams)
    lines.append(header)
    lines.append('-' * len(header))
    for name, real, syn in rows:
        lines.append(f'{name:26s} | {fmt_folds(real, folds)} | ' + ' | '.join(
            fmt_folds(syn[l], folds) for l in lams))

    table = '\n'.join(lines)
    print(table)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(table + '\n')
        print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
