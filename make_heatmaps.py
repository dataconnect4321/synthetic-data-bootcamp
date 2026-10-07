"""Heatmap grid of the final results, in the style of the paper's Adult figure
(Fig. 3: one panel per metric, lambda on the y-axis, darker green = better).

    python make_heatmaps.py [--prefix fin] [--tier1] [--out FILE.png]

The paper's x-axis is the DP budget; these runs use no DP, so the columns are
the three CV folds plus their mean - which also shows the fold spread the
paper's averages hide. Each panel title carries the real-data reference.
A CSV with every plotted value is written next to the PNG (table view).

Panels: Synth. AUC, alpha-precision, beta-recall, statistical parity,
equalized odds, eps-identifiability (full pipeline only; tier-1 decodes
near-copies of training rows, so it is shown as n/a there), true-sex BER
(1 - leakage; tier-1 only), BER, A-NCB (full-cov), and the largest non-sex
category share change (catches categories that vanish).
"""

import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

OUT_DIR = 'adult_results'

# (title, synthetic path, real-reference path, direction)
# direction: 'up' higher is better, 'down' lower is better, 'half' best at 0.5
PANELS_FULL = [
    ('Synth. AUC',          'paper_metrics.downstream_auc',
     'paper_metrics.downstream_auc_real', 'up'),
    ('α-Precision',         'paper_metrics.alpha_precision',
     'paper_metrics.alpha_precision_real', 'up'),
    ('β-Recall',            'paper_metrics.beta_recall',
     'paper_metrics.beta_recall_real', 'up'),
    ('Statistical Parity',  'paper_metrics.downstream_sp_eo.sp_diff',
     None, 'down'),
    ('Equalized Odds',      'paper_metrics.downstream_sp_eo.eo_diff',
     None, 'down'),
    ('ε-Identifiability',   'paper_metrics.eps_identifiability', None, 'down'),
    # Definition 1's BER scored against each decoded row's REAL sex (tier-1
    # only): 1 - leakage. The paper's BER uses the generated sex column,
    # which is largely invented and swings between runs.
    ('True-sex BER',        'true_sex_ber', 'true_sex_ber_real', 'half'),
    ('BER',                 'paper_metrics.syn.ber',
     'paper_metrics.real_test.ber', 'half'),
    ('A-NCB',               'paper_metrics.syn_fullcov.a_ncb|'
                            'paper_metrics.syn.a_ncb',
     'paper_metrics.real_test_fullcov.a_ncb|paper_metrics.real_test.a_ncb',
     'up'),
    ('Max category share change', 'max_nonsex_share_tv', None, 'down'),
]
# why a panel can be empty
NA_REASON = {
    'ε-Identifiability': ('n/a in tier-1\n(decodes near-copies of\n'
                          'training rows)'),
    'True-sex BER': ('n/a in full pipeline\n(no source row with\n'
                     'a real sex to compare)'),
}

GREENS = plt.get_cmap('Greens')
# smallest value range a panel's colour scale may cover (all metrics are on
# a 0-1 scale); smaller differences stay close in colour
MIN_SPAN = 0.10


def get(d, path):
    if path is None:
        return np.nan
    if path in ('true_sex_ber', 'true_sex_ber_real'):
        pm = d.get('paper_metrics', {})
        v = pm.get('decoded_leakage' if path == 'true_sex_ber'
                   else 'heldout_ceiling')
        return 1.0 - float(v) if v is not None else np.nan
    if path == 'max_nonsex_share_tv':
        tv = d.get('paper_metrics', {}).get('all_cat_share_tv') or {}
        s_idx = 0  # 'sex' is the first categorical column
        vals = [v for k, v in tv.items() if int(k) != s_idx]
        return float(max(vals)) if vals else np.nan
    # 'a|b' tries each path in turn (older result files stored the
    # full-covariance A-NCB under syn_fullcov / real_test_fullcov)
    for p in path.split('|'):
        x = d
        try:
            for k in p.split('.'):
                x = x[int(k)] if isinstance(x, list) else x[k]
            return float(x)
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return np.nan


def load(prefix, tier1):
    suffix = '_t1' if tier1 else ''
    runs, skipped = {}, []
    for path in sorted(glob.glob(os.path.join(
            OUT_DIR, f'{prefix}_f?_lam?{suffix}.json'))):
        d = json.load(open(path))
        if d.get('collapsed') or 'paper_metrics' not in d:
            skipped.append(os.path.basename(path))
            continue
        runs[(int(float(d['args']['lam'])), int(d['args']['fold']))] = d
    return runs, skipped


def panel_cmap(direction):
    """Darker = better. 'half' is darkest at the centre (0.5) and light at
    both ends: the Greens ramp mirrored."""
    if direction == 'up':
        return GREENS
    if direction == 'down':
        return GREENS.reversed()
    ramp = GREENS(np.linspace(0.05, 0.95, 128))
    return LinearSegmentedColormap.from_list(
        'half', np.vstack([ramp, ramp[::-1]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--prefix', default='fin')
    ap.add_argument('--tier1', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs, skipped = load(args.prefix, args.tier1)
    if not runs:
        raise SystemExit('no runs found')
    lams = sorted({l for l, _ in runs})
    folds = sorted({f for _, f in runs})
    cols = [f'fold {f}' for f in folds] + ['mean']
    panels = PANELS_FULL

    fig, axes = plt.subplots(2, 5, figsize=(21, 8.2))
    rows_csv = ['panel,lambda,' + ','.join(cols) + ',real_ref']
    for ax, (title, spath, rpath, direction) in zip(axes.flat, panels):
        M = np.full((len(lams), len(cols)), np.nan)
        for i, lam in enumerate(lams):
            for j, f in enumerate(folds):
                if (lam, f) in runs:
                    M[i, j] = get(runs[(lam, f)], spath)
            if np.isfinite(M[i, :len(folds)]).any():
                M[i, -1] = np.nanmean(M[i, :len(folds)])
        real = [get(d, rpath) for d in runs.values()]
        real = np.nanmean(real) if np.isfinite(real).any() else np.nan

        finite = M[np.isfinite(M)]
        if finite.size == 0:
            ax.set_axis_off()
            ax.set_title(title, fontsize=13)
            ax.text(0.5, 0.5, NA_REASON.get(title, 'n/a'), ha='center',
                    va='center', fontsize=11, color='#666666',
                    transform=ax.transAxes)
            continue
        if direction == 'half':
            d = max(np.max(np.abs(finite - 0.5)), 0.05)
            norm = Normalize(0.5 - d, 0.5 + d)
        else:
            # include the real reference and enforce a minimum span, so
            # fold-level noise (e.g. AUC 0.910 vs 0.914) is not drawn as a
            # dark-vs-light contrast
            vals = np.append(finite, real) if np.isfinite(real) else finite
            lo, hi = vals.min(), vals.max()
            if hi - lo < MIN_SPAN:
                mid = (hi + lo) / 2
                lo, hi = mid - MIN_SPAN / 2, mid + MIN_SPAN / 2
            pad = (hi - lo) * 0.05
            lo, hi = lo - pad, hi + pad
            # every metric here lives in [0, 1]: slide the window inside it
            if hi > 1:
                lo, hi = lo - (hi - 1), 1.0
            if lo < 0:
                lo, hi = 0.0, hi - lo
            norm = Normalize(lo, hi)
        cmap = panel_cmap(direction)
        # lambda 0 at the bottom, like the paper
        im = ax.imshow(M, cmap=cmap, norm=norm, origin='lower',
                       aspect='auto')
        for i in range(len(lams)):
            for j in range(len(cols)):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, '–', ha='center', va='center',
                            color='#888888', fontsize=12)
                    continue
                # ink by cell darkness so the number stays legible
                r, g, b, _ = cmap(norm(v))
                dark = (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=12, color='white' if dark else '#1a1a1a')
        # separate the mean column from the folds
        ax.axvline(len(folds) - 0.5, color='white', lw=3)
        ax.set_xticks(range(len(cols)), cols, fontsize=10)
        ax.set_yticks(range(len(lams)), [f'{l:g}' for l in lams],
                      fontsize=11)
        arrow = {'up': '↑', 'down': '↓', 'half': '↕ 0.5'}[direction]
        ref = f'  (real {real:.2f})' if np.isfinite(real) else ''
        ax.set_title(f'{title} {arrow}{ref}', fontsize=13)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=9, length=0)
        for i, lam in enumerate(lams):
            rows_csv.append(f'"{title}",{lam:g},' + ','.join(
                '' if not np.isfinite(v) else f'{v:.4f}' for v in M[i])
                + f',{"" if not np.isfinite(real) else f"{real:.4f}"}')
    for ax in axes[:, 0]:
        ax.set_ylabel('λ (fairness)', fontsize=12)

    kind = 'tier-1 (reconstruct-through-latent)' if args.tier1 \
        else 'full pipeline (VAE + latent diffusion)'
    a = next(iter(runs.values()))['args']
    setup = ', '.join(x for x in [
        'masked proxies' if a.get('mask_proxies') else 'raw proxies',
        (f"decoded-joint {a['decoded_joint']}"
         if a.get('decoded_joint', 'off') != 'off' else None),
        (f"marginal {a.get('marginal_kind', 'tv')} x{a['marginal_weight']:g}"
         if a.get('marginal_weight') else 'marginal off'),
        f"stages {a.get('stage_reduce', 'mean')}",
        'no DP' if a.get('dp_noise') is None else f"DP noise {a['dp_noise']}",
    ] if x)
    fig.suptitle(f'FLIP on Adult, {kind}: {setup}. Darker green = better.',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = args.out or os.path.join(
        OUT_DIR, f'heatmap_{args.prefix}{"_t1" if args.tier1 else ""}.png')
    fig.savefig(out, dpi=150)
    with open(os.path.splitext(out)[0] + '.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows_csv) + '\n')
    print(f'wrote {out} and {os.path.splitext(out)[0]}.csv')
    if skipped:
        print('skipped (collapsed/incomplete): ' + ', '.join(skipped))


if __name__ == '__main__':
    main()
