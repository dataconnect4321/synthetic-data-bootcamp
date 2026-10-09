"""Paper-style (Fig. 3) heatmap grid from paper_grid_report.py's CSV: one
panel per metric, lambda on the y-axis (0 at the bottom), folds + mean on
the x-axis, darker green = better; each title carries the real-data value.
Tier-1 shows true-sex BER (each row's real person); the full pipeline shows
eps-identifiability instead. Writes adult_results/heatmap_PREFIX{SUFFIX|_full}.png.

    python paper_grid_heatmap.py [PREFIX] [SUFFIX]  # defaults: pgrid _t1
"""
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

PREFIX = sys.argv[1] if len(sys.argv) > 1 else 'pgrid'
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else '_t1'   # '' = full pipeline
df = pd.read_csv(f'adult_results/{PREFIX}{SUFFIX or "_full"}_grid_report.csv')
runs = df[df['lam'] != 'real'].copy()
runs['lam'] = runs['lam'].astype(int)
real = df[df['lam'] == 'real']
LAMS = sorted(runs['lam'].unique())
FOLDS = sorted(runs['fold'].unique())

# real references not in the CSV, from the run JSONs (same for every lambda)
ref = {'AUC': [], 'alpha': [], 'beta': [], 'validity': [], 'true-sex BER': []}
for f in FOLDS:
    pm = json.load(open(f'adult_results/{PREFIX}_f{f}_lam0{SUFFIX}.json'))['paper_metrics']
    ref['AUC'].append(pm['downstream_auc_real'])
    ref['alpha'].append(pm['alpha_precision_real'])
    ref['beta'].append(pm['beta_recall_real'])
    ref['validity'].append(pm['validity_real'])
    if pm.get('heldout_ceiling') is not None:
        ref['true-sex BER'].append(1 - pm['heldout_ceiling'])
for c in ('BER (paper)', 'A-NCB ownS full', 'A-NCB real full'):
    ref[c] = list(real[c])

# (title, column, direction); 'half' = best at 0.5
PANELS = [
    ('Synth. AUC', 'AUC', 'up'),
    ('α-Precision', 'alpha', 'up'),
    ('β-Recall', 'beta', 'up'),
    ('Statistical Parity', 'SP', 'down'),
    ('Equalized Odds', 'EO', 'down'),
    ('BER (paper: generated sex)', 'BER (paper)', 'half'),
    ('A-NCB (adversary on\ngenerated sex, full GMM)', 'A-NCB ownS full', 'up'),
    ('A-NCB (adversary trained\non real data, full GMM)', 'A-NCB real full', 'up'),
    ('True-sex BER\n(real person; tier-1)', 'true-sex BER', 'half'),
    ('Validity (learned rules)', 'validity', 'up'),
]
if not SUFFIX:   # full pipeline: no source person; privacy panel instead
    PANELS[8] = ('ε-Identifiability', 'eps-id', 'down')
GREENS = plt.get_cmap('Greens')
MIN_SPAN = 0.10


def cmap_for(direction):
    if direction == 'up':
        return GREENS
    if direction == 'down':
        return GREENS.reversed()
    ramp = GREENS(np.linspace(0.05, 0.95, 128))
    return LinearSegmentedColormap.from_list('half', np.vstack([ramp, ramp[::-1]]))


cols = [f'fold {f}' for f in FOLDS] + ['mean']
fig, axes = plt.subplots(2, 5, figsize=(22, 8.6))
for ax, (title, col, direction) in zip(axes.flat, PANELS):
    M = np.full((len(LAMS), len(cols)), np.nan)
    for i, lam in enumerate(LAMS):
        for j, f in enumerate(FOLDS):
            v = runs[(runs['lam'] == lam) & (runs['fold'] == f)][col]
            M[i, j] = float(v.iloc[0]) if len(v) else np.nan
        M[i, -1] = np.nanmean(M[i, :len(FOLDS)])
    r = np.nanmean(ref[col]) if col in ref else np.nan
    finite = M[np.isfinite(M)]
    if direction == 'half':
        d = max(np.max(np.abs(finite - 0.5)), 0.05)
        norm = Normalize(0.5 - d, 0.5 + d)
    else:
        vals = np.append(finite, r) if np.isfinite(r) else finite
        lo, hi = vals.min(), vals.max()
        if hi - lo < MIN_SPAN:
            mid = (hi + lo) / 2
            lo, hi = mid - MIN_SPAN / 2, mid + MIN_SPAN / 2
        pad = (hi - lo) * 0.05
        lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)
        norm = Normalize(lo, hi)
    cm = cmap_for(direction)
    im = ax.imshow(M, cmap=cm, norm=norm, origin='lower', aspect='auto')
    for i in range(len(LAMS)):
        for j in range(len(cols)):
            v = M[i, j]
            rr, gg, bb, _ = cm(norm(v))
            dark = (0.299 * rr + 0.587 * gg + 0.114 * bb) < 0.5
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=11,
                    color='white' if dark else '#1a1a1a')
    ax.axvline(len(FOLDS) - 0.5, color='white', lw=3)
    ax.set_xticks(range(len(cols)), cols, fontsize=10)
    ax.set_yticks(range(len(LAMS)), [f'{l:g}' for l in LAMS], fontsize=11)
    arrow = {'up': '↑', 'down': '↓', 'half': '↕ 0.5'}[direction]
    refs = f'  (real {r:.2f})' if np.isfinite(r) else ''
    ax.set_title(f'{title} {arrow}{refs}', fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0)
for ax in axes[:, 0]:
    ax.set_ylabel('λ (fairness)', fontsize=12)
fig.suptitle('FLIP as published on Adult, ' + ('tier-1 (no diffusion)' if SUFFIX else 'full pipeline (VAE + latent diffusion on μ+σε)') + ', 3 folds: '
             'unmasked, paper CKAᵀ, ELBO kept in Phase 2, Phase-2 β=1e-2, '
             'argmax sex, no DP. Darker green = better.', fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = f'adult_results/heatmap_{PREFIX}{SUFFIX or "_full"}.png'
fig.savefig(out, dpi=150)
print('wrote', out)
