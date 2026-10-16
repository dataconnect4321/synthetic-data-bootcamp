"""The paper's Fig. 3 layout (Adult, lambda x epsilon, 8 metrics) with our
fold-0 values, each cell also showing the paper's value underneath.

Runs: scripts/run_heatmap_fold0.sh (hm_e{3,2,1}_lam{L}_{argmax,sample},
hmr_einf_lam{L}_{argmax,sample} and the tier-1 runs), in adult_results/.
Also draws the two real-leakage diagnostics the paper's metrics cannot show:
true-sex BER (tier-1 runs) and A-NCB with a real-trained adversary.

    python paper_heatmap_compare.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'adult_results')
OUT = os.path.join(HERE, 'results')
LAMS = [0, 2, 4, 8, 12]
EPS = ['inf', '3', '2', '1']
EPS_LABEL = ['None', '3.0', '2.0', '1.0']

# Paper Fig. 3, read from the published figure: PAPER[metric][lambda] =
# values for eps None, 3, 2, 1
PAPER = {
    'AUC': {0: [.75, .75, .77, .76], 2: [.72, .69, .74, .75], 4: [.70, .66, .75, .71],
            8: [.57, .69, .63, .65], 12: [.58, .54, .61, .51]},
    'alpha': {0: [.90, .85, .78, .79], 2: [.74, .62, .71, .65], 4: [.64, .40, .55, .59],
              8: [.37, .39, .27, .27], 12: [.29, .23, .34, .28]},
    'beta': {0: [.41, .34, .39, .40], 2: [.34, .31, .36, .25], 4: [.27, .21, .25, .27],
             8: [.18, .17, .16, .13], 12: [.15, .08, .14, .13]},
    'SP': {0: [.18, .18, .22, .22], 2: [.19, .15, .16, .30], 4: [.13, .12, .22, .18],
           8: [.04, .16, .13, .26], 12: [.02, .04, .41, .04]},
    'EO': {0: [.12, .10, .17, .25], 2: [.23, .20, .07, .39], 4: [.15, .26, .22, .18],
           8: [.02, .22, .17, .35], 12: [.05, .14, .46, .06]},
    'epsid': {0: [.21, .15, .19, .19], 2: [.14, .12, .14, .09], 4: [.10, .09, .11, .11],
              8: [.06, .06, .05, .04], 12: [.05, .03, .05, .03]},
    'BER': {0: [.41, .59, .36, .30], 2: [.52, .48, .49, .63], 4: [.50, .49, .51, .50],
            8: [.50, .48, .50, .52], 12: [.50, .52, .52, .46]},
    'ANCB': {0: [.32, .26, .35, .20], 2: [.45, .23, .53, .16], 4: [.92, .91, .73, .68],
             8: [.90, .88, .76, .82], 12: [.87, .92, .82, .89]},
}
# (title, key, direction): 'up' higher better, 'down' lower better, 'half' best at 0.5
PANELS = [('Synth. AUC', 'AUC', 'up'), ('α-Precision', 'alpha', 'up'),
          ('β-Recall', 'beta', 'up'), ('Statistical Parity', 'SP', 'down'),
          ('Equalized Odds', 'EO', 'down'), ('ε-Identifiability', 'epsid', 'down'),
          ('BER (generated sex)', 'BER', 'half'),
          ('A-NCB (adversary on generated sex)', 'ANCB', 'up')]
GREENS = plt.get_cmap('Greens')


def cmap_for(direction):
    if direction == 'up':
        return GREENS
    if direction == 'down':
        return GREENS.reversed()
    ramp = GREENS(np.linspace(0.05, 0.95, 128))
    return LinearSegmentedColormap.from_list('half', np.vstack([ramp, ramp[::-1]]))


def metrics(path):
    d = json.load(open(path))
    pm = d['paper_metrics']
    return {'AUC': pm['downstream_auc'], 'alpha': pm['alpha_precision'],
            'beta': pm['beta_recall'], 'SP': pm['downstream_sp_eo']['sp_diff'],
            'EO': pm['downstream_sp_eo']['eo_diff'],
            'epsid': pm['eps_identifiability'], 'BER': pm['syn']['ber'],
            'ANCB': pm['syn_own_adversary']['a_ncb'],
            'ANCB_real': pm['syn']['a_ncb'], 'eps': d.get('eps_total')}


def grid(decode):
    G = {}
    for e in EPS:
        for lam in LAMS:
            # eps = inf: the retrained runs (hmr_*) when present
            p = os.path.join(RES, f'hm_e{e}_lam{lam}_{decode}.json')
            pr = os.path.join(RES, f'hmr_e{e}_lam{lam}_{decode}.json')
            if e == 'inf' and os.path.exists(pr):
                p = pr
            if os.path.exists(p):
                G[(e, lam)] = metrics(p)
    return G


def draw(ax, M, P, title, direction, vmin, vmax):
    """M, P: (len(LAMS), len(EPS)) ours / paper; rows bottom-up = lambda 0..12."""
    ax.imshow(M, cmap=cmap_for(direction), origin='lower', aspect='auto',
              vmin=vmin, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=8,
                        color='#666')
                continue
            frac = (M[i, j] - vmin) / max(vmax - vmin, 1e-9)
            dark = (frac > 0.55) if direction == 'up' else \
                (frac < 0.45) if direction == 'down' else abs(M[i, j] - 0.5) < 0.12
            col = 'white' if dark else '#1a1a1a'
            ax.text(j, i + 0.12, f'{M[i, j]:.2f}', ha='center', va='center',
                    fontsize=10, color=col, fontweight='bold')
            if P is not None and not np.isnan(P[i, j]):
                ax.text(j, i - 0.22, f'({P[i, j]:.2f})', ha='center',
                        va='center', fontsize=7.5, color=col)
    ax.set_xticks(range(len(EPS)), EPS_LABEL)
    ax.set_yticks(range(len(LAMS)), [str(l) for l in LAMS])
    ax.set_xlabel('ε (privacy)')
    ax.set_title(title, fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)


def figure(decode, G, fname, subtitle):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for ax, (title, key, direction) in zip(axes.flat, PANELS):
        M = np.full((len(LAMS), len(EPS)), np.nan)
        P = np.full_like(M, np.nan)
        for i, lam in enumerate(LAMS):
            for j, e in enumerate(EPS):
                if (e, lam) in G:
                    M[i, j] = G[(e, lam)][key]
                P[i, j] = PAPER[key][lam][j]
        both = np.concatenate([M[~np.isnan(M)], P.ravel()])
        vmin, vmax = (0.0, 1.0) if key == 'BER' else (both.min(), both.max())
        arrow = {'up': '↑', 'down': '↓', 'half': '↕ 0.5'}[direction]
        draw(ax, M, P, f'{title} {arrow}', direction, vmin, vmax)
    for ax in axes[:, 0]:
        ax.set_ylabel('λ (fairness)')
    fig.suptitle('FLIP as published on Adult, fold 0, full pipeline - ours (bold) '
                 'vs the paper\'s Fig. 3 (in brackets). ' + subtitle,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    print('wrote', os.path.join(OUT, fname))


def true_sex_ber():
    """Tier-1 true-sex BER of FLIP as published (argmax; decoding does not
    change it): no DP from pgrid_f0_*_t1, DP from e13_paper_e*_t1."""
    T = np.full((len(LAMS), len(EPS)), np.nan)
    for i, lam in enumerate(LAMS):
        for j, e in enumerate(EPS):
            p = (os.path.join(RES, f'pgrid_f0_lam{lam}_t1.json') if e == 'inf'
                 else os.path.join(RES, f'e13_paper_e{e}_lam{lam}_t1.json'))
            if os.path.exists(p):
                T[i, j] = 1 - json.load(open(p))['paper_metrics']['decoded_leakage']
    return T


def diagnostics(Ga, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    T = true_sex_ber()
    draw(axes[0], T, None, 'True-sex BER (real person; tier-1) ↕ 0.5', 'half',
         0.0, 1.0)
    R = np.full((len(LAMS), len(EPS)), np.nan)
    for i, lam in enumerate(LAMS):
        for j, e in enumerate(EPS):
            if (e, lam) in Ga:
                R[i, j] = Ga[(e, lam)]['ANCB_real']
    draw(axes[1], R, None, 'A-NCB (adversary trained on real data) ↑', 'up',
         0.0, 1.0)
    axes[0].set_ylabel('λ (fairness)')
    fig.suptitle('What the paper\'s metrics cannot show: real leakage, '
                 'FLIP as published, fold 0 (real data: true-sex BER 0.14)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    print('wrote', os.path.join(OUT, fname))


if __name__ == '__main__':
    Ga, Gs = grid('argmax'), grid('sample')
    figure('argmax', Ga, 'heatmap_paper_compare_fold0_argmax.png',
           'Sex decoded by argmax (TabSyn).')
    figure('sample', Gs, 'heatmap_paper_compare_fold0_sampled.png',
           'Sex sampled from its probabilities.')
    diagnostics(Ga, 'heatmap_paper_compare_fold0_real_leakage.png')
