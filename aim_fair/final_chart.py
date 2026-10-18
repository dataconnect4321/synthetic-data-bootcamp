"""Final comparison (RESULTS.md): the AIM hybrid with private bins vs our
faithful reproduction of FLIP vs FLIP's reported numbers, at epsilon 3.

  AIM hybrid  run_final.sh tables (3 folds x 10 refits; mean, sd across folds)
  FLIP repro  FLIP as published, lambda 4, fold 0, the paper-closest setup of
              scripts/run_heatmap_fold0.sh (hm_e3_lam4), scored from its
              released CSV like the AIM tables:
                python run_adult_flip.py --score-csv \\
                    adult_results/hm_e3_lam4_argmax_synthetic.csv --fold 0 \\
                    --score-collapsed --no-save-run --tag flipfaithful_e3_f0_lam4
  FLIP paper  values read off the paper's Fig. 3 (Adult, lambda 4, eps 3)

Writes results/final_comparison.csv and results/final_comparison.png.

    python aim_fair/final_chart.py
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
METRICS = [   # key, label, better, getter
    ('income_parity', 'Income parity', 'lower',
     lambda d: d['paper_metrics']['downstream_sp_eo']['sp_diff']),
    ('eo_gap', 'Equalized-odds gap', 'lower',
     lambda d: d['paper_metrics']['downstream_sp_eo']['eo_diff']),
    ('ber', 'BER, synthetic sex label', '0.5 best',
     lambda d: d['paper_metrics']['syn']['ber']),
    ('ancb', "A-NCB (paper's version)*", 'higher',
     lambda d: d['paper_metrics']['syn_own_adversary']['a_ncb']),
    ('auc', 'Income AUC on real people', 'higher',
     lambda d: d['paper_metrics']['downstream_auc']),
    ('alpha', 'Alpha-precision', 'higher',
     lambda d: d['paper_metrics']['alpha_precision']),
    ('beta', 'Beta-recall', 'higher',
     lambda d: d['paper_metrics']['beta_recall']),
    ('eps_id', 'Epsilon-identifiability', 'lower',
     lambda d: d['paper_metrics']['eps_identifiability']),
    ('validity', 'Validity', 'higher',
     lambda d: d['paper_metrics']['validity']),
    ('impossible', 'Impossible rows', 'lower',
     lambda d: d['paper_metrics']['impossible_combination_rate']['any']),
]
# FLIP paper, Fig. 3, Adult, lambda 4, epsilon 3 (not reported: validity,
# impossible rows)
PAPER = {'income_parity': 0.12, 'eo_gap': 0.26, 'ber': 0.49, 'ancb': 0.91,
         'auc': 0.66, 'alpha': 0.40, 'beta': 0.21, 'eps_id': 0.09}
METHODS = ['AIM hybrid, private bins\n(rigorous eps 3; 3 folds x 10 refits)',
           'FLIP, faithful reproduction\n(nominal eps 3; fold 0)',
           'FLIP paper, reported\n(nominal eps 3)']
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']     # categorical slots 1-3
INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'


def load(tag):
    with open(os.path.join(ROOT, 'adult_results', tag + '.json')) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aim-pattern', default='aimfinal_e3_f{f}_r{r}')
    ap.add_argument('--refits', type=int, default=10)
    ap.add_argument('--flip-tag', default='flipfaithful_e3_f0_lam4')
    a = ap.parse_args()

    folds = []
    for f in range(3):
        runs = [load(a.aim_pattern.format(f=f, r=r)) for r in range(a.refits)]
        folds.append([np.mean([g(d) for d in runs]) for _, _, _, g in METRICS])
    folds = np.array(folds)
    aim_m, aim_s = folds.mean(0), folds.std(0, ddof=1)
    flip = load(a.flip_tag)
    flip_v = [g(flip) for _, _, _, g in METRICS]

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', 'final_comparison.csv'), 'w',
              newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'better', 'aim_hybrid_private', 'aim_sd_folds',
                    'flip_faithful_repro', 'flip_paper_reported'])
        for i, (k, lab, better, _) in enumerate(METRICS):
            w.writerow([lab, better, round(aim_m[i], 4), round(aim_s[i], 4),
                        round(flip_v[i], 4), PAPER.get(k, '')])

    plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.6), facecolor=SURFACE)
    for ax, (i, (k, lab, better, _)) in zip(axes.flat, enumerate(METRICS)):
        ax.set_facecolor(SURFACE)
        vals = [aim_m[i], flip_v[i], PAPER.get(k, np.nan)]
        top = np.nanmax(vals) * 1.25 if np.nanmax(vals) > 0 else 1
        for j, v in enumerate(vals):
            if np.isnan(v):
                ax.text(j, top * 0.04, 'not\nreported', ha='center',
                        va='bottom', fontsize=8.5, color=MUTED)
                continue
            ax.bar(j, v, width=0.7, color=COLORS[j], edgecolor=SURFACE,
                   linewidth=2)
            if j == 0:
                ax.errorbar(j, v, yerr=aim_s[i], color=INK2, capsize=3,
                            linewidth=1)
            ax.text(j, v + top * 0.03 + (aim_s[i] if j == 0 else 0),
                    f'{v:.2f}' if k != 'impossible' else f'{v:.0%}',
                    ha='center', va='bottom', fontsize=9.5, color=INK)
        if k == 'ber':
            ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 2)))
            top = max(top, 0.65)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.6, 2.6)
        ax.set_xticks([])
        ax.set_title(f'{lab}\n({better} is better)'
                     if better != '0.5 best' else f'{lab}\n(0.5 is best)',
                     fontsize=10, color=INK)
        ax.grid(axis='y', color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, METHODS, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    fig.text(0.5, 0.005, '* A-NCB with the paper\'s own-sex adversary rewards a '
             'drifting sex label; shown for comparison with the paper only. '
             'Error bars: sd across folds.', ha='center', fontsize=8.5,
             color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    out = os.path.join(ROOT, 'results', 'final_comparison.png')
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    print('wrote', out)


if __name__ == '__main__':
    main()
