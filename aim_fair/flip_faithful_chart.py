"""FLIP-faithful comparison (RESULTS.md): the AIM hybrid vs our faithful
reproduction of FLIP vs FLIP's reported numbers, every number from FLIP's
own evaluation protocol where the paper defines one.

  AIM hybrid  run_adult_refits.sh tables (aimre), fold 0, 3 refits averaged
  FLIP repro  flipfaithful_e3_f0_lam4 (see final_chart.py), fold 0
  FLIP paper  values read off the paper's Fig. 3 (Adult, lambda 4, eps 3)

Data-level metrics (BER, A-NCB in the paper's own-adversary form, tstr,
utility, realism) come from the run_adult_flip.py --score-csv JSONs.
Downstream metrics come from flip_protocol.py (LightGBM, every
non-protected attribute as a target, Eqs. 14 and 15; income at 0.5).

Writes results/flip_faithful_comparison.{csv,png}.

    python aim_fair/flip_faithful_chart.py
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
SOURCES = [   # name, scorer JSONs, flip_protocol.py source, color
    ('FLIP paper (reported)', None, None, '#ffffff'),
    ('FLIP reproduction', 'flipfaithful_e3_f0_lam4.json',
     'FLIP reproduction (lambda 4, nominal eps 3)', '#eb6834'),
    ('AIM hybrid', 'aimre_e3_f0_r*.json', 'AIM hybrid', '#2a78d6'),
]
PM = 'paper_metrics'
METRICS = [   # key, label, better, format, getter (scorer JSON)
    ('ber', 'Sex-inference BER\n(0.5 = sex unrecoverable)', 'higher', '{:.2f}',
     lambda d: d[PM]['syn']['ber']),
    ('ancb', "A-NCB, paper's version\n(1 = full independence)", 'higher',
     '{:.2f}', lambda d: d[PM]['syn_own_adversary']['a_ncb']),
    ('auc', 'Income AUC on real people\n(utility)', 'higher', '{:.2f}',
     lambda d: d[PM]['downstream_auc']),
    ('alpha', 'Alpha-precision\n(realism)', 'higher', '{:.2f}',
     lambda d: d[PM]['alpha_precision']),
    ('impossible', 'Impossible rows (%)\n(e.g. male wives)', 'lower', '{:.1f}%',
     lambda d: 100 * d[PM]['impossible_combination_rate']['any']),
    ('parity', 'Income parity\n(threshold 0.5)', 'lower', '{:.3f}', None),
    ('eo', 'Income EO gap*\n(threshold 0.5)', 'lower', '{:.3f}', None),
    ('sp_cat', 'Mean parity, all categorical\ntargets (FLIP Eq. 14)', 'lower',
     '{:.3f}', None),
    ('w1_num', 'Mean W1/sd, all numeric\ntargets (FLIP Eq. 15)', 'lower',
     '{:.3f}', None),
    ('validity', 'Validity\n(rule-consistent rows)', 'higher', '{:.2f}',
     lambda d: d[PM]['validity']),
]
EXTRA = [     # table only
    ('tstr', 'tstr sex BER (folded; 0.5 best)', 'higher',
     lambda d: d[PM]['tstr_sex_ber_folded']),
    ('beta', 'Beta-recall', 'higher', lambda d: d[PM]['beta_recall']),
    ('eps_id', 'Epsilon-identifiability', 'lower',
     lambda d: d[PM]['eps_identifiability']),
]
# FLIP paper, Fig. 3, Adult, lambda 4, epsilon 3; not reported: tstr,
# impossible rows, validity, per-target means (FLIP's code is not public)
PAPER = {'ber': 0.49, 'ancb': 0.91, 'auc': 0.66, 'alpha': 0.40,
         'parity': 0.12, 'eo': 0.26, 'beta': 0.21, 'eps_id': 0.09}


def values():
    tg = pd.read_csv(os.path.join(ROOT, 'results', 'flip_protocol_targets.csv'))
    inc = pd.read_csv(os.path.join(ROOT, 'results', 'flip_protocol_income.csv'))
    tg, inc = tg[tg.scope == 'fold0'], inc[inc.scope == 'fold0']
    out = {}
    for name, pat, proto, _ in SOURCES:
        if pat is None:
            out[name] = dict(PAPER)
            continue
        runs = [json.load(open(p)) for p in
                sorted(glob.glob(os.path.join(ROOT, 'adult_results', pat)))]
        v = {k: float(np.mean([g(d) for d in runs]))
             for k, *_, g in METRICS + EXTRA if g is not None}
        t = tg[tg.source == proto]
        i = inc[(inc.source == proto) & (inc.threshold.astype(str) == '0.5')]
        v.update(sp_cat=t[t.metric == 'SP (Eq. 14)'].value_sd.mean(),
                 w1_num=t[t.metric == 'W1'].value_sd.mean(),
                 parity=i.parity.iloc[0], eo=i['EO = TPR gap'].iloc[0])
        out[name] = v
    return out


def main():
    v = values()
    names = [s[0] for s in SOURCES]
    rows = [{'metric': lab.replace('\n', ' '), 'better': b,
             **{n: v[n].get(k) for n in names}}
            for k, lab, b, *_ in METRICS + EXTRA]
    pd.DataFrame(rows).round(4).to_csv(os.path.join(
        ROOT, 'results', 'flip_faithful_comparison.csv'), index=False)

    fig, axs = plt.subplots(2, 5, figsize=(19, 6.6))
    y = list(range(len(names)))[::-1]
    for ax, (k, lab, b, fmt, _) in zip(axs.flat, METRICS):
        vals = [v[n].get(k) for n in names]
        vmax = max(x for x in vals if x is not None) or 1
        for yi, vi, (n, _, _, c) in zip(y, vals, SOURCES):
            if vi is None:
                ax.text(vmax * 0.02, yi, 'not reported', va='center',
                        fontsize=9.5, color='#888', style='italic')
                continue
            paper = n.startswith('FLIP paper')
            ax.barh(yi, vi, color=c, height=0.62,
                    edgecolor='#eb6834' if paper else 'white',
                    linewidth=1.4 if paper else 2,
                    hatch='////' if paper else None)
            ax.text(vi + vmax * 0.02, yi, fmt.format(vi), va='center',
                    fontsize=10, color='#222')
        ax.set_title(f'{lab}\n{"↑" if b == "higher" else "↓"} {b} is better',
                     fontsize=10.5, loc='left', color='#222')
        ax.set_xlim(0, vmax * 1.3)
        ax.set_ylim(-0.5, len(names) - 0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax in axs[:, 0] else [], fontsize=10)
        ax.tick_params(axis='x', labelsize=8, colors='#666')
        ax.tick_params(axis='y', length=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#ccc')
        ax.grid(axis='x', color='#eee')
        ax.set_axisbelow(True)
    fig.text(0.01, 0.985, 'FLIP vs AIM hybrid on UCI Adult (sex protected, '
             'ε = 3), scored by FLIP\'s own protocol', fontsize=14,
             weight='bold', va='top')
    fig.text(0.01, 0.945, 'Our sources: fold 0, scored identically (LightGBM, '
             'train on synthetic / test on real, every attribute a target). '
             'FLIP paper: read off its Fig. 3 (λ 4, ε 3), from its own code, '
             'so indicative only.\n*EO: the paper does not define its number; '
             'ours is the TPR gap.', fontsize=10, color='#555', va='top')
    fig.legend(handles=[Patch(facecolor=c, hatch='////' if i == 0 else None,
                              edgecolor='#eb6834' if i == 0 else c, label=n)
                        for i, (n, _, _, c) in enumerate(SOURCES)],
               loc='upper right', ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.995, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.87), h_pad=2.5)
    out = os.path.join(ROOT, 'results', 'flip_faithful_comparison.png')
    fig.savefig(out, dpi=150, facecolor='white')
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(out)


if __name__ == '__main__':
    main()
