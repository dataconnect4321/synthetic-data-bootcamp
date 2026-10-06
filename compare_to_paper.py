"""Compare adult_results/*.json (run_adult_flip.py) with the FLIP paper's
Adult results (arXiv:2508.21815, Fig. 3 heatmap / Table 3) and write
adult_results/benchmark_report.md.

    python compare_to_paper.py
"""

import os
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'adult_results')

# Paper Fig. 3 (Adult, sex protected), read off the heatmap.
# Rows = lambda; columns = DP epsilon (None = no DP, 3, 2, 1).
EPS_COLS = ['none', '3', '2', '1']
PAPER = {
    'BER':     {0: [.41, .59, .36, .30], 2: [.52, .48, .49, .63], 4: [.50, .49, .51, .50],
                8: [.50, .48, .50, .52], 12: [.50, .52, .52, .46]},
    'A-NCB':   {0: [.32, .26, .35, .20], 2: [.45, .23, .53, .16], 4: [.92, .91, .73, .68],
                8: [.90, .88, .76, .82], 12: [.87, .92, .82, .89]},
    'eps-ident': {0: [.21, .15, .19, .19], 2: [.14, .12, .14, .09], 4: [.10, .09, .11, .11],
                  8: [.06, .06, .05, .04], 12: [.05, .03, .05, .03]},
}
# Paper Table 3, FLIP lambda=4, eps=inf: relative change vs real data, averaged over ALL datasets.
PAPER_REL = {'BER': 66.76, 'A-NCB': 26.98}


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(OUT_DIR, '*.json'))):
        with open(p) as f:
            r = json.load(f)
        runs[r['tag']] = r
    return runs


def paper(metric, lam, col='none'):
    return PAPER[metric][lam][EPS_COLS.index(col)]


def mine(r):
    pm = r['paper_metrics']
    return {'BER': pm['syn']['ber'], 'A-NCB': pm['syn']['a_ncb'],
            'eps-ident': pm['eps_identifiability']}


def rel_change(syn, real):
    return 100.0 * (syn - real) / real if real else float('nan')


def main():
    runs = load()
    if not runs:
        raise SystemExit(f'no results in {OUT_DIR}; run run_adult_flip.py first')
    L = ['# FLIP on Adult (sex protected): this branch vs. the paper', '']

    L += ['## Headline: lambda = 4, no DP', '',
          '| Metric | Paper (Fig. 3) | This branch | Delta | Ideal |', '|---|---|---|---|---|']
    ideals = {'BER': '0.5', 'A-NCB': '1', 'eps-ident': 'lower'}
    if 'lam4_nodp' in runs:
        m = mine(runs['lam4_nodp'])
        for k in ('BER', 'A-NCB', 'eps-ident'):
            p = paper(k, 4)
            L.append(f'| {k} | {p:.2f} | {m[k]:.3f} | {m[k] - p:+.3f} | {ideals[k]} |')
    else:
        L.append('| (lam4_nodp not run) | | | | |')

    L += ['', '## Baseline: lambda = 0, no DP (metric-implementation sanity check)', '',
          '| Metric | Paper | This branch | Delta |', '|---|---|---|---|']
    if 'lam0_nodp' in runs:
        m = mine(runs['lam0_nodp'])
        for k in ('BER', 'A-NCB', 'eps-ident'):
            p = paper(k, 0)
            L.append(f'| {k} | {p:.2f} | {m[k]:.3f} | {m[k] - p:+.3f} |')
    else:
        L.append('| (lam0_nodp not run) | | | |')

    L += ['', '## Direction of effect: lambda 0 -> 4', '',
          '| Metric | Paper change | This branch change |', '|---|---|---|']
    if 'lam0_nodp' in runs and 'lam4_nodp' in runs:
        a, b = mine(runs['lam0_nodp']), mine(runs['lam4_nodp'])
        for k in ('BER', 'A-NCB', 'eps-ident'):
            L.append(f'| {k} | {paper(k, 4) - paper(k, 0):+.2f} | {b[k] - a[k]:+.3f} |')

    L += ['', '## Real-data reference (measured by this code)', '',
          '| Run | BER real(test) | A-NCB real(test) | BER syn | A-NCB syn | BER rel. to real | A-NCB rel. to real |',
          '|---|---|---|---|---|---|---|']
    for tag, r in runs.items():
        pm = r['paper_metrics']
        s, rt = pm['syn'], pm['real_test']
        L.append(f'| {tag} | {rt["ber"]:.3f} | {rt["a_ncb"]:.3f} | {s["ber"]:.3f} | {s["a_ncb"]:.3f} '
                 f'| {rel_change(s["ber"], rt["ber"]):+.1f}% | {rel_change(s["a_ncb"], rt["a_ncb"]):+.1f}% |')
    L += ['', f'Paper Table 3 (FLIP lambda=4, no DP; averaged over all four datasets, not Adult alone): '
              f'BER {PAPER_REL["BER"]:+.1f}%, A-NCB {PAPER_REL["A-NCB"]:+.1f}%.']

    L += ['', '## Ablation: effect of the two deviations (lambda = 4, no DP)', '',
          '| Run | BER | A-NCB | eps-ident | Income stat. parity gap (syn) |', '|---|---|---|---|---|']
    for tag in ('lam4_nodp', 'lam4_literal'):
        if tag in runs:
            m = mine(runs[tag])
            L.append(f'| {tag} | {m["BER"]:.3f} | {m["A-NCB"]:.3f} | {m["eps-ident"]:.3f} '
                     f'| {runs[tag]["fairness_extra"]["syn_dp_diff"]:.3f} |')

    L += ['', '## Latent mean-gap diagnostic (normalized ||mu_A - mu_B||^2 / pooled var)', '',
          '| Run | shared-mean ratio | group mean gap |', '|---|---|---|']
    for tag, r in runs.items():
        d = r['latent_diagnostic']
        L.append(f'| {tag} | {d["shared_mean_ratio"]:.1f} | {d["group_mean_gap_normalized"]:.4f} |')

    L += ['', '## Other runs (all configurations found)', '',
          '| Run | BER | A-NCB | eps-ident | total eps | num W1 | cat TV | corr delta |', '|---|---|---|---|---|---|---|---|']
    for tag, r in runs.items():
        m, f = mine(r), r['fidelity']
        eps = f'{r["eps_total"]:.2f}' if 'eps_total' in r else 'none'
        L.append(f'| {tag} | {m["BER"]:.3f} | {m["A-NCB"]:.3f} | {m["eps-ident"]:.3f} | {eps} '
                 f'| {f["num_w1_mean"]:.3f} | {f["cat_tv_mean"]:.3f} | {f["corr_delta"]:.3f} |')

    L += ['', '## Paper grid for reference (Adult)', '']
    for k in PAPER:
        L += [f'**{k}**', '', '| lambda | no DP | eps=3 | eps=2 | eps=1 |', '|---|---|---|---|---|']
        for lam, vals in PAPER[k].items():
            L.append(f'| {lam} | ' + ' | '.join(f'{v:.2f}' for v in vals) + ' |')
        L.append('')

    L += ['## Caveats', '',
          '- One fold of the paper\'s 3-fold CV; differences of a few hundredths are within noise.',
          '- 45,222 rows after dropping `?` rows vs. the paper\'s 43,914; `education` dropped to match 6 numeric + 8 categorical.',
          '- BER, A-NCB and eps-identifiability are re-implemented here (gradient-boosted-tree adversary, own FAMD + diagonal GMM with 2 clusters, own entropy-weighted Gower); they are not the paper\'s / SynthEval code.',
          '- Paper values are read off a heatmap image, so they are accurate to the printed two decimals only.',
          '- DP is off in these runs; the paper\'s eps=3/2/1 columns are not reproduced here.', '']

    path = os.path.join(OUT_DIR, 'benchmark_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('\n'.join(L))
    print(f'\nwritten to {path}')


if __name__ == '__main__':
    main()
