"""Post-hoc (tau) vs in-model (lambda) proxy decoupling on one plot.

    python decouple_curves.py [posthoc_json] [inmodel_glob] [out_png]
    defaults: adult_results/dec_posthoc.json, adult_results/dec_lam*.json,
              adult_results/decouple_curves.png

Both axes are the same for every point, all tier-1 decodes scored against
each row's SOURCE sex (the generated S column is invented by design):
true-sex leakage (balanced accuracy of a real-trained classifier from the
decoded non-sex columns; 0.5 = none) and the (true sex, proxy) joint TV vs
the real data - the fidelity cost of decoupling. The tau curve is the
target; an in-model run matches it if its point sits on or below it.
"""

import glob
import json
import re
import sys
import numpy as np


def main():
    post = sys.argv[1] if len(sys.argv) > 1 else 'adult_results/dec_posthoc.json'
    pattern = sys.argv[2] if len(sys.argv) > 2 else 'adult_results/dec_lam*.json'
    out = sys.argv[3] if len(sys.argv) > 3 else 'adult_results/decouple_curves.png'

    curve = json.load(open(post))['paper_metrics']['decouple_curve']
    print('post-hoc (target):')
    print(f'  {"tau":>5s} {"leak":>6s} {"joint TV":>9s} {"flipped":>8s}')
    for r in curve:
        print(f'  {r["tau"]:5.2f} {r["leakage"]:6.3f} {r["joint_tv"]:9.3f} '
              f'{r["flipped"]:8.3f}')

    pts = []
    for path in sorted(glob.glob(pattern)):
        d = json.load(open(path))
        if d.get('collapsed'):
            print(f'  {path}: collapsed, skipped')
            continue
        pm = d['paper_metrics']
        m = re.search(r'dec_lam([\d.]+)', path)
        pts.append({'lam': float(m.group(1)) if m else d['args']['proxy_weight'],
                    'mode': d['args'].get('proxy_mode'),
                    'leakage': pm['decoded_leakage'],
                    'joint_tv': pm['sex_proxy_joint_tv'],
                    'probe': pm['proxy_latent_probe']})
    pts.sort(key=lambda p: p['lam'])
    print('in-model:')
    print(f'  {"lambda":>6s} {"mode":>5s} {"leak":>6s} {"joint TV":>9s} '
          f'{"latent probe":>13s}  {"tau-curve leak at same TV":>25s}')
    xs = np.array([r['joint_tv'] for r in curve])
    ys = np.array([r['leakage'] for r in curve])
    order = np.argsort(xs)
    for p in pts:
        ref = float(np.interp(p['joint_tv'], xs[order], ys[order]))
        print(f'  {p["lam"]:6g} {p["mode"]:>5s} {p["leakage"]:6.3f} '
              f'{p["joint_tv"]:9.3f} {p["probe"]:13.3f}  {ref:25.3f}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(xs[order], ys[order], '-o', color='#555', label='post-hoc (tau)')
    for r in curve:
        ax.annotate(f'{r["tau"]:g}', (r['joint_tv'], r['leakage']),
                    textcoords='offset points', xytext=(4, -10), fontsize=8,
                    color='#555')
    if pts:
        ax.plot([p['joint_tv'] for p in pts], [p['leakage'] for p in pts], '-s',
                color='#1f6fb4', label='in-model (lambda)')
        for p in pts:
            ax.annotate(f'{p["lam"]:g}', (p['joint_tv'], p['leakage']),
                        textcoords='offset points', xytext=(4, 4),
                        fontsize=8, color='#1f6fb4')
    ax.axhline(0.5, color='#bbb', lw=0.8, ls=':')
    ax.set_xlabel('(true sex, proxy) joint TV vs real  (fidelity cost)')
    ax.set_ylabel('true-sex leakage (balanced acc.)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'plot -> {out}')


if __name__ == '__main__':
    main()
