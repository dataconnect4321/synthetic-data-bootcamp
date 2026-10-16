# Fair and private synthetic tabular data: AIM hybrid vs FLIP (UCI Adult)

This repository generates differentially private, fair synthetic data for
UCI Adult (protected attribute: sex; target: income > $50K; 3-fold CV). Two
methods are compared:

- **AIM hybrid (current method)** — `aim_fair/`.
  - AIM / Private-PGM (McKenna et al., 2022) measures noisy 1- and 2-way
    marginals.
  - Fairness and realism are post-processing of those noisy measurements:
    proxy masking, a Feldman-style repair of every column across sexes,
    consistency rules from a pre-specified test, and pair redraw.
  - The epsilon covers the whole release.
- **Best VAE process (FLIP + joint stage)** — `run_adult_flip.py --preset
  final --lam 4`.
  - FLIP (TabSyn VAE + diffusion, arXiv 2508.21815) with this repository's
    dependence penalty on the decoded output.
  - Its DP is the paper's nominal VAE-only guarantee.

**Verdict: the AIM hybrid wins.** It matches the VAE's income parity while
running at a rigorous epsilon 3 (the VAE has no DP in this comparison). It
produces far more realistic data. The VAE keeps a lower equalized-odds gap.

| 3 folds | AIM hybrid, eps 3 | VAE lambda 4, no DP |
|---|---|---|
| Income parity | 0.139 ± 0.02 | 0.132 ± 0.03 |
| Equalized-odds gap | 0.138 ± 0.03 | **0.044 ± 0.06** |
| Validity / impossible rows | **0.89 / 0%** | 0.72 / 19% |
| BER on the synthetic label / folded tstr | **0.50 / 0.45** | 0.30 / 0.42 |
| Income AUC | 0.889 | **0.897** |
| Privacy | **rigorous eps 3** | none |

Full numbers: `results/final_aim_vs_vae.csv`.

The FLIP replication heatmaps (fold 0, lambda x epsilon, our values against
the paper's Fig. 3) are in `results/heatmap_paper_compare_fold0_*.png`:

- `_argmax`: sex decoded by argmax (TabSyn);
- `_sampled`: sex sampled from its probabilities;
- `_real_leakage`: true-sex BER and real-trained A-NCB, which the paper's
  metrics cannot show.

## Documents

| | |
|---|---|
| `aim_fair/FINAL_COMPARISON.md` | head-to-head results, the winner and why |
| `aim_fair/ARCHITECTURE.md` | architecture diagram of the AIM hybrid (privacy boundary, post-processing steps, code map) |
| `aim_fair/WRITEUP.md` | why we moved away from FLIP; how the AIM hybrid works; A-NCB caveats; limitations |
| `VAE_CHANGES.md` | the VAE changes: Eq. 13 DP, external scoring, metrics |
| `aim_fair/README.md` | AIM setup and files |

## Run

```bash
# main environment (torch): VAE, scoring, tests
pip install torch opacus scikit-learn pandas scipy tqdm
python -m pytest -q tests

# AIM hybrid, 3 folds (separate mbi + JAX environment, see aim_fair/README.md)
bash aim_fair/run_final.sh

# best VAE process: no DP (3 folds) and nominal eps 3 (fold 0)
PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh
bash scripts/run_best_vae_dp.sh

python aim_fair/summarize.py          # -> adult_results/final_aim_vs_vae.csv

# FLIP-as-published heatmap in the paper's Fig. 3 layout (lambda x eps, fold 0)
bash scripts/run_heatmap_fold0.sh && python paper_heatmap_compare.py
```

Adult goes in `Data/adult.data` and `Data/adult.test` (UCI). Outputs go to
`adult_results/` (not committed).

## Layout

| Path | Contents |
|---|---|
| `aim_fair/` | AIM hybrid: `fair_aim.py`, `run_aim.py`, `run_final.sh`, `summarize.py`, vendored AIM reference code |
| `run_adult_flip.py` | Adult driver for the VAE (FLIP presets, DP, `--score-csv` to score any synthetic table) |
| `run_flip_demo.py`, `run_flip_full_pipeline.py` | VAE training and diffusion pipeline used by the driver |
| `fairness_eval.py` | fairness metrics used by the pipeline |
| `calibrate_dp.py` | noise multipliers for a target epsilon (Eq. 13) |
| `paper_heatmap_compare.py` | heatmaps of FLIP as published against the paper's Fig. 3, plus real-leakage panels |
| `tabsyn-main/` | TabSyn (vendored) with the FLIP fairness terms in `tabsyn/vae/` |
| `scripts/` | VAE run scripts (best VAE process, heatmap runs) |
| `tests/` | unit tests |
| `results/` | final comparison table and the fold-0 heatmaps |

Earlier work (HMDA experiments, FLIP replication grids, reports) is on
`main`.
