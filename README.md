# Fair and private synthetic tabular data: AIM hybrid vs FLIP (UCI Adult)

This repository generates differentially private, fair synthetic data for
UCI Adult (protected attribute: sex; target: income > $50K; 3-fold CV). Two
methods are compared:

- **AIM hybrid (current method)** — `aim_fair/`.
  - Numeric bins are learned privately from 5% of the budget.
  - AIM / Private-PGM (McKenna et al., 2022) measures noisy 1- and 2-way
    marginals with the rest.
  - Fairness and realism are post-processing of those noisy measurements:
    proxy masking, a Feldman-style repair of every column across sexes,
    consistency rules from a pre-specified test, and pair redraw.
  - The epsilon covers the whole release, bins included.
- **Best VAE process (FLIP + joint stage)** — `run_adult_flip.py --preset
  final --lam 4`.
  - FLIP (TabSyn VAE + diffusion, arXiv 2508.21815) with this repository's
    dependence penalty on the decoded output.
  - Its DP is the paper's nominal VAE-only guarantee.

**Final result: `RESULTS.md`.** It compares the AIM hybrid with our faithful
reproduction of FLIP and with FLIP's reported numbers (chart and table).

**Verdict: the AIM hybrid wins.**

- It reaches income parity close to the best VAE process while running at a
  rigorous epsilon 3 (the VAE has no DP in this comparison).
- It produces far more realistic data.
- The VAE keeps a lower equalized-odds gap.

| 3 folds | AIM hybrid, eps 3 | VAE lambda 4, no DP |
|---|---|---|
| Income parity | 0.147 ± 0.02 | 0.132 ± 0.03 |
| Equalized-odds gap | 0.126 ± 0.02 | **0.044 ± 0.06** |
| Validity / impossible rows | **0.89 / 0%** | 0.72 / 19% |
| Numeric distortion (mean W1) | **0.024** | 0.051 |
| BER on the synthetic label / folded tstr | **0.50 / 0.43** | 0.30 / 0.42 |
| Income AUC | 0.890 | **0.897** |
| Privacy | **rigorous eps 3** | none |

## Private bins, and why not the public ones

AIM needs numeric columns cut into bins. The first version used fixed
public edges (`adult_domain.EDGES`). They fit Adult badly where people
report heaped values:

- hours of 45, 50 or 60 fell inside wide ranges and were decoded as random
  values (only 69% of the exact hours values reproduced);
- capital gains such as $15,024 became random amounts (non-zero values: 0.6%
  reproduced).

Those edges were also partly chosen with knowledge of Adult, outside the
privacy accounting.

`aim_fair/dp_schema.py` replaces them with a schema learned under DP from
5% of the zCDP budget (AIM keeps 95%, about eps 2.92):

1. Gaussian noisy histograms on a public fine grid.
2. Heaped values become their own bins, found by a pre-specified test (at
   least 0.5% of rows, 3x their neighbours, above the Bonferroni noise
   floor).
3. Ordered equal-frequency bins elsewhere.
4. Decoding by the noisy within-bin shares.

A column switches only if the noisy data show heaped values the public bins
miss. Hours, capital gain and capital loss switch; age, education and fnlwgt
keep the public bins.

The public bins scored slightly better income parity, yet the private ones
are the final setting:

| AIM hybrid, eps 3, 3 folds x 10 refits | Public bins | Private bins |
|---|---|---|
| Income parity | **0.135** | 0.147 |
| Equalized-odds gap | 0.131 | **0.126** (tie) |
| AUC / validity / impossible rows | 0.889 / 0.894 / 0% | 0.890 / 0.890 / 0% |
| Numeric distortion (mean W1) | 0.045 | **0.024** |
| Exact hours values reproduced | 0.69 | **0.98** |
| Non-zero capital gains reproduced | 0.006 | **0.44** |
| Bins inside the privacy accounting | no | **yes** |

**Why the public bins look fairer.** With public bins the synthetic hours
and capital gains are blurred, so an income model trained on them relies
less on "50+ hours" or a $15,024 gain. Real men have those values more
often, so the blur shrinks the parity gap by accident, by making the data
less realistic.

With realistic values the model learns the sharper real link, and parity
rises by about 0.01. Making the repair itself more precise (repair on
coarse groups, smoothed plans) did not change this, so the gap comes from
realism, not from a weaker repair.

EO, AUC and validity are unchanged, so the private bins give a truer
picture of what the method achieves, with a privacy claim that covers
everything. `PUBLIC_BINS=1 bash aim_fair/run_final.sh` reproduces the
public-bin variant.

## Documents

| | |
|---|---|
| `RESULTS.md` | final result: AIM hybrid vs FLIP reproduction vs FLIP's reported numbers (chart and table) |
| `aim_fair/FINAL_COMPARISON.md` | head-to-head with our best VAE process, the winner and why |
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
python aim_fair/final_chart.py        # -> results/final_comparison.{png,csv}

# FLIP-as-published heatmap in the paper's Fig. 3 layout (lambda x eps, fold 0)
bash scripts/run_heatmap_fold0.sh && python paper_heatmap_compare.py
```

Adult goes in `Data/adult.data` and `Data/adult.test` (UCI). Outputs go to
`adult_results/` (not committed).

## Layout

| Path | Contents |
|---|---|
| `aim_fair/` | AIM hybrid: `dp_schema.py` (private bins), `fair_aim.py`, `run_aim.py`, `run_final.sh`, `summarize.py`, `final_chart.py`, vendored AIM reference code |
| `run_adult_flip.py` | Adult driver for the VAE (FLIP presets, DP, `--score-csv` to score any synthetic table) |
| `run_flip_demo.py`, `run_flip_full_pipeline.py` | VAE training and diffusion pipeline used by the driver |
| `fairness_eval.py` | fairness metrics used by the pipeline |
| `calibrate_dp.py` | noise multipliers for a target epsilon (Eq. 13) |
| `paper_heatmap_compare.py` | heatmaps of FLIP as published against the paper's Fig. 3, plus real-leakage panels |
| `tabsyn-main/` | TabSyn (vendored) with the FLIP fairness terms in `tabsyn/vae/` |
| `scripts/` | VAE run scripts (best VAE process, heatmap runs) |
| `tests/` | unit tests |
| `results/` | final comparison chart and table, and the fold-0 FLIP heatmaps |

Earlier work (HMDA experiments, FLIP replication grids, reports) is on
`main`.
