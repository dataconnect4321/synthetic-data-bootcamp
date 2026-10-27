# Fair and private synthetic tabular data: AIM hybrid vs FLIP (UCI Adult)

This repository generates differentially private, fair synthetic data for
UCI Adult (protected attribute: sex; target: income > $50K; 3-fold CV), and
compares two methods:

- **AIM hybrid (the method)** — `aim_fair/`.
  - Numeric bins are learned privately from 5% of the budget.
  - AIM / Private-PGM (McKenna et al., 2022) measures noisy 1- and 2-way
    marginals with the rest.
  - Fairness and realism are post-processing of those noisy measurements:
    proxy masking, a Feldman-style repair of every column across sexes,
    consistency rules from a pre-specified test, and pair redraw.
  - The epsilon covers the whole release, bins included.
- **Best VAE process (FLIP + joint stage)** — `run_adult_flip.py --preset
  final --lam 4`.
  - FLIP (TabSyn VAE + diffusion, arXiv 2508.21815) plus this repository's
    dependence penalty on the decoded output.
  - Its DP is the paper's nominal, VAE-only guarantee.

**Verdict: the AIM hybrid wins.**

- It reaches income parity close to the best VAE process while running at a
  rigorous epsilon 3 (the VAE has no DP in this comparison).
- It produces far more realistic data, and keeps 88–95% of the real data's
  predictive power on most other Adult tasks.
- The VAE keeps a lower equalized-odds gap.

All numbers, the chart and the comparison with FLIP's own reported results
are in `RESULTS.md`.

| 3 folds | AIM hybrid, eps 3 | VAE lambda 4, no DP |
|---|---|---|
| Income parity | 0.147 ± 0.02 | 0.132 ± 0.03 |
| Equalized-odds gap | 0.126 ± 0.02 | **0.044 ± 0.06** |
| Validity / impossible rows | **0.89 / 0%** | 0.72 / 19% |
| Numeric distortion (mean W1) | **0.024** | 0.051 |
| BER on the synthetic label / folded tstr | **0.50 / 0.43** | 0.30 / 0.42 |
| Income AUC | 0.890 | **0.897** |
| Privacy | **rigorous eps 3** | none |

## Documents

| | |
|---|---|
| `RESULTS.md` | all results: vs FLIP (reproduction and reported numbers), vs the best VAE process, task independence, the FLIP heatmaps |
| `aim_fair/WRITEUP.md` | why we moved away from FLIP; how the AIM hybrid works; A-NCB caveats; limitations |
| `aim_fair/ARCHITECTURE.md` | architecture diagram: privacy boundary, post-processing steps, code map |

## The best VAE process

`--preset final` is FLIP as published (`--preset paper`) plus:

- an RFF dependence penalty on the whole decoded row, with the groups' mean
  gap (an MMD between the groups' decoded rows), weighted by lambda;
- a KL guard on the marginals, so the fairness terms cannot delete
  categories.

Under DP (`--dp-group-sigmas`) it follows the paper:

- Eq. 13 per-group noise on the VAE (IDP-SGD clipping per sample and sex
  group).
- The batch-coupled fairness terms go through the same noisy step, which
  gives the paper's **nominal** epsilon: per-sample DP-SGD does not cover
  batch-level terms.
- Diffusion is not private.
- `calibrate_dp.py` finds the multipliers for a target epsilon.

`run_adult_flip.py --score-csv` scores any synthetic table (e.g. the AIM
releases) with exactly the metrics of a VAE run. The metrics include:

- BER and A-NCB;
- tstr sex BER (a classifier trained on synthetic data, tested on real
  people);
- validity and impossible rows;
- alpha-precision / beta-recall and eps-identifiability;
- income parity and EO on real people.

## Private bins, and why not the public ones

AIM needs numeric columns cut into bins. The first version used fixed
public edges (`adult_domain.EDGES`), which fit Adult badly where people
report heaped values:

- hours of 45, 50 or 60 fell inside wide ranges and were decoded as random
  values (69% of the exact hours values reproduced);
- capital gains such as $15,024 became random amounts (0.6% of the non-zero
  values reproduced).

Those edges were also partly chosen with knowledge of Adult, outside the
privacy accounting.

`aim_fair/dp_schema.py` learns the bins under DP from 5% of the zCDP budget
(AIM keeps 95%, about eps 2.92):

1. Gaussian noisy histograms on a public fine grid.
2. Heaped values become their own bins, by a pre-specified test (at least
   0.5% of rows, 3x their neighbours, above the Bonferroni noise floor).
3. Ordered equal-frequency bins elsewhere.
4. Decoding by the noisy within-bin shares.

A column switches only if the noisy data show heaped values the public bins
miss. Hours, capital gain and capital loss switch; age, education and fnlwgt
keep the public bins.

| AIM hybrid, eps 3, 3 folds x 10 refits | Public bins | Private bins |
|---|---|---|
| Income parity | **0.135** | 0.147 |
| Equalized-odds gap | 0.131 | **0.126** (tie) |
| AUC / validity / impossible rows | 0.889 / 0.894 / 0% | 0.890 / 0.890 / 0% |
| Numeric distortion (mean W1) | 0.045 | **0.024** |
| Exact hours values reproduced | 0.69 | **0.98** |
| Non-zero capital gains reproduced | 0.006 | **0.44** |
| Bins inside the privacy accounting | no | **yes** |

**Why the public bins look fairer.** Blurred hours and capital gains make an
income model trained on them rely less on "50+ hours" or a $15,024 gain. Real
men have those values more often, so the blur shrinks the parity gap by
accident.

With realistic values, the model learns the sharper real link and parity
rises by about 0.01. A more precise repair did not change this, so the gap
comes from realism, not from a weaker repair.

EO, AUC and validity are unchanged, so the private bins are the final
setting. `PUBLIC_BINS=1 bash aim_fair/run_final.sh` reproduces the
public-bin variant.

## Setup

```bash
# main environment (torch): VAE, scoring, charts, tests
pip install torch opacus scikit-learn pandas scipy tqdm matplotlib
python -m pytest -q tests

# AIM environment (mbi needs Python >= 3.11 and JAX)
python3.13 -m venv aim_fair/.venv
aim_fair/.venv/Scripts/python -m pip install "git+https://github.com/ryan112358/mbi.git" pandas scikit-learn scipy
aim_fair/.venv/Scripts/python aim_fair/test_aim_fair.py
```

UCI Adult goes in `Data/adult.data` and `Data/adult.test`. Outputs go to
`adult_results/` (not committed). If pip's git clone fails with an SSL
error on Windows, prefix the install with `GIT_CONFIG_COUNT=1
GIT_CONFIG_KEY_0=http.sslBackend GIT_CONFIG_VALUE_0=schannel`.

## Run

```bash
# AIM hybrid, 3 folds x 10 refits + plain AIM (about 1.5 h of CPU per fold)
bash aim_fair/run_final.sh

# best VAE process: no DP (3 folds) and nominal eps 3 (fold 0)
PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh
bash scripts/run_best_vae_dp.sh

# FLIP as published, paper Fig. 3 layout (lambda x eps, fold 0), and the
# faithful reproduction used in RESULTS.md
bash scripts/run_heatmap_fold0.sh && python paper_heatmap_compare.py
python run_adult_flip.py --score-csv adult_results/hm_e3_lam4_argmax_synthetic.csv \
    --fold 0 --score-collapsed --no-save-run --tag flipfaithful_e3_f0_lam4

# results tables and chart, task independence
python aim_fair/final_chart.py
python aim_fair/task_auc.py
```

### The FLIP paper's other datasets

`datasets.py` adds the other datasets of FLIP's Table 2, all with sex
protected: ACS-I Alabama and Utah (the folktables ACSIncome task on the
2018 1-year ACS PUMS) and the Dutch census. `--dataset` selects one in
`export_folds.py`, `run_aim.py` and `run_adult_flip.py`; Adult stays the
default. The AIM hybrid runs with Adult's final settings, unchanged.

Data files (not committed):

- `Data/acs/psam_p01.csv` and `Data/acs/psam_p49.csv`, from `csv_pal.zip`
  and `csv_put.zip` at
  www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/;
- `Data/dutch/dutch.csv`, from the `tailequy/fairness_dataset` repository.

```bash
# 3 datasets x 3 folds: measure, 3 hybrid + 3 plain refits, score; about 1.5 h
bash aim_fair/run_datasets.sh       # writes results/datasets_summary.csv
```

## Layout

| Path | Contents |
|---|---|
| `aim_fair/fair_aim.py` | `FairAIM`: AIM's measuring loop, then masking, repair, consistency rules, pair redraw |
| `aim_fair/dp_schema.py` | private numeric bins (noisy histograms, spike bins, ordered edges, within-bin decoding) |
| `aim_fair/adult_domain.py` | public bins and categories; encoding and decoding |
| `aim_fair/run_aim.py`, `run_final.sh` | measure a fold and write refits; the 3-fold final run |
| `aim_fair/final_chart.py`, `task_auc.py` | results tables and chart; task independence |
| `aim_fair/check_schema.py` | how well public vs private bins reproduce the real numeric values |
| `aim_fair/export_folds.py` | the same 3-fold splits as `run_adult_flip.py` |
| `datasets.py`, `aim_fair/run_datasets.sh`, `aim_fair/datasets_summary.py` | ACS-I Alabama / Utah and Dutch census: loaders, the 3-dataset run, its summary table |
| `aim_fair/vendor/` | AIM reference code (Apache 2.0, see `vendor/NOTICE`) |
| `run_adult_flip.py` | VAE driver: presets `paper` / `final`, Eq. 13 DP, `--score-csv` |
| `run_flip_demo.py`, `run_flip_full_pipeline.py` | VAE training (FLIP phases) and latent diffusion |
| `calibrate_dp.py` | Eq. 13 noise multipliers for a target epsilon |
| `paper_heatmap_compare.py` | FLIP-as-published heatmaps against the paper's Fig. 3 |
| `tabsyn-main/` | the parts of TabSyn used (Apache 2.0), with the FLIP terms in `tabsyn/vae/` |
| `scripts/` | VAE run scripts |
| `tests/` | unit tests (`aim_fair/test_aim_fair.py` for the AIM side) |
| `results/` | committed results (CSV and figures) |
