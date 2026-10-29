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
| `RESULTS.md` | all results: vs FLIP (reproduction and reported numbers, also under FLIP's own evaluation protocol), vs the best VAE process, task independence, the FLIP heatmaps |
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

## Task-agnostic vs task-specific fairness ("Where to Intervene?")

Angelozzi & Arcolezi, *Where to Intervene? Benchmarking Fairness-Aware
Learning on Differentially Private Synthetic Tabular Data* (PoPETs 2026,
arXiv 2607.07471), benchmark the standard **task-specific** fairness
methods on DP synthetic data. We re-ran their benchmark with their code's
settings on their ACS-I Utah setup, and compared it with the AIM hybrid,
which never sees the task (`aim_fair/wti_benchmark.py`).

### Their task-specific fairness, which we replicated

The prediction task is fixed in advance: income above the median, with sex
protected. AIM synthesizes the training data, and a fairness method is then
added at one of three stages, all through AIF360:

| Stage | Method | What it uses | What it fixes |
|---|---|---|---|
| Pre-processing | Reweighing | income labels | weights for an income model |
| | Disparate Impact Remover (repair level 1) | features | the features of the income model's train, calibration and test data |
| | Learning Fair Representations | income labels | an encoding tuned to income |
| In-processing | Exponentiated Gradient, Grid Search (equalized-odds constraint) | income labels | one constrained income classifier |
| Post-processing | Equalized Odds, Reject Option, Calibrated Equalized Odds (`fnr` cost) | a real, labelled 20% calibration split, and sex at prediction time | that classifier's outputs |

Every method is tied to the one income task. Its fix lives in weights, an
encoding or a model, not in the released data. Anyone who predicts another
attribute from the same release gets plain AIM's sex gaps.

### Apples to apples

- **Their data:** ACSIncome Utah 2018, preprocessed as in their repository
  (`acs_ut_wti` in `datasets.py`).
  - Same folktables filter, giving the same 16,337 rows.
  - Target: PINCP above its median.
  - Occupation in their 6 Census groups; birthplace as raw codes; race as
    White alone vs other.
- **Their splits:** for each of 5 seeds, `train_test_split(test_size=0.4,
  random_state=seed)`, then 50/50 into calibration and test.
  - AIM synthesizes the 60% training split only.
  - Every source is scored on the same real test people.
- **One set of AIM measurements per seed (ε 3).** Plain AIM and the hybrid
  are two refits of it, so they differ only by the hybrid's post-processing.
- **Their models:**
  - features are every column but income, sex included, MinMax-scaled;
  - XGBoost (`binary:logistic`) and random forest at their defaults;
  - logistic regression with their tuned settings (saga, elastic net,
    l1_ratio 0.5, C 0.8).
- **Their metrics:** AIF360's `ClassificationMetric`, men privileged.
  Below, they use this README's names:
  - **parity** = |SPD|, the selection-rate gap;
  - **EO** = |EOD|, the true-positive-rate gap;
  - **accuracy gap** = |MAD|.
- **Our metrics on the same splits:** the scorer (`run_adult_flip.py
  --split-csvs`) and FLIP's per-target protocol (`flip_protocol.py
  --dataset acs_ut_wti`).

Deviations:

- AIM from mbi (needed for the hybrid's access to the noisy measurements),
  not SmartNoise.
- Age and hours are binned privately, not used as integer codes.
- 5 seeds and epsilon 3 only; the paper uses 20 seeds and 12 values of
  epsilon.
- AIF360 0.6.1's Learning Fair Representations needs a one-line SciPy
  compatibility shim. With its defaults it still collapses to a single
  class (an error in 39 of 45 runs, as in their code), so it is left out.
- The paper's published numbers are not compared. Their methods were re-run
  on our splits instead.

### Results: the income task

Means over 3 classifiers x 5 seeds. The spread across seeds is about ±0.02
on parity and EO.

| Source | Accuracy | F1 | Parity ↓ | EO ↓ | Accuracy gap ↓ | AIM hybrid vs this |
|---|---|---|---|---|---|---|
| **AIM hybrid (task-agnostic)** | **0.769** | **0.771** | **0.171** | **0.035** | **0.034** | — |
| Plain AIM | 0.804 | 0.802 | 0.382 | 0.267 | 0.028 | hybrid much fairer; 3.5 pts less accurate |
| + Reweighing | 0.790 | 0.790 | 0.206 | 0.062 | 0.030 | hybrid fairer on parity and EO; 2.1 pts less accurate |
| + Disparate Impact Remover | 0.800 | 0.801 | 0.417 | 0.294 | 0.022 | hybrid far fairer; 3.1 pts less accurate |
| + Exponentiated Gradient | 0.795 | 0.794 | 0.268 | 0.138 | 0.032 | hybrid fairer on parity and EO; 2.6 pts less accurate |
| + Grid Search | 0.804 | 0.802 | 0.383 | 0.270 | 0.028 | hybrid far fairer; 3.5 pts less accurate |
| + Equalized Odds | 0.714 | 0.697 | 0.113 | 0.018 | 0.028 | hybrid 5.5 pts more accurate; method fairer |
| + Reject Option | 0.757 | 0.762 | 0.030 | 0.103 | 0.067 | hybrid more accurate, better EO and accuracy gap; method better parity |
| + Calibrated Equalized Odds | 0.699 | 0.742 | 0.705 | 0.371 | 0.165 | hybrid better on every measure |
| Real data (reference) | 0.815 | 0.814 | 0.348 | 0.205 | 0.019 | |

Two notes:

- **Classifier dependence:** the hybrid is fairer than Reweighing with
  XGBoost and random forest, but not with logistic regression (Reweighing:
  parity 0.159, EO 0.022; hybrid: 0.166, 0.027).
- **Combining:** hybrid + Equalized Odds reaches EO 0.024 at 0.754 accuracy,
  against 0.018 at 0.714 for plain AIM + Equalized Odds.

### Results: the release

Only the hybrid changes the data. Every task-specific method leaves plain
AIM's release as it was.

| Release | BER (0.5 = sex hidden) | Folded tstr (0.5 best) | A-NCB ↑ | Mean parity, all categorical targets ↓ | Mean W1/sd, numeric targets ↓ | Outcome AUC | Validity | Impossible rows |
|---|---|---|---|---|---|---|---|---|
| Plain AIM, with or without any task-specific method | 0.300 | 0.286 | 0.616 | 0.088 | 0.228 | 0.894 | 0.96 | 0% |
| **AIM hybrid** | **0.524** | **0.492** | **0.909** | **0.068** | **0.152** | 0.872 | 0.95 | 0% |
| Real data | 0.269 | 0.264 | 0.759 | 0.090 | 0.244 | 0.909 | 0.99 | 0% |

### Analysis: similar results, more efficiently

**Similar.**

- On the income task, the AIM hybrid lands on the same accuracy–fairness
  frontier as the task-specific methods. No method beats it on both
  accuracy and fairness at once.
- It is fairer than every pre- and in-processing method: parity 0.171 and
  EO 0.035, against 0.206 / 0.062 for Reweighing and 0.268 / 0.138 for
  Exponentiated Gradient. The cost is 2–3 points of accuracy.
- It is more accurate than the post-processing methods (0.769 against
  0.714 and 0.757), each of which beats it on one fairness measure.
- It beats Calibrated Equalized Odds, Disparate Impact Remover and Grid
  Search on fairness outright.

**More efficient, because it is task-agnostic.**

- **One repair serves every task.** The repair is done once, inside AIM's
  noisy measurements, and is released with the data.
  - Every model on any target inherits it: across all targets, mean
    parity falls from 0.088 to 0.068 (categorical) and W1/sd from 0.228 to
    0.152 (numeric).
  - A task-specific method fixes one income classifier. Each further task
    needs its own labels, constraint, training run and, for
    post-processing, its own calibration data.
- **It needs less.** It uses no outcome labels for a fairness constraint,
  no labelled real calibration split (post-processing uses 20% extra real
  data), and no sex attribute at prediction time (post-processing needs
  it).
  - So it fits data sharing, where the users and their tasks are not known
    in advance.
- **It fixes the data, not only the predictions.**
  - Sex can't be recovered from the hybrid's release (BER 0.52; a model
    trained on it can't learn sex, tstr 0.49).
  - Plain AIM's release still reveals sex (BER 0.30) whatever fairness
    method is applied downstream.
- **It costs no extra privacy budget or model training.** The repair is
  post-processing of measurements AIM has already made, so the epsilon 3
  guarantee covers it. The task-specific methods each add a separate
  fitting step per task.
- **It combines with them.** A task-specific method can still be applied
  on top of the hybrid when one task matters most.
  - With Equalized Odds, it gives the same EO with 4 points more accuracy
    than on plain AIM.

**The cost:**

- about 3.5 points of accuracy and 0.02 AUC compared with plain AIM;
- 2 points of accuracy compared with the best pre- and in-processing
  methods;
- 1.5 points of validity.

Reproduce:

```bash
bash aim_fair/run_wti_benchmark.sh   # AIM per seed, their benchmark, our scorer and protocol, combined table
```

It writes `results/wti_benchmark_{runs,summary,comparison}.csv` and
`results/flip_protocol_{targets,outcome}_acs_ut_wti.csv`.

## Setup

```bash
# main environment (torch): VAE, scoring, charts, tests
pip install torch opacus scikit-learn pandas scipy tqdm matplotlib
python -m pytest -q tests

# AIM environment (mbi needs Python >= 3.11 and JAX)
python3.13 -m venv aim_fair/.venv
aim_fair/.venv/Scripts/python -m pip install "git+https://github.com/ryan112358/mbi.git" pandas scikit-learn scipy
aim_fair/.venv/Scripts/python aim_fair/test_aim_fair.py

# LightGBM / AIF360 environment: FLIP's evaluation protocol (flip_protocol.py)
# and the "Where to Intervene?" benchmark (wti_benchmark.py)
python3.12 -m venv aim_fair/.venv-aif360
aim_fair/.venv-aif360/Scripts/python -m pip install lightgbm pandas scipy joblib scikit-learn \
    aif360 fairlearn xgboost
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

# FLIP's own evaluation protocol (LightGBM, every attribute as a target):
# fresh refits from the saved measurements, then the protocol and its chart
bash aim_fair/run_adult_refits.sh
aim_fair/.venv-aif360/Scripts/python aim_fair/flip_protocol.py
python aim_fair/flip_faithful_chart.py
# the same protocol on ACS-I Alabama / Utah and Dutch, and the 4-dataset table
for d in acs_al acs_ut dutch; do aim_fair/.venv-aif360/Scripts/python aim_fair/flip_protocol.py --dataset $d; done
python aim_fair/flip_datasets_table.py
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
| `aim_fair/flip_protocol.py`, `flip_faithful_chart.py`, `flip_datasets_table.py`, `run_adult_refits.sh` | FLIP's own evaluation protocol (LightGBM, Eqs. 14 and 15, every attribute a target; `--dataset` for ACS-I / Dutch), the FLIP-faithful comparison chart and the 4-dataset table |
| `aim_fair/wti_benchmark.py`, `run_wti_benchmark.sh` | the "Where to Intervene?" task-specific benchmark on ACS-I Utah vs the AIM hybrid, on the same seed splits |
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
