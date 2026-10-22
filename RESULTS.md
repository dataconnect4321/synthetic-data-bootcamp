# Final result: AIM hybrid vs FLIP (UCI Adult, epsilon 3)

![Final comparison](results/final_comparison.png)

| Metric | AIM hybrid, private bins | FLIP, faithful reproduction | FLIP paper, reported |
|---|---|---|---|
| Privacy | **rigorous eps 3 (whole release)** | nominal eps 3 (VAE only) | nominal eps 3 (VAE only) |
| Income parity (lower better) | 0.147 ± 0.024 | 0.235 | **0.12** |
| Equalized-odds gap (lower better) | 0.126 ± 0.021 | **0.096** | 0.26 |
| BER, synthetic sex label (0.5 best) | **0.505** | 0.162 | 0.49 |
| A-NCB, paper's own-sex version (higher better)* | **0.92** | 0.27 | 0.91 |
| Release teaches real sex, folded tstr (0.5 best) | **0.434** | 0.168 | not reported |
| Income AUC on real people (higher better) | 0.890 | **0.902** | 0.66 |
| Alpha-precision (higher better) | **0.984** | 0.900 | 0.40 |
| Beta-recall (higher better) | **0.385** | 0.186 | 0.21 |
| Validity (higher better; real 0.98) | **0.890** | 0.559 | not reported |
| Impossible rows (lower better; real 0%) | **0%** | 43% | not reported |
| Epsilon-identifiability (lower better) | 0.354 | 0.136 | **0.09** |

Notes on the three columns:

- **AIM hybrid:** 3 folds × 10 refits; ± is the sd across folds.
- **FLIP reproduction:** FLIP as published, lambda 4, fold 0. It uses the
  paper-closest setup of `scripts/run_heatmap_fold0.sh`, scored from its
  released table like AIM.
- **FLIP paper:** values read off its Fig. 3 (Adult, lambda 4, eps 3),
  produced by its own evaluation code, so not strictly comparable.

\* A-NCB with the paper's own-sex adversary rewards a drifting sex label,
and on Adult it reduces to one married/not-married split. It is shown only
because the paper uses it.

## What it shows

- **The AIM hybrid is the only method whose privacy claim holds.** Every
  fairness and realism step is post-processing of AIM's noisy measurements,
  and the numeric bins are learned inside the budget.
- **Our FLIP reproduction never removes the unfairness.**
  - Income parity stays at 0.235, the level with no fairness term at all.
  - The released sex label stays learnable (BER 0.16, tstr 0.17).
  - 43% of its rows are impossible.
- **The paper's FLIP numbers come with a large utility cost.** Its reported
  parity (0.12) is the best, but at AUC 0.66 and alpha-precision 0.40. Its
  reported EO gap is the worst (0.26). It reports no realism or real-leakage
  measure.
- **AIM's weak spot is equalized odds.** The FLIP reproduction's EO gap is
  lower (0.096 vs 0.126), but with parity at 0.235. The remaining gap comes
  from "married": it stays a real-world proxy even after masking.

## Why AIM's epsilon-identifiability is the highest despite formal privacy

Epsilon-identifiability is the share of real people whose nearest synthetic
row is closer than their nearest other real person. It measures **closeness
to real rows, not leakage**, so it rewards realism and penalizes noise:

- AIM samples from a model of the population. Realistic rows land in dense
  regions near many real people, so the score is high: 0.35, and 0.43 for
  plain AIM, the most realistic table. That isn't a leak. DP guarantees the
  same rows would have been generated, with almost the same probability,
  whether or not any one person was in the data.
- FLIP scores low (0.09–0.14) because its rows are far from everyone:
  alpha-precision 0.40–0.90 and 43% impossible rows. Low scores here partly
  reflect unrealistic data, not better privacy.
- The metric comes from non-DP generator papers, where no formal guarantee
  exists. With AIM's formal epsilon 3, it is a realism diagnostic, not the
  privacy measure.

## Task independence: other targets than income

The repair is task-agnostic: it removes sex from every column without
knowing the downstream task. To check how well the release serves other
tasks, a classifier is trained on synthetic data and tested on real people
for several Adult targets. All other columns are features; for "married"
the relationship column is dropped, since it encodes marriage directly. The
metric is ROC AUC: 3 folds, and the first 3 refits per fold for the
synthetic sources.

| Target | Real data (ceiling) | Plain AIM (privacy only) | **AIM hybrid** | Fairness cost | Hybrid keeps |
|---|---|---|---|---|---|
| College degree (education >= 13) | 0.881 | 0.868 | **0.863** | -0.005 | 95% |
| Income >50K | 0.926 | 0.904 | **0.891** | -0.013 | 92% |
| Private sector | 0.761 | 0.741 | **0.732** | -0.008 | 89% |
| Long hours (>= 45) | 0.770 | 0.752 | **0.739** | -0.013 | 89% |
| Age >= 40 | 0.808 | 0.784 | **0.772** | -0.012 | 88% |
| Race White | 0.784 | 0.744 | **0.733** | -0.011 | 82% |
| Occupation (14 classes) | 0.776 | 0.760 | **0.722** | -0.038 | 81% |
| Capital gain > 0 | 0.765 | 0.712 | **0.709** | -0.003 | 79% |
| Married | 0.900 | 0.868 | **0.801** | -0.067 | 75% |
| *Sex (removed by design)* | *0.885* | *0.873* | ***0.582*** | *-0.291* | *21%* |

The fairness cost is the hybrid minus plain AIM. "Hybrid keeps" is the
share of the real data's above-chance AUC retained, (AUC - 0.5) /
(real - 0.5). Fold sd is at most 0.015 for the real targets (0.03 for sex).

- **Largely task-independent.** For 7 of the 9 real targets the repair
  costs at most 0.013 AUC, often less than privacy alone. Income, the task
  the settings were judged on, is not specially favoured.
- **Sex-linked tasks pay most, by design.**
  - Married (-0.067): the repair equalizes marital status across sexes,
    breaking its links to sex-skewed features.
  - Occupation (-0.038): occupations are sex-segregated in real data, and
    the release removes that link.
- **Privacy costs most where data is sparse:** capital gain > 0 (-0.053
  from DP) and race (-0.040), i.e. rare positives and small groups.
- **Sex itself falls to 0.58,** close to chance (the residual matches tstr
  0.43).

Reproduce: `python aim_fair/task_auc.py` (writes `results/task_auc.csv`).

## AIM hybrid vs the best VAE process (FLIP + joint stage)

The best VAE process is FLIP plus this repository's joint decoded-output stage
(`--preset final`, lambda 4); see the README.

### The two contenders

| | AIM hybrid (best AIM process) | VAE (best VAE process) |
|---|---|---|
| Generator | AIM on Private-PGM: noisy 1- and 2-way count tables, a graphical model, then sampling | FLIP (TabSyn VAE + diffusion) with our output-level joint fairness stage |
| Numeric bins | learned privately from 5% of the zCDP budget (`dp_schema.py`): heaped values get their own bins | not applicable |
| Fairness | the noisy tables are repaired before the final fit: every column's distribution is made equal across sexes (Feldman et al., 2015; tau = 1, income included, pooled target) | joint dependence penalty on the decoded output, lambda = 4 |
| Realism | proxy levels (Husband/Wife) merged; consistency rules from a pre-specified test on the noisy tables; broken pairs redrawn from the model; marginals re-matched | none (rejection under DP caused selection bias) |
| Privacy | **epsilon 3, rigorous for the whole release.** All fairness and realism steps are post-processing. | DP-SGD on the VAE only, at a **nominal** epsilon (batch-coupled terms and the diffusion stage are not covered) |
| Evidence | 3 folds x 10 refits from each fold's private measurements | 3 folds without DP; epsilon 3 on fold 0 only |

### Results

**3 folds** (mean ± sd across folds). This comparison favours the VAE:
AIM runs at a rigorous epsilon 3, the VAE without any privacy.

| | AIM hybrid, eps 3 | VAE, lambda 4, no DP | AIM plain, eps 3 | VAE, lambda 0, no DP |
|---|---|---|---|---|
| **Downstream fairness** (income model trained on synthetic, tested on real people) | | | | |
| Income parity (lower better) | 0.147 ± 0.02 | **0.132 ± 0.03** | 0.208 | 0.246 |
| Equalized-odds gap (lower better) | 0.126 ± 0.02 | **0.044 ± 0.06** | 0.143 | 0.125 |
| **Sex leakage** | | | | |
| BER on the synthetic label (0.5 = no link) | **0.50** | 0.30 | 0.16 | 0.17 |
| Release teaches real sex, folded tstr (0.5 = no) | **0.43** | 0.42 | 0.14 | 0.15 |
| **Realism and utility** | | | | |
| Validity (real data 0.98) | **0.89** | 0.72 | 0.91 | 0.97 |
| Impossible rows (real 0%) | **0%** | 19% | 0.7% | 1.3% |
| Numeric distortion, mean W1 (lower better) | **0.024** | 0.051 | 0.024 | 0.036 |
| alpha-precision / beta-recall | **0.984 / 0.39** | 0.885 / 0.30 | 0.993 / 0.46 | 0.990 / 0.48 |
| Income AUC on real people (real-trained 0.93) | 0.890 | **0.897** | 0.904 | 0.910 |
| >50K rate (real 0.248) | **0.247** | 0.250 | 0.247 | 0.249 |
| **Privacy** | **rigorous eps 3** | none | rigorous eps 3 | none |
| eps-identifiability (empirical, lower better) | 0.35 | **0.24** | 0.43 | 0.40 |

**Same privacy level, fold 0 only, epsilon 3** (the VAE's epsilon is
nominal):

| | AIM hybrid | VAE, lambda 4 | FLIP as published, lambda 4 |
|---|---|---|---|
| Income parity | 0.161 | **0.091** | 0.235 |
| EO gap | 0.150 | **0.080** | 0.096 |
| Folded tstr | 0.45 | 0.46 | 0.17 |
| Validity / impossible rows | **0.89 / 0%** | 0.67 / 26% | 0.56 / 43% |
| AUC | **0.902** | 0.887 | 0.902 |
| alpha / beta | **0.985 / 0.39** | 0.89 / 0.20 | 0.90 / 0.19 |

`python aim_fair/final_chart.py` writes this table to
`results/final_aim_vs_vae.csv` once the runs exist (see Reproduce).

### Why the AIM hybrid wins

1. **Privacy that holds.**
   - AIM's epsilon covers the whole release. Every fairness and realism step
     reads only the noisy measurements, so nothing is left unaccounted for.
   - The VAE's epsilon is nominal: per-sample DP-SGD does not cover its
     batch-coupled fairness terms, and the diffusion stage is trained outside
     the accounting.
   - At epsilon 1 the VAE breaks down (income parity 0.33).
2. **Much more realistic data.**
   - Validity is 0.89 vs 0.72 (0.67 at epsilon 3), and impossible rows are
     0% vs 19–26%. Numeric distortion is half the VAE's (W1 0.024 vs
     0.051): heaped hours and real capital-gain amounts are reproduced.
   - alpha-precision 0.984 vs 0.885: synthetic rows sit where real rows
     are.
   - The VAE's fairness comes with a fifth of rows being contradictory
     (e.g. female "Husbands").
3. **Comparable fairness on the 3-fold evidence.**
   - Income parity is 0.147 vs 0.132, within the VAE's own fold-to-fold
     spread (0.10–0.16).
   - Sex leakage is equal or better: BER 0.50 vs 0.30, tstr 0.43 vs 0.42.
4. **Stability.** AIM's realism varies less across folds (validity sd
   0.002 vs 0.02). The VAE's best numbers came from its best fold.
5. **Interpretable and cheap to adjust.**
   - The repair states exactly what changes: P(column | sex) becomes equal.
   - New variants take minutes from the saved private measurements, with no
     new privacy cost. A new VAE configuration needs about an hour of GPU
     training per fold, and each DP run spends budget again.

### Where the VAE is still better

- **Equalized odds.** EO gap 0.044 vs 0.126 on 3 folds, and 0.080 vs 0.150
  at epsilon 3 on fold 0.
  - The VAE's errors are more even between real high-earning men and women,
    though with large fold-to-fold variance (0.006–0.115).
  - AIM's gap likely comes from the merged spouse level. It is still a
    real-world proxy (about 61% of real men vs 15% of real women are
    spouses), so the "married" income premium lands on real men.
- **Income parity at epsilon 3 on fold 0:** 0.091 vs 0.161. This is a
  single run at nominal privacy, but a gap worth noting.
- **Utility and empirical privacy:** slightly higher AUC (0.897 vs 0.890)
  and lower eps-identifiability (0.24 vs 0.35). The latter matters less for
  AIM, whose formal guarantee covers any attacker.
- **Scaling to wide data.** AIM's cost grows with the number of columns and
  categories (2-way candidates grow with d²), not rows. On Adult (14
  columns) that is fine. On very wide or high-cardinality data a neural
  generator scales better.

### Verdict

For a private, fair and realistic release of Adult-like data, use the **AIM
hybrid**. It is the only method whose privacy claim holds. It produces data
that looks real and is fair in the synthetic table itself (sex is unrelated
to every column), with downstream parity on par with the VAE.

Choose the VAE method only if formal privacy is not required and
equalized odds is the priority, or for data too wide for marginal-based
synthesis. Even then, its impossible rows need handling.

### Caveats

- **Unequal privacy in the 3-fold comparison.** Only fold 0 has the VAE at
  epsilon 3; its folds 1–2 exist only without DP. On fold 0, DP cost the VAE
  realism (validity 0.74 → 0.67, impossible rows 17% → 26%) but not
  fairness (parity 0.099 → 0.091), so epsilon-3 runs on all folds would
  sharpen the realism gap; the effect on fairness is unknown.
- **Public vs private bins.** With the fixed public bins the AIM hybrid
  scores parity 0.135 (vs 0.147) at the same EO, AUC and validity, but its
  numeric values are less realistic (W1 0.045). See "Public vs private
  bins" below.
- **A-NCB is excluded from the verdict.**
  - On Adult it reduces to one married/not-married axis.
  - The real-trained version rewards incoherent rows: AIM 0.22, VAE 0.45
    with 19% impossible rows.
  - The own-S version rewards label drift: AIM 0.92, VAE 0.72.
  - See `aim_fair/WRITEUP.md`.
- **Both methods rewrite real sex differences.** The AIM release says men and
  women have the same hours, occupations and incomes. Its income links to
  legitimate features are diluted: married vs unmarried >50K is 0.35 / 0.16
  (real 0.46 / 0.07).
- **Not yet tested:** Angelozzi & Arcolezi's (2026) placement, which repairs
  plain AIM's output instead of inside the generator, on the same folds.
- **Tuning is not covered.** Settings were chosen by scoring on real data,
  which is outside DP. A deployed release must fix them in advance.

## FLIP as published: the paper's Fig. 3 layout

Our reproduction of FLIP as published on fold 0, lambda x epsilon, with the
paper's values in brackets (`scripts/run_heatmap_fold0.sh`,
`paper_heatmap_compare.py`):

- `results/heatmap_paper_compare_fold0_argmax.png`: sex decoded by argmax
  (TabSyn), the setup of the faithful reproduction above.
- `results/heatmap_paper_compare_fold0_sampled.png`: sex sampled from its
  probabilities.
- `results/heatmap_paper_compare_fold0_real_leakage.png`: true-sex BER and
  real-trained A-NCB, which the paper's metrics cannot show.

## Public vs private bins

The AIM hybrid with the fixed public bins scores slightly better income
parity (0.135 vs 0.147), with the same EO, AUC and validity. But its numeric
values are less realistic (W1 0.045 vs 0.024), and part of that parity edge
comes from the blurring. See the README for the full
comparison.

## Reproduce

```bash
bash aim_fair/run_final.sh                       # AIM hybrid, 3 folds
bash scripts/run_heatmap_fold0.sh                # FLIP reproduction runs
python run_adult_flip.py --score-csv adult_results/hm_e3_lam4_argmax_synthetic.csv \
    --fold 0 --score-collapsed --no-save-run --tag flipfaithful_e3_f0_lam4
PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh   # best VAE, no DP
bash scripts/run_best_vae_dp.sh                  # best VAE, nominal eps 3
python aim_fair/final_chart.py                   # results/final_comparison.{png,csv}, final_aim_vs_vae.csv
python aim_fair/task_auc.py                      # results/task_auc.csv
```

## References

- C. Angelozzi, H. H. Arcolezi. *Where to Intervene? Benchmarking
  Fairness-Aware Learning on Differentially Private Synthetic Tabular Data.*
  PoPETs 2026(4). arXiv:2607.07471.
- C. Dwork, A. Roth. *The Algorithmic Foundations of Differential Privacy.*
  2014.
- M. Feldman et al. *Certifying and Removing Disparate Impact.* KDD 2015.
- R. McKenna, B. Mullins, D. Sheldon, G. Miklau. *AIM.* PVLDB 15(11), 2022.
  arXiv:2201.12677.
- R. McKenna, D. Sheldon, G. Miklau. *Graphical-model based estimation and
  inference for differential privacy.* ICML 2019. arXiv:1901.09136.
- FLIP, arXiv:2508.21815.
