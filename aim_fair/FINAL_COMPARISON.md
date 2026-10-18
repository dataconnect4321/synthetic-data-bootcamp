# Final comparison: AIM hybrid vs. VAE (FLIP + joint stage) on UCI Adult

**Winner: the AIM hybrid.** It is the only method with a rigorous privacy
guarantee. It produces far more realistic data and reaches similar income
parity. Our VAE method keeps a real advantage on equalized odds, so the
margin is not total. AIM figures below use the final setting with private
numeric bins; the comparison with FLIP's own reported numbers is in
`RESULTS.md`.

## The two contenders

| | AIM hybrid (best AIM process) | VAE (best VAE process) |
|---|---|---|
| Generator | AIM on Private-PGM: noisy 1- and 2-way count tables, a graphical model, then sampling | FLIP (TabSyn VAE + diffusion) with our output-level joint fairness stage |
| Numeric bins | learned privately from 5% of the zCDP budget (`dp_schema.py`): heaped values get their own bins | not applicable |
| Fairness | the noisy tables are repaired before the final fit: every column's distribution is made equal across sexes (Feldman et al., 2015; tau = 1, income included, pooled target) | joint dependence penalty on the decoded output, lambda = 4 |
| Realism | proxy levels (Husband/Wife) merged; consistency rules from a pre-specified test on the noisy tables; broken pairs redrawn from the model; marginals re-matched | none (rejection under DP caused selection bias) |
| Privacy | **epsilon 3, rigorous for the whole release.** All fairness and realism steps are post-processing. | DP-SGD on the VAE only, at a **nominal** epsilon (batch-coupled terms and the diffusion stage are not covered) |
| Evidence | 3 folds x 10 refits from each fold's private measurements | 3 folds without DP; epsilon 3 on fold 0 only |

## Results

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

Reproduce:

- AIM: `bash aim_fair/run_final.sh`.
- VAE: `PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh` and
  `bash scripts/run_best_vae_dp.sh`.

Then `python aim_fair/summarize.py` writes
`adult_results/final_aim_vs_vae.csv`.

## Why the AIM hybrid wins

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

## Where the VAE is still better

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

## Verdict

For a private, fair and realistic release of Adult-like data, use the **AIM
hybrid**. It is the only method whose privacy claim holds. It produces data
that looks real and is fair in the synthetic table itself (sex is unrelated
to every column), with downstream parity on par with the VAE.

Choose the VAE method only if formal privacy is not required and
equalized odds is the priority, or for data too wide for marginal-based
synthesis. Even then, its impossible rows need handling.

## Caveats

- **Unequal privacy in the 3-fold comparison.** Only fold 0 has the VAE at
  epsilon 3; its folds 1–2 exist only without DP. On fold 0, DP cost the VAE
  realism (validity 0.74 → 0.67, impossible rows 17% → 26%) but not
  fairness (parity 0.099 → 0.091), so epsilon-3 runs on all folds would
  sharpen the realism gap; the effect on fairness is unknown.
- **Public vs private bins.** With the fixed public bins the AIM hybrid
  scores parity 0.135 (vs 0.147) at the same EO, AUC and validity, but its
  numeric values are less realistic (W1 0.045). See the README.
- **A-NCB is excluded from the verdict.**
  - On Adult it reduces to one married/not-married axis.
  - The real-trained version rewards incoherent rows: AIM 0.22, VAE 0.45
    with 19% impossible rows.
  - The own-S version rewards label drift: AIM 0.92, VAE 0.72.
  - See `WRITEUP.md`.
- **Both methods rewrite real sex differences.** The AIM release says men and
  women have the same hours, occupations and incomes. Its income links to
  legitimate features are diluted: married vs unmarried >50K is 0.35 / 0.16
  (real 0.46 / 0.07).
- **Not yet tested:** Angelozzi & Arcolezi's (2026) placement, which repairs
  plain AIM's output instead of inside the generator, on the same folds.
- **Tuning is not covered.** Settings were chosen by scoring on real data,
  which is outside DP. A deployed release must fix them in advance.

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
