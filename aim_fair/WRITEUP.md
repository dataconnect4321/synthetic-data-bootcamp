# Fair, private synthetic Adult data: from FLIP to AIM with marginal repair

*Status: final setting fixed; 3-fold results below (`aim_fair/run_final.sh`).
All results, including the head-to-head with our best VAE method: `RESULTS.md`.*

## Why we moved away from FLIP

Our replication of FLIP (arXiv 2508.21815) on UCI Adult found that its
implementation does not deliver the privacy-preserving fairness it
describes.

- **Incomplete privacy.** DP-SGD covers only the VAE.
  - In our replication, the diffusion model is trained on latents of the
    real data, outside the accounting.
  - The batch-coupled fairness terms are not covered by per-sample DP-SGD
    either, so even the VAE's epsilon is nominal.
- **Metrics that can be satisfied without removing leakage.** BER and A-NCB
  use the generated sex label. When that label drifts from the row, the
  scores improve while true-sex BER stays at about 0.14 for every lambda and
  epsilon.
- **A weak objective.** CKAᵀ compares only the second-moment shape of
  representations, so it misses mean shifts, nonlinear dependence and joint
  structure. It saturates at about 0.996 early in training, and its gradient
  is nearly orthogonal to sex removal (cosine 0.02–0.18).
- **Untouched proxies and latent space.** Sex stays perfectly recoverable
  from the latent (probe BER 0.000), and proxies such as relationship =
  Husband persist.

Adding nonlinear and joint dependence penalties did not fix this. At the
latent and decoder stages, the model moved sex into forms the penalty
couldn't see. Only a penalty on the decoded output reduced real leakage, and
the latent space was still unchanged. With all the fairness work happening
on the output distribution, the VAE latent space was no longer contributing.

## The new approach

We moved to a generator that works on the output distribution directly and
gives a formal guarantee for the whole release: AIM (McKenna et al., 2022)
on Private-PGM (McKenna et al., 2019).

- AIM adaptively measures 1- and 2-way marginal tables with Gaussian noise
  under zero-concentrated DP.
- Private-PGM fits a graphical model to the noisy tables and samples
  synthetic rows from it.

Fairness and realism are added by editing the noisy measurements and the
samples. Every step reads only the noisy measurements or the model fitted
to them, so by DP's post-processing property (Dwork & Roth, 2014) none of it
costs privacy budget.

0. **Private bins.** Numeric bins are learned under DP from 5% of the
   zCDP budget (`dp_schema.py`): Gaussian noisy histograms on a public fine
   grid; heaped values (e.g. 40/50/60 hours, $15,024 capital gain) as their
   own bins; ordered equal-frequency bins elsewhere. This is the only step
   besides AIM's measuring that reads the data, and it is inside the budget.
1. **Proxy masking.** Category levels that are near-deterministic for one
   sex are merged into one level, by summing their cells in every noisy
   table.
   - A level is flagged when P(sex | level) ≥ 0.95, it covers at least 1% of
     rows, and the flagged levels of the column point to different sexes.
   - On Adult this finds exactly Husband + Wife on every fold.
2. **Repair** (adapting the disparate impact remover of Feldman et al.,
   2015). Each attribute's distribution for each sex group, including
   income, is transported to the pooled distribution (tau = 1). Every
   measured table is pushed through the transport plans, and the model is
   refitted.
3. **Pair redraw.**
   - Rows that break a consistency rule have only the broken pair redrawn
     from the model, conditional on the rest of the row. Any combination
     that would break a rule is excluded.
   - The rules are found by a pre-specified test on the noisy non-sex 2-way
     tables: a cell is impossible when, with 95% family-wise confidence
     (Bonferroni), its true count is below 10% of what independence
     predicts. On Adult this gives the same 6 household-consistency rules on
     every fold.
4. **Marginal check.** Columns whose 1-way distribution drifted from the
   repaired model's by more than 0.2 points are re-matched, by redrawing
   only that column.

Unlike CKAᵀ, the repair equalizes the full distribution of every attribute
across sex groups: a nonlinear classifier can no longer recover the
synthetic sex attribute (BER ≈ 0.50).

This differs from Angelozzi & Arcolezi (2026), who benchmark fairness
interventions applied to the output of DP synthesizers, including AIM, or to
the downstream model, and find the disparate impact remover weak under DP.
Here the repair runs inside the generator.

**Our own contributions** (not published): applying Feldman's repair to
noisy marginal tables, proxy masking and consistency rules computed from the
noisy measurements, and the pair redraw with marginal re-matching.

## Results (3 folds, epsilon 3)

Mean ± sd across folds; AIM rows are the mean of 3 refits per fold. Every
variant comes from the same private measurements per fold.

| AIM, epsilon 3 | `run_aim.py` flags | Income parity | EO gap | AUC | Validity | Impossible rows | >50K rate | BER |
|---|---|---|---|---|---|---|---|---|
| plain | `--tau 0 --no-mask --no-fix` | 0.211 | 0.167 | 0.905 | 0.91 | 0.6% | 0.248 | 0.16 |
| repair only | `--no-mask --no-fix` | 0.125 | 0.080 | 0.889 | 0.62 | 16% | 0.248 | 0.51 |
| repair + mask | `--no-fix` | 0.139 | 0.136 | 0.884 | 0.72 | 0% | 0.248 | 0.50 |
| final, public bins (+ rules, pair redraw) | `--public-bins` | **0.139 ± 0.02** | **0.138 ± 0.03** | 0.889 | **0.89** | **0%** | **0.248** | **0.50** |

| **final, private bins** | (defaults) | 0.147 ± 0.02 | 0.126 ± 0.02 | 0.890 | 0.89 | **0%** | 0.247 | **0.50** |

Real data: validity 0.98, 0% impossible rows, >50K rate 0.248. The
private-bin row is 3 folds x 10 refits. It halves numeric distortion
(W1 0.045 → 0.024) at the same EO, AUC and validity; its parity is about
0.01 higher because realistic hours and capital-gain values teach the income
model a sharper real-world link (see the README).

Two dropped alternatives (experiment code, not on this branch):

- **Whole-row rejection** instead of the pair redraw reached the same
  validity but lowered the >50K rate to 0.230 (parity 0.145).
- **Fixed rule thresholds** (3 / 15 noise sd) instead of the test flagged
  3–7 cells per fold that do occur in real data, some involving income.

Summary:

- **Fairness.** The repair cuts income parity from 0.211 to 0.125 and the
  EO gap from 0.167 to 0.080.
- **Realism.** Masking and pair redraw restore realism (validity 0.62 →
  about 0.90, no impossible rows) at a moderate fairness cost (parity
  +0.01–0.02, EO +0.05).

## A note on A-NCB

A-NCB is reported because FLIP uses it, but it is not a reliable fairness
measure on Adult.

- **One axis.** Its 2-cluster split on Adult is the married / not-married
  axis. Excluding relationship alone raises the *real* data from 0.19 to
  0.97.
- **The real-trained version rewards incoherent rows.**
  - Repair-only tables scored 0.85: a third of their rows contradict
    themselves, which breaks the married split.
  - The coherent final tables score about 0.2, because a real-trained
    classifier reads married rows as male.
  - Inside the synthetic table, sex is balanced across the clusters (68% vs
    67% male).
- **The own-S version rewards label drift,** like BER.

Read it only next to validity and impossible rows. Downstream parity and EO,
BER and tstr are the primary measures.

## Sources

| Component | Source |
|---|---|
| Generator | McKenna, Mullins, Sheldon & Miklau, *AIM*, PVLDB 15(11), 2022, arXiv:2201.12677 |
| Model fitting | McKenna, Sheldon & Miklau, *Graphical-model based estimation and inference for differential privacy*, ICML 2019, arXiv:1901.09136; code github.com/ryan112358/mbi |
| Repair | Feldman, Friedler, Moeller, Scheidegger & Venkatasubramanian, *Certifying and Removing Disparate Impact*, KDD 2015 |
| Post-processing | Dwork & Roth, *The Algorithmic Foundations of Differential Privacy*, 2014 |
| Related benchmark | Angelozzi & Arcolezi, *Where to Intervene? Benchmarking Fairness-Aware Learning on Differentially Private Synthetic Tabular Data*, PoPETs 2026(4), arXiv:2607.07471 |
| Related method | Pujol et al., *PreFair*, VLDB 2023 |
| Replicated method | FLIP, arXiv:2508.21815 |

## Limitations

- **Release vs. people.** The repair changes the released data, not real
  people's features at prediction time, so parity cannot reach zero.
  "Married" stays a real-world proxy: the merged spouse level is about 61%
  of real men vs 15% of real women.
- **Every sex difference is rewritten,** including income: the release
  says men and women earn alike. The income repair also dilutes income's
  links to legitimate features:
  - married vs unmarried >50K is 0.35 / 0.16 (real 0.46 / 0.07);
  - top education levels lose about 0.15.

  DP noise adds to this on small categories.
- **Numeric spikes are smeared.** Values are decoded uniformly inside fixed
  public bins, so spikes such as capital-gain 15,024 / 99,999 and 50 or 60
  hours are lost. This comes from our decoding, not from AIM.
- **Rare contradictions remain** where the noisy tables give too little
  evidence at epsilon 3:
  - Married-spouse-absent & spouse, about 0.5% of rows;
  - capital gain and loss together, about 0.5%.
- **Tuning is not covered.** The settings were chosen by scoring on real
  data, which is outside the DP guarantee. A deployed release must fix them
  in advance; the rule test's alpha and kappa were fixed before evaluation.
