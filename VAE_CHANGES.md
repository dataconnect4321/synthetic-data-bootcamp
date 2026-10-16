# VAE (FLIP) changes on this branch

The best VAE process is FLIP on Adult with this repository's output-level
joint stage: `run_adult_flip.py --preset final --lam 4`.

- The `final` preset is already on `main` (README Part 10.5): FLIP as
  published, plus an RFF dependence penalty on the decoded output with the
  mean gap (an MMD between the groups' decoded rows) and a KL guard on the
  marginals.
- This branch adds what was needed to run it under the paper's DP setting
  and to compare it with the AIM hybrid (`aim_fair/`).
- The verdict and the reasons for moving to AIM are in
  `aim_fair/FINAL_COMPARISON.md` and `aim_fair/WRITEUP.md`.

## What changed

**DP as in the paper (Eq. 13, VAE only)**

- `--dp-group-sigmas S0,S1`: per-group noise multipliers, with clipping per
  sample and sensitive group (IDP-SGD style).
- `--dp-coupled paper`: the batch-coupled fairness terms (CKAᵀ, SWD, joint
  stage) are trained with the same noisy step. This gives the paper's
  *nominal* epsilon; per-sample DP-SGD does not cover batch-level terms.
- A grouped RDP accountant.
- The DP metadata of a banked Phase 1 is saved with it. A Phase 2 that does
  not match it is composed separately.
- Diffusion stays non-private, as in the paper.
- `calibrate_dp.py --eq13 --coupled paper` finds the sigmas for a target
  epsilon.
- `--dp-group-noise / --dp-stats-noise / --dp-clip-release /
  --dp-diffusion` are the stricter alternatives that were tested. With them
  the fairness signal drowns.

**Scoring external tables**

- `--score-csv PATH` scores any synthetic CSV (the AIM tables) with exactly
  the metrics of a VAE run; no model is trained.
- With `--mask-proxies`, merged proxy levels are scored against the masked
  real data.

**Metrics**

- `tstr_sex_ber` / `_folded`: a sex classifier trained on synthetic data and
  tested on real people. This is the leakage measure for unpaired tables.
- Validity: plausibility rules learned from the real training fold.
- Impossible (sex, category) rows.
- A-NCB with 2 FAMD components (prince's default). This makes it
  seed-stable; it was unstable with all components.
- Real-data reference scores are cached per fold.

**Stability**

- `--grad-clip`, `--adam-eps` and `--no-recon-clamp` expose the existing
  safeguards. Defaults are unchanged.

**Diagnostic options**

These were kept as off-by-default flags and are not part of the best
configuration. Each was tested and rejected (details in
`aim_fair/WRITEUP.md`):

- inner-stage dependence measures (`--cka-center`
  `pooled_bias|dep_cka|dep_hsic|dep_lf_lin|dep_lf_rbf`): sex moved into
  forms the penalty could not see;
- `--cka-latent`, `--cka-exclude-s`, `--mask-s-token`,
  `--swd-group-center`, `--joint-measure dep_rbf` and `--joint-weighting`;
- `--reject-invalid` / `--repair-invalid` (selection bias under DP);
- `--cat-decode sample` and `--latent-source phase1`;
- `--batch-size`.

## Reproduce

```bash
PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh   # no DP, 3 folds
bash scripts/run_best_vae_dp.sh                          # nominal eps 3, fold 0
python -m pytest -q tests
```

## Result

| | No DP (3 folds) | Nominal epsilon 3 (fold 0) |
|---|---|---|
| Income parity | 0.132 ± 0.03 | 0.091 |
| Equalized-odds gap | 0.044 ± 0.06 | 0.080 |
| Validity | 0.72 | 0.67 |
| Impossible rows | 19% | 26% |
| Income AUC | 0.897 | 0.887 |

It is the only FLIP-based variant that reduced real leakage and income
parity. But its privacy is nominal, and about a fifth of its rows are
contradictory. The AIM hybrid matches its parity at a rigorous epsilon 3
with realistic rows (see the comparison).
