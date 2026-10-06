# EXPERIMENTS — evidence for the cleanup deletions

Code version: `adultDataTest` at `experiments-archive-pre-cleanup` (sweep) and
`dd76822`/`f0ab411` (final 3-fold runs). All sweep runs used the paper config
(`--paper-config --ls-mode batch_mean`), fold 0, split-seed 42, seed 42.

## Caveat on the sweep evidence

The sweep runs below were single-fold (fold 0) and loaded the **older fold-0
bank** (`phase1_pc.pt`), which had been through the flag-ordering bug and
carried ~100 Phase-2 epochs at λ=0 before the sweep settings were applied.
That makes them adequate for "it didn't help / it collapsed, archive it", but
**not headline evidence** — the headline results are the clean 3-fold
λ=0/λ=4 runs from the fresh `pc_clean_f{0,1,2}.pt` banks.

## λ sweep (CKAᵀ strength)

| setting | collapsed | BER | A-NCB | AUC | income rate | W1 |
|---|---|---|---|---|---|---|
| λ=0 | no | 0.463 | 0.937 | 0.913 | 0.248 | 0.029 |
| λ=4 | no | 0.399 | 0.940 | 0.912 | 0.246 | 0.035 |
| λ=8 | no | 0.254 | 0.916 | 0.787 | 0.222 | 0.038 |
| λ=16 | **yes** (income 0.153) | — | — | — | 0.153 | 0.053 |
| λ=32 | **yes** (income 0.151) | — | — | — | 0.151 | 0.086 |

Real income rate ≈ 0.248. λ=8 already costs 13 AUC points for no A-NCB gain;
λ≥16 collapses the target column (income rate halves) and is flagged by the
collapse guard. **λ=4 is the strongest usable setting.**

## MMD (RBF-kernel MMD² on the latent, `--mmd-weight`)

- Tier-1 probes with MMD weight 10: **no effect** on the latent sex probe or
  any reported metric (the CKAᵀ blind spot it targeted was not reachable at
  a weight that training survives).
- Paper-config sweep, MMD weight 20 (`pc_lam4_mmd20`): **collapsed** —
  income rate 0.181 vs real 0.248, flagged by the collapse guard.
- Conclusion: MMD either does nothing (weight ≤ 10) or collapses the run
  (weight ≥ 20). No setting in between helped. **Archived and deleted.**

## Sex adversary (gradient reversal, `--adv-weight`)

- Paper-config sweep, adversary weight 0.01 (`pc_lam4_adv`): **collapsed** —
  W1 0.763 (guard threshold 0.15), income rate 0.396 vs real 0.248. That run
  predates the batch-mean fix in `acd77c1`, so it used the old **summed**
  loss, which is enough by itself to explain the collapse.
- Conclusion: collapsed at weight 0.01 under the old summed-loss version;
  untested since the batch-mean fix. Archived as unpromising, recoverable
  from the tag. (Only `adversary_loss`/`ADV_WEIGHT`/`--adv-weight` and the
  sex-adversary wiring were deleted; `GradientReversal`/`AdversaryHead` are
  kept — `--proxy-mode adv` reuses them).

## S-column tempering (`temper_protected`, `--temper-tau`)

- The temper baseline was superseded by the post-hoc decoupling curve
  (`decouple_curve` in `temper.py`), which sweeps the same BER/leakage-vs-τ
  trade-off against the *true* sex on tier-1 decodes and covers the same
  ground with a cleaner reference.
- Conclusion: `temper_protected` and `temper_curve` are redundant.
  **Archived and deleted** (only those two functions, the `--temper-tau`
  option, and their tests; the rest of `temper.py` — `decouple_curve`,
  `true_sex_joint_tv`, `sublabel_conditional` — stays, it is in active use).

## Merge/expand post-hoc variant (`proxy_decouple.py`)

`merge_proxies`, `fit_member_tables`, `expand_merged` implement a
merge-before-training / expand-after-generation variant of proxy decoupling
that was never wired into `run_adult_flip.py`. The wired path is
`detect_proxy_classes` + `decouple_curve` (post-hoc) + the HSIC/adv in-model
term. **Unwired variant archived and deleted with its tests.**
`detect_proxy_classes`, `describe_classes`, `impossible_combination_rate`,
and `proxy_member_dependence` are in active use and stay.

## `compare_to_paper.py`

Superseded by `aggregate_folds.py` (per-fold relative-to-real values with
fold spreads). **Deleted.**

## Her-PR code (NOT deleted, needs her sign-off)

Her PR contributed: the dim=1 CKAᵀ option, the batch-mean L_S routing,
`run_experiment.py`, the configs, and `fairness_eval.py`. None of the
deletions above touch her code. Deleting or changing any of her
contributions requires her agreement first.
