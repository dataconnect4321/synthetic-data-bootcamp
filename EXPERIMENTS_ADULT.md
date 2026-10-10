# Adult experiments: evidence for the presets and the cleanup deletions

The paper-faithful replication (the current default) is the last section,
"Paper-faithful replication". The sections before it document the earlier
setup, now `--preset ours`.

Code version: `adultDataTest` at `experiments-archive-pre-cleanup` (sweep) and
`dd76822`/`f0ab411` (final 3-fold runs). All sweep runs used the paper's
training setup with the batch-mean L_S (then `--paper-config --ls-mode
batch_mean`), fold 0, split-seed 42, seed 42.

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

## Disha's PR (#1 on `main`) — merged into `adultDataClean`

`main` (PR #1, `23b98fe`: FLIP paper training/eval protocol and Adult
replication attempt) is merged into `adultDataClean`, so merging this branch
into `main` overwrites none of it. Nothing of hers was deleted:

- Kept unchanged: `run_experiment.py`, `configs/flip.yaml` (see below),
  `fairness_eval.py`, `paper_eval.py`, `phase2_sweep.py`,
  `preprocess_adult.py`, `ARCHITECTURE.md`, `CHANGELOG.md`, her
  `EXPERIMENTS.md` (this file was renamed to `EXPERIMENTS_ADULT.md` to
  leave hers intact), `tests/test_metrics.py`, `tests/test_paper_eval.py`,
  and her fixes (per-sample L_S as a mean, non-negative clip, encoder-width
  one-hots, `recon_num_pre`, DP-safe fixed beta decay, diffusion early
  stopping, paper metrics in the HMDA report).
- Her paper-literal method choices became parameters whose DEFAULTS are
  the final Adult setup, and `configs/flip.yaml` sets her values, so her
  experiments run exactly as before: `CKA_CENTER` ('group' | 'row'),
  `NUM_STAGE` ('summed' | 'pre'), `DIV_STAGES` ('latent' |
  'latent+decoder'), `MEAN_SHIFT_WEIGHT`, `BINARY_W1`, `LAMBDA_RAMP_FRAC`
  (0 = no ramp), `BETA_SCHEDULE` ('fixed' | 'decay' | 'plateau'), and
  `train_diffusion(mu, std=None)` (std = her reparameterized latents).
- Her batch-mean L_S routing through the batch-coupled pass is used for
  every run (DP-correct; without DP the gradient is unchanged).
- Four of her tests exercise her paper-literal path; they now pass her
  settings explicitly (`center='row'`, `num_stage='pre'`, ...).

## `adultDataClean` branch — streamlined to the final setup

Branched from `adultDataTest` at `97bd885`. Everything the final run plan
(README Part 10) does not use was removed; `adultDataTest` keeps it all.

| Removed | Why |
|---|---|
| In-model proxy decoupling (`--proxy-weight/-mode/-exclude-disent`, HSIC and gradient-reversal adversary, `proxy_sublabel`, token helpers, `exclude_cat`) | Fold-1 screens: no effect on leakage at weight <= 32, collapse at 512; superseded by masking |
| Post-hoc decoupling curve (`temper.py`, `--decouple-curve`, `decouple_curves.py`) | Superseded by masking (pre-training) and the repair baseline |
| Random re-draw of proxy members (`--reencode-tau`) | Created impossible rows (male wives); replaced by masking |
| Mean-softmax TV marginal penalty | Let sex-skewed categories vanish (married, Craft-repair, race=Black); KL guard only |
| Linear joint stage, `--cat-decode sample` (all categoricals) | Linear fooled only the soft probabilities; sampling all columns did not stop deletion |
| Diagonal-covariance A-NCB, paper-style (own-S) A-NCB | Diag split off a cluster unrelated to sex (A-NCB ~1 for real and synthetic); own-S adversary is trivially fair under an invented S column. Full-covariance A-NCB is now the default and only one |
| Diagnostics: latent probes, shared-mean ratio, label validity, inferred group gap, tier-1 group agreement, relationship faithfulness, proxy ranking, proxy-member dependence, promotion gates, HMDA fairness extras, `--diagnose-only`, `--paper-literal`, `--recon-s`, `--proxy-col/-labels`, `--fast` subsample | Not part of the final report |
| `aggregate_folds.py` | Superseded by `final_table.py` |
| Tracked `__pycache__/*.pyc` files | Build artifacts (already in `.gitignore`) |

Behaviour changes for anyone reusing the flags (see also the next section):
`--paper-config` and `--ls-mode` are gone (the paper's training setup and
the batch-mean L_S are always on); the method is chosen with `--preset
paper|ours` plus per-piece flags; `--decoded-joint` is a switch (RFF lift
always).

## Paper-faithful replication (`--preset paper`)

Code: `adultDataClean` after the merge of PR #2. All runs unmasked unless
stated, split-seed 42, seed 42; "tier-1" decodes each real row's latent (so
every row has a known real person), "full" samples latents from the
diffusion model. Fold-0 screens first, then a 3-fold λ grid. The cleaned-up
default reproduces the grid's fold-0 λ=4 tier-1 run to four decimals.

### Corrections to the implementation

| Issue | Finding | Fix |
|---|---|---|
| Phase 2 objective | Sec. 4.3.2 keeps L_quality (ELBO + L_S) in Phase 2 as the detokenizer-stage divergence term; Eq. 10 alone reads as if it were dropped. An ablation that dropped it collapsed (λ=4 numeric W1 0.76, validity 0.20 without our guards) | ELBO kept (the only option now; the ablation was removed) |
| β in Phase 2 from a bank | a bank stores only the weights, so Phase 2 restarted β at 1e-2 instead of the decayed ~8e-6 a single run would carry | `--phase2-beta` (number or `carry`; banks write their final β to `BANK.pt.json`) |
| SWD stages | the paper anchors the latent AND decoder; we anchored the latent only | `--div-stages latent+decoder` (paper preset) |
| Diffusion input | the paper trains diffusion on μ + σε; we trained on μ | μ + σε, always |

### What moves (fold 0, tier-1, λ = 4 unless stated)

| Setting | True-sex BER | Paper's BER (argmax) | Note |
|---|---|---|---|
| Paper disentanglement, β carried (8e-6), λ 0 → 4 | 0.139 → 0.140 | 0.131 → 0.156 | CKAᵀ 0.88 → 0.95 |
| … summed stages, λ 4/8, 100/400 epochs | 0.139–0.141 | 0.089–0.162 | CKAᵀ up to 0.996; sex *more* separable inside the model (other-token probe 0.81 → 0.99) |
| … Phase-2 β = 1e-2 | 0.140 | 0.413 (λ 0: 0.164) | sex token gap/σ 0.8; reproduces the paper's BER trend |
| … + dim-0 CKAᵀ + mean gap + joint stage | 0.43–0.48 | 0.10–0.43 | `relationship` SP by real sex 0.54 → 0.07–0.23; W1 ~×2 |
| Masked bank, paper disentanglement | 0.194 → 0.194 | — | masking alone does not let λ act |
| Repair τ=1 bank (masked + scrambled) | 0.502 | — | repair removes the real proxies before training |

Diagnostics on the saved models (scratch scripts, not kept): the latent's
sex token is linearly recoverable at 1.000 at every λ; removing it changes
CKAᵀ by 0.004–0.010; the decoder writes extra sex signal into columns that
carry none in the real data (capital-gain/loss, age, fnlwgt: probe accuracy
0.51 → 0.59–0.67); reconstruction does not use the sex token (collapsing or
scrambling it changes the ELBO by ≤ 1e-4) while the SWD anchor and L_S hold
its group gap in place (activation-space gradient effects +0.045 and +0.044
vs CKAᵀ −0.035 at the end of λ=4 training).

### 3-fold grid (λ 0/2/4/8/12, Phase-2 β = 1e-2, argmax)

`results/paper_faithful_{tier1,full}_grid.{csv,txt}` and the heatmaps. Both
of the paper's headline measures rise with λ — BER (generated sex) 0.27 →
0.42 and A-NCB with the adversary trained on the generated sex (full GMM)
0.28 → 0.58 (tier-1) / 0.49 → 0.71 (full, non-monotone) — while true-sex
BER stays 0.139–0.140, the A-NCB with a real-trained adversary shows no
trend, and a diagonal GMM reverses the A-NCB trend. α-precision falls with
λ as in the paper; AUC and ε-identifiability stay flat.

### Removed after these experiments

| Removed | Why |
|---|---|
| ELBO-free Phase 2 (`--phase2-objective fair`) | not the paper (Sec. 4.3.2); collapses |
| Group-conditional L_S (`--ls-mode group_mean`) | removes the person's sex from the generated label, but the label then follows other columns (synthetic BER 0.235 at λ=4); neither preset uses it |
| Share-matched argmax (`--s-decode share`) | recovers the person's sex from the tilt (100% on a bank model); diagnostic only |
| `--paper-config`, `--ls-mode`, `--paper-disent`, `scripts/run_final_sweep.sh` | folded into the defaults / `--preset` / `scripts/run_paper_grid.sh` |

## Sensitivity sweep of the paper's unstated choices, and the final configuration

All fold 0, tier-1, λ = 4 unless stated, unmasked bank, paper preset
otherwise. "Real sex" metrics use each tier-1 row's source person.

### Checks on the evaluation and the inputs

| Check | Result |
|---|---|
| Income in the fairness terms (`--target-in-fairness`) | identical to the digit: income has two levels, and row-centred CKAᵀ is constant for two columns, so the paper's method cannot score it either way |
| `--binary-w1` (W1 on two-level columns, ours) | λ 12: income gap between true-sex groups 0.198 → 0.105, AUC 0.914 → 0.903; true-sex BER 0.140 → 0.143 |
| Group centring (dim 0) instead of row | AUC flat; income gap unchanged (a rate difference is a mean offset, which dim-0 removes); the paper's BER/A-NCB trend weaker (0.24/0.25 at λ 4 vs 0.41/0.43) |
| Downstream AUC with sex as a feature | identical to ±0.001 at every λ (real data too: sex adds nothing beyond relationship / marital status) |
| The paper's Fig. 3 numbers (no-DP column) | AUC 0.75 → 0.57, α 0.90 → 0.29, β 0.41 → 0.15 with λ 0 → 12, BER 0.41 → 0.50, A-NCB 0.32 → 0.92: their λ degrades the data broadly |

### Readings of the method that could strengthen λ

| Setting | True-sex BER | `relationship` SP, real sex | Note |
|---|---|---|---|
| Paper preset (reference) | 0.140 | 0.528 | |
| Phase-2 LR ×3 / ×10 | 0.140 / 0.140 | 0.532 / 0.570 | scrambles the sex label even at λ 0 (paper BER 0.38) |
| Phase-2 reconstruction weight 0.3 / 0.1 / 0.03 | 0.141 / 0.145 / 0.150 | 0.52 / 0.52 / 0.51 | W1 0.045 → 0.079: untargeted drift |
| SWD weight 0.3 / 0.1 / 0 | 0.140 / 0.139 / 0.139 | 0.53 | the paper's BER/A-NCB improve further (0.49 / 0.79) |
| SWD 10 / 500 projections; latent SWD on z | 0.140 | 0.52–0.53 | |
| Per-token CKAᵀ (λ 4 / 12) | 0.140 / 0.140 | 0.53 | relationship token's CKAᵀ 0.65 → 0.97 while its sex probe stays 0.72–0.74; the paper's BER trend vanishes |
| RBF-kernel CKAᵀ (λ 4 / 12) | 0.140 / 0.142 | 0.52–0.53 | gradient alignment 0.33 at the bank, ~0.07 after training |
| RFF-lift CKAᵀ (λ 4 / 12) | 0.142 / 0.172 | 0.52 / 0.51 | latent probe 1.00 → 0.83; λ 12 trips the collapse guard (W1 0.156) |

The global optimizer-level gradient clip (ours, norm 5) is post-processing
under DP and was not limiting (Adam undoes a uniform rescale); checkpoint
selection is not used (Phase 2 returns its final weights; diffusion early
stopping is off).

### What CKAᵀ sees (permutation and gradient tests on saved models)

The real men/women split lowers CKAᵀ far below random 50/50 splits at every
stage (latent 0.876 vs 1.000; relationship token 0.658 vs 0.999), and
removing each group's mean representation brings it back to ~1 and a linear
sex probe to chance: CKAᵀ registers the groups' mean differences. The cosine
between the gradient that raises CKAᵀ and the gradient that defeats a sex
probe is 0.02–0.18 at the bank and ~0.05 after Phase 2, so optimizing CKAᵀ
barely acts on that information.

### Ablation of the additions on the paper backbone

| Arm | True-sex BER | Synthetic-trained BER | `relationship` SP, real sex | Income SP | AUC | Largest category shift |
|---|---|---|---|---|---|---|
| Backbone | 0.140 | 0.163 | 0.546 | 0.204 | 0.912 | 0.003 |
| + mean gap at the stages | 0.139 | 0.163 | 0.558 | 0.163 | 0.912 | 0.003 |
| + dim-0 CKAᵀ + mean gap | 0.148 | 0.166 | 0.553 | 0.177 | 0.901 | 0.012 |
| + joint stage (income in, no guard) | 0.396 | 0.302 | 0.175 | 0.295 | 0.892 | 0.242 |
| + mean gap + joint (income in, no guard) | 0.494 | 0.297 | 0.068 | 0.000 | 0.768 | 0.439 (collapsed) |
| + joint, income out, no guard | 0.451 | 0.292 | 0.100 | 0.036 | 0.889 | 0.415 |
| **+ mean gap + joint + KL guard, income in (final)** | **0.265** | **0.230** | **0.299** | **0.056** | **0.894** | **0.160** |
| + mean gap + joint + KL guard, income out | 0.283 | 0.254 | 0.370 | 0.130 | 0.905 | 0.150 |
| + joint + KL guard, masked, income in | 0.235 (masked real 0.195) | 0.168 | 0.439 | 0.282 | 0.888 | 0.104 |

The joint stage is the active ingredient; the guard is what lets income stay
in the fairness terms without the target collapsing; masking adds no real
reduction on top. The 3-fold tier-1 check of the final configuration is in
README Part 10.5 and `results/final_tier1_check.{csv,txt}`.
