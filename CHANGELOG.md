# Changelog

## 2026-10-16: Pivot from FLIP to the AIM hybrid (`aim_fair/`)

- **`aim_fair/`: AIM / Private-PGM at a rigorous ε.** Fairness and realism
  are post-processing of its noisy measurements:
  - proxy masking;
  - Feldman-style repair of every column, income included, to the pooled
    distribution;
  - consistency rules from a pre-specified test;
  - pair redraw from the model conditional with marginal re-matching.

  Write-up: `aim_fair/WRITEUP.md`. Verdict against the best VAE process:
  `aim_fair/FINAL_COMPARISON.md`.
- **VAE.**
  - Paper-style Eq. 13 per-group DP (`--dp-group-sigmas`, `--dp-coupled
    paper`, `calibrate_dp.py`).
  - `--score-csv` for external tables.
  - New metrics: tstr sex BER, validity and impossible rows; A-NCB on 2
    FAMD components.
  - Off-by-default diagnostic options.

  See `VAE_CHANGES.md`.
- **Scripts:** `scripts/run_best_vae_dp.sh` and `aim_fair/run_final.sh`.

## 2026-10-09: Adult reproduction (`run_adult_flip.py`, PR #3)

The Adult work moved from `configs/flip.yaml` experiments to a dedicated
driver, `run_adult_flip.py`, with one 3-fold-CV fold per run. Full details,
results and commands are in README Part 10; what was tried and removed is in
`EXPERIMENTS_ADULT.md`.

### Presets

- **`--preset paper` (default): FLIP as published.**
  - Phase 1: ELBO + batch-mean L_S (Eq. 3), β plateau-decayed 1e-2 → ~1e-5.
  - Phase 2: L_quality kept at the detokenizer (Sec. 4.3.2), SWD at the latent and decoder, λ·(−CKAᵀ) at all three stages with row centring (Eqs. 5–7).
  - Diffusion on μ + σε (now always on); argmax decoding; income stays in the fairness terms (task-agnostic).
- **`--preset final` (recommended):** the paper backbone plus an RFF joint stage on the decoded row with a mean gap (≈ MMD between the groups' decoded rows) and a KL marginal guard.
- **`--preset ours`:** the earlier additions (group centring, mean gap, binary W1, proxy masking), kept for comparison.

Every piece of a preset is also its own flag. The paper's unstated choices are flags too: `--phase2-beta`, `--stage-reduce`, `--phase2-lr-scale`, `--phase2-recon-weight`, `--div-weight`, `--swd-projections`, `--swd-latent`, `--cka-tokens`, `--cka-kernel {linear,rbf,rff_lift}`.

### Evaluation

- **True-sex BER (tier-1):** each real row's latent is decoded, and a classifier trained on real data predicts the row's real sex. This measures leakage that the paper's BER, computed on the generated sex column, cannot see.
- **A-NCB with two adversaries:** one trained on the synthetic table's own generated sex column (the paper's reading) and one trained on the real training fold.
- Downstream AUC now includes sex (no-sex AUC recorded too).
- New metrics: validity (rules learned from real data) and the largest category share change.
- Released CSVs are rounded to each real column's format.
- Each run is saved to `adult_results/runs/`; `--from-run` re-decodes a saved run without retraining.

### Findings

- On Adult, FLIP as published leaves true-sex BER at the real data's 0.139 for every λ (0–12).
- The paper's BER and A-NCB trends do reproduce. They come from noise in the generated sex label, not from removing proxies.
- CKAᵀ detects group differences, but its gradient is nearly orthogonal to removing sex information.
- With `--preset final` at λ = 4 (3 folds):
  - true-sex BER is 0.26, against 0.14 for the backbone;
  - income SP falls 0.21 → 0.12;
  - real-trained A-NCB rises to 0.61;
  - AUC is 0.896, against 0.91 for the backbone.

### Removed

`--paper-config`, `--ls-mode`, `--paper-disent`, share-matched decoding, the fair Phase-2 objective, group-mean L_S, MMD, the sex adversary, S-column tempering, the merge/expand variant and `compare_to_paper.py`. The evidence for each removal is in `EXPERIMENTS_ADULT.md`.

### Tests and scripts

- 130 tests.
- Sweeps: `scripts/run_paper_grid.sh`.
- Tables and heatmaps: `paper_grid_report.py`, `paper_grid_heatmap.py`, `final_table.py`, `make_heatmaps.py`.

## 2026-10-03: the paper's training setup and evaluation protocol

This change lets the pipeline run the way the FLIP paper describes, so the Adult results can be checked against it before the method is applied to HMDA. Every new behaviour is opt-in through config. The defaults are unchanged, and `baseline --predict-eps` still gives 197 / 261.

The paper publishes no hyperparameters. Settings it leaves unstated follow TabSyn, which it says it adapts.

### Training

- **β schedule.**
  - Where: `beta_at()` in `run_flip_demo.py`, and the keys `BETA_SCHEDULE`, `BETA_MAX`, `BETA_MIN`, `BETA_DECAY`, `BETA_DECAY_EVERY`.
  - Why: the paper decreases β adaptively, as TabSyn does (Sec. 4.3.1). TabSyn triggers each decay on a training-loss plateau, which reads raw data outside the DP accounting. Here the schedule is fixed in advance.
- **VAE width.**
  - Where: new key `FACTOR`, used by both `Model_VAE` constructions in `train_flip`.
  - Why: it was hard-coded to 8, while TabSyn uses 32.
- **Diffusion early stopping.**
  - Where: new key `EARLY_STOP_PATIENCE` in `train_diffusion`.
  - Why: TabSyn trains for up to 10,000 epochs and stops after 500 without improvement. Setting it together with diffusion DP raises an error, because the stopping decision reads the training loss.

### Evaluation (paper Sec. 5.3)

- **3-fold cross-validation.**
  - Where: `run_experiment.py --fold k` (`make_fold`) and `--summarize`.
  - Why: the paper trains on two folds and scores against the held-out third. The split is stratified on target × group with seed 42, so every experiment sees the same folds.
- **New `paper_eval.py` and an `eval` stage.**
  - What it adds:
    - synthetic AUC, statistical parity and equalized odds, from a model trained on synthetic data and scored on the real test fold;
    - α-precision and β-recall (synthcity's naive variant, as TabSyn uses);
    - ε-identifiability (Gower distance with entropy weights, as in SynthEval);
    - BER, A-NCB and task fairness, reused from `fairness_eval.py`.
  - Checked on real data scored as synthetic: α-precision 0.997, ε-identifiability 0.495, SP 0.176 (the paper's λ=0 value is 0.18).
  - Deviation: HistGradientBoosting instead of LightGBM.

### The paper's uniform attribute loss (Eq. 3)

- **New option `LS_VARIANT: 'batch_mean'`.** The default stays `'per_sample'`.
  - Where: `train_flip` in `run_flip_demo.py`; ε prediction in `run_experiment.py`.
  - Why: Eq. 3 only requires the batch-average softmax of S to be uniform. The per-sample variant pushes every row to 50/50, and argmax decoding then collapsed the synthetic sex column to one group: 99.6% Female in the first λ=0 gate run, kept as `runs/adult_paper/old_per_sample_ls_fold0/`.
  - How: the batch-mean loss couples the batch, so it joins the SWD and CKA terms in pass 2, which now runs in both phases. Under DP it goes through `GroupLevelDPMechanism`, and `--predict-eps` counts those extra steps. The per-sample term stays in pass 1 with weight 0, because otherwise the S head gets no per-sample gradient and Opacus raises an error.

### Fast Phase-2 debugging

- **`PHASE1_CHECKPOINT`.**
  - Where: `train_flip` and `save_phase1_checkpoint` in `run_flip_demo.py`.
  - What: the model and optimizer are saved at the phase boundary. If the file already exists, Phase 1 is skipped.
  - Limits: refused under DP, because skipping Phase 1 would skip its privacy accounting.
- **New `phase2_sweep.py`.**
  - What: trains Phase 1 once, then runs only Phase 2 for each λ, in about 15 min instead of 1.7 h.
  - Probes on the VAE directly, without diffusion:
    - latent BER: predicting the protected attribute from the encoder means, with and without that attribute's own token;
    - CKAᵀ at each stage (latent, detokenizer, decoder).
  - Why: λ=4 showed no fairness effect, and CKAᵀ was already about 0.9 when Phase 2 started.

### Config and tests

- **Experiments.** `adult_paper` (λ=0), `adult_paper_lam4`, `adult_paper_eps3`, `adult_paper_lam4_eps3` and `adult_paper_smoke`. DP is on the VAE only (paper Sec. 4). The paper runs use `ls_variant: batch_mean`. The ε=3 runs use batch 4096 with σ 64 and group σ 256, which predicts a composed ε of 2.9 on a fold.
- **Tests.** New `tests/test_paper_eval.py` covers the metrics, the folds and the β schedule (91 tests in total).

## 2026-10-01: crash fixes, result bugs, and alignment with the FLIP paper

This change fixes bugs that crashed the pipeline or corrupted its results, and places where the code departed from the paper (Hyrup et al., arXiv:2508.21815). It also adds the paper's fairness metrics. Each entry gives what changed, where, and why.

**Path shorthand.** `vae/` means `tabsyn-main/tabsyn-main/tabsyn/vae/`.

**Scope.** The live training loop is `train_flip()` in `run_flip_demo.py`. Both the demo and the full pipeline use it.

### Crashes

- **One-hot widths come from the encoders, not `max()+1`.**
  - Where: [run_flip_demo.py:793](run_flip_demo.py#L793) (`_downstream_fairness`) and [run_flip_demo.py:734](run_flip_demo.py#L734) (`_attribute_probe_auc`).
  - Why: when the synthetic data never produced a category, the real and synthetic feature matrices had different widths and `X_real @ w` raised `ValueError`. Every demo run crashed this way before writing its report, with or without DP (`runs/hmda/logs/demo.log`, `runs/baseline_before/no_dp_demo_before.log`).
- **Standalone scripts default to repo-relative paths.**
  - Where: [preprocess.py:7](preprocess.py#L7), [fairness_audit.py:23](fairness_audit.py#L23), [fairness_report.py:18](fairness_report.py#L18).
  - Why: they hardcoded `c:\Users\db234\...`, so running them directly failed on any other machine. `run_experiment.py` already overrode these paths.

### Bugs that corrupted results

- **The non-negativity clamp happens at real-unit 0.**
  - Where: new `clip_nonnegative()` at [run_flip_demo.py:574](run_flip_demo.py#L574), used by `generate()` and by `decode_latents()` at [run_flip_full_pipeline.py:282](run_flip_full_pipeline.py#L282).
  - Why: `np.clip(syn_num, 0, None)` ran on *standardized* values, where 0 is the column mean, so every numeric column was floored at its mean. The clamp is now at −mean/std.
- **The downstream fairness probe excludes the target from its features.**
  - Where: [run_flip_demo.py:789](run_flip_demo.py#L789).
  - Why: the model predicting the target was given the target as an input, so it read the label back and could never show bias.
- **Per-row L_S is averaged, not summed.**
  - Where: [vae/flip_fairness.py:470](tabsyn-main/tabsyn-main/tabsyn/vae/flip_fairness.py#L470).
  - Why: MSE and CE are means. A sum outweighed reconstruction by about the batch size (L_S 23 vs MSE 2.2 at batch 64), and under Opacus's mean-reduction scaling it dominated each row's clipped gradient.
- *(Reverted later the same day; see "Fairness focus" below.)* The diffusion latent mean's DP release and the composed-ε report headline.

### Deviations from the paper

- **CKAᵀ centers each row across activations (`dim=1`).**
  - Where: [vae/flip_fairness.py:279](tabsyn-main/tabsyn-main/tabsyn/vae/flip_fairness.py#L279).
  - Why (Sec. 4.2.2, Eqs. 5–7): CKAᵀ(A,B) := CKA(Aᵀ,Bᵀ). The activations are the "samples", so K = AᵀA and the centering matrix H is p×p, and A·H subtracts each row's mean.
  - The old `dim=0` centered each activation over samples. That discarded group means: a mean-shifted group scored 0.997 instead of 0.76.
  - A test checks the code against the explicit tr(KHLH) form.
- **Inputs with ≤ 2 columns are skipped in the disentanglement loss.**
  - Where: [vae/flip_fairness.py:305](tabsyn-main/tabsyn-main/tabsyn/vae/flip_fairness.py#L305).
  - Why: after centering across two columns, every group's Gram matrix is ∝ [[1,−1],[−1,1]], so CKAᵀ is constant and carries no signal. This affects the binary `target` logits.
- **The detokenizer stage uses pre-aggregation numerical logits, one feature at a time.**
  - Where: `recon_num_pre` at [vae/model.py:386](tabsyn-main/tabsyn-main/tabsyn/vae/model.py#L386), used at [vae/flip_fairness.py:369](tabsyn-main/tabsyn-main/tabsyn/vae/flip_fairness.py#L369).
  - Why (Sec. 4.3.2): the paper uses "the pre-aggregation numerical logits", feature-wise and mean-aggregated. The code had used the summed output of all numeric features as one matrix.
- **The SWD divergence penalty applies at both the latent and decoder stages, against a frozen Phase-1 VAE.**
  - Where: [run_flip_demo.py:390](run_flip_demo.py#L390) and [:408](run_flip_demo.py#L408).
  - Why (Sec. 4.3.2): the paper approximates the divergence penalty with SWD at the latent and decoder stages. The code had only the latent stage and froze only the encoder.
  - The frozen copy is built from `state_dict`; a `deepcopy` would also copy Opacus's per-sample hooks.
- **Diffusion trains on reparameterized latents μ + σ·ε, with fresh ε per batch.**
  - Where: [run_flip_full_pipeline.py:121](run_flip_full_pipeline.py#L121), [:208](run_flip_full_pipeline.py#L208), [:351](run_flip_full_pipeline.py#L351).
  - Why (Sec. 4.1.2): the paper trains on reparameterized latents, not μ. Note that `forward_with_stages`' `std_z` is a log-variance, so σ = exp(½·logvar).

### Paper fairness metrics

- **New `fairness_eval.py`: BER (Def. 1), A-NCB (Def. 2) and per-feature task fairness (Eqs. 14/15).**
  - Where: [fairness_eval.py](fairness_eval.py), called from `fairness_metrics` at [run_flip_demo.py:696](run_flip_demo.py#L696); printed by `print_paper_metrics` ([:703](run_flip_demo.py#L703)) and shown in both HTML reports.
  - Why: the paper evaluates fairness task-agnostically. The repo only measured approval-rate parity, which a randomized race column can game.
  - Extended from binary to multi-group:
    - BER's ideal is 1 − 1/|S|.
    - NCB takes the minimum over group pairs.
    - Eq. 14 uses the largest gap across groups.
  - Uses scikit-learn histogram gradient boosting in place of LightGBM, so there's no new dependency.
  - Groups the adversary assigns fewer than 30 rows are left out of NCB. In the smoke run, one row predicted as one group forced A-NCB to 0.

### Docs

- **README.md:**
  - CKAᵀ derivation and centering (2.1.1).
  - Per-row L_S as a deviation (2.1.5).
  - Pre-aggregation detokenizer and decoder-stage SWD (2.1.3, 2.1.4, 2.3).
  - The `tabsyn/vae/main.py` loop marked unused (2.3).
  - New deviation items 6–12 (Part 4).
  - Stale results replaced (8.1).
  - Removed the claim that the paper shows fairness survives DP noise; the paper says DP weakens it (Secs. 6.3, 7.2).
  - Added the paper's baselines (8.3).
  - Corrected the mean-shift limitation (Part 9 item 8).
  - Updated test counts and run commands.
- **ARCHITECTURE.md:**
  - Issue status (#1, #2, #3, #7, #10 fixed; #9 partly fixed).
  - New issues #14–16.
  - Module map and paper mapping.
  - ε table, plus a note that the recorded HMDA results predate these fixes.
- **EXPERIMENTS.md:** ε table, test count, outputs, and known limitations.

### Deliberately not changed

- **The FLIP loop in `vae/main.py`.** It is unused and doesn't run: the bad `ref_mu` unpack (`:323`), the second backward through a freed graph (`:340/361`), `ReduceLROnPlateau(verbose=True)` rejected by torch 2.14 (`:257`), and `utils.get_args()` lacking the FLIP flags. These are documented only, as decided; only its `compute_loss()` is live.
- **`run_flip_demo_tmp.py`:** a stale copy, unused, documented only.
- **Kept paper deviations**, each with its reason in README Part 4:
  - per-row L_S (DP-decomposable)
  - DP-trained diffusion (not post-processing)
  - one σ at γ_max (valid by Prop. 1)
  - a separate DP mechanism for the batch-coupled fairness gradient
  - Eq. 11 ceiling
  - fixed β (an adaptive schedule would be data-dependent)
  - three groups
- **Statistics outside the privacy accounting** (issue #9): the preprocessing statistics in `load_sample` and the full pipeline's latent mean.

### Verification

- **Unit tests:** `pytest tests` → **80 passed**: 71 in `test_flip_fairness.py`, 9 in the new `test_metrics.py`.
  - Updated tests that encoded the old `dim=0` centering, the summed L_S, and post-aggregation numerics.
  - Added tests: CKAᵀ vs Eqs. 5–6, mean-shift sensitivity, ≤ 2-column skip, L_S batch-size invariance, the `recon_num_pre` sum, the one-hot crash regression, the clip floor, target exclusion, BER, NCB/A-NCB.
- **`run_experiment.py --experiment smoke`:** every stage finished, and the demo now writes `runs/smoke/flip_report.html`.
- **Before/after, `no_dp` demo, scored with the same script.** The before CSV is in `runs/baseline_before/`.

  | Measure | Before | After | Real |
  |---|---|---|---|
  | Numeric W1 / categorical TV | 0.38 / 0.44 | 0.19 / 0.18 | — |
  | Numeric std ratio (syn / real) | 0.00–0.11 | 0.07–0.58 | 1 |
  | Overall origination | 0.99 (collapsed) | 0.61 | 0.59 |
  | Approval parity ratio | 1.00 (from collapse) | 0.80 | 0.72 |
  | BER (ideal 0.667) | 0.66 | 0.55 | 0.59 |
  | A-NCB (ideal 1) | 0.97 | 0.13 | 0.006 |

  - **Before:** the output had collapsed, and its near-ideal fairness scores only reflect columns that barely vary.
  - **After:** no collapse, and fidelity is much better. But on the paper's metrics, race is *more* predictable from the other synthetic columns than from the real ones (BER 0.55 vs 0.59; attribute-probe AUC 0.66).
  - So at these settings (λ 2, 60 Phase-2 epochs, no DP), the disentanglement does not yet remove group information.
  - This is a single run, so it doesn't isolate which change drives what. One untested hypothesis: with per-row L_S, the race column is the argmax of near-equal logits that depend on the same latent as the other columns, which would make it predictable from them.

## 2026-10-01 (later): fairness focus, privacy changes reverted

Scope narrowed to the fairness component, with privacy set aside.

### Reverted (privacy-only changes from the entry above)

- **The diffusion latent mean is a plain mean again**, not DP-released (`run_flip_full_pipeline.py`, `run_experiment.predict_eps`). The reparameterized-latent change (paper Sec. 4.1.2) is kept.
- **The report's headline ε is Stage 1's per-row value again** (`generate_html_report`).
- **Docs restored:** the ε tables in ARCHITECTURE.md and EXPERIMENTS.md, issue #9/#10 status, and the README ε row. `--predict-eps` again gives 9.6 / 9.7 for `eps_10`.
- **Kept:** L_S is still averaged rather than summed. It sets how strongly the fairness loss pulls, so it belongs to the fairness component.

### Fixed

- **`fairness_eval.py` crashed when a synthetic category had a single row.**
  - Where: `_clf`, `_fit_predict` in [fairness_eval.py](fairness_eval.py).
  - Why: above 10k rows, sklearn's gradient boosting holds out a stratified validation split by default, and that split raises on a one-row class.
  - Fix: `early_stopping=False`.
  - Test: `test_task_fairness_tolerates_single_row_class`.

### Added

- **`fairness_only` experiment** in `configs/flip.yaml`: `no_dp` with λ 8 and 600 Phase-2 epochs at full LR.
  - Why: an epoch is ceil(smallest group / per-group batch) = 2 steps, so `no_dp`'s 60 Phase-2 epochs are only 120 steps at 0.2× LR.

### End-to-end fairness check (DP off, 12k HMDA rows, seed 42, single runs)

**Each component does its local job:**
- **Balanced sampling:** ≈ 330 rows per group per batch, against population shares of 83 / 12 / 5%.
- **L_S:** race-head softmax ≈ uniform per row (0.013 from uniform) and per batch (0.001).
- **CKAᵀ is optimized.** With default Phase-2 length, latent CKAᵀ is 0.88 / 0.92 / 0.95 / 0.98 at λ = 0 / 2 / 8 / 32. With 600 Phase-2 epochs at λ = 8, latent and decoder CKAᵀ both reach 0.99.

**But race is not removed from the representation.**
- A classifier recovers real race from the encoder latent almost perfectly at every setting.
- BER is 0.002 with all tokens, and 0.01–0.13 with the race token removed (ideal 0.667).
- High CKAᵀ (similar Gram-matrix patterns across groups) does not imply that race can't be decoded.

**What the synthetic data shows depends on the generator:**

| Run | BER (real 0.59, ideal 0.667) | A-NCB (real 0.006) | Approval parity (real 0.72) | Notes |
|---|---|---|---|---|
| demo, `fairness_only` | 0.65 | 0.59 | 0.66 | Numeric task gaps ≈ 0, but target / product-type gaps grew |
| full, `fairness_only`, 10 diffusion epochs | 0.60 | 0.04 | 0.77 | Diffusion undertrained |
| full, `fairness_only`, 1000 diffusion epochs | 0.59 | 0.37 | 0.56 | Good fidelity (W1 0.08, TV 0.10); task gaps close to real's |
| full, same but λ = 0 (control) | 0.49 | 0.92 | 0.97 | **Not usable:** diffusion loss rose (0.65 → 1.10) and no Black rows were generated |

**Reading:**
- The demo's Gaussian prior samples latent dimensions independently. That scrambles the race structure still present in the latent, so the demo looks fair partly by accident.
- The diffusion prior models the latent faithfully and brings the race structure back.
- In the paper's architecture (VAE + diffusion), at these settings, FLIP does not reduce race information in the synthetic data.
- These are single runs with high variance, and the λ = 0 control failed, so a multi-seed comparison is needed before drawing firm conclusions.

### Diagnosis: why CKAᵀ ≈ 1 while race stays decodable (no code changes)

**Setup:** two models trained with the same seed and DP off. One is Phase 1 only; the other is `fairness_only` (λ 8, 600 Phase-2 epochs). Both are inspected on the encoder latent of the 12k real rows.

**1. CKAᵀ is dominated by structure every group shares.**
- With the paper's centering (across activations), the latent's second-moment matrix AᵀA is **92% shared mean**. The small β (1e-3) leaves the latents far from 0.
- 5 of the 80 latent dimensions hold 76–83% of its mass.
- The dimensions that separate races best are among the lowest-variance ones (variance rank 56–76 of 80).
- Group-mean offsets make up 0.04–2% of ‖AᵀA‖ after Phase 2.
- So CKAᵀ, a magnitude-weighted cosine, sits near 1 whatever those dimensions do: 0.993 as is, 0.994 with group means equalized.

**2. CKAᵀ is second-order.**
- After Phase 2, equalizing the group means removes *linear* decodability (BER 0.002 → 0.51).
- Race is still decodable *nonlinearly* (BER 0.004), and CKAᵀ can't see that structure.

**3. Race is an encoder input, and Phase 2 spreads it further.**
- The race token alone decodes race perfectly (BER ≈ 0), as expected: it's the input's embedding.
- Phase 2 made race *more* decodable from the other tokens:

  | Measure | Phase 1 | After Phase 2 |
  |---|---|---|
  | BER, all but race token | 0.057 | 0.010 |
  | BER, CLS token | 0.14 | 0.03 |
  | BER, loan_amount token | 0.37 | 0.08 |

- Causal check: changing only a row's race input, with everything else fixed, now moves the other tokens' latents **4× more** (0.04 → 0.16 of their spread).
- It also moves the decoded numerics more (loan_amount 0.01 → 0.14 std; tract_population 0.02 → 0.24 std).
- The likely mechanism: the encoder knows each row's group, so the cheapest way to make the groups' second-moment patterns match is a group-*specific* correction. That satisfies CKAᵀ while encoding race more deeply.

**4. Consequence for generation.**
- The diffusion prior learns the joint latent, including the race token and its links to the other tokens, so the synthetic data inherits them.
- The demo's independent Gaussian prior breaks those links, which is why the demo looked fair.

**Caveat:** single seed. Per-token BERs come from 3-fold gradient boosting on 12k rows.
