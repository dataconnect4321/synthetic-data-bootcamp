# Running experiments

This guide shows how to run and add FLIP experiments. Every setting lives in [`configs/flip.yaml`](configs/flip.yaml), and [`run_experiment.py`](run_experiment.py) runs the pipeline from it. The design and known issues are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Setup (one time)

```bash
cd hilbert_schmidt_fairness_syn
python3 -m venv .venv
.venv/bin/pip install torch opacus pytest numpy pandas scipy scikit-learn fairlearn tqdm pyyaml \
    icecream category_encoders tomli tomli-w pynvml
.venv/bin/pip install --no-deps libzero==0.0.8   # NOT `zero`: that PyPI package is unrelated
```

- **Input data:** put the raw HMDA extract at `hmda_bmo_data.csv` in the repo root, or change `paths.raw_data` in the config.
- **Preprocessing:** run it once. Its output, `Data/bmo_nationwide_preprocessed.csv`, is shared by all experiments.

```bash
.venv/bin/python run_experiment.py --stages preprocess
```

## Quick start

```bash
.venv/bin/python run_experiment.py --list                                   # available experiments
.venv/bin/python run_experiment.py --experiment smoke                       # ~1 min check of every stage
.venv/bin/python run_experiment.py --experiment eps_10 --predict-eps        # privacy budget, no training
.venv/bin/python run_experiment.py --experiment eps_10 --stages demo        # fast generator
.venv/bin/python run_experiment.py --experiment eps_10 --stages full        # VAE + diffusion (slow)
```

## Command reference

| Option | Default | Meaning |
|---|---|---|
| `--experiment NAME` | `baseline` | Which entry under `experiments:` to apply |
| `--stages A,B` | all | A subset of `preprocess,audit,report,demo,full`, run in that order |
| `--predict-eps` | — | Print the composed ε this experiment will report, then exit (no training) |
| `--list` | — | List experiments and what each one extends |
| `--config PATH` | `configs/flip.yaml` | Use a different config file |

### Stages

| Stage | What it does | Typical time (this Mac, CPU) |
|---|---|---|
| `preprocess` | Raw HMDA → binary target, drops other actions and rows with missing demographics | seconds |
| `audit` | Real-data fairness metrics printed to the console | seconds |
| `report` | Real-data fairness HTML report | seconds |
| `demo` | FLIP VAE plus Gaussian latent prior → synthetic CSV | minutes |
| `full` | FLIP VAE plus DP diffusion prior → synthetic CSV and HTML report | about 1.3 h per diffusion epoch with DP |

If a stage fails, the runner prints the error and continues with the next stage, then exits non-zero and lists the failed stages.

## Included experiments

Predicted composed ε is at δ = 1e-5. The first value is for the demo (`--stages demo`), the second for the full pipeline (`--stages full`).

| Experiment | What it changes from the base | Predicted ε (demo / full) | Use it for |
|---|---|---|---|
| `baseline` | nothing (repo defaults) | 197 / 261 | the reference point; no meaningful privacy |
| `hmda_2026_09_29` | diffusion epochs 10 | 197 / 194 | reproducing the recorded HMDA run |
| `smoke` | 2 + 2 VAE epochs, 2k rows, 1 diffusion epoch, small network | 28 / 23 | checking the plumbing; results are meaningless |
| `no_dp` | privacy off in both stages | none | the fairness effect alone, fastest |
| `fairness_only` | `no_dp` with λ_fair 8, 600 Phase-2 epochs at full LR | none | **fairness without privacy**: enough Phase-2 steps for the CKA term to act (no_dp's Phase 2 is only 120 steps). The full pipeline shows no fairness gain yet; see CHANGELOG.md |
| `eps_10` | batch 96, noise 2.0, group noise 8, stats noise 10, diffusion noise 2.0, 10 epochs | 9.6 / 9.7 | **recommended starting point** |
| `eps_6` | `eps_10` with noise 3.0 | 6.3 / 6.4 | stronger privacy |
| `eps_10_strong_fairness` | `eps_10` with λ_fair 5.0 | 9.6 / 9.7 | more bias removal at the same ε |
| `eps_10_full_data` | `eps_10` on all ~24.7k rows | 6.5 / 6.5 | more data; also lowers ε |
| `my_experiment` | a template listing every setting (`eps_10` values) | 9.6 / 9.7 | copying to make your own |

### Commands for each included experiment

```bash
# Check ε for any of them first (seconds, no training)
.venv/bin/python run_experiment.py --experiment eps_10 --predict-eps

# Demo generator (minutes)
.venv/bin/python run_experiment.py --experiment baseline               --stages demo
.venv/bin/python run_experiment.py --experiment no_dp                  --stages demo
.venv/bin/python run_experiment.py --experiment fairness_only          --stages demo,full
.venv/bin/python run_experiment.py --experiment eps_10                 --stages demo
.venv/bin/python run_experiment.py --experiment eps_6                  --stages demo
.venv/bin/python run_experiment.py --experiment eps_10_strong_fairness --stages demo
.venv/bin/python run_experiment.py --experiment eps_10_full_data       --stages demo
.venv/bin/python run_experiment.py --experiment my_experiment          --stages demo

# Full pipeline (hours): run in the background with a log
nohup .venv/bin/python run_experiment.py --experiment eps_10 --stages full > runs/eps_10.log 2>&1 &
tail -f runs/eps_10.log

# Everything at once (preprocess, audit, report, demo, full)
.venv/bin/python run_experiment.py --experiment smoke

# Reproduce the recorded HMDA run (~13 h)
nohup .venv/bin/python run_experiment.py --experiment hmda_2026_09_29 --stages full > runs/hmda_2026_09_29.log 2>&1 &
```

### Other useful commands

```bash
# Unit tests (91 should pass)
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest tests -q -p no:cacheprovider

# Is a background run still going? Stop it if needed.
pgrep -fl run_experiment.py
pkill -f "run_experiment.py --experiment eps_10"

# Use a different config file
.venv/bin/python run_experiment.py --config configs/my_flip.yaml --experiment eps_10 --stages demo
```

## Adult (UCI): reproducing the FLIP paper

> **Current Adult work is in `run_adult_flip.py`, not these config experiments.**
> README Part 10 describes the setup, results and commands (`--preset paper`
> by default, `--preset final` recommended); `EXPERIMENTS_ADULT.md` records
> what was tried; `results/` holds the 3-fold tables and heatmaps. The
> `adult*` experiments below (2026-10-01 to 10-04) are kept as a record of the
> first attempts. Their main conclusion, that linear CKAᵀ alone does not make
> the data independent of sex, still holds; the batch-mean L_S they
> introduced is now the only L_S in `run_adult_flip.py`.

These experiments reproduce the paper's headline Adult result (Figure 3 and the FLIP rows of Table 3): λ=4 with no DP and with ε≈3, plus λ=0 as a reference. Put the UCI files in `Data/adult/`; you need `adult.data` and `adult.test`.

| Experiment | What it changes | Predicted ε (demo / full) |
|---|---|---|
| `adult` | Adult data, protected `sex`, all 45,222 rows, λ 0, no DP | none |
| `adult_smoke` | `adult` with λ 4, 2k rows, 2 + 2 epochs; results are meaningless | none |
| `adult_lam4` | `adult` with λ 4 | none |
| `adult_lam4_eps3` | `adult_lam4` with DP: batch 256, noise 3.0, group noise 8, stats noise 10, diffusion noise 2.0 | 2.8 / 2.6 |

```bash
.venv/bin/python preprocess_adult.py          # -> Data/adult_preprocessed.csv
.venv/bin/python run_experiment.py --experiment adult           --stages demo
.venv/bin/python run_experiment.py --experiment adult_lam4      --stages demo,full
.venv/bin/python run_experiment.py --experiment adult_lam4_eps3 --stages demo,full
```

Use `--stages demo` / `full` only. The `preprocess`, `audit` and `report` stages run the HMDA scripts.

Paper values to compare against (Figure 3, Adult). Synthetic AUC, α-precision, β-recall, equalized odds and ε-identifiability are not computed here.

| Metric | Paper λ=0, ε=∞ | Paper λ=4, ε=∞ | Paper λ=4, ε=3 | Where it appears in our output |
|---|---|---|---|---|
| BER (ideal 0.5) | 0.41 | 0.50 | 0.49 | `PAPER FAIRNESS METRICS` → BER (synthetic) |
| A-NCB ↑ | 0.32 | 0.92 | 0.91 | `PAPER FAIRNESS METRICS` → A-NCB (synthetic) |
| Statistical parity ↓ | 0.18 | 0.13 | 0.12 | `Downstream model ... DP diff` (trained on synthetic, evaluated on real) |

Differences from the paper's protocol:
- a single seed, where the paper uses 3-fold cross-validation;
- metrics computed against the training data, not a held-out fold;
- scikit-learn HistGradientBoosting instead of LightGBM;
- 45,222 rows after dropping `?`, against the paper's 43,914, whose filtering is not stated.

### First results (2026-10-01/02, seed 42, single runs)

Real data: target rate 0.248 (Male 0.312, Female 0.114), BER 0.155, A-NCB 0.934.

| Run | Synthetic target rate (M / F) | BER (paper) | A-NCB (paper) | Valid? |
|---|---|---|---|---|
| `adult` demo (λ=0) | 0.279 / no Female rows | NaN (0.41) | NaN (0.32) | **No**: the synthetic `sex` column is 100% Male |
| `adult_lam4` demo | 0.184 / 0.358 | 0.370 (0.50) | 0.464 (0.92) | Partly: the gap reverses direction |
| `adult_lam4` full | 0.207 / 0.282 | **0.437** (0.50) | **0.509** (0.92) | **Yes**: the only usable comparison |
| `adult_lam4_eps3` demo | 0.005 / 0.010; 98.9% Male | 0.500 (0.49) | 1.000 (0.91) | **No**: the target collapsed to ~0 |
| `adult_lam4_eps3` full | 0.004 / 0.004 | 0.487 (0.49) | 0.202 (0.91) | **No**: the target collapsed to ~0 |

Takeaways:
- At λ=4 without DP, the full pipeline moves BER most of the way to 0.5 and narrows the income-rate gap from 0.199 to 0.076. A-NCB, however, falls well short of the paper (0.51 vs 0.92).
- The ε≈3 runs are not a reproduction. The synthetic target is almost all 0, so the parity metrics look perfect only because nobody is positive (ARCHITECTURE.md known issue #12).
- The paper's λ=0 reference still needs a full-pipeline run, because the demo generator drops the Female group.

## Adult: paper setup (3-fold CV, paper metrics)

These experiments follow the paper's training and evaluation as closely as the paper allows. The earlier `adult*` runs above are kept as a record.

| Experiment | λ | DP | Predicted ε (full, per fold) |
|---|---|---|---|
| `adult_paper` | 0 | off | none |
| `adult_paper_lam4` | 4 | off | none |
| `adult_paper_eps3` | 0 | VAE only, batch 4096, σ 64 | 2.9 |
| `adult_paper_lam4_eps3` | 4 | VAE only, batch 4096, σ 64 | 2.9 |
| `adult_paper_smoke` | 4 | off; 5 VAE + 3 diffusion epochs | none (plumbing check) |

```bash
.venv/bin/python preprocess_adult.py                       # once
for k in 0 1 2; do
  .venv/bin/python run_experiment.py --experiment adult_paper --fold $k --stages full,eval
done
.venv/bin/python run_experiment.py --experiment adult_paper --summarize   # mean ± std beside the paper's Figure 3
```

Each fold writes `runs/<experiment>/fold<k>/` with `train.csv`, `test.csv`, the synthetic CSV, the report and `metrics.json`. `eval` also scores a demo CSV if the fold has one.

**Settings:**
- VAE: 3,000 + 1,000 epochs, batch 4,096, d_token 4, factor 32, β decayed from 1e-2 by ×0.7 every 150 epochs down to 1e-5, no Phase-2 LR drop.
- Diffusion: up to 10,000 epochs, batch 4,096, early stop after 500 epochs without improvement.

**Our assumptions,** because the paper doesn't state them: the 3,000 + 1,000 phase split, and a β schedule fixed in advance rather than triggered by the loss.

**Deviations:**
- HistGradientBoosting instead of LightGBM.
- Accounting at the largest group's sample rate instead of per group (Eq. 13), which is more conservative.
- The privacy additions described in ARCHITECTURE.md: the fairness-gradient mechanism and the DP-released statistics. Both are included in ε.
- Like the paper, the diffusion step trains on encoded real rows without DP, so the reported ε covers the VAE only.

**Run time (MPS, measured 2026-10-03 on a fold's ~30k training rows):**
- VAE: 1.7 s/epoch without DP and 2.0 s/epoch with DP, so 4,000 epochs takes about 1.9 h / 2.2 h.
- Diffusion: 1.15 s/epoch, so at most about 3.2 h; less if early stopping triggers.
- Sampling, report and `eval`: about 6 min.

That is at most about 5.5 h per fold run: about 22 h for fold 0 of all four experiments, and about 66 h for every fold. The paper experiments set `device: mps`. CPU diffusion was 2–5× slower and its timings were inconsistent between identical runs.

**Real data scored as "synthetic"** (fold 0; this is the ceiling for synthetic data): AUC 0.922, α-precision 0.997, β-recall 0.499, SP 0.176, EO 0.066, ε-identifiability 0.495, BER 0.161, A-NCB 0.084.

### Gate run: `adult_paper` (λ=0, no DP), fold 0, 2026-10-03

The run took 1 h 52 min on MPS. Diffusion stopped early at epoch 1,757.

| Metric | Ours | Paper |
|---|---|---|
| Synthetic AUC | 0.865 | 0.75 |
| α-precision | 0.806 | 0.90 |
| β-recall | 0.177 | 0.41 |
| SP | 0.024 | 0.18 |
| EO | 0.373 | 0.12 |
| ε-identifiability | 0.085 | 0.21 |
| BER | 0.318 | 0.41 |
| A-NCB | 0.744 | 0.32 |

**The run failed the gate.** The synthetic `sex` column is 99.6% Female (the training fold is 32.5% Female), so SP, EO, BER and A-NCB are not meaningful.

Likely cause: L_S pushes each row's predicted sex probabilities towards 50/50, and decoding takes the argmax of each categorical (as TabSyn does). With every row near 50/50, a tiny shared bias sends almost all rows to the same class. Earlier runs show the same pattern: the first demo was 100% Male, and the ε≈3 demo was 98.9% Male.

Not yet confirmed: the pipeline does not log the decoder's sex probabilities.

**Cause found: our L_S did not follow the paper.** The paper's Eq. 3 makes only the batch-average softmax of S uniform. The code used a per-sample variant, written for DP, that pushes every row to 50/50. The paper experiments now set `ls_variant: batch_mean` (see CHANGELOG.md). The old results are in `runs/adult_paper/old_per_sample_ls_fold0/`.

### Gate rerun with the paper's L_S (`ls_variant: batch_mean`), fold 0, 2026-10-04

The run took 1 h 42 min on MPS. Diffusion stopped early at epoch 1,801. The synthetic sex split is 51.8% Female / 48.2% Male, so the collapse is fixed.

| Metric | Ours | Paper λ=0 | Match |
|---|---|---|---|
| β-recall | 0.397 | 0.41 | ✓ |
| SP | 0.152 | 0.18 | ✓ |
| BER | 0.448 | 0.41 | ✓ |
| α-precision | 0.966 | 0.90 | +0.07 |
| EO | 0.058 | 0.12 | −0.06 |
| ε-identifiability | 0.141 | 0.21 | −0.07 |
| Synthetic AUC | 0.905 | 0.75 | **+0.16** |
| A-NCB | 0.634 | 0.32 | **+0.31** |

Six of the eight metrics are within about 0.07 of the paper. AUC and A-NCB are far off, and both gaps look like differences in how the metric is computed rather than in the generator:
- **AUC:** real data alone scores 0.92-0.93 on Adult with our protocol. The paper's 0.75 for synthetic data at λ=0 implies a weaker classifier or a different AUC protocol.
- **A-NCB:** our implementation scores the *real* test fold at 0.87. The paper's 0.32 for λ=0 synthetic data implies a different clustering setup, for example the number of GMM components or the FAMD details, neither of which the paper states.

Next: λ=4 on fold 0. The change from λ=0 to λ=4 does not depend on these metric offsets.

### λ=4 (`adult_paper_lam4`), fold 0, 2026-10-04: the fairness effect does not appear

| Metric | Ours λ=0 | Ours λ=4 | Paper λ=0 → λ=4 | Same direction? |
|---|---|---|---|---|
| BER (ideal 0.5) | 0.448 | 0.435 | 0.41 → 0.50 | ✗ |
| A-NCB | 0.634 | 0.582 | 0.32 → 0.92 | ✗ |
| SP | 0.152 | 0.240 | 0.18 → 0.13 | ✗ |
| α-precision | 0.966 | 0.969 | 0.90 → 0.64 | ✗ (unchanged) |
| β-recall | 0.397 | 0.354 | 0.41 → 0.27 | ✓ |
| Synthetic AUC | 0.905 | 0.901 | 0.75 → 0.70 | ✓ (barely) |
| ε-identifiability | 0.141 | 0.121 | 0.21 → 0.10 | ✓ |
| EO | 0.058 | 0.323 | 0.12 → 0.15 | ✓ (much larger) |

**Replication fails at the method level.** λ=4 barely changes the data: α-precision and AUC are flat, and the task-agnostic fairness metrics do not improve.

Why: the Phase-2 loss shows CKAᵀ (the mean over the three stages) already at about 0.89 when Phase 2 starts, and only about 0.90 at the end. The objective is close to saturated, so it gives almost no gradient, while BER shows sex is still predictable from the data.

Open questions:
- Our CKAᵀ follows Eqs. 5-7, but the paper leaves details unstated: how the stages are combined (mean or sum), the scale of λ relative to the other losses, and the batch size used for CKA.
- Phase 1 and Phase 2 together take about 1.7 h per attempt, which makes testing slow.

### Phase-2 λ sweep (`phase2_sweep.py`), fold 0, 2026-10-04

Phase 1 was trained once (checkpointed); then Phase 2 ran for each λ. The probes run on the VAE directly. "BER w/o S" predicts sex from the encoder means with sex's own token removed; 0.5 would mean no sex information.

| Run | BER, all tokens | BER w/o S | CKAᵀ latent / detokenizer / decoder |
|---|---|---|---|
| End of Phase 1 | 0.000 | 0.018 | 0.88 / 0.96 / 0.77 |
| λ=0 | 0.000 | 0.023 | 0.88 / 0.96 / 0.78 |
| λ=4 | 0.000 | 0.031 | 0.96 / 0.99 / 0.94 |
| λ=12 | 0.000 | 0.000 | 0.99 / 1.00 / 0.99 |
| λ=32 | 0.000 | 0.000 | 1.00 / 1.00 / 1.00 |

**Conclusion.** The optimizer pushes CKAᵀ to its maximum of 1.0 at every stage, yet sex remains perfectly predictable from the latent, and becomes more predictable at high λ. Maximizing linear CKAᵀ between the groups, as written in Eqs. 5-7 and implemented here, does not make the latent independent of sex. On Adult, the replication fails at the method level.

No public code exists for the paper (searched 2026-10-04), so we cannot tell whether the authors' implementation differs in some way the paper does not describe.

## Adding your own experiment

1. Open `configs/flip.yaml` and add a block under `experiments:`. List **only the settings you change**; everything else keeps the base value.

   ```yaml
   experiments:
     lambda4_batch192:
       stage1:
         fairness: {lambda_fair: 4.0}
         training: {batch_size: 192}
       diffusion: {epochs: 5}
   ```

   To build on an existing experiment, use `extends:`. The overrides apply on top of it, and chains work (`eps_6` extends `eps_10`):

   ```yaml
     eps_10_longer_phase2:
       extends: eps_10
       stage1:
         fairness: {phase2_epochs: 120}
   ```

   To see every available setting in one place, copy the `my_experiment` block and edit it.

2. **Check ε before training.** Epochs, batch size and every noise setting change it:

   ```bash
   .venv/bin/python run_experiment.py --experiment lambda4_batch192 --predict-eps
   ```

3. **Run.** Put long runs in the background with a log:

   ```bash
   .venv/bin/python run_experiment.py --experiment lambda4_batch192 --stages demo
   nohup .venv/bin/python run_experiment.py --experiment lambda4_batch192 --stages full \
       > runs/lambda4_batch192.log 2>&1 &
   tail -f runs/lambda4_batch192.log
   ```

**Rules**
- **Nesting:** keep the base config's structure. Under `stage1:`, a setting goes in its subsection (`fairness`, `privacy`, `fidelity`, `stability`, `training`). Under `diffusion:`, settings go directly.
- **Key names:** they must match the base config. A misspelling stops the run with the list of valid keys.
- **Other sections:** `data:` and `paths:` can be overridden too, for example `data: {sensitive_col: derived_sex, groups: [Male, Female]}`.

## Which settings to change

| Goal | Change | Cost |
|---|---|---|
| More fairness | raise `stage1.fairness.lambda_fair` (e.g. 3–5) or `phase2_epochs` | lower fidelity; more epochs raise ε |
| More privacy (lower ε) | lower `stage1.training.batch_size`; raise `noise_multiplier`, `group_noise_multiplier`, `diffusion.noise_multiplier`; fewer epochs | lower fidelity |
| More fidelity | more `phase1_epochs`, lower `beta`, more `diffusion.epochs`, less noise | higher ε, longer runs |
| Faster runs | smaller `diffusion.dim_t` (e.g. 256), fewer `diffusion.epochs`, or `diffusion.use_dp: false` | lower quality; no DP for Stage 2 if turned off |

Batch size is the biggest privacy lever. Each batch takes the same number of rows from every group, so rows from the smallest group are picked most often. That group (576 rows in the 12k sample) is included in about 59% of steps at batch 1,024, but about 6% at batch 96.

## Outputs

Each run writes to `runs/<experiment>/`:

| File | Contents |
|---|---|
| `resolved_config.yaml` | Every value used, for reproducing the run |
| `flip_synthetic.csv` | Synthetic data from the demo generator |
| `flip_full_synthetic.csv` | Synthetic data from the full pipeline |
| `flip_full_report.html` | Full pipeline report: fidelity, privacy (ε per mechanism and composed), fairness |
| `flip_report.html` | Demo report: fidelity, privacy, fairness, including the paper's BER / A-NCB / per-feature task fairness |
| `fairness_report.html` | Real-data fairness report (from the `report` stage) |

Running the same experiment again overwrites its folder. To keep old results, rename the folder first or use a new experiment name.

## Known limitations when reading results

See ARCHITECTURE.md → Known issues, and `CHANGELOG.md` for what was fixed.

- **Good fairness scores can hide collapse** (issue #12): a generator that approves everyone scores a parity ratio of 1.0, and one whose columns barely vary also scores near-ideal BER, A-NCB and task fairness. Always check the overall origination rate and the fidelity metrics as well.
- **The synthetic protected column is close to random by design** (per-row L_S, README Part 4 item 6), so read BER and A-NCB, which test the *other* columns, rather than the race mix.
- **Every ε above excludes some raw-data statistics** (issue #9). The HTML report's headline ε is the Stage-1 per-row value, not the composed total (issue #10); use the "TOTAL composed" line in the console or log.
