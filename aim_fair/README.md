# aim_fair: the AIM hybrid

Differentially private synthetic Adult data from AIM / Private-PGM:

- Numeric bins are learned privately from 5% of the budget (`dp_schema.py`).
- Fairness (proxy repair) and realism (proxy masking, consistency rules, pair
  redraw) are applied to AIM's noisy measurements, as post-processing.

The epsilon covers the whole release, bins included.

- Architecture diagram: `ARCHITECTURE.md`.
- What it does and why: `WRITEUP.md`.
- Head-to-head with the best VAE method: `FINAL_COMPARISON.md`.

## Files

| File | Purpose |
|---|---|
| `vendor/` | AIM reference code from https://github.com/ryan112358/mbi (Apache 2.0, see `vendor/NOTICE`) |
| `fair_aim.py` | `FairAIM`: AIM's measuring loop, then masking, repair, rules and pair redraw |
| `dp_schema.py` | private numeric bins: noisy histograms (5% of rho), spike bins, ordered edges, within-bin decoding |
| `adult_domain.py` | public bin edges, encoding and decoding (with or without the private schema) |
| `check_schema.py` | how well public vs private bins reproduce the real numeric values |
| `export_folds.py` | writes the same 3-fold splits as `run_adult_flip.py` (main environment) |
| `run_aim.py` | measure one fold (private bins by default, `--public-bins` for the fixed edges) and write the refits |
| `run_final.sh` | final setting on 3 folds x 10 refits plus the plain-AIM baseline, scored |
| `summarize.py` | AIM-vs-VAE table |
| `final_chart.py` | final chart and table vs the FLIP reproduction and the paper (`RESULTS.md`) |
| `test_aim_fair.py` | self-tests |

## Setup

mbi needs Python >= 3.11 and JAX. A separate environment keeps the torch
environment intact.

```bash
C:/Python313/python.exe -m venv aim_fair/.venv
aim_fair/.venv/Scripts/python -m pip install "git+https://github.com/ryan112358/mbi.git" pandas scikit-learn
aim_fair/.venv/Scripts/python aim_fair/test_aim_fair.py
bash aim_fair/run_final.sh
```

If pip's git clone fails with an SSL error on Windows, prefix the install
with `GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.sslBackend
GIT_CONFIG_VALUE_0=schannel`.

Scoring goes through `run_adult_flip.py --score-csv` (main environment), so
AIM tables get exactly the metrics the VAE runs get. AIM rows are not
paired with real people, so true-sex BER does not apply. The leakage
measures are `tstr_sex_ber` (a classifier trained on synthetic data, tested
on real people) and BER on the synthetic label.
