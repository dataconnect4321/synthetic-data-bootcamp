#!/bin/bash
# Best VAE process (FLIP + output-level joint stage, --preset final, lambda 4)
# under DP: paper-style Eq. 13 per-group noise on the VAE (nominal epsilon 3;
# sigmas from `python calibrate_dp.py --eq13 --coupled paper --eps 3 --fold 0`,
# which reserves 10% of epsilon for an optional DP rule release). Banks a DP Phase 1
# per fold, then runs the full pipeline (diffusion 500 epochs) at lambda 0
# and 4. The no-DP version is:
#   PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh
#
#   bash scripts/run_best_vae_dp.sh       # FOLDS="0 1 2" for all folds
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
FOLDS=${FOLDS:-"0"}
DP="--dp-coupled paper --dp-group-sigmas 44.6096,21.3794"
L=adult_results/logs; C=adult_results/ckpt; mkdir -p $L $C

for f in $FOLDS; do
  ck=$C/dp_e3_f$f.pt
  [ -f $ck ] || $PY run_adult_flip.py --preset paper --fold $f --tier1 \
      --phase2-epochs 0 $DP --save-phase1 $ck --tag dp_e3_f${f}_bank \
      --score-collapsed > $L/dp_e3_f${f}_bank.log 2>&1
  for lam in 0 4; do
    t=best_vae_e3_f${f}_lam$lam
    [ -f adult_results/$t.json ] || $PY run_adult_flip.py --preset final \
        --lam $lam --fold $f --diff-epochs 500 --phase1-checkpoint $ck $DP \
        --score-collapsed --tag $t > $L/$t.log 2>&1
    echo "$t exit $? $(date +%H:%M)"
  done
done
