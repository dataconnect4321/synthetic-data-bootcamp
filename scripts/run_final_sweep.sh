#!/bin/bash
# Final Adult sweep (see README Part 10). For each fold: bank Phase 1 on the
# masked data, then FLIP Phase 2 at lambda 0/2/4/8 (tier-1, then the full
# pipeline); plus the scrambling-repair baseline (tau=1, lambda=0).
# Folds run in parallel. Results: adult_results/{flip,rpb}_f*_lam*.json
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
C="--paper-config --ls-mode batch_mean --mask-proxies --decoded-joint --marginal-weight 1.0 --stage-reduce sum --s-decode sample"
L=adult_results/logs
mkdir -p $L adult_results/ckpt

flip_chain() {
  f=$1; ck=adult_results/ckpt/fin_f$f.pt
  [ -f $ck ] || $PY run_adult_flip.py $C --fold $f --phase2-epochs 0 --tier1 \
      --save-phase1 $ck --tag fin_f${f}_bank > $L/fin_f${f}_bank.log 2>&1
  [ -f $ck ] || { echo "fold $f: no bank"; return; }
  for lam in 0 2 4 8; do
    $PY run_adult_flip.py $C --fold $f --lam $lam --tier1 --phase1-checkpoint $ck \
        --tag flip_f${f}_lam${lam}_t1 > $L/flip_f${f}_lam${lam}_t1.log 2>&1
    echo "flip fold $f lam $lam tier1 exit $?"
  done
  for lam in 0 2 4 8; do
    $PY run_adult_flip.py $C --fold $f --lam $lam --phase1-checkpoint $ck \
        --tag flip_f${f}_lam${lam} > $L/flip_f${f}_lam${lam}.log 2>&1
    echo "flip fold $f lam $lam full exit $?"
  done
}

base_chain() {
  f=$1; ck=adult_results/ckpt/rpb_f$f.pt
  [ -f $ck ] || $PY run_adult_flip.py $C --fold $f --repair-tau 1.0 \
      --phase2-epochs 0 --tier1 --save-phase1 $ck --tag rpb_f${f}_bank \
      > $L/rpb_f${f}_bank.log 2>&1
  [ -f $ck ] || { echo "repair fold $f: no bank"; return; }
  $PY run_adult_flip.py $C --fold $f --repair-tau 1.0 --lam 0 --tier1 \
      --phase1-checkpoint $ck --tag rpb_f${f}_lam0_t1 > $L/rpb_f${f}_lam0_t1.log 2>&1
  echo "repair fold $f tier1 exit $?"
  $PY run_adult_flip.py $C --fold $f --repair-tau 1.0 --lam 0 \
      --phase1-checkpoint $ck --tag rpb_f${f}_lam0 > $L/rpb_f${f}_lam0.log 2>&1
  echo "repair fold $f full exit $?"
}

for f in 0 1 2; do flip_chain $f & base_chain $f & done
wait
echo ALL DONE
