#!/bin/bash
# Repair amount (tau) grid for the leakage explorer's scorecard: refits of
# the AIM hybrid from each fold's saved private measurements (Adult, eps 3),
# so only tau varies and the grid spends no privacy budget. Masking and
# consistency rules stay on. Each refit, and plain AIM from the same
# measurements, is scored with run_adult_flip.py; aim_fair/tau_grid_summary.py
# then writes results/utility_tau_grid.csv (mean over refits in each fold,
# then mean and sd across folds: the scorecard's bands).
#
# Default: run_final.sh's measurements (aimfinal_e3_f0..2), 3 refits per tau,
# so tau 1 and plain AIM are the RESULTS.md releases. A fold without saved
# measurements is measured first (about 1.5 h of CPU).
#
#   bash scripts/run_tau_grid.sh
#   FOLDS=0 REFITS=0 TAG=lk PLAIN=lk_plain bash scripts/run_tau_grid.sh   # one run
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# several JAX processes at once: keep each one's CPU threads small
export XLA_FLAGS=${XLA_FLAGS:-"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"}
PY=${PY:-$(command -v python || command -v python3)}   # main environment
# mbi + JAX environment: the venv's python on macOS / Linux, or on Windows
[ -x aim_fair/.venv/bin/python ] && V=aim_fair/.venv/bin/python || V=aim_fair/.venv/Scripts/python.exe
AIMPY=${AIMPY:-$V}
AIMPY=$(cd "$(dirname "$AIMPY")" && pwd)/$(basename "$AIMPY")   # absolute
TAG=${TAG:-aimfinal}                               # measurements out/TAG_e3_fF
PLAIN=${PLAIN:-${TAG}plain}
FOLDS=${FOLDS:-"0 1 2"}
REFITS=${REFITS:-"0 1 2"}
TAUS=${TAUS:-"0 0.25 0.5 0.75 1"}
JOBS=${JOBS:-4}                                    # processes at once
LAST=${REFITS##* }
L=adult_results/logs; mkdir -p $L

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done; }

# measurements (only for folds that have none), then the refits
for f in $FOLDS; do
  M=out/${TAG}_e3_f${f}_measurements.pkl
  [ -f aim_fair/$M ] && continue
  throttle
  (cd aim_fair && $AIMPY -W ignore run_aim.py --fold $f \
      --epsilon 3 --refits --tag ${TAG}_e3_f$f > ../$L/${TAG}_e3_f$f.log 2>&1
   echo "${TAG}_e3_f$f measured exit $? $(date +%H:%M)") &
done
wait
for f in $FOLDS; do
  M=out/${TAG}_e3_f${f}_measurements.pkl
  for t in $TAUS plain; do
    if [ $t = plain ]; then T=${PLAIN}_e3_f$f; X="--tau 0 --no-mask --no-fix"
    else T=${TAG}_tau${t}_e3_f$f; X="--tau $t"; fi
    [ -f aim_fair/out/${T}_r${LAST}_synthetic.csv ] && continue
    throttle
    (cd aim_fair && $AIMPY -W ignore run_aim.py --fold $f --epsilon 3 \
        --load $M $X --refits $REFITS --tag $T > ../$L/$T.log 2>&1
     echo "$T refits exit $? $(date +%H:%M)") &
  done
done
wait

score() {  # fold csv tag [extra flags]
  [ -f adult_results/$3.json ] || $PY -W ignore run_adult_flip.py --fold $1 \
      --score-csv $2 --score-collapsed --no-save-run --tag $3 "${@:4}" \
      > $L/$3_scored.log 2>&1
  echo "$3 scored exit $? $(date +%H:%M)"
}
for f in $FOLDS; do
  first=1
  for t in $TAUS plain; do
    if [ $t = plain ]; then T=${PLAIN}_e3_f$f; else T=${TAG}_tau${t}_e3_f$f; fi
    # merged levels (MASK) need the same masking of the real data
    mask=$(grep -q '"merged": {}' aim_fair/out/$T.json || echo --mask-proxies)
    [ $t = plain ] && mask=""
    for r in $REFITS; do
      if [ $first = 1 ]; then   # alone: it writes the fold's real-reference cache
        score $f aim_fair/out/${T}_r${r}_synthetic.csv ${T}_r$r $mask; first=0
      else
        throttle; score $f aim_fair/out/${T}_r${r}_synthetic.csv ${T}_r$r $mask &
      fi
    done
  done
done
wait
$PY aim_fair/tau_grid_summary.py --tag $TAG --plain $PLAIN --folds $FOLDS --refits $REFITS
