#!/bin/bash
# Repair amount (tau) grid for the leakage explorer's scorecard: refits of
# the AIM hybrid from ONE set of saved private measurements (Adult, fold 0,
# eps 3), so only tau varies and no privacy budget is spent. Masking and
# consistency rules stay on. Each refit, and plain AIM from the same
# measurements, is scored with run_adult_flip.py; aim_fair/tau_grid_summary.py
# then writes results/utility_tau_grid.csv.
#
#   bash scripts/run_tau_grid.sh
#   M=out/aimfinal_e3_f0_measurements.pkl TAG=aimfinal bash scripts/run_tau_grid.sh
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# several JAX processes at once: keep each one's CPU threads small
export XLA_FLAGS=${XLA_FLAGS:-"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"}
PY=${PY:-python}                                   # main environment
AIMPY=${AIMPY:-aim_fair/.venv/Scripts/python.exe}  # mbi + JAX environment
AIMPY=$(cd "$(dirname "$AIMPY")" && pwd)/$(basename "$AIMPY")   # absolute
M=${M:-out/lk_e3_f0_measurements.pkl}              # relative to aim_fair/
TAG=${TAG:-lk}
TAUS=${TAUS:-"0 0.25 0.5 0.75 1"}
L=adult_results/logs; mkdir -p $L

for t in $TAUS; do
  (cd aim_fair
   T=${TAG}_tau${t}_e3_f0
   [ -f out/${T}_r0_synthetic.csv ] || $AIMPY -W ignore run_aim.py --fold 0 \
       --epsilon 3 --load $M --tau $t --refits 0 --tag $T > ../$L/$T.log 2>&1
   echo "$T refit exit $? $(date +%H:%M)") &
done
wait

score() {  # csv tag [extra flags]
  [ -f adult_results/$2.json ] || $PY -W ignore run_adult_flip.py --fold 0 \
      --score-csv $1 --score-collapsed --no-save-run --tag $2 "${@:3}" \
      > $L/$2_scored.log 2>&1
  echo "$2 scored exit $? $(date +%H:%M)"
}
# merged levels (MASK) need the same masking of the real data
for t in $TAUS; do
  T=${TAG}_tau${t}_e3_f0
  mask=$(grep -q '"merged": {}' aim_fair/out/$T.json || echo --mask-proxies)
  score aim_fair/out/${T}_r0_synthetic.csv ${T}_r0 $mask
done
score aim_fair/out/${TAG}_plain_e3_f0_r0_synthetic.csv ${TAG}_plain_e3_f0_r0
$PY aim_fair/tau_grid_summary.py --tag $TAG
