#!/bin/bash
# Fresh hybrid and plain AIM refits of Adult from saved measurements (the
# measuring stage is unchanged), for analyses that need the release tables
# (flip_protocol.py: 3 hybrid + 3 plain refits per fold, tags aimre and
# aimreplain, from run_final.sh's saved measurements):
#   bash aim_fair/run_adult_refits.sh
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export XLA_FLAGS=${XLA_FLAGS:-"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"}
PY=${PY:-$(command -v python || command -v python3)}   # main environment
# mbi + JAX environment: the venv's python on macOS / Linux, or on Windows
[ -x aim_fair/.venv/bin/python ] && V=aim_fair/.venv/bin/python || V=aim_fair/.venv/Scripts/python.exe
AIMPY=${AIMPY:-$V}
AIMPY=$(cd "$(dirname "$AIMPY")" && pwd)/$(basename "$AIMPY")
MEAS=${MEAS:-aimfinal}; T=${T:-aimre}
FOLDS=${FOLDS:-"0 1 2"}; R=${R:-"0 1 2"}
L=adult_results/logs; mkdir -p $L
for f in $FOLDS; do
  M=out/${MEAS}_e3_f${f}_measurements.pkl
  (cd aim_fair; [ -f out/${T}_e3_f${f}_r2_synthetic.csv ] || $AIMPY -W ignore \
       run_aim.py --fold $f --epsilon 3 --load $M --refits $R --tag ${T}_e3_f$f \
       > ../$L/${T}_e3_f$f.log 2>&1) &
  (cd aim_fair; [ -f out/${T}plain_e3_f${f}_r2_synthetic.csv ] || $AIMPY -W ignore \
       run_aim.py --fold $f --epsilon 3 --load $M --refits $R --tau 0 --no-mask \
       --no-fix --tag ${T}plain_e3_f$f > ../$L/${T}plain_e3_f$f.log 2>&1) &
done
wait
for f in $FOLDS; do
  for t in $T ${T}plain; do
    mask=$(grep -q '"merged": {}' aim_fair/out/${t}_e3_f$f.json \
           || [ $t = ${T}plain ] || echo --mask-proxies)
    for r in $R; do
      tag=${t}_e3_f${f}_r$r
      [ -f adult_results/$tag.json ] || $PY -W ignore run_adult_flip.py --fold $f \
          --score-csv aim_fair/out/${tag}_synthetic.csv --score-collapsed $mask \
          --no-save-run --tag $tag > $L/${tag}_scored.log 2>&1 &
      # r0 alone first: it writes the fold's real-reference cache
      [ $r = 0 ] && wait
    done
  done
  wait
done
echo REFITS_SCORED
