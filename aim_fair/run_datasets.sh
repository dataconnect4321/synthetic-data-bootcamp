#!/bin/bash
# The AIM hybrid's final setting, unchanged, on the FLIP paper's other
# datasets (datasets.py: ACS-I Alabama and Utah, Dutch census; sex
# protected), epsilon 3, 3 folds: measure once, 3 hybrid refits and 3 plain
# AIM refits from the same measurements, then score them and the real
# training fold (the real-data reference) with run_adult_flip.py.
#
#   bash aim_fair/run_datasets.sh                  # all three datasets
#   DATASETS="dutch" FOLDS="0" bash aim_fair/run_datasets.sh
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
# several JAX processes at once: keep each one's CPU threads small
export XLA_FLAGS=${XLA_FLAGS:-"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"}
PY=${PY:-$(command -v python || command -v python3)}   # main environment
# mbi + JAX environment: the venv's python on macOS / Linux, or on Windows
[ -x aim_fair/.venv/bin/python ] && V=aim_fair/.venv/bin/python || V=aim_fair/.venv/Scripts/python.exe
AIMPY=${AIMPY:-$V}
AIMPY=$(cd "$(dirname "$AIMPY")" && pwd)/$(basename "$AIMPY")   # absolute
DATASETS=${DATASETS:-"acs_al acs_ut dutch"}
FOLDS=${FOLDS:-"0 1 2"}
R="0 1 2"
L=adult_results/logs; mkdir -p $L

for d in $DATASETS; do
  [ -f aim_fair/data/${d}_fold0_train.csv ] || $PY aim_fair/export_folds.py --dataset $d
  for f in $FOLDS; do
    (cd aim_fair
     T=${d}_aim_e3_f$f; M=out/${T}_measurements.pkl
     [ -f $M ] || $AIMPY -W ignore run_aim.py --dataset $d --fold $f \
         --epsilon 3 --refits --tag $T > ../$L/$T.log 2>&1
     [ -f out/${T}_r2_synthetic.csv ] || $AIMPY -W ignore run_aim.py \
         --dataset $d --fold $f --epsilon 3 --load $M --refits $R --tag $T \
         >> ../$L/$T.log 2>&1
     P=${d}_aimplain_e3_f$f
     [ -f out/${P}_r2_synthetic.csv ] || $AIMPY -W ignore run_aim.py \
         --dataset $d --fold $f --epsilon 3 --load $M --refits $R --tau 0 \
         --no-mask --no-fix --tag $P > ../$L/$P.log 2>&1) &
  done
done
wait

score() {  # dataset fold csv tag [extra flags]
  [ -f adult_results/$4.json ] || $PY -W ignore run_adult_flip.py --dataset $1 \
      --fold $2 --score-csv $3 --score-collapsed --no-save-run --tag $4 "${@:5}" \
      > $L/$4_scored.log 2>&1
}
for d in $DATASETS; do
  for f in $FOLDS; do
    # the real training fold first, alone: it writes the fold's reference cache
    score $d $f aim_fair/data/${d}_fold${f}_train.csv ${d}_real_f$f
    for t in aim aimplain; do
      for r in $R; do
        tag=${d}_${t}_e3_f${f}_r$r
        # merged levels (MASK) need the same masking of the real data
        mask=$(grep -q '"merged": {}' aim_fair/out/${d}_${t}_e3_f$f.json \
               || [ $t = aimplain ] || echo --mask-proxies)
        score $d $f aim_fair/out/${tag}_synthetic.csv $tag $mask &
      done
    done
    wait
  done
done
$PY aim_fair/datasets_summary.py
