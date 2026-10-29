#!/bin/bash
# wti_benchmark.py: per seed, AIM measures ACS-I Utah's training split once
# (acs_ut_wti, epsilon 3); a plain refit and an AIM hybrid refit come from
# the same measurements; then the "Where to Intervene?" evaluation, our
# scorer and FLIP protocol on the same seed splits, and the combined table.
#
#   bash aim_fair/run_wti_benchmark.sh
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export XLA_FLAGS=${XLA_FLAGS:-"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=3"}
PY=${PY:-python}
AIMPY=${AIMPY:-aim_fair/.venv/Scripts/python.exe}
AIMPY=$(cd "$(dirname "$AIMPY")" && pwd)/$(basename "$AIMPY")
FAIRPY=${FAIRPY:-aim_fair/.venv-aif360/Scripts/python.exe}
SEEDS=${SEEDS:-"0 1 2 3 4"}
D=acs_ut_wti
L=adult_results/logs; mkdir -p $L
[ -f aim_fair/data/${D}_fold0_train.csv ] || $PY aim_fair/wti_benchmark.py export
for s in $SEEDS; do
  (cd aim_fair
   T=${D}_aim_e3_s$s; M=out/${T}_measurements.pkl
   [ -f $M ] || $AIMPY -W ignore run_aim.py --dataset $D --fold $s --epsilon 3 \
       --refits --tag $T > ../$L/$T.log 2>&1
   [ -f out/${T}_r0_synthetic.csv ] || $AIMPY -W ignore run_aim.py --dataset $D \
       --fold $s --epsilon 3 --load $M --refits 0 --tag $T >> ../$L/$T.log 2>&1
   P=${D}_aimplain_e3_s$s
   [ -f out/${P}_r0_synthetic.csv ] || $AIMPY -W ignore run_aim.py --dataset $D \
       --fold $s --epsilon 3 --load $M --refits 0 --tau 0 --no-mask --no-fix \
       --tag $P > ../$L/$P.log 2>&1) &
done
wait
$FAIRPY -W ignore aim_fair/wti_benchmark.py evaluate
score() {  # seed csv tag [flags]: our scorer on the seed's train / test split
  [ -f adult_results/$3.json ] || $PY -W ignore run_adult_flip.py --dataset $D       --fold $1 --split-csvs aim_fair/data/${D}_fold$1_train.csv       aim_fair/data/${D}_fold$1_test.csv --score-csv $2 --score-collapsed       --no-save-run --tag $3 "${@:4}" > $L/$3_scored.log 2>&1
}
for s in $SEEDS; do
  ( # the real training split first, alone: it writes the reference cache
    score $s aim_fair/data/${D}_fold${s}_train.csv ${D}_real_s$s
    for t in aim aimplain; do
      mask=""
      [ $t = aim ] && ! grep -q '"merged": {}' aim_fair/out/${D}_aim_e3_s$s.json           && mask=--mask-proxies
      score $s aim_fair/out/${D}_${t}_e3_s${s}_r0_synthetic.csv           ${D}_${t}_e3_s${s}_r0 $mask &
    done
    wait ) &
done
wait
$FAIRPY -W ignore aim_fair/flip_protocol.py --dataset $D > $L/flip_protocol_$D.out 2>&1
$PY aim_fair/wti_benchmark.py combine
