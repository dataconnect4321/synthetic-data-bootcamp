#!/bin/bash
# AIM hybrid, final setting, epsilon 3, 3 folds: private numeric bins (5% of
# rho) + AIM measuring (saved), then 10 refits per fold from the saved
# measurements (masking, repair tau 1, tested consistency rules, pair redraw)
# and plain AIM refits from the SAME measurements as the baseline. Scores
# every table with run_adult_flip.py --score-csv, then summarizes.
# About 1.5 h of CPU per fold for the measuring, minutes per refit.
#
#   bash aim_fair/run_final.sh                 # FOLDS="0" for one fold
#   PUBLIC_BINS=1 bash aim_fair/run_final.sh   # fixed public bins instead
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}                                   # main environment
AIMPY=${AIMPY:-aim_fair/.venv/Scripts/python.exe}  # mbi + JAX environment
FOLDS=${FOLDS:-"0 1 2"}
R="0 1 2 3 4 5 6 7 8 9"
A=aimfinal; X=""
[ -n "$PUBLIC_BINS" ] && { A=aimpublic; X=--public-bins; }
L=adult_results/logs; mkdir -p $L

[ -f aim_fair/data/fold0_train.csv ] || $PY aim_fair/export_folds.py
for f in $FOLDS; do
  (cd aim_fair
   M=out/${A}_e3_f${f}_measurements.pkl
   [ -f $M ] || ../$AIMPY -W ignore run_aim.py --fold $f --epsilon 3 $X \
       --refits --tag ${A}_e3_f$f > ../$L/${A}_e3_f$f.log 2>&1
   [ -f out/${A}_e3_f${f}_r9_synthetic.csv ] || ../$AIMPY -W ignore run_aim.py \
       --fold $f --epsilon 3 --load $M --refits $R --tag ${A}_e3_f$f \
       >> ../$L/${A}_e3_f$f.log 2>&1
   [ -f out/${A}plain_e3_f${f}_r9_synthetic.csv ] || ../$AIMPY -W ignore run_aim.py \
       --fold $f --epsilon 3 --load $M --refits $R --tau 0 --no-mask --no-fix \
       --tag ${A}plain_e3_f$f > ../$L/${A}plain_e3_f$f.log 2>&1) &
done
wait
for f in $FOLDS; do
  for t in $A ${A}plain; do
    mask=""; [ $t = $A ] && mask=--mask-proxies
    for r in $R; do
      tag=${t}_e3_f${f}_r$r
      [ -f adult_results/$tag.json ] || $PY run_adult_flip.py --fold $f \
          --score-csv aim_fair/out/${tag}_synthetic.csv --score-collapsed $mask \
          --no-save-run --tag $tag > $L/${tag}_scored.log 2>&1 &
      # r0 alone first: it writes the fold's real-reference cache
      { [ $r = 0 ] || [ $((r % 5)) = 4 ]; } && wait
    done
  done
done
wait
$PY aim_fair/final_chart.py
