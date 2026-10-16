#!/bin/bash
# AIM hybrid, final setting, epsilon 3, 3 folds: AIM measuring (saved), then
# 3 refits per fold with masking, repair (tau 1), tested consistency rules
# and pair redraw; plain AIM refits from the SAME measurements as the
# baseline. Scores every table with run_adult_flip.py --score-csv.
# About 1.5 h of CPU per fold for the measuring, minutes per refit.
#
#   bash aim_fair/run_final.sh            # FOLDS="0" for one fold
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}                                   # main environment
AIMPY=${AIMPY:-aim_fair/.venv/Scripts/python.exe}  # mbi + JAX environment
FOLDS=${FOLDS:-"0 1 2"}
L=adult_results/logs; mkdir -p $L

[ -f aim_fair/data/fold0_train.csv ] || $PY aim_fair/export_folds.py
for f in $FOLDS; do
  (cd aim_fair
   [ -f out/aim_e3_f${f}_r2_synthetic.csv ] || ../$AIMPY -W ignore run_aim.py \
       --fold $f --epsilon 3 --tag aim_e3_f$f > ../$L/aim_e3_f$f.log 2>&1
   [ -f out/aimplain_e3_f${f}_r2_synthetic.csv ] || ../$AIMPY -W ignore run_aim.py \
       --fold $f --epsilon 3 --load out/aim_e3_f${f}_measurements.pkl \
       --tau 0 --no-mask --no-fix --tag aimplain_e3_f$f \
       > ../$L/aimplain_e3_f$f.log 2>&1) &
done
wait
for f in $FOLDS; do for r in 0 1 2; do
  for t in aim aimplain; do
    tag=${t}_e3_f${f}_r$r; mask=""; [ $t = aim ] && mask=--mask-proxies
    # r0 first, alone: it writes the fold's real-reference cache
    [ -f adult_results/$tag.json ] || $PY run_adult_flip.py --fold $f \
        --score-csv aim_fair/out/${tag}_synthetic.csv --score-collapsed $mask \
        --no-save-run --tag $tag > $L/${tag}_scored.log 2>&1 &
    [ $r = 0 ] && wait
  done
done; wait; done
$PY aim_fair/summarize.py
