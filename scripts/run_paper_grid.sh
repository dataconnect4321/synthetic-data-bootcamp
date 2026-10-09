#!/bin/bash
# Adult lambda grid (README Part 10). For each fold: bank Phase 1, then
# Phase 2 at every lambda, tier-1 first (fast, scored against each row's real
# person) and then the full pipeline (VAE + latent diffusion), at most
# $JOBS runs at a time. Re-running skips finished runs.
#
#   bash scripts/run_paper_grid.sh                 # FLIP as published
#   PRESET=ours bash scripts/run_paper_grid.sh     # this repository's additions
#   REPAIR=1 bash scripts/run_paper_grid.sh        # + scrambling-repair baseline
#   LAMS="0 4" FOLDS="0" TIER1_ONLY=1 bash scripts/run_paper_grid.sh
#
# Results: adult_results/{prefix}_f{fold}_lam{lambda}[_t1].json, prefix pgrid
# (paper) or flip (ours); tables / heatmaps: paper_grid_report.py +
# paper_grid_heatmap.py, or final_table.py / make_heatmaps.py.
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
PRESET=${PRESET:-paper}
LAMS=${LAMS:-"0 2 4 8 12"}
FOLDS=${FOLDS:-"0 1 2"}
JOBS=${JOBS:-3}
case $PRESET in
  paper) P=pgrid; BANK=raw ;;   # unmasked banks
  ours)  P=flip;  BANK=fin ;;   # masked banks (the preset masks proxies)
  *) echo "PRESET must be paper or ours"; exit 1 ;;
esac
L=adult_results/logs
mkdir -p $L adult_results/ckpt

bank() {   # $1 = fold, $2 = bank name, $3.. = extra flags
  f=$1; ck=adult_results/ckpt/$2_f$f.pt; shift 2
  [ -f $ck ] || $PY run_adult_flip.py --preset $PRESET "$@" --fold $f \
      --phase2-epochs 0 --tier1 --save-phase1 $ck --tag $(basename $ck .pt)_bank \
      > $L/$(basename $ck .pt)_bank.log 2>&1
  [ -f $ck ] && echo "bank $ck ready" || echo "bank $ck FAILED"
}
run() {    # $1 = tag, $2 = fold, $3 = bank, rest = flags
  t=$1; f=$2; ck=adult_results/ckpt/$3_f$f.pt; shift 3
  [ -f adult_results/$t.json ] && return
  $PY run_adult_flip.py --preset $PRESET --fold $f --phase1-checkpoint $ck \
      "$@" --tag $t > $L/$t.log 2>&1
  echo "$t exit $? $(date +%H:%M)"
}
export -f run; export PY PRESET L

for f in $FOLDS; do bank $f $BANK & done
[ -n "$REPAIR" ] && for f in $FOLDS; do bank $f ${BANK}rpb --repair-tau 1.0 & done
wait

jobs_list() {   # one line per run: tag fold bank flags...
  for t1 in _t1 ""; do
    [ -n "$TIER1_ONLY" ] && [ -z "$t1" ] && continue
    fl=""; [ -n "$t1" ] && fl="--tier1"
    for lam in $LAMS; do for f in $FOLDS; do
      echo "${P}_f${f}_lam${lam}${t1} $f $BANK --lam $lam $fl"
    done; done
    [ -n "$REPAIR" ] && for f in $FOLDS; do
      echo "rpb_f${f}_lam0${t1} $f ${BANK}rpb --lam 0 --repair-tau 1.0 $fl"
    done
  done
}
jobs_list | xargs -P $JOBS -L 1 bash -c 'run "$@"' _
echo "GRID DONE $(date +%H:%M)"
