#!/bin/bash
# Adult lambda grid without DP. For each fold: bank Phase 1, then Phase 2 at
# every lambda, tier-1 first (fast, scored against each row's real person)
# and then the full pipeline (VAE + latent diffusion), at most $JOBS runs at
# a time. Re-running skips finished runs.
#
#   bash scripts/run_paper_grid.sh                             # FLIP as published
#   PRESET=final LAMS="0 4" bash scripts/run_paper_grid.sh     # best VAE process
#   LAMS="0 4" FOLDS="0" TIER1_ONLY=1 bash scripts/run_paper_grid.sh
#
# Results: adult_results/{pgrid|final}_f{fold}_lam{lambda}[_t1].json
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
PRESET=${PRESET:-paper}
LAMS=${LAMS:-"0 2 4 8 12"}
FOLDS=${FOLDS:-"0 1 2"}
JOBS=${JOBS:-3}
case $PRESET in
  paper) P=pgrid ;;
  final) P=final ;;
  *) echo "PRESET must be paper or final"; exit 1 ;;
esac
BANK=raw   # both presets share the paper's Phase 1
L=adult_results/logs
mkdir -p $L adult_results/ckpt

bank() {   # $1 = fold
  f=$1; ck=adult_results/ckpt/${BANK}_f$f.pt
  [ -f $ck ] || $PY run_adult_flip.py --preset paper --fold $f \
      --phase2-epochs 0 --tier1 --save-phase1 $ck --tag ${BANK}_f${f}_bank \
      > $L/${BANK}_f${f}_bank.log 2>&1
  [ -f $ck ] && echo "bank $ck ready" || echo "bank $ck FAILED"
}
run() {    # $1 = tag, $2 = fold, rest = flags
  t=$1; f=$2; ck=adult_results/ckpt/${BANK}_f$f.pt; shift 2
  [ -f adult_results/$t.json ] && return
  $PY run_adult_flip.py --preset $PRESET --fold $f --phase1-checkpoint $ck \
      "$@" --tag $t > $L/$t.log 2>&1
  echo "$t exit $? $(date +%H:%M)"
}
export -f run; export PY PRESET L BANK

for f in $FOLDS; do bank $f & done
wait

jobs_list() {   # one line per run: tag fold flags...
  for t1 in _t1 ""; do
    [ -n "$TIER1_ONLY" ] && [ -z "$t1" ] && continue
    fl=""; [ -n "$t1" ] && fl="--tier1"
    for lam in $LAMS; do for f in $FOLDS; do
      echo "${P}_f${f}_lam${lam}${t1} $f --lam $lam $fl"
    done; done
  done
}
jobs_list | xargs -P $JOBS -L 1 bash -c 'run "$@"' _
echo "GRID DONE $(date +%H:%M)"
