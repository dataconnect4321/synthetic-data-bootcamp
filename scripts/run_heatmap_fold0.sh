#!/bin/bash
# Runs for the paper's Fig. 3 layout on Adult (paper_heatmap_compare.py):
# FLIP as published, fold 0, full pipeline, lambda 0/2/4/8/12 x epsilon
# inf/3/2/1. DP: Eq. 13 per-group noise, paper-style coupling, VAE only
# (nominal epsilon; sigmas from calibrate_dp.py --eq13 --coupled paper).
# Every cell is decoded twice from the same model: argmax sex (TabSyn) and
# sampled sex (--s-decode sample). Tier-1 runs give the true-sex BER panel.
# One run at a time; re-running skips finished runs.
#
#   bash scripts/run_heatmap_fold0.sh && python paper_heatmap_compare.py
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=${PY:-python}
L=adult_results/logs; C=adult_results/ckpt; mkdir -p $L $C
dp() {
  case "$1" in
    3) echo "--dp-coupled paper --dp-group-sigmas 44.6096,21.3794";;
    2) echo "--dp-coupled paper --dp-group-sigmas 65.6506,31.4526";;
    1) echo "--dp-coupled paper --dp-group-sigmas 128.7097,61.6466";;
  esac
}
run() {   # tag, flags...
  t=$1; shift
  [ -f adult_results/$t.json ] || $PY run_adult_flip.py --preset paper --fold 0 \
      --score-collapsed "$@" --tag $t > $L/$t.log 2>&1
  echo "$t exit $? $(date +%H:%M)"
}

# Phase-1 banks: no DP and one per epsilon
[ -f $C/raw_f0.pt ] || run raw_f0_bank --tier1 --phase2-epochs 0 --save-phase1 $C/raw_f0.pt
for e in 3 2 1; do
  [ -f $C/e13_e${e}_f0.pt ] || run e13_e${e}_f0_bank --tier1 --phase2-epochs 0 \
      $(dp $e) --save-phase1 $C/e13_e${e}_f0.pt
done

for lam in 0 2 4 8 12; do
  # epsilon = inf
  run hmr_einf_lam$lam --lam $lam --phase1-checkpoint $C/raw_f0.pt
  run pgrid_f0_lam${lam}_t1 --lam $lam --tier1 --phase1-checkpoint $C/raw_f0.pt
  for dec in argmax sample; do
    run hmr_einf_lam${lam}_$dec --from-run hmr_einf_lam$lam --s-decode $dec
  done
  # epsilon = 3 / 2 / 1
  for e in 3 2 1; do
    run hm_e${e}_lam$lam --lam $lam $(dp $e) --phase1-checkpoint $C/e13_e${e}_f0.pt
    run e13_paper_e${e}_lam${lam}_t1 --lam $lam --tier1 $(dp $e) \
        --phase1-checkpoint $C/e13_e${e}_f0.pt
    for dec in argmax sample; do
      run hm_e${e}_lam${lam}_$dec --from-run hm_e${e}_lam$lam --s-decode $dec
    done
  done
done
echo "HEATMAP RUNS DONE $(date +%H:%M)"
