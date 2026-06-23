#!/bin/bash
# Two-pass driver for a SERIES of plt snapshots through the quokka2s pipeline.
#
#   Pass 1:  every dataset  --mode compute   (heavy physics → task_intermediates/)
#   Pass 2:  every dataset  --mode plot      (render figures from the stored dicts)
#
# Each dataset runs as a separate OS process, so there is no in-process state
# bleed and no memory accumulation across the series (safe on the 16 GB Mac).
# The dataset is chosen via the YT_DATASET env var (see prep/config.py);
# `_DATASET_BASENAME` and `OUTPUT_DIR` derive from it, so each dataset lands in
# its own output dir and its own field-cache root — nothing collides.  The SAME
# env (YT_DATASET / LEXT_KPC / RUN_TAG) is used in both passes, so pass 2's
# --mode plot reloads exactly the task_intermediates pass 1 wrote.
#
# Usage:
#   run_dataset_series.sh [plt0655228 plt0857000 ...]   # basenames under $ROOT
#   (no args → the DEFAULT_DATASETS list below)
#
# Env overrides:
#   LEXT_KPC  (default 15)              RUN_TAG  (default v4)
#   MODE      (compute | plot | all | both;  default both)
#
# Examples:
#   scripts/run_dataset_series.sh
#   LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh plt0655228 plt0857000
#   MODE=plot scripts/run_dataset_series.sh             # re-plot all from store only

set +e
ROOT=/Users/baochen/quokka_postprocessing
PY=/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python
LOGS=$ROOT/logs/dataset_series
mkdir -p "$LOGS"

LEXT_KPC=${LEXT_KPC:-15}
RUN_TAG=${RUN_TAG:-v4}
MODE=${MODE:-both}

DEFAULT_DATASETS=(plt0655228 plt0857000 plt263168)
if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

MASTER=$LOGS/MASTER_dataset_series.log
> "$MASTER"
echo "[$(date)] === dataset series: ${DATASETS[*]}  L_ext=$LEXT_KPC  tag=$RUN_TAG  MODE=$MODE ===" | tee -a "$MASTER"

run_pass () {
  PASS_MODE=$1
  echo "[$(date)] --- PASS: --mode $PASS_MODE ---" | tee -a "$MASTER"
  for D in "${DATASETS[@]}"; do
    DPATH=$ROOT/$D
    if [ ! -e "$DPATH" ]; then
      echo "[$(date)] [$D] SKIP — not found at $DPATH" | tee -a "$MASTER"
      continue
    fi
    LOG=$LOGS/${D}_Lext${LEXT_KPC}_${RUN_TAG}_${PASS_MODE}.log
    echo "[$(date)] [$D] --mode $PASS_MODE  →  $LOG" | tee -a "$MASTER"
    cd "$ROOT/quokka2s/src" || exit 1
    YT_DATASET=$DPATH LEXT_KPC=$LEXT_KPC RUN_TAG=$RUN_TAG \
      $PY -u -m quokka2s.pipeline.tasks.run_pipeline --mode "$PASS_MODE" > "$LOG" 2>&1
    echo "[$(date)] [$D] --mode $PASS_MODE  RC=$?" | tee -a "$MASTER"
  done
}

case "$MODE" in
  both)              run_pass compute; run_pass plot ;;
  compute|plot|all)  run_pass "$MODE" ;;
  *) echo "unknown MODE=$MODE (use compute|plot|all|both)"; exit 1 ;;
esac

echo "[$(date)] === ALL DONE ===" | tee -a "$MASTER"
