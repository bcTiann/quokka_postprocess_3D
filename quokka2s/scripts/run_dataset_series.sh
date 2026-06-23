#!/bin/bash
# Driver for the quokka2s pipeline that runs each task (or a small batch) in its
# OWN process, so memory is fully released between tasks.  This avoids the
# out-of-memory crashes that happen when all ~13 tasks run in a single process
# at down=1 on a 16 GB Mac (the spectra task alone peaks ~13.5 GB).
#
# It ALSO loops over a series of plt snapshots (forward-looking; the default is
# just the one active dataset).  The dataset is selected via the YT_DATASET env
# var (see prep/config.py); OUTPUT_DIR and the field-cache root derive from it,
# so each dataset stays in its own directories — nothing collides.
#
# Tasks run in dependency order.  Memory-heavy tasks (VelocityPhase, spectra,
# slices) each get a fresh process; light tasks share one process.
#
# Usage:
#   scripts/run_dataset_series.sh [plt0655228 plt0857000 ...]   # datasets under $ROOT
#
# Env:
#   LEXT_KPC  (default 15)   RUN_TAG  (default v4)
#   MODE      (all | compute | plot;  default all)
#     all     = each task computes + stores result + plots          (one pass)
#     compute = each task computes + stores result only, NO figures (do the
#               heavy physics once, then iterate figures with MODE=plot)
#     plot    = re-render figures from the stored results (fast, light)
#
# Examples:
#   scripts/run_dataset_series.sh                        # one dataset, compute+plot
#   MODE=compute scripts/run_dataset_series.sh           # heavy physics only
#   MODE=plot    scripts/run_dataset_series.sh           # re-plot from store
#   scripts/run_dataset_series.sh plt0655228 plt0857000  # a 2-dataset series

set +e
ROOT=/Users/baochen/quokka_postprocessing
PY=/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python
LOGS=$ROOT/logs/dataset_series
mkdir -p "$LOGS"

LEXT_KPC=${LEXT_KPC:-15}
RUN_TAG=${RUN_TAG:-v4}
MODE=${MODE:-all}

DEFAULT_DATASETS=(plt0655228)
if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Task groups, in dependency order.  Format: "tag|--task A [--task B]".
# Each group runs as ONE separate `run_pipeline` process; the OS reclaims all
# its memory before the next group starts.  Heavy tasks run alone; the light
# PhaseHist tasks share a process, as do the two plot-only aggregators.
#   velocity  → produces sigma_v + velocity PDFs (read by spectra + aggregate)
#   spectra   → needs velocity's result; heaviest task (~13.5 GB) → solo
#   phasehist → 7 PhaseHist + NH-rho (light; feed PhaseCombinedPlot)
#   slices    → MultiFieldSlices (reads many 3D fields → solo)
#   aggregate → PhaseSpectrumOverlay + PhaseCombinedPlot (plot-only; run last)
TASK_GROUPS=(
  "velocity|--task VelocityPhaseTask"
  "spectra|--task SpeciesSpectrumTask"
  "phasehist|--task PhaseHistTask --task PhaseHistNHRhoTask"
  "slices|--task MultiFieldSlicesTask"
  "aggregate|--task PhaseSpectrumOverlayTask --task PhaseCombinedPlotTask"
)

MASTER=$LOGS/MASTER_dataset_series.log
> "$MASTER"
echo "[$(date)] === datasets: ${DATASETS[*]}  L_ext=$LEXT_KPC  tag=$RUN_TAG  MODE=$MODE ===" | tee -a "$MASTER"

for D in "${DATASETS[@]}"; do
  DPATH=$ROOT/$D
  if [ ! -e "$DPATH" ]; then
    echo "[$(date)] [$D] SKIP — not found at $DPATH" | tee -a "$MASTER"
    continue
  fi
  echo "[$(date)] [$D] === start (MODE=$MODE) ===" | tee -a "$MASTER"
  for entry in "${TASK_GROUPS[@]}"; do
    tag=${entry%%|*}
    args=${entry#*|}
    LOG=$LOGS/${D}_Lext${LEXT_KPC}_${RUN_TAG}_${MODE}_${tag}.log
    echo "[$(date)] [$D] $tag  ($args)  →  $LOG" | tee -a "$MASTER"
    cd "$ROOT/quokka2s/src" || exit 1
    # $args is intentionally unquoted so it splits into separate --task tokens.
    YT_DATASET=$DPATH LEXT_KPC=$LEXT_KPC RUN_TAG=$RUN_TAG \
      $PY -u -m quokka2s.pipeline.tasks.run_pipeline --mode "$MODE" $args > "$LOG" 2>&1
    echo "[$(date)] [$D] $tag  RC=$?" | tee -a "$MASTER"
  done
  echo "[$(date)] [$D] === done ===" | tee -a "$MASTER"
done

echo "[$(date)] === ALL DONE ===" | tee -a "$MASTER"
