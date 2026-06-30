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
#   LEXT_KPC  (default 15)   RUN_TAG  (default: none)
#   PYTHON    (default: `python` from the active PATH/conda environment)
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
# Portable roots: repo derived from the script location; interpreter defaults to
# `python` from the active PATH (including an activated conda environment).
# Override with QUOKKA_ROOT= / PYTHON=.
ROOT="${QUOKKA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python}"
PY_RESOLVED="$(command -v "$PY")"
if [ -z "$PY_RESOLVED" ]; then
  echo "ERROR: Python interpreter not found: $PY" >&2
  exit 127
fi
PY="$PY_RESOLVED"
export MPLBACKEND="${MPLBACKEND:-Agg}"   # headless-safe on compute nodes
LOGS=$ROOT/logs/dataset_series
mkdir -p "$LOGS"

LEXT_KPC=${LEXT_KPC:-15}
RUN_TAG=${RUN_TAG:-}
MODE=${MODE:-all}

DEFAULT_DATASETS=(plt0655228)
if [ "$#" -gt 0 ]; then
  DATASETS=("$@")
else
  DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Task groups, in dependency order.  Format: "tag|--task A [--task B]".
# Each group runs as ONE separate `run_pipeline` process; the OS reclaims all
# its memory before the next group starts.  Each group pairs a Build_ (compute)
# with its Plot_ where one exists; under --mode compute only the Build_ runs,
# under --mode plot only the Plot_, under --mode all both (Build before Plot).
#   velocity  → Build_VelocityPhase (sigma_v + PDFs, read by spectra+aggregate) + its plot
#   spectra   → Build_SpeciesSpectrum (heaviest ~13.5 GB; needs velocity) + its plot
#   phasehist → Build_PhaseHist ×7 + Build_PhaseHistNHRho (light; feed Plot_PhaseCombined)
#   slices    → Build_MultiFieldSlices (reads many 3D fields) + its plot
#   aggregate → Plot_PhaseCombined + Plot_PhaseSpectrumOverlay (plot-only; run last)
TASK_GROUPS=(
  "velocity|--task Build_VelocityPhase --task Plot_VelocityPhase"
  "spectra|--task Build_SpeciesSpectrum --task Plot_SpeciesSpectrum"
  "phasehist|--task Build_PhaseHist --task Build_PhaseHistNHRho"
  "slices|--task Build_MultiFieldSlices --task Plot_MultiFieldSlices"
  "aggregate|--task Plot_PhaseCombined --task Plot_PhaseSpectrumOverlay"
)

MASTER=$LOGS/MASTER_dataset_series.log
> "$MASTER"
echo "[$(date)] === datasets: ${DATASETS[*]}  L_ext=$LEXT_KPC  tag=$RUN_TAG  MODE=$MODE  python=$PY ===" | tee -a "$MASTER"

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
    cd "$ROOT/src" || exit 1
    # $args is intentionally unquoted so it splits into separate --task tokens.
    YT_DATASET=$DPATH LEXT_KPC=$LEXT_KPC RUN_TAG=$RUN_TAG \
      $PY -u -m quokka2s.pipeline.tasks.run_pipeline --mode "$MODE" $args > "$LOG" 2>&1
    echo "[$(date)] [$D] $tag  RC=$?" | tee -a "$MASTER"
  done
  echo "[$(date)] [$D] === done ===" | tee -a "$MASTER"
done

echo "[$(date)] === ALL DONE ===" | tee -a "$MASTER"
