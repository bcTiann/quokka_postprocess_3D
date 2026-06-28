#!/bin/bash
# Standardised pipeline for "table plot with sim-cell overlay + MP4 sweep".
# 4 steps per (table, L_ext):
#   1.  Ensure log_samples_3d_L<LEXT>.npy exists.  If not, run pipeline
#       TemperatureSlicesTask to build field caches at that L_ext, then run
#       build_log_samples_3d.py to dump the (log nH, log NH, log dVdr) array.
#   2.  Copy log_samples_3d_L<LEXT>.npy → log_samples_3d.npy (view_table
#       reads this canonical name).
#   3.  Run quokka2s.tables.view_table --all --table <PATH> -o
#       output/table_plots/<TABLE_TAG>_L<LEXT>/  → 35 dVdr × 10 PNG.
#   4.  Stitch 10 fields × 35 dVdr frames into 10 MP4s under
#       output/table_plots/sweeps/<TABLE_TAG>_L<LEXT>/.
#
# Usage:
#   run_table_overlay_sweep.sh <TABLE_PATH> <TABLE_TAG> <LEXT1> [<LEXT2> ...]
#
# Example:
#   run_table_overlay_sweep.sh \
#     /Users/baochen/quokka_postprocessing/output_tables_3D_NL99_GC_LVG/despotic_table.npz \
#     NL99_GC 0 15

set +e
# Portable roots (see run_dataset_series.sh): repo from script location;
# interpreter = local macOS yt-env, falling back to PATH `python` if absent.
ROOT="${QUOKKA_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOGS=$ROOT/logs/v4_pipeline_runs
PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/yt-env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python)"

TABLE_PATH=$1
TABLE_TAG=$2
shift 2
LEXTS=("$@")

# MASS_MODE=1 → mass-weighted contours; output dirs get "_mass" suffix.
MASS_MODE=${MASS_MODE:-0}
if [ "$MASS_MODE" = "1" ]; then
  SAMPLE_SUFFIX="_4d"
  BUILD_FLAGS="--include-mass"
  OUT_SUFFIX="_mass"
else
  SAMPLE_SUFFIX="_3d"
  BUILD_FLAGS=""
  OUT_SUFFIX=""
fi

if [ -z "$TABLE_PATH" ] || [ -z "$TABLE_TAG" ] || [ ${#LEXTS[@]} -eq 0 ]; then
  echo "Usage: $0 <TABLE_PATH> <TABLE_TAG> <LEXT1> [<LEXT2> ...]"
  echo "  Set MASS_MODE=1 to weight contours by cell mass (output goes to *_mass/ dirs)."
  exit 1
fi

MASTER=$LOGS/MASTER_overlay_sweep_${TABLE_TAG}${OUT_SUFFIX}.log
> $MASTER
echo "[$(date)] === overlay sweep: table=$TABLE_PATH  tag=$TABLE_TAG${OUT_SUFFIX}  L_ext=${LEXTS[*]}  MASS_MODE=$MASS_MODE ===" >> $MASTER

FIELDS=(tg_final species_CO_abundance species_CO_lumPerH \
        species_C_abundance species_C_lumPerH \
        species_C+_abundance species_C+_lumPerH \
        species_HCO+_abundance species_HCO+_lumPerH \
        species_e-_abundance)

for LEXT in "${LEXTS[@]}"; do
  SAMPLES=$ROOT/log_samples${SAMPLE_SUFFIX}_L${LEXT}.npy
  echo "[$(date)] [L=$LEXT] step 1: ensure samples exist" >> $MASTER
  if [ ! -f "$SAMPLES" ]; then
    echo "  building cache + samples at L=$LEXT" >> $MASTER
    cd $ROOT/src
    LEXT_KPC=$LEXT RUN_TAG=v4 $PY -u -m quokka2s.pipeline.tasks.run_pipeline \
      --task TemperatureSlicesTask >> $LOGS/overlay_sweep_${TABLE_TAG}${OUT_SUFFIX}_L${LEXT}_cache.log 2>&1
    cd $ROOT
    $PY -u scripts/build_log_samples_3d.py --out $SAMPLES $BUILD_FLAGS \
      >> $LOGS/overlay_sweep_${TABLE_TAG}${OUT_SUFFIX}_L${LEXT}_samples.log 2>&1
  else
    echo "  samples already exist at $SAMPLES" >> $MASTER
  fi
  cp $SAMPLES $ROOT/log_samples_3d.npy   # canonical name view_table reads
  echo "[$(date)] [L=$LEXT] step 1 done" >> $MASTER

  echo "[$(date)] [L=$LEXT] step 2: view_table --all" >> $MASTER
  OUT_TP=$ROOT/output/table_plots/${TABLE_TAG}_L${LEXT}${OUT_SUFFIX}
  mkdir -p $ROOT/output/table_plots
  rm -rf $OUT_TP $ROOT/src/TablePlots_${TABLE_TAG}_LVG_with_sim_overlay_L${LEXT}${OUT_SUFFIX}
  cd $ROOT/src
  $PY -u -m quokka2s.tables.view_table --all \
    --table $TABLE_PATH \
    -o $OUT_TP \
    > $LOGS/overlay_sweep_${TABLE_TAG}${OUT_SUFFIX}_L${LEXT}_viewtable.log 2>&1
  echo "[$(date)] [L=$LEXT] view_table RC=$?  ($(find $OUT_TP -name '*.png' | wc -l) PNG)" >> $MASTER

  # Regenerate /tmp/all_dvdr.txt from THIS table (it has the dVdr values
  # the view_table dirs are named after).
  $PY <<PYEOF > /tmp/all_dvdr.txt 2>/dev/null
import numpy as np
tbl = np.load("$TABLE_PATH", allow_pickle=True)
fm = tbl['failure_mask']; Tg = tbl['tg_final']
samples = np.load("$ROOT/log_samples_3d.npy")
dV_v = tbl['dVdr_values']; nH_v = tbl['nH_values']; NH_v = tbl['col_density_values']
def _log_edges(v):
    lv = np.log10(v); d = np.diff(lv); e = np.empty(v.size + 1)
    e[1:-1] = lv[:-1] + d/2; e[0] = lv[0] - d[0]/2; e[-1] = lv[-1] + d[-1]/2
    return e
nH_e=_log_edges(nH_v); NH_e=_log_edges(NH_v); dV_e=_log_edges(dV_v)
for k in range(len(dV_v)):
    sel = (samples[:,2] >= dV_e[k]) & (samples[:,2] < dV_e[k+1])
    n_sim = int(sel.sum())
    if n_sim == 0:
        print(f"{k:02d}\\t{dV_v[k]:.2e}\\t0\\t0"); continue
    s = samples[sel]
    counts, _, _ = np.histogram2d(s[:,0], s[:,1], bins=[nH_e, NH_e])
    T_slice = np.where(~fm[:,:,k], Tg[:,:,k], np.nan)
    n_in_T = int(counts.astype(int)[(T_slice>=100)&(T_slice<=200)].sum())
    print(f"{k:02d}\\t{dV_v[k]:.2e}\\t{n_sim}\\t{n_in_T}")
PYEOF

  echo "[$(date)] [L=$LEXT] step 3: build 10 MP4s" >> $MASTER
  OUT_MP4=$ROOT/output/table_plots/sweeps/${TABLE_TAG}_L${LEXT}${OUT_SUFFIX}
  rm -rf $OUT_MP4 && mkdir -p $OUT_MP4
  for FIELD in "${FIELDS[@]}"; do
    STAGE=$(mktemp -d)
    while IFS=$'\t' read -r idx dvdr n_sim n_in_T; do
      if [ -f "$OUT_TP/dVdr_${dvdr}/${FIELD}.png" ]; then
        cp "$OUT_TP/dVdr_${dvdr}/${FIELD}.png" "$(printf '%s/%02d.png' "$STAGE" $((10#$idx)))"
      fi
    done < /tmp/all_dvdr.txt
    out_name=$(echo "$FIELD" | tr '+' 'p' | tr '-' 'm')
    ffmpeg -y -framerate 2 -i "$STAGE/%02d.png" \
      -c:v libx264 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
      -crf 18 -preset slow "$OUT_MP4/sweep_${out_name}_${TABLE_TAG}_L${LEXT}${OUT_SUFFIX}.mp4" 2>/dev/null
    rm -rf $STAGE
  done
  echo "[$(date)] [L=$LEXT] step 3 done  ($(ls $OUT_MP4/*.mp4 | wc -l) MP4)" >> $MASTER
done
echo "[$(date)] === ALL DONE ===" >> $MASTER
