#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRACKLE_DIR="$PROJECT_ROOT/work/cloudy_cooling_tools_history/examples/grackle"
CIALOOP="$PROJECT_ROOT/work/cloudy_cooling_tools_history/CIAOLoop_lines"
PARAM_DIR="$PROJECT_ROOT/vendor/cloudy_cooling_tools/examples/grackle"
STEM="hm2012_native_plus_filtered_ism_cmb_cr_mol_ct_defaultabund_sixline"
RUN_DIR="$GRACKLE_DIR/${STEM}_run"
WORKERS="${1:-11}"

if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "workers must be a positive integer" >&2
  exit 2
fi

COLUMN_PAR="$PARAM_DIR/${STEM}_column_10x10x21.par"
JEANS_PAR="$PARAM_DIR/${STEM}_jeans_10x21.par"
COLUMN_OUTPUT="$GRACKLE_DIR/${STEM}_column_10x10x21_output"
JEANS_OUTPUT="$GRACKLE_DIR/${STEM}_jeans_10x21_output"

for required in "$CIALOOP" "$COLUMN_PAR" "$JEANS_PAR" \
  "$GRACKLE_DIR/HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed"; do
  [[ -e "$required" ]] || { echo "missing required file: $required" >&2; exit 2; }
done

for output_dir in "$COLUMN_OUTPUT" "$JEANS_OUTPUT"; do
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to mix old and new output: $output_dir" >&2
    exit 2
  fi
done

mkdir -p "$RUN_DIR"
cd "$GRACKLE_DIR"

run_table() {
  local parameter_file="$1"
  local log_file="$2"
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu "$CIALOOP" -np "$WORKERS" "$parameter_file"
  else
    "$CIALOOP" -np "$WORKERS" "$parameter_file"
  fi 2>&1 | tee "$log_file"
}

run_table "$COLUMN_PAR" "$RUN_DIR/column.log"
run_table "$JEANS_PAR" "$RUN_DIR/jeans.log"

echo "Native HM2012 + filtered ISM + CMB + CR + molecules + CT tables completed."
