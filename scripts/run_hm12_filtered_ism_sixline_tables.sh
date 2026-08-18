#!/usr/bin/env bash

# Build both Cloudy-native HM2012 + foreground-filtered ISM six-line tables.
#
# In one command this script:
#   1. uses the requested number of parallel CIAOLoop workers;
#   2. prevents macOS from sleeping while Cloudy is running, when `caffeinate`
#      is available;
#   3. builds the explicit-column table (n_H, N_H, T) first;
#   4. builds the Jeans-length table (n_H, T) after the column table finishes;
#   5. saves the terminal output from each table to a separate log; and
#   6. refuses to mix a new run with files in an existing non-empty output
#      directory.
#
# Usage from the repository root:
#
#   ./scripts/run_hm12_filtered_ism_sixline_tables.sh 11
#
# Here, `11` is the maximum number of parallel CIAOLoop workers.  If this
# argument is omitted, the script also defaults to 11 workers.

# Exit on a failed command (-e), an undefined variable (-u), or a failure in
# any command within a pipeline (-o pipefail).
set -euo pipefail

# Resolve every path from the location of this script, so the command works
# even when it is launched from a different current directory.
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GRACKLE_DIR="$PROJECT_ROOT/work/cloudy_cooling_tools_history/examples/grackle"
CIALOOP="$PROJECT_ROOT/work/cloudy_cooling_tools_history/CIAOLoop_lines"
PARAM_DIR="$PROJECT_ROOT/vendor/cloudy_cooling_tools/examples/grackle"
RUN_DIR="$GRACKLE_DIR/hm2012_native_plus_filtered_ism_sixline_run"

# The first command-line argument is the worker count; use 11 when absent.
WORKERS="${1:-11}"

# The column model has independent n_H, N_H, and T axes.  The Jeans model has
# n_H and T axes and obtains its stopping length from the Jeans prescription.
COLUMN_PAR="$PARAM_DIR/hm2012_native_plus_filtered_ism_defaultabund_sixline_column_10x10x21.par"
JEANS_PAR="$PARAM_DIR/hm2012_native_plus_filtered_ism_defaultabund_sixline_jeans_10x21.par"
COLUMN_OUTPUT="$GRACKLE_DIR/hm2012_native_plus_filtered_ism_defaultabund_sixline_column_10x10x21_output"
JEANS_OUTPUT="$GRACKLE_DIR/hm2012_native_plus_filtered_ism_defaultabund_sixline_jeans_10x21_output"

# Reject zero, negative, fractional, or non-numeric worker counts.
if [[ ! "$WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "workers must be a positive integer" >&2
  exit 2
fi

# Fail before starting if CIAOLoop, either parameter file, or the combined
# incident SED is missing.
for required in "$CIALOOP" "$COLUMN_PAR" "$JEANS_PAR" \
  "$GRACKLE_DIR/HM12_NATIVE_ISM_NH21/z_0.0000e+00.sed"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

# Never append a new calculation to a non-empty scientific output directory.
# This protects against silently mixing files from different configurations.
for output_dir in "$COLUMN_OUTPUT" "$JEANS_OUTPUT"; do
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "refusing to mix old and new output: $output_dir" >&2
    exit 2
  fi
done

# RUN_DIR stores the console logs, while the scientific tables are written to
# COLUMN_OUTPUT and JEANS_OUTPUT by the corresponding parameter files.
mkdir -p "$RUN_DIR"

# CIAOLoop is run here because the custom SED path in the parameter files is
# relative to the Grackle example directory.
cd "$GRACKLE_DIR"

# Run one CIAOLoop parameter file and preserve its complete stdout and stderr.
run_table() {
  local parameter_file="$1"
  local log_file="$2"

  # `caffeinate -dimsu` keeps the Mac awake until this CIAOLoop process exits.
  # `-np` sets the maximum number of parallel CIAOLoop workers.
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -dimsu "$CIALOOP" -np "$WORKERS" "$parameter_file"
  else
    "$CIALOOP" -np "$WORKERS" "$parameter_file"
  fi 2>&1 | tee "$log_file"  # Show output in the terminal and save it.
}

# These calls are sequential: the Jeans table starts only after the full
# explicit-column table has returned successfully.
run_table "$COLUMN_PAR" "$RUN_DIR/column.log"
run_table "$JEANS_PAR" "$RUN_DIR/jeans.log"

echo "All Cloudy-native HM2012 + filtered ISM six-line tables completed."
