#!/bin/zsh
set -eu

root="/Users/tianbaochen/quokka_postprocess_3D"
examples="$root/work/cloudy_cooling_tools_history/examples/grackle"
python_bin="/Users/tianbaochen/miniforge3/envs/quokka/bin/python"
export MPLCONFIGDIR="/private/tmp/matplotlib-defaultabund-codex"
mkdir -p "$MPLCONFIGDIR"
cd "$root"

logs=(
  "$examples/hm2012_defaultabund_cii_column_10x10x21.log"
  "$examples/hm2012_plus_draine_defaultabund_cii_column_10x10x21.log"
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_column_10x10x21.log"
  "$examples/hm2012_defaultabund_cii_jeans_10x21.log"
  "$examples/hm2012_plus_draine_defaultabund_cii_jeans_10x21.log"
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_jeans_10x21.log"
)

while true; do
  complete=1
  for log in "${logs[@]}"; do
    if [[ ! -f "$log" ]] || ! grep -q "Run completed successfully" "$log"; then
      complete=0
      break
    fi
  done
  (( complete == 1 )) && break
  sleep 60
done

build_column() {
  "$python_bin" scripts/build_cloudy_cii_column_single_state.py \
    --input-dir "$1" --parameter-file "$2" --output "$3" \
    --failure-manifest "$4" --state-label "$5" \
    --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
    --radiation-field "$6" --cosmic-ray-rate "$7"
}

build_column \
  "$examples/hm2012_defaultabund_cii_column_10x10x21_output" \
  "$examples/hm2012_defaultabund_cii_column_10x10x21.par" \
  data/cloudy_cii_hm2012_z0_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_z0_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  hm2012_defaultabund "HM2012 z=0" 0

build_column \
  "$examples/hm2012_plus_draine_defaultabund_cii_column_10x10x21_output" \
  "$examples/hm2012_plus_draine_defaultabund_cii_column_10x10x21.par" \
  data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  hm2012_draine_defaultabund "HM2012 z=0 plus Draine" 0

build_column \
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_column_10x10x21_output" \
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_column_10x10x21.par" \
  data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  hm2012_draine_cr_defaultabund "HM2012 z=0 plus Draine" 2e-17

build_jeans() {
  "$python_bin" scripts/build_cloudy_cii_jeans_table.py \
    --input "$1" --parameter-file "$2" --output "$3" \
    --failure-report "$4" --state-label "$5" \
    --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
    --radiation-field "$6" --cosmic-ray-h0-ionization-rate "$7" \
    --carbon-abundance-log10 nan
}

build_jeans \
  "$examples/hm2012_defaultabund_cii_jeans_10x21_output" \
  "$examples/hm2012_defaultabund_cii_jeans_10x21.par" \
  data/cloudy_cii_hm2012_z0_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_z0_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  hm2012_defaultabund "HM2012 z=0" 0

build_jeans \
  "$examples/hm2012_plus_draine_defaultabund_cii_jeans_10x21_output" \
  "$examples/hm2012_plus_draine_defaultabund_cii_jeans_10x21.par" \
  data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  hm2012_draine_defaultabund "HM2012 z=0 plus Draine" 0

build_jeans \
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_jeans_10x21_output" \
  "$examples/hm2012_plus_draine_cr_defaultabund_cii_jeans_10x21.par" \
  data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  hm2012_draine_cr_defaultabund "HM2012 z=0 plus Draine" 2e-17

"$python_bin" scripts/plot_cii_defaultabund_radiation_cr_comparisons.py \
  --column-hm2012 data/cloudy_cii_hm2012_z0_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  --column-hm2012-draine data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  --column-hm2012-draine-cr data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  --jeans-hm2012 data/cloudy_cii_hm2012_z0_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  --jeans-hm2012-draine data/cloudy_cii_hm2012_plus_draine_z0_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  --jeans-hm2012-draine-cr data/cloudy_cii_hm2012_plus_draine_z0_cr2e-17_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  --workers 11
