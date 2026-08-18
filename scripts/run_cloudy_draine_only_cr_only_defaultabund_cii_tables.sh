#!/bin/zsh
set -eu
unsetopt BG_NICE

root="/Users/tianbaochen/quokka_postprocess_3D"
examples="$root/work/cloudy_cooling_tools_history/examples/grackle"
python_bin="/Users/tianbaochen/miniforge3/envs/quokka/bin/python"

cd "$examples"
../../CIAOLoop_lines -rx -np 5 draine_only_defaultabund_cii_column_10x10x21.par > draine_only_defaultabund_cii_column_10x10x21.log 2>&1 &
p1=$!
../../CIAOLoop_lines -rx -np 4 cr_only_defaultabund_cii_column_10x10x21.par > cr_only_defaultabund_cii_column_10x10x21.log 2>&1 &
p2=$!
../../CIAOLoop_lines -rx -np 1 draine_only_defaultabund_cii_jeans_10x21.par > draine_only_defaultabund_cii_jeans_10x21.log 2>&1 &
p3=$!
../../CIAOLoop_lines -rx -np 1 cr_only_defaultabund_cii_jeans_10x21.par > cr_only_defaultabund_cii_jeans_10x21.log 2>&1 &
p4=$!

run_status=0
wait "$p1" || run_status=1
wait "$p2" || run_status=1
wait "$p3" || run_status=1
wait "$p4" || run_status=1
(( run_status == 0 )) || exit "$run_status"

cd "$root"
build_column() {
  "$python_bin" scripts/build_cloudy_cii_column_single_state.py \
    --input-dir "$examples/$1" --parameter-file "$examples/$2" \
    --output "data/$3" --failure-manifest "data/$4" \
    --state-label "$5" \
    --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
    --radiation-field "$6" --cosmic-ray-rate "$7"
}
build_jeans() {
  "$python_bin" scripts/build_cloudy_cii_jeans_table.py \
    --input "$examples/$1" --parameter-file "$examples/$2" \
    --output "data/$3" --failure-report "data/$4" \
    --state-label "$5" \
    --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
    --radiation-field "$6" --cosmic-ray-h0-ionization-rate "$7" \
    --carbon-abundance-log10 nan
}

build_column draine_only_defaultabund_cii_column_10x10x21_output draine_only_defaultabund_cii_column_10x10x21.par \
  cloudy_cii_draine_only_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  cloudy_cii_draine_only_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  draine_only_defaultabund "Draine only; no HM2012" 0
build_column cr_only_defaultabund_cii_column_10x10x21_output cr_only_defaultabund_cii_column_10x10x21.par \
  cloudy_cii_cr2e-17_only_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  cloudy_cii_cr2e-17_only_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  cr_only_defaultabund "No incident radiation; no HM2012" 2e-17
build_jeans draine_only_defaultabund_cii_jeans_10x21_output draine_only_defaultabund_cii_jeans_10x21.par \
  cloudy_cii_draine_only_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  cloudy_cii_draine_only_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  draine_only_defaultabund "Draine only; no HM2012" 0
build_jeans cr_only_defaultabund_cii_jeans_10x21_output cr_only_defaultabund_cii_jeans_10x21.par \
  cloudy_cii_cr2e-17_only_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  cloudy_cii_cr2e-17_only_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  cr_only_defaultabund "No incident radiation; no HM2012" 2e-17
