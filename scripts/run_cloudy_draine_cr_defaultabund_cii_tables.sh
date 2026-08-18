#!/bin/zsh
set -eu
unsetopt BG_NICE

root="/Users/tianbaochen/quokka_postprocess_3D"
examples="$root/work/cloudy_cooling_tools_history/examples/grackle"
python_bin="/Users/tianbaochen/miniforge3/envs/quokka/bin/python"
column_par="draine_cr_defaultabund_cii_column_10x10x21.par"
jeans_par="draine_cr_defaultabund_cii_jeans_10x21.par"

cd "$examples"
column_resume=()
jeans_resume=()
[[ -d draine_cr_defaultabund_cii_column_10x10x21_output ]] && column_resume=(-rx)
[[ -d draine_cr_defaultabund_cii_jeans_10x21_output ]] && jeans_resume=(-rx)

../../CIAOLoop_lines "${column_resume[@]}" -np 10 "$column_par" > "${column_par%.par}.log" 2>&1 &
column_pid=$!
../../CIAOLoop_lines "${jeans_resume[@]}" -np 1 "$jeans_par" > "${jeans_par%.par}.log" 2>&1 &
jeans_pid=$!

run_status=0
wait "$column_pid" || run_status=1
wait "$jeans_pid" || run_status=1
(( run_status == 0 )) || exit "$run_status"

cd "$root"
"$python_bin" scripts/build_cloudy_cii_column_single_state.py \
  --input-dir "$examples/draine_cr_defaultabund_cii_column_10x10x21_output" \
  --parameter-file "$examples/$column_par" \
  --output data/cloudy_cii_draine_cr2e-17_defaultabund_column_10x10x21_T3p6_to1e9.npz \
  --failure-manifest data/cloudy_cii_draine_cr2e-17_defaultabund_column_10x10x21_T3p6_to1e9_failures.json \
  --state-label draine_cr_defaultabund \
  --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
  --radiation-field "Draine only; no HM2012" \
  --cosmic-ray-rate 2e-17

"$python_bin" scripts/build_cloudy_cii_jeans_table.py \
  --input "$examples/draine_cr_defaultabund_cii_jeans_10x21_output" \
  --parameter-file "$examples/$jeans_par" \
  --output data/cloudy_cii_draine_cr2e-17_defaultabund_jeans_10x21_T3p6_to1e9.npz \
  --failure-report data/cloudy_cii_draine_cr2e-17_defaultabund_jeans_10x21_T3p6_to1e9_failures.json \
  --state-label draine_cr_defaultabund \
  --composition-label "Cloudy 17.02 default.abn; no element abundance overrides" \
  --radiation-field "Draine only; no HM2012" \
  --cosmic-ray-h0-ionization-rate 2e-17 \
  --carbon-abundance-log10 nan
