#!/bin/zsh
set -u

example_dir="/Users/tianbaochen/quokka_postprocess_3D/work/cloudy_cooling_tools_history/examples/grackle"
cd "$example_dir"

run_group() {
  local first="$1"
  local second="$2"
  local third="$3"
  local first_log="${first%.par}.log"
  local second_log="${second%.par}.log"
  local third_log="${third%.par}.log"
  local group_status=0

  ../../CIAOLoop_lines -np 4 "$first" > "$first_log" 2>&1 &
  local first_pid=$!
  ../../CIAOLoop_lines -np 4 "$second" > "$second_log" 2>&1 &
  local second_pid=$!
  ../../CIAOLoop_lines -np 3 "$third" > "$third_log" 2>&1 &
  local third_pid=$!

  wait "$first_pid" || group_status=1
  wait "$second_pid" || group_status=1
  wait "$third_pid" || group_status=1
  return "$group_status"
}

run_group \
  hm2012_defaultabund_cii_column_10x10x21.par \
  hm2012_plus_draine_defaultabund_cii_column_10x10x21.par \
  hm2012_plus_draine_cr_defaultabund_cii_column_10x10x21.par || exit 1

run_group \
  hm2012_defaultabund_cii_jeans_10x21.par \
  hm2012_plus_draine_defaultabund_cii_jeans_10x21.par \
  hm2012_plus_draine_cr_defaultabund_cii_jeans_10x21.par
