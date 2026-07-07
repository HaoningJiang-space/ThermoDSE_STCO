#!/usr/bin/env bash
set -euo pipefail

config_file=${1:?missing HotSpot config path}
flp_file=${2:?missing floorplan path}
ptrace_file=${3:?missing ptrace path}
interposer_side=${4:-0.02}
sim_path=${5:-$(pwd)}
thermal_side=$(awk -v side="$interposer_side" 'BEGIN { printf "%.6f", side + 0.001 }')

hotspot_dir=${HOTSPOT_PATH:-/data/ziheng/Open3DBench/OpenROAD-3D/flow/HotSpot}
hotspot_bin="$hotspot_dir/hotspot"

mkdir -p "$sim_path/outputs"
rm -f "$sim_path"/outputs/gcc*

"$hotspot_bin" \
  -c "$config_file" \
  -p "$ptrace_file" \
  -grid_layer_file "$sim_path/floorplan/example.lcf" \
  -materials_file "$sim_path/example.materials" \
  -model_type grid \
  -detailed_3D on \
  -s_sink "$thermal_side" \
  -s_spreader "$thermal_side" \
  -steady_file "$sim_path/outputs/gcc.steady" \
  -grid_steady_file "$sim_path/outputs/gcc.grid.steady"
