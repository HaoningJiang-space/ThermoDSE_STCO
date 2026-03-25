#!/usr/bin/env bash

# Remove results from previous simulations
rm -f *.init
rm -f outputs/*

# Create outputs directory if it doesn't exist
mkdir outputs

../../HotSpot/hotspot -c test.config -p test.ptrace -f floorplan2.flp -grid_layer_file test.lcf -materials_file test.materials -model_type grid -detailed_3D on -steady_file outputs/test.steady -grid_steady_file outputs/test.grid.steady

# # Copy steady-state results over to initial temperatures
# cp outputs/test.steady test.init

# # Transient simulation
# ../../HotSpot/hotspot -c test.config -p test.ptrace  -f floorplan2.flp -grid_layer_file test.lcf -materials_file test.materials -model_type grid -detailed_3D on -o outputs/test.ttrace -grid_transient_file outputs/test.grid.ttrace

# Visualize Heat Map of Layer 0 with Perl and with Python script
# python ../../HotSpot7/scripts/split_grid_steady.py outputs/test.grid.steady 6 64 64
python ../../HotSpot7/scripts/grid_thermal_map.py floorplan2.flp outputs/test.grid.steady.layer_2 64 64 outputs/core_thermal.png
../../HotSpot7/scripts/grid_thermal_map.pl floorplan2.flp outputs/test.grid.steady.layer_2  64 64 > outputs/core_thermal.svg

python ../../HotSpot7/scripts/grid_thermal_map.py floorplan1.flp outputs/test.grid.steady.layer_0  64 64 outputs/cache_thermal.png
../../HotSpot7/scripts/grid_thermal_map.pl floorplan1.flp outputs/test.grid.steady.layer_0  64 64 > outputs/cache_thermal.svg