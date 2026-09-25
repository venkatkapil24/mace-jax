#!/usr/bin/env bash
set -euo pipefail

root="$HOME/scratch/MACE-POLAR-JAX"
source "$root/gpu-env.sh"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
cd "$root/mace-jax"

initial="$root/Examples/HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz"
bundle="$root/artifacts/MACE-POLAR-1-M-jax-fixed.msgpack"
output="$root/Examples/HydrogenMaleate-Mechanical/md-100ps"
charges="$root/Examples/HydrogenMaleate-Mechanical/fixed-qm-charges.json"
mkdir -p "$output"

if [[ ! -s "$charges" ]]; then
  taskset -c 8-15 python \
    examples/HydrogenMaleate-MACE-POLAR/prepare_mechanical_charges.py \
    --initial "$initial" --output "$charges" \
    --qm-water-count 13 \
    > "$root/Examples/HydrogenMaleate-Mechanical/prepare-charges.log" 2>&1
fi

taskset -c 8-15 python examples/HydrogenMaleate-MACE-POLAR/md.py \
  --mode mechanical --initial "$initial" --bundle "$bundle" \
  --qm-water-count 13 --mechanical-qm-charges "$charges" \
  --restraint-radius 4.2 --restraint-k 0.2 --output "$output" \
  --steps 200000 --timestep-fs 0.5 --temperature-k 330 --seed 20260922

python examples/HydrogenMaleate-MACE-POLAR/proton_histogram.py \
  --examples "$root/Examples" \
  --output "$root/Examples/HydrogenMaleate-Analysis/proton-histogram-three-way.png" \
  > "$root/Examples/HydrogenMaleate-Analysis/proton-histogram-three-way.log" 2>&1
