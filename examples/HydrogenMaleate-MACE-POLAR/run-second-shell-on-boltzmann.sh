#!/usr/bin/env bash
# Separate detached second-shell ML/MM trajectory on otherwise unused CPU cores.
set -euo pipefail

root="$HOME/scratch/MACE-POLAR-JAX"
source "$root/gpu-env.sh"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
cd "$root/mace-jax"

base="$root/Examples/HydrogenMaleate-MLMM-SecondShell"
original="$root/Examples/HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz"
initial="$base/structure/md-initial.xyz"
bundle="$root/artifacts/MACE-POLAR-1-M-jax-fixed.msgpack"
mkdir -p "$base/structure" "$base/md-100ps"

python examples/HydrogenMaleate-MACE-POLAR/prepare_second_shell.py \
  --initial "$original" --output "$initial" --cutoff 3.5 \
  --temperature-k 330 --seed 20260922 \
  > "$base/structure/prepare.log" 2>&1
qm_count=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["qm_water_count"])' "$base/structure/partition.json")
first_count=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["first_shell_water_count"])' "$base/structure/partition.json")
echo "Starting 100 ps second-shell ML/MM: $qm_count QM waters, $first_count in the first shell"

if taskset -c 16-23 python examples/HydrogenMaleate-MACE-POLAR/md.py \
  --mode mlmm --initial "$initial" \
  --initial-velocities "$base/structure/initial-velocities.npy" \
  --bundle "$bundle" --qm-water-count "$qm_count" \
  --first-shell-water-count "$first_count" \
  --restraint-radius 4.2 --restraint-k 0.2 \
  --output "$base/md-100ps" --steps 200000 --timestep-fs 0.5 \
  --temperature-k 330 --seed 20260922 \
  > "$base/md-100ps/run.log" 2>&1; then
  printf 'exit=0\n' > "$base/run-status.txt"
else
  status=$?
  printf 'exit=%s\n' "$status" > "$base/run-status.txt"
  exit "$status"
fi
