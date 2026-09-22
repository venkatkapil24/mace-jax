#!/usr/bin/env bash
# Run from a detached shell on Boltzmann; safe to disconnect the client.
set -euo pipefail

root="$HOME/scratch/MACE-POLAR-JAX"
source "$root/gpu-env.sh"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
cd "$root/mace-jax"

base="$root/Examples/Glycine-MACE-POLAR"
mm_base="$root/Examples/Glycine-MLMM"
initial="$base/structure/initial-neutral.xyz"
optimized="$base/bfgs/optimized-neutral.xyz"
bundle="$root/artifacts/MACE-POLAR-1-M-jax-fixed.msgpack"
mkdir -p "$base/bfgs" "$base/md-100ps" "$mm_base/md-100ps"

if [[ ! -s "$optimized" ]]; then
  echo "Starting constrained neutral glycine BFGS on CPU"
  taskset -c 0-7 python examples/Glycine-MACE-POLAR/optimize.py \
    --initial "$initial" --bundle "$bundle" --output "$base/bfgs" \
    --fmax 0.1 --max-steps 500 > "$base/bfgs/run.log" 2>&1
fi

echo "Starting both CPU trajectories from $optimized"
taskset -c 0-7 python examples/Glycine-MACE-POLAR/md.py \
  --mode polar --initial "$optimized" --bundle "$bundle" \
  --output "$base/md-100ps" --steps 200000 --timestep-fs 0.5 \
  --temperature-k 330 --seed 20260922 \
  > "$base/md-100ps/run.log" 2>&1 &
polar_pid=$!

taskset -c 8-15 python examples/Glycine-MACE-POLAR/md.py \
  --mode mlmm --initial "$optimized" --bundle "$bundle" \
  --output "$mm_base/md-100ps" --steps 200000 --timestep-fs 0.5 \
  --temperature-k 330 --seed 20260922 \
  > "$mm_base/md-100ps/run.log" 2>&1 &
mm_pid=$!

echo "POLAR pid=$polar_pid; POLAR/MM pid=$mm_pid"
set +e
wait "$polar_pid"; polar_status=$?
wait "$mm_pid"; mm_status=$?
set -e
printf 'POLAR exit=%s\nPOLAR/MM exit=%s\n' "$polar_status" "$mm_status" > "$base/run-status.txt"
echo "Finished: POLAR exit=$polar_status, POLAR/MM exit=$mm_status"
if (( polar_status || mm_status )); then exit 1; fi
