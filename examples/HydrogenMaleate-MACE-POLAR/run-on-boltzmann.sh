#!/usr/bin/env bash
set -euo pipefail
root="$HOME/scratch/MACE-POLAR-JAX"
source "$root/gpu-env.sh"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8
cd "$root/mace-jax"
base="$root/Examples/HydrogenMaleate-MACE-POLAR"
mm_base="$root/Examples/HydrogenMaleate-MLMM"
initial="$root/mace-jax/examples/HydrogenMaleate-MACE-POLAR/initial-anion.xyz"
optimized="$base/bfgs/optimized-anion.xyz"
md_initial="$base/structure/md-initial.xyz"
bundle="$root/artifacts/MACE-POLAR-1-M-jax-fixed.msgpack"
mkdir -p "$base/bfgs" "$base/structure" "$base/md-100ps" "$mm_base/md-100ps"
if [[ ! -s "$optimized" ]]; then
  echo "Starting constrained hydrogen maleate BFGS on CPU"
  taskset -c 0-7 python examples/HydrogenMaleate-MACE-POLAR/optimize.py \
    --initial "$initial" --bundle "$bundle" --output "$base/bfgs" \
    --fmax 0.1 --max-steps 500 > "$base/bfgs/run.log" 2>&1
fi
python examples/HydrogenMaleate-MACE-POLAR/prepare_md_start.py \
  --optimized "$optimized" --output "$md_initial" > "$base/structure/partition.log" 2>&1
qm_water_count=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["qm_water_count"])' "$base/structure/partition.json")
echo "Starting both trajectories; $qm_water_count first-shell QM waters"
taskset -c 0-7 python examples/HydrogenMaleate-MACE-POLAR/md.py \
  --mode polar --initial "$md_initial" --bundle "$bundle" --output "$base/md-100ps" \
  --steps 200000 --timestep-fs 0.5 --temperature-k 330 --seed 20260922 \
  > "$base/md-100ps/run.log" 2>&1 &
polar_pid=$!
taskset -c 8-15 python examples/HydrogenMaleate-MACE-POLAR/md.py \
  --mode mlmm --initial "$md_initial" --bundle "$bundle" --qm-water-count "$qm_water_count" \
  --restraint-radius 4.2 --restraint-k 0.2 --output "$mm_base/md-100ps" \
  --steps 200000 --timestep-fs 0.5 --temperature-k 330 --seed 20260922 \
  > "$mm_base/md-100ps/run.log" 2>&1 &
mm_pid=$!
printf '%s\n' "$polar_pid" > "$base/md-100ps/pid"
printf '%s\n' "$mm_pid" > "$mm_base/md-100ps/pid"
echo "POLAR pid=$polar_pid; POLAR/MM pid=$mm_pid"
set +e
wait "$polar_pid"; polar_status=$?
wait "$mm_pid"; mm_status=$?
set -e
printf 'POLAR exit=%s\nPOLAR/MM exit=%s\n' "$polar_status" "$mm_status" > "$base/run-status.txt"
if (( polar_status || mm_status )); then exit 1; fi
