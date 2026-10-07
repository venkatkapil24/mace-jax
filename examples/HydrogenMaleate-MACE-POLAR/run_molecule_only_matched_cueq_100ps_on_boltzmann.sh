#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
source gpu-env.sh
mode=${1:-mechanical}
gpu=${2:-0}
if [[ "$mode" != mechanical && "$mode" != mlmm ]]; then
  printf "Use mechanical or mlmm\n" >&2
  exit 2
fi
label=mechanical
if [[ "$mode" == mlmm ]]; then label=electrostatic; fi
export CUDA_VISIBLE_DEVICES="$gpu"
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
example=Examples/HydrogenMaleate-MoleculeOnly
output="$example/$label-matched24-cueq-100ps-20261001"
venv/bin/python -u mace-jax/examples/HydrogenMaleate-MACE-POLAR/md.py \
  --initial "$example/structure/initial-100ps-full-ml.xyz" \
  --bundle artifacts/cueq-benchmark/MACE-POLAR-1-M-jax-fused.msgpack \
  --output "$output" \
  --mode "$mode" --qm-water-count 0 --restraint-mode none \
  --mechanical-qm-charges "$example/structure/molecule-fixed-charges.json" \
  --steps 200000 --chunk 200 --save-interval 20 --report-interval 200 \
  --timestep-fs 0.5 --temperature-k 330 --thermostat-tau-ps 1 \
  --seed 20260922 --thermostat-seed 20260922 --integrator langevin-middle \
  --compatibility-profile generalized-accurate \
  --reciprocal-method pme --ml-reciprocal-method pme \
  --pme-mesh-spacing 0.5 --pme-assignment-order 8 \
  --pme-error-tolerance 5e-5 --nonbonded-cutoff 5.8
printf 'Completed\n' > "$output/COMPLETE"
