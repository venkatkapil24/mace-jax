#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
source gpu-env.sh
export PYTHONPATH="$root/third_party/openmm-ml${PYTHONPATH:+:$PYTHONPATH}"
selection=${1:-all}
export CUDA_VISIBLE_DEVICES="${2:-0}"
export OMP_NUM_THREADS=2
export OPENMM_CPU_THREADS=2
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
if [[ "$selection" != all && "$selection" != mlmm && "$selection" != mechanical && "$selection" != openmm ]]; then
  printf 'Use all, mlmm, mechanical or openmm\n' >&2
  exit 2
fi

example=Examples/HydrogenMaleate-FIRES-25W
source_chunk=Examples/HydrogenMaleate-MACE-POLAR/md-500ps/chunks/00020000.xyz
initial="$example/structure/initial-10ps-25w.xyz"
charges="$example/structure/fixed-charges-25w.json"
driver=mace-jax/examples/HydrogenMaleate-MACE-POLAR/md.py
mkdir -p "$example/structure"

if [[ ! -s "$initial" ]]; then
  venv/bin/python mace-jax/examples/HydrogenMaleate-MACE-POLAR/prepare_fires_start.py \
  --trajectory-chunk "$source_chunk" \
  --water-count 25 \
  --output "$initial" \
    >"$example/structure/prepare.log" 2>&1
fi

if [[ ! -s "$charges" ]]; then
  venv/bin/python mace-jax/examples/HydrogenMaleate-MACE-POLAR/prepare_mechanical_charges.py \
  --initial "$initial" \
  --qm-water-count 25 \
  --output "$charges" \
    >"$example/structure/charges.log" 2>&1
fi

# Dispatch only after short benchmarks and a GPU availability check.
# 500 kcal mol^-1 Angstrom^-2 converted using the shared energy unit constant.
fires_k=21.68205212090047

run_jax() {
  local mode=$1
  local output=$2
  shift 2
  if [[ -s "$output/final.xyz" ]]; then
    return
  fi
  JAX_PLATFORMS=cuda venv/bin/python -u "$driver" \
    --initial "$initial" \
    --bundle artifacts/cueq-benchmark/MACE-POLAR-1-M-jax-fused.msgpack \
    --output "$output" \
    --mode "$mode" \
    --qm-water-count 25 \
    --restraint-mode fires \
    --restraint-k "$fires_k" \
    --steps 200000 \
    --chunk 200 \
    --save-interval 20 \
    --report-interval 200 \
    --timestep-fs 0.5 \
    --temperature-k 330 \
    --seed 20260922 \
    --thermostat-seed 20260922 \
    --integrator langevin-middle \
    --compatibility-profile generalized-accurate \
    --nonbonded-cutoff 5.8 \
    --ewald-alpha 0.5232507342707401 \
    --pme-error-tolerance 5e-5 \
    --reciprocal-method pme \
    --pme-mesh-spacing 0.5 \
    --pme-assignment-order 8 \
    "$@"
}

run_openmm() {
  if [[ -s "$example/openmm-mechanical-500kcal-cueq-100ps/final.xyz" ]]; then
    return
  fi
  venv/bin/python -u \
    mace-jax/examples/HydrogenMaleate-MACE-POLAR/openmm_mechanical.py \
    --initial "$initial" \
    --model artifacts/cueq-torch/MACE-POLAR-1-M-cueq-fused-fp64.model \
    --charges "$charges" \
    --output "$example/openmm-mechanical-500kcal-cueq-100ps" \
    --qm-water-count 25 \
    --restraint-mode fires \
    --restraint-k "$fires_k" \
    --platform CUDA \
    --platform-precision double \
    --compatibility-profile openmm \
    --nonbonded-cutoff 5.8 \
    --ewald-alpha 0.5232507342707401 \
    --point-pme-grid 30 30 30 \
    --pme-error-tolerance 5e-5 \
    --thermostat-seed 20260922 \
    --timestep-fs 0.5 \
    --temperature-k 330 \
    --thermostat-tau-ps 1 \
    --seed 20260922 \
    --steps 200000 \
    --report-interval 20 \
    --checkpoint-interval 2000 \
    --trajectory-interval 400 \
    --torch-device cuda:0 \
    >"$example/openmm-mechanical-500kcal-cueq-100ps.log" 2>&1
}

# Protect each selection against duplicate launches. Model and trajectory
# checkpoints are written by the drivers, independently of the SSH connection.
exec 9>"$example/$selection-500kcal-cueq.launch.lock"
flock -n 9 || { printf 'This selection is already running\n' >&2; exit 2; }
printf '%s\n' "$$" >"$example/$selection-500kcal-cueq.launch.pid"
if [[ "$selection" == all || "$selection" == mlmm ]]; then
  run_jax mlmm "$example/electrostatic-500kcal-cueq-100ps" \
    >"$example/electrostatic-500kcal-cueq-100ps.log" 2>&1
  printf 'Completed\n' >"$example/electrostatic-500kcal-cueq-100ps/COMPLETE"
fi
if [[ "$selection" == all || "$selection" == mechanical ]]; then
  run_jax mechanical "$example/mechanical-500kcal-cueq-100ps" \
    --mechanical-qm-charges "$charges" \
    >"$example/mechanical-500kcal-cueq-100ps.log" 2>&1
  printf 'Completed\n' >"$example/mechanical-500kcal-cueq-100ps/COMPLETE"
fi
if [[ "$selection" == all || "$selection" == openmm ]]; then
  run_openmm
  printf 'Completed\n' >"$example/openmm-mechanical-500kcal-cueq-100ps/COMPLETE"
fi

date -Is >"$example/COMPLETE-$selection-500kcal-cueq"
