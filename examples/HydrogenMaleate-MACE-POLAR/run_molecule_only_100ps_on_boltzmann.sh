#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
source gpu-env.sh
export PYTHONPATH="$root/third_party/openmm-ml${PYTHONPATH:+:$PYTHONPATH}"

example=Examples/HydrogenMaleate-MoleculeOnly
initial="$example/structure/initial-100ps-full-ml.xyz"
charges="$example/structure/molecule-fixed-charges.json"
driver=mace-jax/examples/HydrogenMaleate-MACE-POLAR/md.py

# Keep the four-panel OpenMM trajectory rerun at full speed.  GPU 0 becomes
# available when that process exits; rendering that follows is CPU-only.
while pgrep -f "openmm_mechanical.py.*video-rerun-100ps" >/dev/null; do
  sleep 30
done

run_jax() {
  local mode=$1
  local output=$2
  shift 2
  CUDA_VISIBLE_DEVICES=0 venv/bin/python -u "$driver" \
    --initial "$initial" \
    --bundle artifacts/MACE-POLAR-1-M-jax-fixed.msgpack \
    --output "$output" \
    --mode "$mode" \
    --qm-water-count 0 \
    --steps 200000 \
    --chunk 200 \
    --save-interval 20 \
    --report-interval 200 \
    --timestep-fs 0.5 \
    --temperature-k 330 \
    --seed 20260922 \
    --reciprocal-method pme \
    --pme-mesh-spacing 0.5 \
    --pme-assignment-order 8 \
    "$@"
}

run_jax mlmm "$example/electrostatic-100ps" \
  >"$example/electrostatic-100ps.log" 2>&1

run_jax mechanical "$example/mechanical-100ps" \
  --mechanical-qm-charges "$charges" \
  >"$example/mechanical-100ps.log" 2>&1

CUDA_VISIBLE_DEVICES=0 OPENMM_CPU_THREADS=8 venv/bin/python -u \
  mace-jax/examples/HydrogenMaleate-MACE-POLAR/openmm_mechanical.py \
  --initial "$initial" \
  --model artifacts/MACE-POLAR-1-M.model \
  --charges "$charges" \
  --output "$example/openmm-mechanical-100ps" \
  --qm-water-count 0 \
  --steps 200000 \
  --report-interval 20 \
  --checkpoint-interval 2000 \
  --trajectory-interval 400 \
  --torch-device cuda:0 \
  >"$example/openmm-mechanical-100ps.log" 2>&1

date -Is >"$example/COMPLETE"
