#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$root"
source gpu-env.sh
export PYTHONPATH="$root/third_party/openmm-ml${PYTHONPATH:+:$PYTHONPATH}"

output=Examples/HydrogenMaleate-OpenMM-Mechanical/video-rerun-100ps
CUDA_VISIBLE_DEVICES=0 OPENMM_CPU_THREADS=8 venv/bin/python -u \
  mace-jax/examples/HydrogenMaleate-MACE-POLAR/openmm_mechanical.py \
  --initial Examples/HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz \
  --model artifacts/MACE-POLAR-1-M.model \
  --charges Examples/HydrogenMaleate-Mechanical/fixed-qm-charges.json \
  --output "$output" \
  --steps 200000 \
  --report-interval 20 \
  --checkpoint-interval 2000 \
  --trajectory-interval 400 \
  --torch-device cuda:0

venv/bin/python -u \
  mace-jax/examples/HydrogenMaleate-MACE-POLAR/render_four_method_comparison.py \
  --examples Examples \
  --output Examples/HydrogenMaleate-Analysis/four-panel-100ps-top-view.mp4 \
  --duration-ps 100 \
  --stride-fs 200 \
  --fps 20
