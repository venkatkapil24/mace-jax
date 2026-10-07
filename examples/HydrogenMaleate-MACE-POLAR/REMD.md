# Hydrogen maleate: CSVR temperature replica exchange

Driver: `remd.py`; remote launcher: `run_remd_on_boltzmann.sh`.
Requires `venkatkapil24/jax-md`, `working-branch`, commit
`677d06efe597277dab0eb959df353a9a73aa6682`, loaded through `PYTHONPATH`.

## Physical settings

- Same periodic 167-atom starting configuration for all methods: hydrogen
  maleate and 52 waters, from the 10 ps full-ML frame.
- Full ML: all 167 atoms described by MACE-POLAR.
- Electrostatic and mechanical ML/MM: 86 ML atoms (11 solute atoms plus
  25 waters); 81 MM atoms (27 waters). FIRES: 500 kcal/mol/Angstrom².
- Fused cuEquivariance MACE-POLAR-1-M weights; float64 arithmetic.
- Generalized PME, 24³ mesh, order 8; MM real-space cutoff 5.8 Angstrom,
  Ewald alpha 0.5232507342707401 inverse Angstrom, neighbor skin 0.25 Angstrom.
- Fixed volume, CSVR thermostat, coupling time 0.1 ps, timestep 0.5 fs,
  center-of-mass momentum removed. No bond constraints.
- Exchanges every 10 MD steps (5 fs), using the user's JAX-MD algorithm:
  each unordered temperature pair is attempted once per sweep.

## Ladder and duration

Start with six replicas at 300, 320, 340, 360, 380 and 400 K.
Run a 1 ps pilot. Discard its first 0.2 ps when checking exchange counts.
Retain this ladder if every adjacent pair has acceptance >=10%; otherwise
try eleven replicas, 300 through 400 K at 10 K intervals. If that ladder also
fails this check, stop at the pilot for inspection.

A successful pilot is resumed from its exact state to a **total of 100 ps per
replica**, including the pilot. No trajectory exceeds 100 ps. Six-replica runs
contain 600 ps in aggregate; eleven-replica runs contain 1100 ps in aggregate.
The six-replica ladder has no 330 K temperature: analysis at 330 K requires
statistical reweighting. Eleven replicas include 330 K directly.

## State, sampling and restart

Temperature labels move between trajectories; coordinates, neighbor lists and
thermostat random-number streams stay with their continuous trajectories.
Velocities are scaled when labels change. Exchange probabilities use the
complete potential energy, including FIRES for the embedded methods.
The native JAX-MD CSVR thermostat is followed by velocity Verlet. The bridge
retains energy and forces from one model evaluation per physical MD step.
Model weights are passed as shared runtime inputs, outside the replicated state.

`chunks/*.npz` stores positions and energies every 20 steps (10 fs), in
**continuous trajectory order**. `ensemble_indices` gives the temperature label
of each saved trajectory; `swap_steps`/`swap_indices` store every exchange event.
Data saved at an exchange step uses the post-exchange label.

`checkpoint.npz` stores every dynamic state leaf, including forces, neighbors,
velocities, all random-number generator keys, temperature labels and exchange
counts. `settings.json` fixes the physical settings and hashes the inputs.
Resuming under different physical settings is rejected. Chunks and checkpoints
are written atomically; an invalid chunk leaves the last valid checkpoint intact.

`progress.json` and `progress.jsonl` contain timing and acceptance measurements.
Throughput is reported both per replica and summed over replicas, in ns/day.
`compilation.json` records CUDA call targets and compiler memory estimates.
Production GPU runs require the cuEquivariance `uniform_1d_cuda` target.
`COMPLETE` is written only after reaching 100 ps per replica.

## Boltzmann

Outputs: `~/scratch/MACE-POLAR-JAX/Examples/HydrogenMaleate-REMD-CSVR-20261006`.
Full ML uses GPU 0; the two embedded methods share GPU 1. Launchers use `nohup`
and a per-method lock; closing the laptop does not stop them. Original MD runs
and reference trajectories remain untouched.
