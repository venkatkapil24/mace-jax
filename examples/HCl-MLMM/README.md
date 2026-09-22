# HCl in water: MACE-POLAR / AMBER TIP3P

This example repeats the [all-POLAR 50 fs trajectory](../HCl-MACE-POLAR/README.md)
from the same ASE-optimized 63-water HCl structure and the same 300 K velocity
seed. It uses one fixed partition throughout the run:

- MACE-POLAR-1-M: chloride, all waters with O within 3.8 Å of Cl, and the
  complete hydronium carrying the excess proton. For the included starting
  structure this is seven shell waters plus one hydronium, 26 atoms total.
- AMBER14 flexible TIP3P: the remaining 55 complete water molecules, 165 atoms.

The POLAR model's periodic Fourier electrostatics includes the MM point
charges and their effect on its learned QM density. MM–MM point-charge
electrostatics uses an Ewald sum with intramolecular exclusions. Flexible
TIP3P bond and angle terms and O–O Lennard-Jones interactions use
[OpenMM's AMBER14 TIP3P parameters](https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/data/amber14/tip3p.xml).
The QM/MM boundary also has O–O and Cl–O Lennard-Jones interactions, using
Lorentz–Berthelot mixing and the chloride ion parameters in that file.
This is a short fixed-partition prototype; there are no constraints or
thermostat.

```bash
python examples/HCl-MLMM/md.py \
  --initial examples/HCl-MACE-POLAR/run-omol-bfgs/preoptimized.xyz \
  --bundle /tmp/MACE-POLAR-1-M-jax-fixed.msgpack \
  --output examples/HCl-MLMM/run-polar-mm-nve-100 \
  --steps 100 --timestep-fs 0.5 --temperature-k 300 --seed 20260922
```

`partition.json` records the atom mapping. `trajectory.xyz`, `final.xyz`,
and `results.json` record the 50 fs NVE run. All potential-energy and
velocity-Verlet steps are JIT compiled. The partition is selected once at the
start; check for shell exchange and proton transfer across the QM/MM boundary
before extending the trajectory.
