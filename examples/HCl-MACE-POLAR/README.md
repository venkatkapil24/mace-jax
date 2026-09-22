# HCl in water

The included `equilibrated-63h2o-hcl.xyz` is the final configuration of a
published 0.9 M HCl simulation: 63 waters, one excess proton, and one chloride
in a 12.4187 Å cubic cell (191 atoms). The coordinates come from
[Yang et al.](https://www.nature.com/articles/s41467-025-60794-2),
[Zenodo record 15490289](https://doi.org/10.5281/zenodo.15490289), file
`0.9M_HCl_final.txt`. The paper reports 0.6 ns equilibration followed by
production sampling. This is an equilibrated geometry under the authors'
force field, not a MACE-POLAR minimum.

First relax the published coordinates with the OMOL head of MACE-MH-1 using
ASE BFGS. This produces an intermediate geometry that can be read by the
MACE-POLAR-1-M optimizer or used for the short NVE examples. The cell stays
fixed.

```bash
python examples/HCl-MACE-POLAR/preopt_omol.py \
  --initial examples/HCl-MACE-POLAR/equilibrated-63h2o-hcl.xyz \
  --output examples/HCl-MACE-POLAR/run-omol-bfgs \
  --device cpu --fmax 0.05 --max-steps 200

python examples/HCl-MACE-POLAR/optimize.py \
  --initial examples/HCl-MACE-POLAR/run-omol-bfgs/preoptimized.xyz \
  --bundle /tmp/MACE-POLAR-1-M-jax-fixed.msgpack \
  --output examples/HCl-MACE-POLAR/run-polar-from-omol-bfgs
```

Install `jax-md` in the project environment before the second stage. Its
force threshold is specified in meV/Å (default: 0.01 meV/Å = 1e-5 eV/Å).
The second stage uses a JIT-compiled BFGS step with backtracking line search
and neighbor-list updates.
`run-omol-bfgs/forces.csv` records the first-stage force series; both
directories contain a `results.json` summary and a final extended XYZ.
An optimization that reaches its step limit is reported as unconverged.

Omitting `--initial` instead generates a random 72-water, one-HCl configuration
in a 13 Å cube. This random configuration is not equilibrated.

To run the 100-step, 0.5 fs all-POLAR trajectory from the ASE-optimized
geometry, use:

```bash
python examples/HCl-MACE-POLAR/md.py \
  --initial examples/HCl-MACE-POLAR/run-omol-bfgs/preoptimized.xyz \
  --bundle /tmp/MACE-POLAR-1-M-jax-fixed.msgpack \
  --output examples/HCl-MACE-POLAR/run-polar-nve-100 \
  --steps 100 --timestep-fs 0.5 --temperature-k 300 --seed 20260922
```

The MD script uses JIT-compiled velocity Verlet with JAX-MD neighbor lists.
It initializes 300 K Maxwell-Boltzmann velocities, removes center-of-mass
translation, and evolves NVE. `trajectory.xyz` contains frames every five
steps, and `results.json` contains every step's energy, temperature, and
maximum force. The [POLAR/MM example](../HCl-MLMM/README.md) uses the same
starting structure and velocity seed.
