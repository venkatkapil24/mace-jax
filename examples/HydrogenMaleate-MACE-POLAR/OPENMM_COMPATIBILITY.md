# OpenMM mechanical embedding compatibility

Use `--compatibility-profile openmm` in `md.py` for comparison with the
independent `openmm_mechanical.py` driver. The existing `generalized` profile
remains the JAX default for full ML and electrostatic embedding.

## Shared settings

Both drivers import `simulation_settings.py`: identical atom masses from ASE,
flexible TIP3P water, solute Lennard-Jones parameters, charge arrays, mixing
rules and initial velocities. All water bonds remain flexible. There are no
bond constraints, barostat, continued center-of-mass momentum removal,
Lennard-Jones switching function or dispersion correction. Initial velocities
have zero total momentum and their kinetic energy is rescaled using `3N-3`
degrees of freedom, where `N` is the number of atoms. Reported temperatures use
the same convention; the thermostat acts on all `3N` velocity components.

For the 11.76 Å cubic hydrogen maleate cell, the compatibility defaults are:

| Setting | Value |
|---|---|
| Classical cutoff | 5.8 Å |
| Ewald splitting parameter | 0.5232507342707401 Å⁻¹ |
| Point-charge mesh | 30 × 30 × 30 |
| Point-charge interpolation | Cardinal B-spline, order 5 |
| Requested OpenMM error tolerance | 0.00005 |
| Temperature | 330 K |
| Time step | 0.5 fs |
| Thermostat friction | 1 ps⁻¹ |
| Integrator | Langevin middle; saved velocities are at half time steps |
| Initial velocity seed and thermostat seed | 20260922 |
| ML-region long-range solver | Direct reciprocal sum, matching Torch |

`--point-pme-grid`, `--ewald-alpha`, and `--nonbonded-cutoff` override the
classical settings in both drivers. OpenMM platforms can choose different
meshes when parameters are automatic, so explicitly export and supply the
actual mesh when comparing a different reference. The mesh-size default uses
OpenMM's prescribed minimum rounded to a size with factors 2, 3 and 5.

The point-charge mesh implementation is independent of the generalized
Gaussian-multipole solver. It uses the discrete spline Fourier correction,
OpenMM's Coulomb constant, a uniform neutralizing background for a charged
cell, the Ewald self term and water nonbonded exclusions. Mechanical embedding
also includes the small beyond-cutoff ML–ML Coulomb tail left by OpenMM-ML's
exception-and-subtraction construction. The local ML graph contains all
periodic copies inside the ML cutoff, including additional copies when that
cutoff exceeds half the box side; its spatial search uses JAX-MD.

The current OpenMM-ML MACE adapter converts model energies using
96.4853 kJ/mol per eV. The JAX compatibility profile reproduces that rounded
factor. The generalized profile retains the native model energy convention.
This adapter convention should be rechecked if OpenMM-ML is upgraded.

## Selecting solvers

For JAX compatibility, classical point charges use the independent order-5
mesh and the ML model uses direct reciprocal summation. The optional
`--ml-reciprocal-method pme` selects the generalized mesh for the model; this
is an accuracy/performance choice and no longer exactly matches Torch's
reciprocal calculation.

The default `generalized` profile with `--mode mlmm --reciprocal-method pme`
retains the existing electrostatic embedding: ML Gaussian multipoles and MM
point charges share the field calculation, and the classical MM energy is
restored with consistent background subtraction. `--mode mechanical` instead
uses fixed ML-region charges for classical ML–MM coupling, with no MM field
passed into the model in the compatibility profile.

## Replaying noise and restarting

Both drivers accept `--noise-file FILE.npy`, a NumPy array of standard normal
samples with shape `(steps, N, 3)`. The same sample at each step permits a
short deterministic integration comparison. OpenMM uses a custom integrator
with the same unconstrained middle update for this check. Without replay,
OpenMM uses its built-in LangevinMiddleIntegrator and JAX generates its own
noise. Equal seeds do **not** mean equal random numbers between the libraries.

JAX stores half-step velocities in Å/ps for the middle integrator, and ASE
velocity units for its legacy BAOAB integrator. Both drivers record the
physical and integrator settings and reject incompatible checkpoint restarts.
Legacy JAX checkpoints are accepted only by the generalized BAOAB path;
historical OpenMM checkpoints require their historical driver. Use new output
directories when switching profiles. Final snapshots record the actual final
time and energy rather than metadata inherited from the starting frame.

## Verification

Run the isolated checks wherever OpenMM is installed:

```sh
python -m pytest tests/test_openmm_compatibility.py -q
```

They compare point-charge energies and forces against OpenMM's double-precision
Reference platform, including a charged cell, exclusions, nonneutral region
subtraction and the beyond-cutoff exception. They also compare middle updates
with supplied noise and compare image graphs against ASE, including periodic
self images. Optional OpenMM tests skip when OpenMM is unavailable.

The complete hydrogen maleate check uses the existing saved starting and final
coordinates and performs 20 steps with identical noise:

```sh
python examples/HydrogenMaleate-MACE-POLAR/verify_openmm_compatibility.py --steps 20
```

This requires the existing model, converted bundle, molecule-only structures
and charge file under the parent project directory, plus its installed
OpenMM-ML source. It writes
`Examples/HydrogenMaleate-Analysis/openmm-compatibility-verification.json`.
The JSON lists energy terms, force differences and integration differences.
Verification trajectories and logs are stored separately under
`Examples/HydrogenMaleate-Analysis/openmm-compatibility/`.

On boltzmann (JAX 0.10.2, Torch 2.9.1, OpenMM 8.6.1), the independent checks
passed. Across three molecule-only saved configurations, the maximum per-atom
force-vector difference was 0.0000848 meV/Å and the maximum classical energy
difference was 2.77 × 10⁻¹³ eV. The remaining complete-model difference is in
the Torch/JAX ML calculation. Over 20 steps with identical noise, the maximum
coordinate difference was 3.83 × 10⁻⁹ Å. This verifies the current cubic,
unconstrained hydrogen maleate setup; it does not establish identical long
trajectories or identical sampling from one 100 ps run. CPU and GPU OpenMM
platforms can have different numerical precision from the Reference platform.

Both actual drivers also passed a checkpoint continuation from step 20 to step
40 with the same supplied noise. Final coordinates saved with eight decimal
places differed by at most 1.00 × 10⁻⁸ Å; final potential and kinetic energies
differed by 1.19 × 10⁻⁹ eV and 6.13 × 10⁻⁹ eV, respectively.

For the existing generalized mesh solver with tighter Ewald settings, use
`--compatibility-profile generalized-accurate`; see [numerical accuracy measurements](GENERALIZED_ACCURACY.md).
