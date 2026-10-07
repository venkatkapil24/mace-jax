# More accurate generalized mesh electrostatics

Use the existing MD command with:

```sh
--compatibility-profile generalized-accurate
```

This retains the existing generalized PME solver for Gaussian ML multipoles
and MM point charges. The mesh algorithm and the model's reciprocal cutoff
are unchanged. The new profile changes numerical settings and aligns the
mechanical calculation's units and dynamics with the OpenMM reference.

## Default settings for the 11.76 Å hydrogen maleate box

| Parameter | Previous generalized profile | New accuracy profile |
|---|---:|---:|
| Classical real-space cutoff | 5.7624 Å | 5.80 Å |
| Ewald splitting parameter | 0.500 Å⁻¹ | 0.6245655654795327 Å⁻¹ |
| Requested mesh spacing | 0.500 Å | 0.400 Å |
| Actual mesh | 24³ | 30³ |
| B-spline interpolation order | 8 | 8 |
| Ewald real-space error control parameter | Fixed splitting parameter | 0.000001 |
| Integrator | BAOAB | Langevin middle |

The splitting parameter controls how much of the Coulomb interaction is
calculated in real space and how much on the reciprocal mesh. The new value
reduces the interaction omitted beyond the finite real-space cutoff. It is
chosen from the same cutoff/error relation used by OpenMM, with a tighter
error control parameter. That parameter is not a universal bound on force
error; the force errors were measured independently below.

The accuracy profile uses OpenMM's Coulomb constant consistently across the
mechanical calculation's real, reciprocal, self, exclusion and background
terms. Its full-ML and mechanical model energy conversion matches the current
OpenMM-ML adapter. The mechanical calculation also retains OpenMM-ML's small
beyond-cutoff ML–ML exception contribution. It uses shared initial velocities,
Boltzmann constant, temperature, thermostat friction and half-step velocity
convention. Identical seeds still produce different random numbers in JAX and
OpenMM; supplied noise is required for a deterministic short-step comparison.

For electrostatic embedding (`--mode mlmm`), the original model energy and
Coulomb conventions are retained consistently for the combined Gaussian and
point-charge field, its subtracted MM contribution, and the restored MM Ewald
energy. The tighter splitting parameter and finer generalized mesh apply in
that mode too. This profile requires generalized PME, rather than a finite
point-charge reciprocal sum.

## Measurements

At the same three molecule-only saved configurations, force errors relative
to a converged direct Ewald calculation were:

| Point-charge electrostatics | RMS force-component error (meV/Å) |
|---|---:|
| Previous generalized JAX | 0.149–0.168 |
| Previous OpenMM settings | 0.0629–0.0672 |
| Generalized JAX, OpenMM-matched splitting and cutoff | 0.0626–0.0659 |
| New generalized accuracy profile | 0.00135–0.00140 |

RMS is the square root of the mean squared error over every atomic Cartesian
force component. These point-charge calculations include MM–MM and fixed-charge
ML–MM interactions. Two independently converged direct Ewald settings agree
to less than 0.000000001 meV/Å in the largest atomic force-vector difference.
Using direct reciprocal evaluation while keeping the previous real-space
cutoff gives essentially the same previous error: the real-space truncation
was the dominant error, rather than mesh interpolation.

The complete mechanical model was also checked, including the Gaussian ML
mesh against the previously verified direct Gaussian calculation. Its RMS
force error is 0.00135–0.00140 meV/Å against direct Gaussian ML plus converged
point-charge Ewald. The Gaussian mesh contribution alone differs by at most
0.0000000379 meV/Å RMS. These are numerical solver checks at three structures;
they do not establish convergence of a free-energy curve or chemical accuracy.

Refining the mesh further to 40³ changes the total force error only slightly,
while increasing memory and fast Fourier transform (FFT) work. The 30³ grid has 1.95 times as many cells
as the previous 24³ grid; this is a mesh-size comparison, not a measured ratio
of MD runtimes.

## Controls and reports

For OpenMM-level numerical accuracy with the original 24³ mesh, use:

```sh
--compatibility-profile generalized-accurate --pme-error-tolerance 5e-5 --pme-mesh-spacing 0.5
```

This selects a 5.80 Å cutoff, splitting parameter 0.5232507342707401 Å⁻¹,
and order-8 interpolation for the 11.76 Å cell. The name of the profile does
not imply that its tighter default remains active after these overrides.

For the tighter default use the profile alone. Override `--pme-mesh-spacing`,
`--pme-assignment-order`, `--ewald-alpha` or `--nonbonded-cutoff` when needed.
The actual mesh, splitting parameter, integrator and constants are recorded in
progress and checkpoint metadata. Use a new output directory for this profile;
a checkpoint with different force settings is rejected.

Reproduce the point-charge parameter study and the complete-model check with:

```sh
python examples/HydrogenMaleate-MACE-POLAR/compare_previous_electrostatics.py --tune
python examples/HydrogenMaleate-MACE-POLAR/compare_previous_electrostatics.py --verify-model
```

The reports are in the parent project directory:

- `Examples/HydrogenMaleate-Analysis/generalized-electrostatics-tuning.json`
- `Examples/HydrogenMaleate-Analysis/generalized-accurate-model-verification.json`

The independent OpenMM-compatible order-5 point-charge solver remains
available through `--compatibility-profile openmm` for strict numerical
comparisons with OpenMM's own mesh algorithm.

Verification also passed 13 independent compatibility/profile tests, a 20-step
fully compiled mechanical MD check, and one fully compiled electrostatic
embedding force evaluation with finite energy and forces. These are short
implementation checks, rather than new production trajectories.

## Measured GPU cost

The selected GPU comparison now includes previous numerical settings and
OpenMM-matched 24³ settings, each with the same integrator. Both mechanical
and electrostatic embedding were also benchmarked with fused cuEquivariance
CUDA kernels.

| Embedding | Settings | Force call (ms) | MD step (ms) | Throughput (ns/day) | Change in MD cost vs previous |
|---|---|---:|---:|---:|---:|
| Mechanical | Previous settings | 25.4 | 10.1 | 4.30 | Reference |
| Mechanical | Matched, 24³ | 25.9 | 10.1 | 4.27 | +0.584% |
| Mechanical | Matched, 24³ + cuEquivariance | 25.2 | 9.47 | 4.56 | −5.83% |
| Electrostatic | Previous settings | 25.8 | 10.3 | 4.21 | Reference |
| Electrostatic | Matched, 24³ | 26.0 | 10.2 | 4.22 | −0.142% |
| Electrostatic | Matched, 24³ + cuEquivariance | 24.8 | 9.65 | 4.48 | −5.88% |

All selected runs completed 600 steps at 330 K and a 0.500 fs timestep on an
RTX A6000, with 11 ML atoms and 156 MM atoms. Values are medians of five
100-step windows after 100 warmup steps; compilation and disk writes are
excluded. Differences below 1% are comparable to timing variation. The
accelerated kernels passed energy and force parity checks and are present
in the compiled calculations.

The selected results are saved in the parent project directory at
`Examples/HydrogenMaleate-Analysis/settings-benchmark-20261001/comparison.json`,
with a self-contained `README.md` beside it. The accelerated model is an
optional separate bundle on boltzmann.
