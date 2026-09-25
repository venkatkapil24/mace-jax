# Hydrogen maleate mechanical embedding control

Output directory for the OpenMM-style mechanical embedding trajectory launched
by `../HydrogenMaleate-MACE-POLAR/run-mechanical-on-boltzmann.sh`.

The 50-atom QM region contains hydrogen maleate and 13 fixed first-shell waters.
Its intraregion energy is MACE-POLAR. The remaining 39 waters use flexible
AMBER14 TIP3P. The cross-boundary terms are fixed-point-charge Ewald
electrostatics and Lennard-Jones interactions; MM charges do not enter the
MACE-POLAR field. The run uses the same initial coordinates, velocities,
temperature, random seed, and flat-bottom QM-water restraint as electrostatic
embedding.
