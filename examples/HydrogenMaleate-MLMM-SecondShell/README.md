# Hydrogen maleate with two QM water shells

This separate 330 K, 100 ps ML/MM trajectory starts from the same optimized
hydrogen maleate/water geometry as the existing all-POLAR and first-shell
ML/MM runs. The first 13 waters are the original first QM shell. A water joins
the second QM shell when its oxygen is within 3.5 Å (minimum image) of any
first-shell water oxygen in the initial structure. For this cell that adds 25
waters, giving 38 QM waters and 14 MM waters. The shell identities remain
fixed during MD.

The initial velocities and thermostat noise are permuted by atom identity to
match the original runs. Flat-bottom restraints act at 4.2 Å from solute
oxygens for first-shell QM waters, and at 4.2 Å from first-shell water oxygens
for second-shell QM waters. The run uses Boltzmann CPU cores 16–23 and writes
restart checkpoints and trajectory chunks to `md-100ps/`.
