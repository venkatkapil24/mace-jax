# Glycine MACE-POLAR/MM

The [glycine setup](../Glycine-MACE-POLAR/README.md) runs this 100 ps CPU
trajectory from the same neutral, BFGS-optimized glycine/water coordinates
and initial velocities as the all-POLAR trajectory. The fixed POLAR region
contains glycine and the first-shell waters selected from the optimized
structure. The remaining waters are flexible AMBER14 TIP3P. A weak flat-bottom
restraint keeps the selected water oxygens within the first shell; see the
linked setup for its equation and sampling limitation.

Remote output: `~/scratch/MACE-POLAR-JAX/Examples/Glycine-MLMM/md-100ps/`.
