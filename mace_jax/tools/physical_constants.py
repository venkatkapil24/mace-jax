"""Unit constants shared with the OpenMM compatibility implementation."""

KJ_MOL_PER_EV = 96.48533212331002
OPENMM_COULOMB = 138.93545764438198 * 10 / KJ_MOL_PER_EV
OPENMM_BOLTZMANN = 0.00831446261815324 / KJ_MOL_PER_EV
ACCELERATION = KJ_MOL_PER_EV * 100
# Existing OpenMM-ML MACE adapter uses this rounded energy conversion.
OPENMM_MACE_SCALE = 96.4853 / KJ_MOL_PER_EV
