"""Unconstrained Langevin updates with explicit, replayable Gaussian noise.

Units: positions Angstrom, velocities Angstrom/ps, time ps, masses dalton,
forces eV/Angstrom. Stored velocities in the middle scheme are half-step
velocities, following OpenMM's LangevinMiddleIntegrator convention.
"""

import jax.numpy as jnp

from mace_jax.tools.physical_constants import ACCELERATION, OPENMM_BOLTZMANN


def middle_step(
    positions,
    velocities,
    forces,
    masses,
    noise,
    *,
    timestep_ps,
    temperature,
    friction_ps,
    shift=None,
):
    """Full force kick, half drift, thermal velocity update, half drift."""
    masses = jnp.asarray(masses)
    if masses.ndim == 1:
        masses = masses[:, None]
    inverse = jnp.where(masses > 0, 1 / jnp.where(masses > 0, masses, 1), 0.0)
    velocity = velocities + timestep_ps * ACCELERATION * forces * inverse
    displacement = 0.5 * timestep_ps * velocity
    decay = jnp.exp(-timestep_ps * friction_ps)
    velocity = (
        decay * velocity
        + jnp.sqrt(
            (1 - decay * decay)
            * OPENMM_BOLTZMANN
            * temperature
            * ACCELERATION
            * inverse
        )
        * noise
    )
    displacement += 0.5 * timestep_ps * velocity
    positions = (
        positions + displacement if shift is None else shift(positions, displacement)
    )
    return positions, velocity


def kinetic_energy(velocities, masses):
    masses = jnp.asarray(masses)
    if masses.ndim == 1:
        masses = masses[:, None]
    return 0.5 * jnp.sum(masses * velocities**2) / ACCELERATION
