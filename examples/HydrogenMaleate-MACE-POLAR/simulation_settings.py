"""Shared physical parameters and initial velocities for both MD drivers."""

import math
from dataclasses import asdict, dataclass

import numpy as np
from ase import units

from mace_jax.tools.physical_constants import (
    KJ_MOL_PER_EV,
    OPENMM_BOLTZMANN,
    OPENMM_MACE_SCALE,
)

TIP3P_Q = (-0.834, 0.417, 0.417)
TIP3P_SIGMA = 3.150752406575124
TIP3P_EPS_KJ_MOL = 0.635968
OH_R0 = 0.9572
OH_K_KJ_MOL_ANGSTROM2 = 4627.504
HOH_THETA0 = 1.82421813418
HOH_K_KJ_MOL_RAD2 = 836.8
SOLUTE_SIGMA = np.asarray(
    [
        3.066473387839048,
        2.959921901149463,
        2.959921901149463,
        2.959921901149463,
        3.399669508423535,
        3.399669508423535,
        3.399669508423535,
        3.399669508423535,
        2.471353044121301,
        2.471353044121301,
        10.0,
    ]
)
SOLUTE_EPS_KJ_MOL = np.asarray(
    [
        0.8803136,
        0.87864,
        0.87864,
        0.87864,
        0.359824,
        0.359824,
        0.4577296,
        0.4577296,
        0.0656888,
        0.0656888,
        0.0,
    ]
)


def generate_velocities(atoms, temperature, seed, profile='generalized'):
    """Return ASE velocities; the OpenMM driver only converts their units."""
    if (
        profile not in ('generalized', 'generalized-accurate', 'openmm')
        or not np.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError('Invalid velocity profile or temperature')
    masses = atoms.get_masses()
    if np.any(masses <= 0):
        raise ValueError('Initial velocities require positive masses')
    if temperature == 0:
        return np.zeros((len(atoms), 3))
    kB = units.kB if profile == 'generalized' else OPENMM_BOLTZMANN
    velocities = np.random.default_rng(seed).normal(size=(len(atoms), 3)) * np.sqrt(
        kB * temperature / masses[:, None]
    )
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / masses.sum()
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    return velocities * np.sqrt(0.5 * (3 * len(atoms) - 3) * kB * temperature / kinetic)


@dataclass(frozen=True)
class ForceSettings:
    profile: str
    cutoff: float
    alpha: float
    grid_shape: tuple[int, int, int] | None
    order: int
    ml_energy_scale: float

    def to_dict(self):
        return asdict(self)


def resolve_force_settings(
    profile, box, *, cutoff=None, alpha=None, grid_shape=None, order=8, tolerance=None
):
    if not np.isfinite(box) or box <= 0:
        raise ValueError('Box side must be positive and finite')
    if alpha is not None and (not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('Ewald alpha must be positive and finite')
    if profile not in ('generalized', 'generalized-accurate', 'openmm'):
        raise ValueError('Unknown compatibility profile')
    if cutoff is None:
        cutoff = (
            min(9.0, 0.49 * box)
            if profile == 'generalized'
            else min(5.8, 0.5 * box - 1e-6)
        )
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box / 2:
        raise ValueError('Classical cutoff must be smaller than half the box')
    if grid_shape is not None and (
        len(grid_shape) != 3
        or any(not isinstance(n, (int, np.integer)) or n <= 5 for n in grid_shape)
    ):
        raise ValueError('Point PME grid needs three integer dimensions greater than 5')
    tolerance = (
        (1e-6 if profile == 'generalized-accurate' else 5e-5)
        if tolerance is None
        else tolerance
    )
    if profile in ('openmm', 'generalized-accurate'):
        if not 0 < tolerance < 0.5:
            raise ValueError('PME error tolerance must be between zero and 0.5')
        alpha = math.sqrt(-math.log(2 * tolerance)) / cutoff if alpha is None else alpha
        if profile == 'generalized-accurate':
            if grid_shape is not None:
                raise ValueError('Use --pme-mesh-spacing for the generalized mesh')
            return ForceSettings(profile, cutoff, alpha, None, order, OPENMM_MACE_SCALE)
        if grid_shape is None:
            # OpenMM's prescribed minimum size. Its platforms may round this
            # up; export actual grid sizes from the reference for exact matching.
            size = math.ceil(2 * alpha * box / (3 * tolerance**0.2))

            def smooth(n):
                while True:
                    remaining = n
                    for p in (2, 3, 5):
                        while remaining % p == 0:
                            remaining //= p
                    if remaining == 1:
                        return n
                    n += 1

            grid_shape = (smooth(size),) * 3
        return ForceSettings(
            profile, cutoff, alpha, tuple(grid_shape), 5, OPENMM_MACE_SCALE
        )
    return ForceSettings(
        profile,
        cutoff,
        0.5 if alpha is None else alpha,
        tuple(grid_shape) if grid_shape is not None else None,
        order,
        1.0,
    )
