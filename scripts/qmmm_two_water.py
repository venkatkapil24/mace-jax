#!/usr/bin/env python3
"""Periodic POLAR/TIP3P two-water proof of concept (one water per region).

The MM parameters are AMBER14 TIP3P values from OpenMM's amber14/tip3p.xml.
Only this fixed two-water topology is supported. MM electrostatics uses a
point-charge Ewald sum with all three intramolecular pairs excluded; QM/MM
electrostatics shares POLAR's Fourier grid and polarizes its learned density.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy.special import erf

from mace_jax.data.utils import (
    AtomicNumberTable,
    Configuration,
    graph_from_configuration,
)
from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.tools.bundle import load_model_bundle

COULOMB_EV_ANGSTROM = FIELD_CONSTANT / (4 * math.pi)
KJ_MOL_TO_EV = 1.0 / 96.48533212331002
TIP3P_CHARGES = (-0.834, 0.417, 0.417)
TIP3P_O_SIGMA = 3.150752406575124  # Å
TIP3P_O_EPSILON = 0.635968 * KJ_MOL_TO_EV  # eV
TIP3P_BOND_R0 = 0.9572  # Å
TIP3P_BOND_K = 462750.4 * KJ_MOL_TO_EV / 100.0  # eV/Å²
TIP3P_ANGLE_R0 = 1.82421813418  # rad
TIP3P_ANGLE_K = 836.8 * KJ_MOL_TO_EV  # eV/rad²


def tip3p_bonded_energy(mm_positions: jnp.ndarray) -> jnp.ndarray:
    v1 = mm_positions[1] - mm_positions[0]
    v2 = mm_positions[2] - mm_positions[0]
    r1 = jnp.linalg.norm(v1)
    r2 = jnp.linalg.norm(v2)
    angle = jnp.arccos(jnp.clip(jnp.dot(v1, v2) / (r1 * r2), -1.0, 1.0))
    return (
        0.5 * TIP3P_BOND_K * ((r1 - TIP3P_BOND_R0) ** 2 + (r2 - TIP3P_BOND_R0) ** 2)
        + 0.5 * TIP3P_ANGLE_K * (angle - TIP3P_ANGLE_R0) ** 2
    )


def tip3p_periodic_electrostatics(
    mm_positions: jnp.ndarray,
    charges: jnp.ndarray,
    box_length: float,
    *,
    alpha: float = 0.5,
    kmax: int = 6,
) -> jnp.ndarray:
    """Point-charge Ewald for one neutral water, excluding its internal pairs.

    At this box size there are no distinct-molecule real-space pairs. The
    reciprocal term includes all image interactions; the erf exception removes
    the three base-cell intramolecular interactions from that term.
    """
    indices = jnp.arange(-kmax, kmax + 1)
    integer_k = jnp.stack(jnp.meshgrid(indices, indices, indices, indexing='ij'), -1)
    k = (2 * math.pi / box_length) * integer_k.reshape(-1, 3)
    k2 = jnp.sum(k * k, axis=-1)
    nonzero = k2 > 0
    phase = k @ mm_positions.T
    structure_real = jnp.cos(phase) @ charges
    structure_imag = jnp.sin(phase) @ charges
    weight = jnp.where(
        nonzero,
        jnp.exp(-k2 / (4 * alpha**2)) / jnp.where(nonzero, k2, 1.0),
        0.0,
    )
    reciprocal = (
        COULOMB_EV_ANGSTROM
        * 2
        * math.pi
        / box_length**3
        * jnp.sum(weight * (structure_real**2 + structure_imag**2))
    )
    self_energy = (
        -COULOMB_EV_ANGSTROM * alpha / math.sqrt(math.pi) * jnp.sum(charges**2)
    )
    pair_i = jnp.asarray([0, 0, 1])
    pair_j = jnp.asarray([1, 2, 2])
    dr = mm_positions[pair_i] - mm_positions[pair_j]
    dr = dr - box_length * jnp.round(dr / box_length)
    distances = jnp.linalg.norm(dr, axis=-1)
    exceptions = -COULOMB_EV_ANGSTROM * jnp.sum(
        charges[pair_i] * charges[pair_j] * erf(alpha * distances) / distances
    )
    return reciprocal + self_energy + exceptions


def cross_oxygen_lj(
    qm_positions: jnp.ndarray, mm_positions: jnp.ndarray, box_length: float
) -> jnp.ndarray:
    """Provisional TIP3P O--O Lennard-Jones boundary term."""
    dr = qm_positions[0] - mm_positions[0]
    dr = dr - box_length * jnp.round(dr / box_length)
    r = jnp.linalg.norm(dr)
    sigma_over_r6 = (TIP3P_O_SIGMA / r) ** 6
    return 4 * TIP3P_O_EPSILON * (sigma_over_r6**2 - sigma_over_r6)


def initial_positions() -> tuple[np.ndarray, np.ndarray]:
    angle = TIP3P_ANGLE_R0
    qm_o = np.asarray([3.0, 3.0, 3.0])
    qm = np.stack(
        [
            qm_o,
            qm_o + [TIP3P_BOND_R0, 0.0, 0.0],
            qm_o
            + [TIP3P_BOND_R0 * math.cos(angle), TIP3P_BOND_R0 * math.sin(angle), 0.0],
        ]
    )
    mm_o = np.asarray([5.9, 3.0, 3.0])
    mm = np.stack(
        [
            mm_o,
            mm_o
            + [
                TIP3P_BOND_R0 * math.cos(angle / 2),
                TIP3P_BOND_R0 * math.sin(angle / 2),
                0.0,
            ],
            mm_o
            + [
                TIP3P_BOND_R0 * math.cos(angle / 2),
                -TIP3P_BOND_R0 * math.sin(angle / 2),
                0.0,
            ],
        ]
    )
    return qm, mm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bundle', type=Path, default=Path('/tmp/MACE-POLAR-1-M-jax-fixed.msgpack')
    )
    parser.add_argument('--box', type=float, default=10.0)
    args = parser.parse_args()
    if args.box < 8.0:
        parser.error('Use a box of at least 8 Å for this fixed Ewald setup')

    jax.config.update('jax_enable_x64', True)
    bundle = load_model_bundle(str(args.bundle), 'float64')
    model = nnx.merge(bundle.graphdef, bundle.params)
    if model.__class__.__name__ != 'PolarMACE':
        parser.error('The bundle must contain POLAR-MACE')
    qm_np, mm_np = initial_positions()
    config = Configuration(
        atomic_numbers=np.asarray([8, 1, 1]),
        positions=qm_np,
        cell=np.eye(3) * args.box,
        pbc=(True, True, True),
        total_charge=0.0,
        total_spin=1.0,
        external_field=np.zeros(3),
    )
    z_table = AtomicNumberTable(bundle.config['atomic_numbers'])
    graph = graph_from_configuration(
        config, cutoff=float(bundle.config['r_max']), z_table=z_table
    )
    # The generic graph adapter assumes the final graph is padding. This is a
    # single physical graph, so construct its data dictionary without that mask.
    species = jnp.asarray(graph.nodes.species, dtype=jnp.int32)
    data = {
        'positions': jnp.asarray(qm_np),
        'node_attrs': jax.nn.one_hot(
            species, len(bundle.config['atomic_numbers']), dtype=jnp.float64
        ),
        'node_attrs_index': species,
        'edge_index': jnp.stack(
            (jnp.asarray(graph.senders), jnp.asarray(graph.receivers)), axis=0
        ),
        'shifts': jnp.asarray(graph.edges.shifts, dtype=jnp.float64),
        'unit_shifts': jnp.asarray(graph.edges.unit_shifts, dtype=jnp.float64),
        'batch': jnp.zeros(3, dtype=jnp.int32),
        'ptr': jnp.asarray([0, 3], dtype=jnp.int32),
        'cell': jnp.asarray(config.cell)[None],
        'pbc': jnp.asarray([[True, True, True]]),
        'head': jnp.asarray([0], dtype=jnp.int32),
        'total_charge': jnp.asarray([0.0]),
        'total_spin': jnp.asarray([1.0]),
        'external_field': jnp.zeros((1, 3)),
    }
    mode, data = model.prepare_jit_data(data, pbc_handling='pbc')
    charges = jnp.asarray(TIP3P_CHARGES, dtype=jnp.float64)
    qm_start = jnp.asarray(qm_np)
    mm_start = jnp.asarray(mm_np)

    def components(qm_positions, mm_positions, *, embedded=True):
        current_data = dict(data, positions=qm_positions)
        if embedded:
            current_data['mm_positions'] = mm_positions
            current_data['mm_charges'] = charges
        qm_result = model(current_data, compute_force=False, pbc_handling=mode)
        mm_bonded = tip3p_bonded_energy(mm_positions)
        mm_electrostatic = tip3p_periodic_electrostatics(
            mm_positions, charges, args.box
        )
        oxygen_lj = cross_oxygen_lj(qm_positions, mm_positions, args.box)
        total = qm_result['energy'][0] + mm_bonded + mm_electrostatic + oxygen_lj
        return total, {
            'qm_backbone_and_electron': qm_result['energy'][0]
            - qm_result['electrostatic_energy'][0],
            'qm_and_cross_electrostatic': qm_result['electrostatic_energy'][0],
            'qm_mm_cross_electrostatic': qm_result['cross_electrostatic_energy'][0],
            'mm_electrostatic': mm_electrostatic,
            'mm_bonded': mm_bonded,
            'qm_mm_oxygen_lj': oxygen_lj,
            'qm_density': qm_result['density_coefficients'],
        }

    (energy, terms), (grad_qm, grad_mm) = jax.value_and_grad(
        components, argnums=(0, 1), has_aux=True
    )(qm_start, mm_start)
    isolated_energy, isolated = components(qm_start, mm_start, embedded=False)
    forces = jnp.concatenate((-grad_qm, -grad_mm), axis=0)
    result = {
        'total_energy_eV': float(energy),
        'without_qm_mm_electrostatics_eV': float(isolated_energy),
        'electrostatic_embedding_shift_eV': float(energy - isolated_energy),
        'qm_plus_cross_electrostatic_eV': float(terms['qm_and_cross_electrostatic']),
        'qm_mm_cross_electrostatic_eV': float(terms['qm_mm_cross_electrostatic']),
        'mm_electrostatic_eV': float(terms['mm_electrostatic']),
        'mm_bonded_eV': float(terms['mm_bonded']),
        'qm_mm_oxygen_lj_eV': float(terms['qm_mm_oxygen_lj']),
        'qm_force_eV_per_A': np.asarray(-grad_qm).tolist(),
        'mm_force_eV_per_A': np.asarray(-grad_mm).tolist(),
        'net_force_norm_eV_per_A': float(jnp.linalg.norm(jnp.sum(forces, axis=0))),
        'qm_density_response_l2': float(
            jnp.linalg.norm(terms['qm_density'] - isolated['qm_density'])
        ),
        'finite': bool(
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(grad_qm))
            & jnp.all(jnp.isfinite(grad_mm))
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
