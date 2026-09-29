#!/usr/bin/env python3
"""Benchmark electrostatic MACE-POLAR/MM memory at enzyme-scale atom counts.

The generated systems are deliberately synthetic.  They reproduce the atom
counts, density, PME grid, ML-region graph size, MM charge spreading, MM
real-space neighbor list, and reverse-mode force calculation needed for a
memory feasibility test.  They are not chemically meaningful enzyme models.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.scipy.special import erfc
from jax_md import partition, space

from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.tools.bundle import load_model_bundle

COULOMB = FIELD_CONSTANT / (4 * math.pi)


def synthetic_sites(count: int, box: float, seed: int) -> np.ndarray:
    """Create approximately uniform sites with a deterministic small jitter."""
    side = math.ceil(count ** (1 / 3))
    spacing = box / side
    axis = (np.arange(side) + 0.5) * spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    full_grid = grid.reshape(-1, 3)
    # Taking the first ``count`` entries would leave an empty slab whenever
    # ``side**3`` is appreciably larger than ``count``.  Select throughout the
    # cube so that the compact QM region and the MM density are representative.
    selection = np.linspace(0, len(full_grid) - 1, count, dtype=np.int64)
    positions = full_grid[selection].copy()
    rng = np.random.default_rng(seed)
    positions += rng.uniform(-0.08, 0.08, positions.shape) * spacing
    return positions % box


def select_qm_region(
    positions: np.ndarray, qm_atoms: int, box: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select the sites nearest the box center as a compact ML region."""
    center = np.full(3, 0.5 * box)
    delta = positions - center
    delta -= box * np.rint(delta / box)
    order = np.argsort(np.sum(delta * delta, axis=-1))
    qm_index = order[:qm_atoms]
    mm_index = order[qm_atoms:]
    return positions[qm_index], positions[mm_index]


def directed_qm_edges(
    positions: np.ndarray, cutoff: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a fixed directed graph for a static compact QM region."""
    displacement = positions[:, None, :] - positions[None, :, :]
    distance2 = np.sum(displacement * displacement, axis=-1)
    receiver, sender = np.nonzero(
        (distance2 < cutoff**2) & ~np.eye(len(positions), dtype=bool)
    )
    edge_index = jnp.asarray(np.stack((sender, receiver)), dtype=jnp.int32)
    zeros = jnp.zeros((edge_index.shape[1], 3), dtype=jnp.float64)
    return edge_index, zeros, zeros


def model_data(
    qm_positions: np.ndarray,
    box: float,
    atomic_numbers: list[int],
) -> dict[str, jnp.ndarray]:
    """Construct one-graph MACE input with representative enzyme elements."""
    n = len(qm_positions)
    # Include the difficult enzyme elements P and Mg whenever the region permits.
    representative = np.asarray([6, 1, 8, 7, 6, 1, 8, 16, 15, 12], dtype=int)
    numbers = np.resize(representative, n)
    z_to_index = {int(z): i for i, z in enumerate(atomic_numbers)}
    species = jnp.asarray([z_to_index[int(z)] for z in numbers], dtype=jnp.int32)
    return {
        'positions': jnp.asarray(qm_positions, dtype=jnp.float64),
        'node_attrs': jax.nn.one_hot(
            species, len(atomic_numbers), dtype=jnp.float64
        ),
        'node_attrs_index': species,
        'edge_index': jnp.zeros((2, 1), dtype=jnp.int32),
        'shifts': jnp.zeros((1, 3), dtype=jnp.float64),
        'unit_shifts': jnp.zeros((1, 3), dtype=jnp.float64),
        'batch': jnp.zeros(n, dtype=jnp.int32),
        'ptr': jnp.asarray([0, n], dtype=jnp.int32),
        'cell': jnp.eye(3, dtype=jnp.float64)[None] * box,
        'pbc': jnp.asarray([[True, True, True]]),
        'head': jnp.asarray([0], dtype=jnp.int32),
        'total_charge': jnp.asarray([0.0], dtype=jnp.float64),
        'total_spin': jnp.asarray([1.0], dtype=jnp.float64),
        'external_field': jnp.zeros((1, 3), dtype=jnp.float64),
    }


def ordered_pairs(neighbor_index: jnp.ndarray, n_atoms: int) -> jnp.ndarray:
    atom_i = jnp.broadcast_to(
        jnp.arange(n_atoms, dtype=neighbor_index.dtype)[:, None],
        neighbor_index.shape,
    )
    keep = atom_i < neighbor_index
    padding = jnp.asarray(n_atoms, dtype=neighbor_index.dtype)
    return jnp.stack(
        (
            jnp.where(keep, atom_i, padding).reshape(-1),
            jnp.where(keep, neighbor_index, padding).reshape(-1),
        )
    )


def memory_stats(device: jax.Device) -> dict[str, int | float | None]:
    raw = device.memory_stats() or {}
    result: dict[str, int | float | None] = {}
    for key in ('bytes_in_use', 'peak_bytes_in_use', 'bytes_limit'):
        value = raw.get(key)
        result[key] = None if value is None else int(value)
        result[f'{key}_gib'] = None if value is None else value / 2**30
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--system', required=True)
    parser.add_argument('--qm-atoms', type=int, required=True)
    parser.add_argument('--mm-atoms', type=int, required=True)
    parser.add_argument('--full-system-atoms', type=int, required=True)
    parser.add_argument('--atom-density', type=float, default=0.100)
    parser.add_argument('--mesh-spacing', type=float, default=0.5)
    parser.add_argument('--assignment-order', type=int, default=8)
    parser.add_argument('--real-cutoff', type=float, default=9.0)
    parser.add_argument('--neighbor-skin', type=float, default=0.25)
    parser.add_argument('--neighbor-capacity', type=float, default=1.3)
    parser.add_argument('--ewald-alpha', type=float, default=0.5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    if args.mm_atoms + args.qm_atoms > args.full_system_atoms:
        raise ValueError('QM plus MM atoms exceed the full-system atom count')
    if args.mm_atoms < 1 or args.qm_atoms < 2:
        raise ValueError('Both regions must be nonempty')

    jax.config.update('jax_enable_x64', True)
    device = jax.devices()[0]
    box = (args.full_system_atoms / args.atom_density) ** (1 / 3)
    all_positions = synthetic_sites(args.full_system_atoms, box, args.seed)
    qm_positions, candidate_mm = select_qm_region(
        all_positions, args.qm_atoms, box
    )
    # Uniformly sample the entire environment at intermediate sweep sizes.
    selection = np.linspace(
        0, len(candidate_mm) - 1, args.mm_atoms, dtype=np.int64
    )
    mm_positions = candidate_mm[selection]
    positions = jnp.asarray(
        np.concatenate((qm_positions, mm_positions), axis=0), dtype=jnp.float64
    )

    bundle = load_model_bundle(str(args.bundle), 'float64')
    model = nnx.merge(bundle.graphdef, bundle.params)
    if model.__class__.__name__ != 'PolarMACE':
        raise ValueError('Expected a MACE-POLAR model bundle')
    data = model_data(qm_positions, box, bundle.config['atomic_numbers'])
    mode, data = model.prepare_jit_data(
        data,
        pbc_handling='pbc',
        generalized_pme=True,
        pme_mesh_spacing=args.mesh_spacing,
        pme_assignment_order=args.assignment_order,
    )
    graphdef, params = nnx.split(model)
    edge_index, shifts, unit_shifts = directed_qm_edges(
        qm_positions, float(bundle.config['r_max'])
    )

    n_qm = args.qm_atoms
    n_mm = args.mm_atoms
    mm_charges_np = np.where(np.arange(n_mm) % 2 == 0, 0.4, -0.4)
    mm_charges_np -= mm_charges_np.mean()
    mm_charges = jnp.asarray(mm_charges_np, dtype=jnp.float64)

    displacement, _ = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        min(args.real_cutoff, 0.49 * box),
        dr_threshold=args.neighbor_skin,
        capacity_multiplier=args.neighbor_capacity,
        format=partition.Dense,
    )
    allocation_start = time.perf_counter()
    # The production ML/MM path builds this list over the complete system so
    # it contains both MM--MM and QM--MM nonbonded pairs.
    nonbonded_neighbors = neighbor_fn.allocate(positions)
    jax.block_until_ready(nonbonded_neighbors.idx)
    allocation_seconds = time.perf_counter() - allocation_start
    if bool(nonbonded_neighbors.did_buffer_overflow):
        raise RuntimeError('Neighbor-list capacity overflow during allocation')
    n_total = n_qm + n_mm
    pair_index = ordered_pairs(nonbonded_neighbors.idx, n_total)

    def total_energy(
        current_positions: jnp.ndarray,
        model_params,
        current_pair_index: jnp.ndarray,
    ) -> jnp.ndarray:
        qm_r = current_positions[:n_qm]
        mm_r = current_positions[n_qm:]
        inputs = dict(
            data,
            positions=qm_r,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            mm_positions=mm_r,
            mm_charges=mm_charges,
            mm_ewald_alpha=jnp.asarray(args.ewald_alpha, dtype=qm_r.dtype),
        )
        result = nnx.merge(graphdef, model_params)(
            inputs, pbc_handling=mode
        )

        atom_i, atom_j = current_pair_index
        valid = (atom_i < n_total) & (atom_j < n_total)
        atom_i = jnp.where(valid, atom_i, 0)
        atom_j = jnp.where(valid, atom_j, 0)
        delta = current_positions[atom_i] - current_positions[atom_j]
        delta -= box * jnp.round(delta / box)
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-24)
        safe_distance = jnp.where(valid, distance, 1.0)
        mm_pair = valid & (atom_i >= n_qm) & (atom_j >= n_qm)
        charge_i = mm_charges[jnp.maximum(atom_i - n_qm, 0)]
        charge_j = mm_charges[jnp.maximum(atom_j - n_qm, 0)]
        real_space = COULOMB * jnp.sum(
            jnp.where(
                mm_pair,
                charge_i
                * charge_j
                * erfc(args.ewald_alpha * safe_distance)
                / safe_distance,
                0.0,
            )
        )
        # A lightweight Lennard-Jones term makes the reverse pass exercise the
        # full QM--MM and MM--MM pair list, as the production energy does.
        non_qm_pair = valid & ~((atom_i < n_qm) & (atom_j < n_qm))
        inverse_r6 = (3.2 / safe_distance) ** 6
        lennard_jones = jnp.sum(
            jnp.where(non_qm_pair, 4 * 0.006 * (inverse_r6**2 - inverse_r6), 0.0)
        )
        return (
            result['energy'][0]
            + result['mm_ewald_reciprocal_energy'][0]
            + real_space
            + lennard_jones
        )

    force_function = jax.jit(jax.value_and_grad(total_energy))
    before_compile = memory_stats(device)
    compile_start = time.perf_counter()
    compiled = force_function.lower(positions, params, pair_index).compile()
    compile_seconds = time.perf_counter() - compile_start
    after_compile = memory_stats(device)

    energy, gradient = compiled(positions, params, pair_index)
    energy.block_until_ready()
    gradient.block_until_ready()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        energy, gradient = compiled(positions, params, pair_index)
        energy.block_until_ready()
        gradient.block_until_ready()
        times.append(time.perf_counter() - start)
    after_evaluation = memory_stats(device)

    mesh_shape = tuple(int(x) for x in data['pme_mesh_template'].shape)
    result = {
        'system': args.system,
        'synthetic_memory_test': True,
        'device': str(device),
        'qm_atoms': n_qm,
        'mm_atoms': n_mm,
        'evaluated_atoms': n_qm + n_mm,
        'full_system_atoms': args.full_system_atoms,
        'box_angstrom': box,
        'mesh_shape': mesh_shape,
        'mesh_points': int(np.prod(mesh_shape)),
        'assignment_order': args.assignment_order,
        'qm_directed_edges': int(edge_index.shape[1]),
        'nonbonded_neighbor_capacity_per_atom': int(
            nonbonded_neighbors.idx.shape[1]
        ),
        'nonbonded_pair_slots': int(pair_index.shape[1]),
        'neighbor_allocation_seconds': allocation_seconds,
        'compile_seconds': compile_seconds,
        'evaluation_seconds': times,
        'median_evaluation_seconds': float(np.median(times)),
        'estimated_ps_per_day_at_0p5fs': 86400 * 0.0005 / np.median(times),
        'energy_eV': float(energy),
        'finite_energy': bool(jnp.isfinite(energy)),
        'finite_gradient': bool(jnp.all(jnp.isfinite(gradient))),
        'memory_before_compile': before_compile,
        'memory_after_compile': after_compile,
        'memory_after_evaluation': after_evaluation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
