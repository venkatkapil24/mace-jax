#!/usr/bin/env python3
"""Fixed-cell periodic aqueous HCl relaxation with MACE-POLAR and JAX BFGS."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.io import read, write
from flax import nnx
from jax_md import partition, space

from mace_jax.tools.bundle import load_model_bundle

BOX = 13.0
N_WATER = 72
WATER_OH = 0.9572
WATER_ANGLE = math.radians(104.52)
HCL_BOND = 1.275


def _rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    q[:, 0] *= np.linalg.det(q)
    return q


def _minimum_distances(candidate: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = candidate[:, None, :] - previous[None, :, :]
    delta -= BOX * np.rint(delta / BOX)
    return np.linalg.norm(delta, axis=-1)


def make_initial_cell(seed: int = 20260922) -> Atoms:
    """Pack intact, randomly oriented molecules without severe close contacts."""
    rng = np.random.default_rng(seed)
    cl = np.array([BOX / 2] * 3)
    hcl_h = cl + np.array([HCL_BOND, 0.0, 0.0])
    positions = [cl, hcl_h]
    atomic_numbers = [17, 1]
    # Begin from a jittered 5x5x5 grid. Its 2.6 Å spacing avoids the very
    # long rejection tail of random sequential packing at liquid density.
    grid = np.stack(np.meshgrid(*([np.arange(5)] * 3), indexing='ij'), -1)
    candidates = (grid.reshape(-1, 3) + 0.5) * (BOX / 5)
    candidates += rng.uniform(-0.10, 0.10, size=candidates.shape)
    near_cl = _minimum_distances(candidates, cl[None, :])[:, 0] < 3.15
    candidate_indices = np.flatnonzero(~near_cl)
    rng.shuffle(candidate_indices)
    oxygen_positions = candidates[candidate_indices[:N_WATER]]
    water_template = np.array(
        [
            [0.0, 0.0, 0.0],
            [WATER_OH, 0.0, 0.0],
            [WATER_OH * math.cos(WATER_ANGLE), WATER_OH * math.sin(WATER_ANGLE), 0.0],
        ]
    )
    for oxygen in oxygen_positions:
        for attempt in range(200):
            water = water_template @ _rotation(rng).T + oxygen
            prior = np.asarray(positions)
            distances = _minimum_distances(water, prior)
            # This is only a packing filter; POLAR supplies the actual forces.
            limits = np.where(np.asarray(atomic_numbers) == 17, 1.75, 1.05)
            if np.any(distances[1:] < limits[None, :]):
                continue
            positions.extend(water)
            atomic_numbers.extend([8, 1, 1])
            break
        else:
            raise RuntimeError(f'Could not orient water molecule after {attempt} tries')
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=np.asarray(positions) % BOX,
        cell=np.eye(3) * BOX,
        pbc=True,
    )
    assert len(atoms) == 2 + 3 * N_WATER
    return atoms


def _model_data(atoms: Atoms, config: dict) -> dict[str, jax.Array]:
    dtype = jnp.float64
    numbers = np.asarray(atoms.numbers)
    box = float(atoms.cell.lengths()[0])
    z_to_index = {int(z): i for i, z in enumerate(config['atomic_numbers'])}
    species = jnp.asarray([z_to_index[int(z)] for z in numbers], dtype=jnp.int32)
    n = len(numbers)
    return {
        'positions': jnp.asarray(atoms.positions, dtype=dtype),
        'node_attrs': jax.nn.one_hot(species, len(z_to_index), dtype=dtype),
        'node_attrs_index': species,
        'edge_index': jnp.zeros((2, 1), dtype=jnp.int32),
        'shifts': jnp.zeros((1, 3), dtype=dtype),
        'unit_shifts': jnp.zeros((1, 3), dtype=dtype),
        'batch': jnp.zeros(n, dtype=jnp.int32),
        'ptr': jnp.asarray([0, n], dtype=jnp.int32),
        'cell': jnp.eye(3, dtype=dtype)[None] * box,
        'pbc': jnp.asarray([[True, True, True]]),
        'head': jnp.asarray([0], dtype=jnp.int32),
        'total_charge': jnp.asarray([0.0], dtype=dtype),
        'total_spin': jnp.asarray([1.0], dtype=dtype),
        'external_field': jnp.zeros((1, 3), dtype=dtype),
    }


def _edges(positions: jax.Array, neighbors: partition.NeighborList, box: float):
    """Turn a fixed-capacity JAX-MD list into MACE directed periodic edges."""
    n = positions.shape[0]
    sender, target = neighbors.idx
    valid = (sender < n) & (target < n)
    sender = jnp.where(valid, sender, 0).astype(jnp.int32)
    receiver = jnp.where(valid, target, 0).astype(jnp.int32)
    delta = positions[receiver] - positions[sender]
    unit_shifts = -jnp.round(delta / box)
    shifts = unit_shifts * box
    # Invalid slots are moved beyond the cutoff; their radial weights vanish.
    shifts = jnp.where(valid[:, None], shifts, jnp.array([box, 0.0, 0.0]))
    unit_shifts = jnp.where(valid[:, None], unit_shifts, jnp.array([1.0, 0.0, 0.0]))
    return jnp.stack((sender, receiver)), shifts, unit_shifts


def run(args: argparse.Namespace) -> dict:
    jax.config.update('jax_enable_x64', True)
    atoms = (
        read(args.initial) if args.initial is not None else make_initial_cell(args.seed)
    )
    cell = np.asarray(atoms.cell)
    box = float(cell[0, 0])
    if not np.allclose(cell, np.eye(3) * box) or not np.all(atoms.pbc):
        raise ValueError('The initial structure must have a cubic periodic cell')
    numbers = np.asarray(atoms.numbers)
    n_water = int(np.sum(numbers == 8))
    if int(np.sum(numbers == 17)) != 1 or int(np.sum(numbers == 1)) != 2 * n_water + 1:
        raise ValueError('Expected one HCl equivalent and intact water stoichiometry')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'initial.xyz', atoms, format='extxyz')

    bundle = load_model_bundle(str(args.bundle), 'float64')
    model = nnx.merge(bundle.graphdef, bundle.params)
    if model.__class__.__name__ != 'PolarMACE':
        raise ValueError('The supplied bundle is not MACE-POLAR')
    mode, data = model.prepare_jit_data(
        _model_data(atoms, bundle.config), pbc_handling='pbc'
    )
    graphdef, model_state = nnx.split(model)
    displacement, shift_fn = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        float(bundle.config['r_max']),
        dr_threshold=0.25,
        capacity_multiplier=1.15,
        format=partition.Sparse,
    )
    positions = jnp.asarray(atoms.positions)
    neighbors = neighbor_fn.allocate(positions)

    def energy_fn(r, nbrs, params):
        edge_index, shifts, unit_shifts = _edges(r, nbrs, box)
        inputs = dict(
            data,
            positions=r,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
        )
        return nnx.merge(graphdef, params)(
            inputs, compute_force=False, pbc_handling=mode
        )['energy'][0]

    force_eval = jax.jit(jax.value_and_grad(energy_fn))
    compile_start = time.perf_counter()
    (initial_energy, initial_grad) = force_eval(positions, neighbors, model_state)
    initial_energy.block_until_ready()
    compile_seconds = time.perf_counter() - compile_start

    # The force benchmark synchronizes every call and excludes JIT compilation.
    benchmark_times = []
    for _ in range(args.benchmark_calls):
        start = time.perf_counter()
        energy, grad = force_eval(positions, neighbors, model_state)
        grad.block_until_ready()
        benchmark_times.append(time.perf_counter() - start)

    dimension = positions.size
    identity = jnp.eye(dimension, dtype=positions.dtype)
    initial_inverse_hessian = identity / args.initial_hessian
    state = (
        positions,
        neighbors,
        initial_energy,
        initial_grad,
        initial_inverse_hessian,
    )
    force_threshold = args.force_mev_per_a / 1000.0

    def bfgs_step(state, params):
        r, nbrs, energy, grad, inverse_hessian = state
        direction = -(inverse_hessian @ grad.reshape(-1)).reshape(r.shape)
        max_displacement = jnp.max(jnp.linalg.norm(direction, axis=1))
        direction = direction / jnp.maximum(1.0, max_displacement / args.maxstep)
        directional_derivative = jnp.sum(grad * direction)

        def backtrack(_, trial):
            alpha, accepted, trial_r, trial_nbrs, trial_energy, trial_grad = trial

            def evaluate():
                new_r = shift_fn(r, alpha * direction)
                new_nbrs = neighbor_fn.update(new_r, nbrs)
                new_energy, new_grad = force_eval(new_r, new_nbrs, params)
                good = jnp.isfinite(new_energy) & (
                    new_energy <= energy + 1e-4 * alpha * directional_derivative
                )
                return (
                    jnp.where(good, alpha, alpha / 2),
                    good,
                    new_r,
                    new_nbrs,
                    new_energy,
                    new_grad,
                )

            return jax.lax.cond(accepted, lambda: trial, evaluate)

        trial = (
            jnp.asarray(1.0, dtype=r.dtype),
            jnp.asarray(False),
            r,
            nbrs,
            energy,
            grad,
        )
        _, accepted, new_r, new_nbrs, new_energy, new_grad = jax.lax.fori_loop(
            0, args.max_backtracks, backtrack, trial
        )
        displacement_step = jax.vmap(displacement)(new_r, r).reshape(-1)
        gradient_change = (new_grad - grad).reshape(-1)
        curvature = jnp.dot(displacement_step, gradient_change)

        def update_hessian():
            h_y = inverse_hessian @ gradient_change
            updated = (
                inverse_hessian
                + (curvature + jnp.dot(gradient_change, h_y))
                / (curvature * curvature)
                * jnp.outer(displacement_step, displacement_step)
                - (
                    jnp.outer(h_y, displacement_step)
                    + jnp.outer(displacement_step, h_y)
                )
                / curvature
            )
            return (updated + updated.T) / 2

        new_inverse_hessian = jax.lax.cond(
            accepted & (curvature > 1e-12),
            update_hessian,
            lambda: initial_inverse_hessian,
        )
        new_state = jax.lax.cond(
            accepted,
            lambda: (new_r, new_nbrs, new_energy, new_grad, new_inverse_hessian),
            lambda: state,
        )
        return new_state, accepted

    @jax.jit
    def optimize_chunk(state, params):
        return jax.lax.scan(
            lambda carry, _: bfgs_step(carry, params), state, None, length=args.chunk
        )

    compile_start = time.perf_counter()
    compiled_chunk = optimize_chunk.lower(state, model_state).compile()
    bfgs_compile_seconds = time.perf_counter() - compile_start
    started = time.perf_counter()
    steps = 0
    converged = False
    force_history = []
    while steps < args.max_steps:
        state, accepted = compiled_chunk(state, model_state)
        positions, neighbors, _, gradient, _ = state
        force_max = float(jnp.max(jnp.linalg.norm(gradient, axis=1)))
        steps += args.chunk
        force_history.append([steps, force_max])
        print(f'step {steps}: max |F| = {1000 * force_max:.4f} meV/Å', flush=True)
        if not bool(jnp.all(accepted)):
            raise RuntimeError('BFGS line search failed to find a decreasing step')
        if bool(neighbors.did_buffer_overflow):
            raise RuntimeError('JAX-MD neighbor-list capacity overflowed')
        if not math.isfinite(force_max):
            raise RuntimeError('Non-finite force in optimization')
        if force_max <= force_threshold:
            converged = True
            break
    optimize_seconds = time.perf_counter() - started
    final_positions = np.asarray(positions) % box
    atoms.positions = final_positions
    write(args.output / 'optimized.xyz', atoms, format='extxyz')
    final_energy, final_grad = force_eval(positions, neighbors, model_state)
    final_energy.block_until_ready()
    result = {
        'atoms': len(atoms),
        'water_molecules': n_water,
        'hcl_molecules': 1,
        'box_angstrom': box,
        'steps': steps,
        'converged': converged,
        'force_threshold_mev_per_angstrom': args.force_mev_per_a,
        'initial_energy_eV': float(initial_energy),
        'final_energy_eV': float(final_energy),
        'initial_max_force_mev_per_angstrom': float(
            1000 * jnp.max(jnp.linalg.norm(initial_grad, axis=1))
        ),
        'final_max_force_mev_per_angstrom': float(
            1000 * jnp.max(jnp.linalg.norm(final_grad, axis=1))
        ),
        'compile_seconds': compile_seconds,
        'bfgs_compile_seconds': bfgs_compile_seconds,
        'force_eval_seconds': benchmark_times,
        'force_eval_median_seconds': float(np.median(benchmark_times)),
        'optimization_seconds': optimize_seconds,
        'force_history': force_history,
        'neighbor_slots': int(neighbors.idx.shape[1]),
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bundle', type=Path, default=Path('/tmp/MACE-POLAR-1-M-jax-fixed.msgpack')
    )
    parser.add_argument(
        '--output', type=Path, default=Path('examples/HCl-MACE-POLAR/run')
    )
    parser.add_argument(
        '--initial',
        type=Path,
        help='ASE-readable periodic starting structure; default is a random 72-water box',
    )
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--force-mev-per-a', type=float, default=0.01)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=5)
    parser.add_argument(
        '--maxstep', type=float, default=0.2, help='Maximum atom step in Å'
    )
    parser.add_argument(
        '--initial-hessian',
        type=float,
        default=70.0,
        help='Initial Hessian diagonal in eV/Å²',
    )
    parser.add_argument('--max-backtracks', type=int, default=12)
    parser.add_argument('--benchmark-calls', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
