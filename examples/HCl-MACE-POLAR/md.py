#!/usr/bin/env python3
"""JIT-compiled NVE MACE-POLAR trajectory from an aqueous HCl structure."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from flax import nnx
from jax_md import partition, space
from optimize import _edges, _model_data

from mace_jax.tools.bundle import load_model_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--timestep-fs', type=float, default=0.5)
    parser.add_argument('--temperature-k', type=float, default=300.0)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--benchmark-calls', type=int, default=3)
    args = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    atoms = read(args.initial)
    cell = np.asarray(atoms.cell)
    box = float(cell[0, 0])
    if not np.allclose(cell, np.eye(3) * box) or not np.all(atoms.pbc):
        raise ValueError('The initial structure must have a cubic periodic cell')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'initial.xyz', atoms, format='extxyz')

    masses_np = atoms.get_masses()
    rng = np.random.default_rng(args.seed)
    velocities_np = rng.normal(size=(len(atoms), 3)) * np.sqrt(
        units.kB * args.temperature_k / masses_np[:, None]
    )
    velocities_np -= np.sum(masses_np[:, None] * velocities_np, axis=0) / np.sum(
        masses_np
    )
    kinetic_np = 0.5 * np.sum(masses_np[:, None] * velocities_np**2)
    target_kinetic = 0.5 * (3 * len(atoms) - 3) * units.kB * args.temperature_k
    velocities_np *= np.sqrt(target_kinetic / kinetic_np)

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
    positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
    velocities = jnp.asarray(velocities_np, dtype=jnp.float64)
    masses = jnp.asarray(masses_np[:, None], dtype=jnp.float64)
    neighbors = neighbor_fn.allocate(positions)
    dt = args.timestep_fs * units.fs

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
    initial_energy, initial_grad = force_eval(positions, neighbors, model_state)
    initial_energy.block_until_ready()
    force_compile_seconds = time.perf_counter() - compile_start

    benchmark_times = []
    for _ in range(args.benchmark_calls):
        started = time.perf_counter()
        _, grad = force_eval(positions, neighbors, model_state)
        grad.block_until_ready()
        benchmark_times.append(time.perf_counter() - started)

    def md_step(carry, _, params):
        r, v, nbrs, energy, grad = carry
        half_v = v - 0.5 * dt * grad / masses
        new_r = shift_fn(r, dt * half_v)
        new_nbrs = neighbor_fn.update(new_r, nbrs)
        new_energy, new_grad = force_eval(new_r, new_nbrs, params)
        new_v = half_v - 0.5 * dt * new_grad / masses
        kinetic = 0.5 * jnp.sum(masses * new_v**2)
        max_force = jnp.max(jnp.linalg.norm(new_grad, axis=1))
        sample = (
            new_r,
            new_v,
            new_energy,
            kinetic,
            max_force,
            new_nbrs.did_buffer_overflow,
        )
        return (new_r, new_v, new_nbrs, new_energy, new_grad), sample

    @jax.jit
    def run_md(state, params):
        return jax.lax.scan(
            lambda carry, step: md_step(carry, step, params),
            state,
            None,
            length=args.steps,
        )

    initial_state = (positions, velocities, neighbors, initial_energy, initial_grad)
    compile_start = time.perf_counter()
    compiled_md = run_md.lower(initial_state, model_state).compile()
    md_compile_seconds = time.perf_counter() - compile_start
    started = time.perf_counter()
    final_state, samples = compiled_md(initial_state, model_state)
    samples[2].block_until_ready()
    md_seconds = time.perf_counter() - started

    (
        positions_series,
        velocities_series,
        potential_series,
        kinetic_series,
        force_series,
        overflow_series,
    ) = (np.asarray(x) for x in samples)
    if np.any(overflow_series):
        raise RuntimeError('JAX-MD neighbor-list capacity overflowed during MD')
    potential_initial = float(initial_energy)
    kinetic_initial = float(target_kinetic)
    total_initial = potential_initial + kinetic_initial
    total_final = float(potential_series[-1] + kinetic_series[-1])
    temperatures = 2 * kinetic_series / ((3 * len(atoms) - 3) * units.kB)

    frames = []
    for i in range(args.steps):
        if (i + 1) % args.save_interval and i + 1 != args.steps:
            continue
        frame = atoms.copy()
        frame.positions = positions_series[i]
        frame.set_velocities(velocities_series[i])
        frame.info['time_fs'] = (i + 1) * args.timestep_fs
        frame.info['potential_energy_eV'] = float(potential_series[i])
        frame.info['kinetic_energy_eV'] = float(kinetic_series[i])
        frame.info['temperature_K'] = float(temperatures[i])
        frames.append(frame)
    write(args.output / 'trajectory.xyz', frames, format='extxyz')
    final_atoms = frames[-1]
    write(args.output / 'final.xyz', final_atoms, format='extxyz')

    result = {
        'atoms': len(atoms),
        'box_angstrom': box,
        'steps': args.steps,
        'timestep_fs': args.timestep_fs,
        'duration_fs': args.steps * args.timestep_fs,
        'ensemble': 'NVE',
        'seed': args.seed,
        'initial_structure': str(args.initial),
        'initial_temperature_k': args.temperature_k,
        'final_temperature_k': float(temperatures[-1]),
        'initial_potential_eV': potential_initial,
        'final_potential_eV': float(potential_series[-1]),
        'initial_total_eV': total_initial,
        'final_total_eV': total_final,
        'total_energy_drift_meV_per_atom': 1000
        * (total_final - total_initial)
        / len(atoms),
        'initial_max_force_mev_per_angstrom': float(
            1000 * jnp.max(jnp.linalg.norm(initial_grad, axis=1))
        ),
        'final_max_force_mev_per_angstrom': float(1000 * force_series[-1]),
        'force_compile_seconds': force_compile_seconds,
        'md_compile_seconds': md_compile_seconds,
        'force_eval_seconds': benchmark_times,
        'force_eval_median_seconds': float(np.median(benchmark_times)),
        'md_seconds': md_seconds,
        'mean_md_step_seconds': md_seconds / args.steps,
        'neighbor_slots': int(final_state[2].idx.shape[1]),
        'samples': [
            {
                'step': i + 1,
                'time_fs': (i + 1) * args.timestep_fs,
                'potential_eV': float(potential_series[i]),
                'kinetic_eV': float(kinetic_series[i]),
                'temperature_k': float(temperatures[i]),
                'max_force_mev_per_angstrom': float(1000 * force_series[i]),
            }
            for i in range(args.steps)
        ],
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != 'samples'}, indent=2
        )
    )


if __name__ == '__main__':
    main()
