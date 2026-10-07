#!/usr/bin/env python3
"""Restartable, fixed-volume CSVR REMD using the user's JAX-MD branch.

Trajectories retain walker order; ensemble_indices records their temperatures.
The trajectory cap includes the pilot. Model weights are shared runtime inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write

from md import build_energy
from simulation_settings import generate_velocities
from mace_jax.modules.langevin import kinetic_energy
from mace_jax.modules.replica_exchange import PolarReplicaState, make_exchange
from mace_jax.tools.physical_constants import OPENMM_BOLTZMANN


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def save_state(path, state):
    leaves = jax.tree.leaves(state)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **{f'leaf_{i}': np.asarray(x) for i, x in enumerate(leaves)})
    tmp.replace(path)


def restore_state(path, template):
    leaves, tree = jax.tree.flatten(template)
    with np.load(path, allow_pickle=False) as data:
        restored = [data[f'leaf_{i}'] for i in range(len(leaves))]
    if any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(leaves, restored)):
        raise ValueError('Checkpoint shapes or dtypes differ from the current state')
    return tree.unflatten([jnp.asarray(x) for x in restored])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['polar', 'mlmm', 'mechanical'], required=True)
    parser.add_argument('--mechanical-qm-charges', type=Path)
    parser.add_argument('--temperatures', nargs='+', type=float,
                        default=[300, 320, 340, 360, 380, 400])
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--chunk', type=int, default=200)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--timestep-fs', type=float, default=.5)
    parser.add_argument('--thermostat-tau-ps', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    temps = np.asarray(args.temperatures)
    if (len(temps) < 2 or not np.all(np.isfinite(temps)) or np.any(temps <= 0)
            or np.any(np.diff(temps) <= 0)):
        parser.error('Temperatures must be positive and strictly increasing')
    if (args.steps <= 0 or args.chunk <= 0 or args.stride <= 0
            or args.timestep_fs <= 0 or args.thermostat_tau_ps <= 0
            or args.save_interval <= 0 or args.chunk % args.stride
            or args.save_interval % args.stride or args.chunk % args.save_interval
            or args.steps % args.chunk):
        parser.error('Steps/chunk must align with exchange and saving intervals')
    if args.steps * args.timestep_fs / 1000 > 100 + 1e-10:
        parser.error('The total trajectory, including pilot steps, cannot exceed 100 ps')
    if args.mode == 'mechanical' and args.mechanical_qm_charges is None:
        parser.error('Mechanical embedding requires fixed QM charges')
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_compilation_cache_dir', str(args.output.parent / 'jit-cache'))
    devices = jax.devices()
    if args.require_gpu and any(x.platform != 'gpu' for x in devices):
        raise RuntimeError(f'GPU required; got {devices}')
    import jax_md
    branch_dir = Path(jax_md.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ['git', '-C', str(branch_dir), 'rev-parse', 'HEAD'], text=True).strip()
    settings = dict(
        mode=args.mode, temperatures_K=temps.tolist(), timestep_fs=args.timestep_fs,
        thermostat='CSVR', thermostat_tau_ps=args.thermostat_tau_ps,
        fix_center_of_mass=True, exchange_stride=args.stride,
        exchange_pairs='all unordered pairs once per sweep (JAX-MD implementation)',
        chunk=args.chunk, save_interval=args.save_interval, seed=args.seed,
        initial_sha256=digest(args.initial), bundle_sha256=digest(args.bundle),
        charges_sha256=digest(args.mechanical_qm_charges)
        if args.mechanical_qm_charges else None,
        jax_md_revision=revision, jax_version=jax.__version__,
        qm_water_count=52 if args.mode == 'polar' else 25,
        restraint=None if args.mode == 'polar' else 'FIRES',
        restraint_k_eV_A2=None if args.mode == 'polar' else 21.68205212090047,
        compatibility_profile='generalized-accurate', pme_mesh_spacing_A=.5,
        pme_assignment_order=8, mm_cutoff_A=5.8,
        ewald_alpha_A_inv=.5232507342707401, neighbor_skin_A=.25,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / 'settings.json'
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError('Run settings changed; use a separate output directory')
    save_json(config, settings)
    atoms = read(args.initial)
    print(json.dumps({'phase': 'build', 'settings': settings, 'devices': str(devices)}), flush=True)
    force_eval, params, neighbor_fn, shift_fn, n_qm = build_energy(
        atoms, args.bundle, args.mode, .5232507342707401, 4, 25,
        'fires', 4.2, 21.68205212090047,
        mechanical_qm_charges=(
            np.asarray(json.loads(args.mechanical_qm_charges.read_text())['charges'])
            if args.mechanical_qm_charges else None),
        nonbonded_cutoff=5.8, neighbor_skin=.25, reciprocal_method='pme',
        pme_mesh_spacing=.5, pme_assignment_order=8,
        compatibility_profile='generalized-accurate', pme_error_tolerance=5e-5,
    )
    r = jnp.asarray(atoms.positions)
    masses = jnp.asarray(atoms.get_masses())[:, None]
    neighbors = neighbor_fn.allocate(r)
    started = time.perf_counter()
    energy, gradient = force_eval(r, neighbors, params)
    energy.block_until_ready()
    if bool(neighbors.did_buffer_overflow) or not np.isfinite(float(energy)):
        raise RuntimeError('Invalid initial energy or neighbor capacity')
    print(f'Initial force compiled in {time.perf_counter()-started:.1f} s; energy={float(energy):.9f} eV', flush=True)
    count = len(temps)
    broadcast = lambda x: jnp.broadcast_to(x, (count,) + x.shape)
    velocities = jnp.asarray(np.stack([
        generate_velocities(atoms, t, args.seed + i, 'generalized-accurate')
        * units.fs * 1000 for i, t in enumerate(temps)
    ]))
    replicas = PolarReplicaState(
        broadcast(r), velocities, jax.tree.map(broadcast, neighbors),
        broadcast(energy), broadcast(gradient),
        jax.random.split(jax.random.PRNGKey(args.seed + 10000), count),
        jnp.zeros(count, dtype=bool), jnp.zeros(count, dtype=jnp.int32),
    )
    init, apply = make_exchange(
        force_eval, neighbor_fn, shift_fn, masses, neighbors, temps,
        timestep_ps=args.timestep_fs / 1000,
        thermostat_tau_ps=args.thermostat_tau_ps, stride=args.stride,
    )
    state = init(jax.random.PRNGKey(args.seed + 20000), replicas)
    checkpoint = args.output / 'checkpoint.npz'
    if checkpoint.exists():
        state = restore_state(checkpoint, state)
    start_step = int(state.step)
    if not np.all(np.asarray(state.replicas.md_steps) == start_step):
        raise ValueError('Exchange and MD step counters disagree')
    (args.output / 'chunks').mkdir(exist_ok=True)

    @jax.jit
    def run_chunk(state, model_params):
        def event(state, unused):
            state = jax.lax.fori_loop(
                0, args.stride, lambda _, s: apply(s, model_params=model_params), state)
            kinetic = jax.vmap(lambda v: kinetic_energy(v, masses))(state.replicas.velocity)
            return state, (state.step, state.replicas.position,
                           state.replicas.potential_energy, kinetic,
                           state.ensemble_indices, state.replicas.invalid)
        return jax.lax.scan(event, state, None, length=args.chunk // args.stride)

    print('Compiling batched REMD chunk...', flush=True)
    started = time.perf_counter()
    lowered = run_chunk.lower(state, params)
    targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"',
                                   lowered.compiler_ir('hlo').as_hlo_text())))
    if args.require_gpu and 'uniform_1d_cuda' not in targets:
        raise RuntimeError('Expected cuEquivariance CUDA kernels in the batched model')
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - started
    memory = compiled.memory_analysis()
    save_json(args.output / 'compilation.json', dict(
        seconds=compile_seconds, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        custom_call_targets=targets,
        argument_bytes=memory.argument_size_in_bytes,
        temporary_bytes=memory.temp_size_in_bytes,
        output_bytes=memory.output_size_in_bytes,
        alias_bytes=memory.alias_size_in_bytes,
    ))
    print(f'Compiled REMD in {compile_seconds:.1f} s', flush=True)
    started = time.perf_counter()
    accepts_start = np.asarray(state.swap_accepts)
    history = []
    for offset in range(start_step, args.steps, args.chunk):
        tick = time.perf_counter()
        next_state, output = compiled(state, params)
        steps, positions, potentials, kinetics, labels, invalid = jax.tree.map(np.asarray, output)
        elapsed = time.perf_counter() - tick
        if (np.any(invalid) or not all(np.all(np.isfinite(x)) for x in
                                    (positions, potentials, kinetics))):
            raise RuntimeError(f'Nonfinite state or neighbor overflow after step {offset}; last valid checkpoint retained')
        if offset == start_step:
            single_neighbors = jax.tree.map(lambda x: x[0], next_state.replicas.neighbors)
            single_energy, single_gradient = force_eval(
                next_state.replicas.position[0], single_neighbors, params)
            energy_error = abs(float(single_energy - next_state.replicas.potential_energy[0]))
            force_error = float(jnp.max(jnp.abs(single_gradient - next_state.replicas.gradient[0])))
            save_json(args.output / 'batch-verification.json', dict(
                step=int(next_state.step), energy_error_eV=energy_error,
                max_force_component_error_eV_A=force_error))
            if energy_error > 2e-7 or force_error > 1e-6:
                raise RuntimeError(f'Batched/single force disagreement: {energy_error=} {force_error=}')
        state = next_state
        selected = steps % args.save_interval == 0
        filename = args.output / 'chunks' / f'{int(state.step):08d}.npz'
        tmp = filename.with_suffix('.tmp.npz')
        np.savez(tmp, steps=steps[selected], positions=positions[selected],
                 potential_energy=potentials[selected], kinetic_energy=kinetics[selected],
                 ensemble_indices=labels[selected], swap_steps=steps, swap_indices=labels,
                 swap_accepts=np.asarray(state.swap_accepts))
        tmp.replace(filename)
        save_state(checkpoint, state)
        sweeps = int(state.step) // args.stride
        counts = np.asarray(state.swap_accepts)
        # Every unordered temperature pair is attempted once per exchange sweep.
        adjacent = counts[np.arange(count-1), np.arange(1, count)] / sweeps
        recent_events = args.chunk // args.stride
        previous_counts = accepts_start if not history else history[-1]['swap_accepts']
        recent_adjacent = ((counts - previous_counts)[np.arange(count-1), np.arange(1, count)] / recent_events)
        report = dict(step=int(state.step), time_ps=int(state.step)*args.timestep_fs/1000,
                      target_ps=args.steps*args.timestep_fs/1000, replicas=count,
                      n_qm=n_qm, n_mm=len(atoms)-n_qm, compile_seconds=compile_seconds,
                      elapsed_seconds_this_process=time.perf_counter()-started,
                      chunk_seconds=elapsed, ms_per_step_all_replicas=1000*elapsed/args.chunk,
                      aggregate_ns_day=count*args.chunk*args.timestep_fs*0.0864/elapsed,
                      ns_day_per_replica=args.chunk*args.timestep_fs*0.0864/elapsed,
                      swap_accepts=counts.tolist(), adjacent_acceptance=adjacent.tolist(),
                      recent_adjacent_acceptance=recent_adjacent.tolist(),
                      ensemble_indices=labels[-1].tolist(),
                      last_temperature_K=(2*kinetics[-1]/((3*len(atoms)-3)*OPENMM_BOLTZMANN)).tolist())
        save_json(args.output / 'progress.json', report)
        with (args.output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(report) + '\n')
        history = [dict(swap_accepts=counts)]
        print(json.dumps(report), flush=True)
    for i, positions in enumerate(np.asarray(state.replicas.position)):
        final = atoms.copy()
        final.positions = positions
        final.info.update(walker=i, temperature_K=float(temps[int(state.ensemble_indices[i])]),
                          time_ps=int(state.step)*args.timestep_fs/1000)
        write(args.output / f'final-walker-{i}.xyz', final)
    if int(state.step)*args.timestep_fs/1000 >= 100-1e-10:
        (args.output / 'COMPLETE').write_text('100 ps per replica, including pilot\n')
    print(f'Finished at {int(state.step)*args.timestep_fs/1000:g} ps per replica', flush=True)


if __name__ == '__main__':
    main()
