#!/usr/bin/env python3
"""Measure force calls and actual compiled MD for generalized PME settings.

All cases use the same updated force conventions, Langevin-middle integrator,
initial velocities and thermostat seed. Only cutoff, alpha and mesh change.
Compilation, force-only calls and MD windows are measured separately. Each
timed GPU operation is synchronized. MD timings include device-to-host copies
of the per-step positions, energies, kinetic energies and overflow flags, as
in the production driver; disk writes are outside the timed interval.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
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

from mace_jax.modules.langevin import kinetic_energy, middle_step


def gpu_status():
    return subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,utilization.gpu,clocks.sm',
            '--format=csv',
        ],
        text=True,
    ).strip()


def statistics(values):
    a = np.asarray(values)
    return dict(
        median=float(np.median(a)),
        mean=float(a.mean()),
        minimum=float(a.min()),
        maximum=float(a.max()),
        std=float(a.std(ddof=1)),
        samples=a.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    source = root / 'Examples/HydrogenMaleate-MoleculeOnly/structure'
    parser.add_argument(
        '--initial', type=Path, default=source / 'initial-100ps-full-ml.xyz'
    )
    parser.add_argument(
        '--charges', type=Path, default=source / 'molecule-fixed-charges.json'
    )
    parser.add_argument(
        '--bundle',
        type=Path,
        default=root / 'artifacts/MACE-POLAR-1-M-jax-fixed.msgpack',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=['mechanical', 'mlmm', 'polar'],
        default=['mechanical', 'mlmm'],
    )
    parser.add_argument('--force-calls', type=int, default=20)
    parser.add_argument('--windows', type=int, default=5)
    parser.add_argument('--chunk', type=int, default=100)
    parser.add_argument(
        '--cases',
        nargs='+',
        choices=['original_numerics', 'openmm_matched'],
        help='Select already defined settings; useful for a backend comparison',
    )
    parser.add_argument('--require-cueq', action='store_true')
    parser.add_argument(
        '--audit-only',
        action='store_true',
        help='Compile and inspect without MD or timings',
    )
    parser.add_argument('--qm-water-count', type=int, default=0)
    parser.add_argument('--restraint-mode', choices=['none', 'fires'], default='none')
    parser.add_argument('--restraint-k', type=float, default=0.0)
    parser.add_argument(
        '--wait-for-pid',
        type=int,
        action='append',
        default=[],
        help='Compile now, but wait for these processes before measuring',
    )
    parser.add_argument(
        '--reference-forces',
        type=Path,
        help='Compare with saved nonaccelerated initial forces before running MD',
    )
    args = parser.parse_args()
    if min(args.force_calls, args.chunk) <= 0 or args.windows < 2:
        raise ValueError('Need positive call/chunk counts and at least two windows')
    jax.config.update('jax_enable_x64', True)
    devices = jax.devices()
    if any(d.platform != 'gpu' for d in devices):
        raise RuntimeError(f'GPU benchmark requires GPU, found {devices}')
    args.output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.initial)
    full_ml = args.modes == ['polar']
    if 'polar' in args.modes and not full_ml:
        raise ValueError('Benchmark full ML in a separate invocation')
    if full_ml:
        args.qm_water_count = 52
        if args.restraint_mode != 'none' or args.restraint_k != 0:
            raise ValueError('Full ML has no region restraint')
    charges_data = None if full_ml else json.loads(args.charges.read_text())
    charges = None if full_ml else np.asarray(charges_data['charges'])
    n_qm = len(atoms) if full_ml else 11 + 3 * args.qm_water_count
    if not 0 <= args.qm_water_count <= (52 if full_ml else 51) or args.restraint_k < 0:
        raise ValueError('Invalid QM water count or restraint strength')
    if len(atoms) != 167 or (
        not full_ml
        and (
            charges.shape != (n_qm,)
            or not np.array_equal(atoms.numbers[:n_qm], charges_data['atomic_numbers'])
            or int(atoms.info.get('qm_water_count', -1)) != args.qm_water_count
        )
    ):
        raise ValueError('Structure, partition metadata and fixed charges disagree')
    box = float(atoms.cell[0, 0])
    matched_alpha = np.sqrt(-np.log(2 * 5e-5)) / 5.8
    cases = [
        ('original_numerics', 0.49 * box, 0.5, 0.5),
        ('openmm_matched', 5.8, matched_alpha, 0.5),
    ]
    if args.cases:
        cases = [case for case in cases if case[0] in args.cases]
    report = dict(
        initial=str(args.initial),
        bundle=str(args.bundle),
        atoms=len(atoms),
        ml_atoms=n_qm,
        mm_atoms=len(atoms) - n_qm,
        qm_water_count=args.qm_water_count,
        restraint_mode=args.restraint_mode,
        restraint_k_eV_per_angstrom2=args.restraint_k,
        box_angstrom=box,
        dtype='float64',
        jax_version=jax.__version__,
        python=platform.python_version(),
        devices=[str(d) for d in devices],
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        temperature_K=330.0,
        timestep_fs=0.5,
        friction_per_ps=1.0,
        seed=20260922,
        integrator='langevin-middle',
        force_calls_per_window=args.force_calls,
        md_steps_per_window=args.chunk,
        measured_windows=args.windows,
        convention='generalized-accurate force conventions for every case; only numerical electrostatics settings differ',
        md_timing='compiled integration + neighbor updates + force evaluations + host copies; excludes compilation and disk IO',
        results=[],
        gpu_status_start=gpu_status(),
    )
    output = args.output / 'benchmark.json'

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    save()
    masses = jnp.asarray(atoms.get_masses()[:, None])
    initial_positions = jnp.asarray(atoms.positions)
    velocities = jnp.asarray(
        generate_velocities(atoms, 330.0, 20260922, 'generalized-accurate')
        * units.fs
        * 1000.0
    )
    key = jax.random.PRNGKey(20260922)
    for mode in args.modes:
        for name, cutoff, alpha, spacing in cases:
            print(f'START {mode} {name}', flush=True)
            started = time.perf_counter()
            force, params, neighbors_fn, shift, nq = build_energy(
                atoms,
                args.bundle,
                mode,
                alpha,
                6,
                args.qm_water_count,
                args.restraint_mode,
                4.2,
                args.restraint_k,
                charges if mode == 'mechanical' else None,
                nonbonded_cutoff=cutoff,
                pme_mesh_spacing=spacing,
                pme_assignment_order=8,
                compatibility_profile='generalized-accurate',
            )
            nbrs = neighbors_fn.allocate(initial_positions)
            if bool(nbrs.did_buffer_overflow):
                raise RuntimeError('Initial neighbor overflow')
            row = dict(
                mode=mode,
                name=name,
                cutoff_angstrom=cutoff,
                alpha_per_angstrom=float(alpha),
                mesh=list(neighbors_fn.model_mesh_shape),
                spline_order=8,
                gpu_before=gpu_status(),
                setup_seconds=time.perf_counter() - started,
            )
            start = time.perf_counter()
            ef, grad = force(initial_positions, nbrs, params)
            jax.block_until_ready((ef, grad))
            row['force_first_call_seconds'] = time.perf_counter() - start
            if not np.isfinite(float(ef)) or not np.all(np.isfinite(np.asarray(grad))):
                raise RuntimeError('Nonfinite initial force')
            if args.require_cueq:
                hlo = (
                    force.lower(initial_positions, nbrs, params)
                    .compiler_ir(dialect='hlo')
                    .as_hlo_text()
                )
                targets = sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))
                row['custom_call_targets'] = targets
                if not any(
                    'cuequivariance' in target.lower() or target == 'uniform_1d_cuda'
                    for target in targets
                ):
                    raise RuntimeError(
                        f'No cuEquivariance CUDA call found; targets={targets}'
                    )
            if args.reference_forces:
                reference = np.load(
                    args.reference_forces / f'{mode}-{name}-initial-forces.npz'
                )
                diff = -np.asarray(grad) - reference['forces_eV_per_angstrom']
                row['backend_force_rms_difference_meV_per_angstrom'] = float(
                    1000 * np.sqrt(np.mean(diff**2))
                )
                row['backend_max_atom_force_difference_meV_per_angstrom'] = float(
                    1000 * np.linalg.norm(diff, axis=1).max()
                )
                row['backend_energy_difference_meV'] = float(
                    1000 * (float(ef) - reference['energy_eV'])
                )
                if (
                    np.sqrt(np.mean(diff**2)) > 1e-6
                    or abs(float(ef) - reference['energy_eV']) > 1e-6
                ):
                    raise RuntimeError(f'Backend parity failed: {row}')
            print(
                f'FORCE READY {mode} {name}: {row["force_first_call_seconds"]:.1f}s',
                flush=True,
            )
            np.savez(
                args.output / f'{mode}-{name}-initial-forces.npz',
                energy_eV=float(ef),
                forces_eV_per_angstrom=-np.asarray(grad),
            )

            def step(
                state,
                absolute_step,
                shift_fn=shift,
                neighbor_fn=neighbors_fn,
                force_fn=force,
                model_params=None,
            ):
                r, v, nb, energy, g = state
                noise = jax.random.normal(
                    jax.random.fold_in(key, absolute_step), r.shape, dtype=r.dtype
                )
                r, v = middle_step(
                    r,
                    v,
                    -g,
                    masses,
                    noise,
                    timestep_ps=0.0005,
                    temperature=330.0,
                    friction_ps=1.0,
                    shift=shift_fn,
                )
                nb = neighbor_fn.update(r, nb)
                energy, g = force_fn(r, nb, model_params)
                return (r, v, nb, energy, g), (
                    r,
                    energy,
                    kinetic_energy(v, masses),
                    nb.did_buffer_overflow,
                )

            @jax.jit
            def run_chunk(state, offset, model_params, step_fn=step):
                return jax.lax.scan(
                    lambda carry, sample: step_fn(
                        carry, sample, model_params=model_params
                    ),
                    state,
                    offset + jnp.arange(args.chunk, dtype=jnp.int32),
                )

            state = (initial_positions, velocities, nbrs, ef, grad)
            start = time.perf_counter()
            compiled = run_chunk.lower(
                state, jnp.asarray(0, dtype=jnp.int32), params
            ).compile()
            row['md_compile_seconds'] = time.perf_counter() - start
            print(
                f'COMPILED {mode} {name}: force {row["force_first_call_seconds"]:.1f}s; MD {row["md_compile_seconds"]:.1f}s',
                flush=True,
            )
            row['parameter_leaf_count'] = len(jax.tree_util.tree_leaves(params))
            row['parameter_bytes_as_runtime_inputs'] = sum(
                leaf.nbytes for leaf in jax.tree_util.tree_leaves(params)
            )
            row['model_neighbor_capacity'] = list(nbrs.model.idx.shape)
            row['model_valid_edges'] = int(
                np.asarray(jnp.sum(jnp.all(nbrs.model.idx < nq, axis=0)))
            )
            row['gpu_after_compile'] = gpu_status()
            if args.audit_only:
                row['audit_only_no_md'] = True
                report['results'].append(row)
                save()
                print(f'AUDIT_COMPLETE {json.dumps(row)}', flush=True)
                return
            if args.wait_for_pid:
                print(f'WAIT before measurement: {args.wait_for_pid}', flush=True)
                while True:
                    active = []
                    for pid in args.wait_for_pid:
                        try:
                            os.kill(pid, 0)
                            active.append(pid)
                        except ProcessLookupError:
                            pass
                    if not active:
                        break
                    time.sleep(15)
                row['waited_for_pids'] = args.wait_for_pid
            for _ in range(5):
                jax.block_until_ready(force(initial_positions, nbrs, params))
            force_windows = []
            for _ in range(args.windows):
                start = time.perf_counter()
                for _ in range(args.force_calls):
                    jax.block_until_ready(force(initial_positions, nbrs, params))
                force_windows.append((time.perf_counter() - start) / args.force_calls)
            row['force_seconds_per_call'] = statistics(force_windows)

            md_windows = []
            positions = []
            potential = []
            kinetic = []
            for window in range(args.windows + 1):
                offset = jnp.asarray(window * args.chunk, dtype=jnp.int32)
                start = time.perf_counter()
                state, outputs = compiled(state, offset, params)
                r_np, e_np, ke_np, overflow = [np.asarray(x) for x in outputs]
                elapsed = time.perf_counter() - start
                if np.any(overflow) or not all(
                    np.all(np.isfinite(x))
                    for x in (
                        r_np,
                        e_np,
                        ke_np,
                        np.asarray(state[1]),
                        np.asarray(state[4]),
                    )
                ):
                    raise RuntimeError(
                        f'Nonfinite MD or neighbor overflow: {mode} {name} {window}'
                    )
                if window:
                    md_windows.append(elapsed / args.chunk)
                positions.append(r_np)
                potential.append(e_np)
                kinetic.append(ke_np)
                print(
                    f'WINDOW {mode} {name} {window}: {1000 * elapsed / args.chunk:.3f} ms/step',
                    flush=True,
                )
            row['md_seconds_per_step'] = statistics(md_windows)
            row['throughput_ns_per_day'] = (
                0.0005 * 86.4 / row['md_seconds_per_step']['median']
            )
            row['simulated_ps'] = (args.windows + 1) * args.chunk * 0.0005
            row['all_finite_no_neighbor_overflow'] = True
            row['gpu_after'] = gpu_status()
            np.savez(
                args.output / f'{mode}-{name}-short-md.npz',
                positions_angstrom=np.concatenate(positions),
                potential_eV=np.concatenate(potential),
                kinetic_eV=np.concatenate(kinetic),
            )
            final = atoms.copy()
            final.positions = np.asarray(state[0])
            write(args.output / f'{mode}-{name}-final.xyz', final, format='extxyz')
            report['results'].append(row)
            save()
            print(f'RESULT {json.dumps(row)}', flush=True)
            del (
                compiled,
                run_chunk,
                step,
                state,
                outputs,
                force,
                params,
                neighbors_fn,
                nbrs,
                shift,
            )
            jax.clear_caches()
            gc.collect()
    report['gpu_status_end'] = gpu_status()
    save()
    print('BENCHMARK COMPLETE', flush=True)


if __name__ == '__main__':
    main()
