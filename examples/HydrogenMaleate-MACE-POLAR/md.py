#!/usr/bin/env python3
"""Restartable 330 K periodic hydrogen maleate MD: all POLAR or POLAR/TIP3P."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from common import graph_edges, initialize_model
from flax import nnx
from jax.scipy.special import erf, erfc

from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.modules.polar_periodic import ewald_neutralizing_background_energy

COULOMB = FIELD_CONSTANT / (4 * math.pi)
KJ_MOL_TO_EV = 1 / 96.48533212331002
TIP3P_Q = (-0.834, 0.417, 0.417)
TIP3P_SIGMA = 3.150752406575124  # Å, OpenMM amber14/tip3p.xml
TIP3P_EPS = 0.635968 * KJ_MOL_TO_EV
OH_R0 = 0.9572
OH_K = 462750.4 * KJ_MOL_TO_EV / 100
HOH_THETA0 = 1.82421813418
HOH_K = 836.8 * KJ_MOL_TO_EV
# Approximate AMBER-like cross Lennard-Jones types for PubChem atom order:
# four carboxyl oxygens, four carbons, two vinyl hydrogens, acidic hydrogen.
SOLUTE_SIGMA = np.array([
    3.066473387839048, 2.959921901149463, 2.959921901149463,
    2.959921901149463, 3.399669508423535, 3.399669508423535,
    3.399669508423535, 3.399669508423535, 2.471353044121301,
    2.471353044121301, 10.0,
])
SOLUTE_EPS = np.array([
    0.8803136, 0.87864, 0.87864, 0.87864, 0.359824, 0.359824,
    0.4577296, 0.4577296, 0.0656888, 0.0656888, 0.0,
]) * KJ_MOL_TO_EV


def minimum_image(delta, box):
    return delta - box * jnp.round(delta / box)


def generate_velocities(atoms, temperature, seed):
    masses = atoms.get_masses()
    rng = np.random.default_rng(seed)
    velocities = rng.normal(size=(len(atoms), 3)) * np.sqrt(
        units.kB * temperature / masses[:, None]
    )
    velocities -= np.sum(masses[:, None] * velocities, axis=0) / masses.sum()
    kinetic = 0.5 * np.sum(masses[:, None] * velocities**2)
    target = 0.5 * (3 * len(atoms) - 3) * units.kB * temperature
    return velocities * np.sqrt(target / kinetic)


def build_energy(atoms, bundle, mode_name, alpha, kmax, qm_water_count,
                 restraint_radius, restraint_k):
    box = float(atoms.cell[0, 0])
    n_qm = len(atoms) if mode_name == 'polar' else 11 + 3*qm_water_count
    qm_atoms = atoms[:n_qm]
    box, periodic_mode, data, graphdef, params, neighbor_fn, shift_fn = (
        initialize_model(qm_atoms, bundle)
    )
    if mode_name == 'mlmm' and (len(atoms) - n_qm) % 3:
        raise ValueError('MM region must contain complete water triplets')
    mm_count = (len(atoms) - n_qm) // 3
    mm_charges = jnp.asarray(TIP3P_Q * mm_count, dtype=jnp.float64)
    mm_molecule = np.repeat(np.arange(mm_count), 3)
    real_mask = jnp.asarray(np.triu(mm_molecule[:, None] != mm_molecule[None], k=1))
    mm_o_pairs = np.triu_indices(mm_count, k=1)
    exception_i = jnp.asarray([3*i+a for i in range(mm_count) for a in (0, 0, 1)])
    exception_j = jnp.asarray([3*i+b for i in range(mm_count) for b in (1, 2, 2)])
    k_axis = np.arange(-kmax, kmax + 1)
    k_int = np.stack(np.meshgrid(k_axis, k_axis, k_axis, indexing='ij'), -1).reshape(-1, 3)
    k_int = k_int[np.any(k_int != 0, axis=1)]
    k_vectors = jnp.asarray(2 * math.pi / box * k_int, dtype=jnp.float64)
    k2 = jnp.sum(k_vectors**2, axis=1)
    k_weights = jnp.exp(-k2 / (4 * alpha**2)) / k2
    cross_sigma = jnp.asarray((SOLUTE_SIGMA + TIP3P_SIGMA) / 2)
    cross_epsilon = jnp.asarray(np.sqrt(SOLUTE_EPS * TIP3P_EPS))

    def lj(r, sigma, epsilon):
        x6 = (sigma / r) ** 6
        return 4 * epsilon * (x6*x6 - x6)

    def mm_terms(mm_r, qm_r):
        waters = mm_r.reshape(-1, 3, 3)
        oh1 = minimum_image(waters[:, 1] - waters[:, 0], box)
        oh2 = minimum_image(waters[:, 2] - waters[:, 0], box)
        r1, r2 = jnp.linalg.norm(oh1, axis=-1), jnp.linalg.norm(oh2, axis=-1)
        cosine = jnp.sum(oh1 * oh2, axis=-1) / (r1*r2)
        angle = jnp.arccos(jnp.clip(cosine, -1+1e-12, 1-1e-12))
        bonded = 0.5*OH_K*jnp.sum((r1-OH_R0)**2 + (r2-OH_R0)**2)
        bonded += 0.5*HOH_K*jnp.sum((angle-HOH_THETA0)**2)
        mm_o = waters[:, 0]
        oo = minimum_image(mm_o[mm_o_pairs[0]] - mm_o[mm_o_pairs[1]], box)
        mm_lj = jnp.sum(lj(jnp.linalg.norm(oo, axis=-1), TIP3P_SIGMA, TIP3P_EPS))
        solute_cross = minimum_image(qm_r[:11, None] - mm_o[None], box)
        solute_cross_r = jnp.linalg.norm(solute_cross, axis=-1)
        cross_lj = jnp.sum(lj(
            solute_cross_r, cross_sigma[:, None], cross_epsilon[:, None]
        ))
        # QM water O - MM water O uses the same TIP3P Lennard-Jones type.
        qm_water_o = qm_r[11::3]
        water_cross = minimum_image(qm_water_o[:, None] - mm_o[None], box)
        cross_lj += jnp.sum(lj(
            jnp.linalg.norm(water_cross, axis=-1), TIP3P_SIGMA, TIP3P_EPS
        ))

        delta = minimum_image(mm_r[:, None] - mm_r[None], box)
        distances = jnp.sqrt(jnp.sum(delta*delta, axis=-1) + 1e-24)
        safe = jnp.where(real_mask, distances, 1.0)
        pair_q = mm_charges[:, None] * mm_charges[None]
        real = COULOMB*jnp.sum(jnp.where(real_mask, pair_q*erfc(alpha*safe)/safe, 0.0))
        phase = k_vectors @ mm_r.T
        c, s = jnp.cos(phase) @ mm_charges, jnp.sin(phase) @ mm_charges
        reciprocal = COULOMB*2*math.pi/box**3*jnp.sum(k_weights*(c*c+s*s))
        self_energy = -COULOMB*alpha/math.sqrt(math.pi)*jnp.sum(mm_charges**2)
        background = ewald_neutralizing_background_energy(
            jnp.sum(mm_charges), box**3, alpha
        )
        excluded = minimum_image(mm_r[exception_i] - mm_r[exception_j], box)
        excluded_r = jnp.linalg.norm(excluded, axis=-1)
        exceptions = -COULOMB*jnp.sum(
            mm_charges[exception_i]*mm_charges[exception_j]
            *erf(alpha*excluded_r)/excluded_r
        )
        return (
            bonded + mm_lj + cross_lj + real + reciprocal + self_energy
            + background + exceptions
        )

    def boundary_restraint(qm_r):
        # Restrain only first-shell water oxygens, with zero energy/force
        # inside the shell. The closest solute oxygen moves with the solute.
        delta = minimum_image(
            qm_r[11::3, None] - qm_r[jnp.asarray([0, 1, 2, 3])][None], box
        )
        nearest = jnp.linalg.norm(delta, axis=-1).min(axis=-1)
        excess = jnp.maximum(nearest - restraint_radius, 0.0)
        return 0.5 * restraint_k * jnp.sum(excess**2)

    def energy_fn(r, neighbors, model_params):
        qm_r = r[:n_qm]
        edge_index, shifts, unit_shifts = graph_edges(qm_r, neighbors, box)
        inputs = dict(
            data, positions=qm_r, edge_index=edge_index,
            shifts=shifts, unit_shifts=unit_shifts,
        )
        if mode_name == 'mlmm':
            inputs['mm_positions'] = r[n_qm:]
            inputs['mm_charges'] = mm_charges
        polar_energy = nnx.merge(graphdef, model_params)(
            inputs, compute_force=False, pbc_handling=periodic_mode
        )['energy'][0]
        return (polar_energy if mode_name == 'polar' else
                polar_energy + mm_terms(r[n_qm:], qm_r) + boundary_restraint(qm_r))

    return jax.jit(jax.value_and_grad(energy_fn)), params, neighbor_fn, shift_fn, n_qm


def write_chunk(args, atoms, position_series, energy_series, kinetic_series, start_step):
    frames = []
    samples = []
    box = float(atoms.cell[0, 0])
    for i in range(len(energy_series)):
        step = start_step + i + 1
        positions = position_series[i]
        h_to_o = positions[:4] - positions[10]
        h_to_o -= box*np.rint(h_to_o/box)
        h_to_o = np.linalg.norm(h_to_o, axis=1)
        left_h = float(h_to_o[:2].min())
        right_h = float(h_to_o[2:].min())
        proton_coordinate = right_h - left_h
        if step % args.save_interval == 0 or step == args.steps:
            frame = atoms.copy()
            frame.positions = positions
            frame.info['time_fs'] = step * args.timestep_fs
            frame.info['potential_energy_eV'] = float(energy_series[i])
            frame.info['kinetic_energy_eV'] = float(kinetic_series[i])
            frame.info['left_carboxyl_H_angstrom'] = left_h
            frame.info['right_carboxyl_H_angstrom'] = right_h
            frame.info['proton_coordinate_angstrom'] = proton_coordinate
            frames.append(frame)
        if step % args.report_interval == 0 or step == args.steps:
            samples.append({
                'step': step,
                'time_ps': step*args.timestep_fs/1000,
                'potential_eV': float(energy_series[i]),
                'temperature_K': float(2*kinetic_series[i]/((3*len(atoms)-3)*units.kB)),
                'left_carboxyl_H_angstrom': left_h,
                'right_carboxyl_H_angstrom': right_h,
                'proton_coordinate_angstrom': proton_coordinate,
            })
    if frames:
        tmp = args.output / 'chunks' / f'{start_step+len(energy_series):08d}.tmp.xyz'
        final = args.output / 'chunks' / f'{start_step+len(energy_series):08d}.xyz'
        write(tmp, frames, format='extxyz')
        tmp.replace(final)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['polar', 'mlmm'], required=True)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--timestep-fs', type=float, default=0.5)
    parser.add_argument('--temperature-k', type=float, default=330.0)
    parser.add_argument('--thermostat-tau-ps', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--chunk', type=int, default=200)
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--report-interval', type=int, default=200)
    parser.add_argument('--ewald-alpha', type=float, default=0.5)
    parser.add_argument('--ewald-kmax', type=int, default=6)
    parser.add_argument('--qm-water-count', type=int, default=0)
    parser.add_argument('--restraint-radius', type=float, default=4.2)
    parser.add_argument('--restraint-k', type=float, default=0.2)
    args = parser.parse_args()
    if args.steps % args.chunk or args.chunk % args.save_interval:
        raise ValueError('steps must divide by chunk and chunk by save-interval')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'chunks').mkdir(exist_ok=True)
    atoms = read(args.initial)
    if len(atoms) != 167:
        raise ValueError('Expected 11 hydrogen maleate atoms and 52 waters')
    if args.mode == 'mlmm':
        if not 1 <= args.qm_water_count < 52:
            raise ValueError('ML/MM requires a nonempty first-shell QM water region')
        if int(atoms.info.get('qm_water_count', -1)) != args.qm_water_count:
            raise ValueError('QM water count differs from initial structure metadata')
    initial_h_to_o = np.linalg.norm(atoms.positions[:4]-atoms.positions[10], axis=1)
    if not initial_h_to_o[:2].min() < initial_h_to_o[2:].min():
        raise ValueError('Initial hydrogen maleate proton is not on the left carboxyl')
    if int(atoms.info.get('qm_charge', atoms.info.get('charge', 0))) != -1:
        raise ValueError('Hydrogen maleate QM region must have charge -1')
    write(args.output / 'initial.xyz', atoms, format='extxyz')
    force_eval, params, neighbor_fn, shift_fn, n_qm = build_energy(
        atoms, args.bundle, args.mode, args.ewald_alpha, args.ewald_kmax,
        args.qm_water_count, args.restraint_radius, args.restraint_k
    )
    checkpoint = args.output / 'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            start_step = int(saved['step'])
            positions = jnp.asarray(saved['positions'])
            velocities = jnp.asarray(saved['velocities'])
        print(f'Restarting from step {start_step}', flush=True)
    else:
        start_step = 0
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        velocities = jnp.asarray(generate_velocities(atoms, args.temperature_k, args.seed))
    masses = jnp.asarray(atoms.get_masses()[:, None], dtype=jnp.float64)
    neighbors = neighbor_fn.allocate(positions[:n_qm])
    compile_start = time.perf_counter()
    initial_energy, initial_grad = force_eval(positions, neighbors, params)
    initial_energy.block_until_ready()
    print(f'Compiled force in {time.perf_counter()-compile_start:.1f} s; initial energy {float(initial_energy):.6f} eV', flush=True)
    dt = args.timestep_fs * units.fs
    c = math.exp(-args.timestep_fs / (1000*args.thermostat_tau_ps))
    noise_scale = jnp.sqrt((1-c*c)*units.kB*args.temperature_k/masses)

    def step(carry, absolute_step):
        r, v, nbrs, energy, grad = carry
        v = v - 0.5*dt*grad/masses
        r = shift_fn(r, 0.5*dt*v)
        noise = jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(args.seed+1), absolute_step), r.shape)
        v = c*v + noise_scale*noise
        r = shift_fn(r, 0.5*dt*v)
        nbrs = neighbor_fn.update(r[:n_qm], nbrs)
        new_energy, new_grad = force_eval(r, nbrs, params)
        v = v - 0.5*dt*new_grad/masses
        kinetic = 0.5*jnp.sum(masses*v*v)
        return (r, v, nbrs, new_energy, new_grad), (
            r, new_energy, kinetic, nbrs.did_buffer_overflow,
        )

    @jax.jit
    def run_chunk(state, offset):
        return jax.lax.scan(step, state, offset+jnp.arange(args.chunk))

    state = (positions, velocities, neighbors, initial_energy, initial_grad)
    compile_start = time.perf_counter()
    compiled = run_chunk.lower(state, jnp.asarray(start_step, dtype=jnp.int32)).compile()
    print(f'Compiled MD chunk in {time.perf_counter()-compile_start:.1f} s', flush=True)
    started = time.perf_counter()
    for offset in range(start_step, args.steps, args.chunk):
        state, output = compiled(state, jnp.asarray(offset, dtype=jnp.int32))
        positions_np, potential_np, kinetic_np, overflow_np = (np.asarray(x) for x in output)
        if np.any(overflow_np) or not np.all(np.isfinite(potential_np)):
            raise RuntimeError(f'Nonfinite energy or neighbor overflow in chunk {offset}')
        samples = write_chunk(args, atoms, positions_np, potential_np, kinetic_np, offset)
        new_step = offset + args.chunk
        tmp = args.output / 'checkpoint.tmp.npz'
        np.savez(tmp, step=new_step, positions=np.asarray(state[0]), velocities=np.asarray(state[1]))
        tmp.replace(checkpoint)
        progress = {
            'mode': args.mode, 'step': new_step,
            'time_ps': new_step*args.timestep_fs/1000,
            'target_time_ps': args.steps*args.timestep_fs/1000,
            'elapsed_seconds_this_process': time.perf_counter()-started,
            'last_sample': samples[-1] if samples else None,
            'cpu_only': os.environ.get('JAX_PLATFORMS') == 'cpu',
            'qm_water_count': args.qm_water_count if args.mode == 'mlmm' else 52,
            'restraint_radius_angstrom': args.restraint_radius if args.mode == 'mlmm' else None,
            'restraint_k_eV_per_angstrom2': args.restraint_k if args.mode == 'mlmm' else None,
        }
        (args.output / 'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print(json.dumps(progress), flush=True)
    final = atoms.copy()
    final.positions = np.asarray(state[0])
    write(args.output / 'final.xyz', final, format='extxyz')
    print(f'Completed {args.steps*args.timestep_fs/1000:.3f} ps', flush=True)


if __name__ == '__main__':
    main()
