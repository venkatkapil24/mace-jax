#!/usr/bin/env python3
"""Restartable 330 K hydrogen maleate MD with full or embedded MACE-POLAR."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from common import graph_edges, initialize_model
from flax import nnx
from jax.scipy.special import erf, erfc
from jax_md import partition, space

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


class EnergyNeighbors(NamedTuple):
    """Fixed-capacity neighbor lists carried through the compiled MD loop."""

    model: Any
    nonbonded: Any | None

    @property
    def did_buffer_overflow(self):
        overflow = self.model.did_buffer_overflow
        if self.nonbonded is not None:
            overflow = overflow | self.nonbonded.did_buffer_overflow
        return overflow


class EnergyNeighborFns:
    """Allocate and update the model and MM nonbonded neighbor lists together."""

    def __init__(self, model, nonbonded, n_qm):
        self.model = model
        self.nonbonded = nonbonded
        self.n_qm = n_qm

    def allocate(self, positions):
        model_neighbors = self.model.allocate(positions[:self.n_qm])
        nonbonded_neighbors = (
            None if self.nonbonded is None else self.nonbonded.allocate(positions)
        )
        return EnergyNeighbors(model_neighbors, nonbonded_neighbors)

    def update(self, positions, neighbors):
        model_neighbors = self.model.update(
            positions[:self.n_qm], neighbors.model
        )
        nonbonded_neighbors = (
            None
            if self.nonbonded is None
            else self.nonbonded.update(positions, neighbors.nonbonded)
        )
        return EnergyNeighbors(model_neighbors, nonbonded_neighbors)


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


def sparse_nonbonded_terms(
    positions,
    pair_index,
    n_qm,
    mm_charges,
    alpha,
    cutoff,
    box,
    cross_sigma,
    cross_epsilon,
    mechanical_qm_charges=None,
):
    """Evaluate real-space MM terms from an unordered JAX-MD edge list."""
    n_atoms = positions.shape[0]
    n_mm = mm_charges.shape[0]
    atom_i, atom_j = pair_index
    valid = (atom_i < n_atoms) & (atom_j < n_atoms)
    atom_i = jnp.where(valid, atom_i, 0)
    atom_j = jnp.where(valid, atom_j, 0)
    delta = minimum_image(positions[atom_i] - positions[atom_j], box)
    distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-24)
    within_cutoff = valid & (distance < cutoff)

    i_is_qm = atom_i < n_qm
    j_is_qm = atom_j < n_qm
    i_mm = jnp.clip(atom_i - n_qm, 0, n_mm - 1)
    j_mm = jnp.clip(atom_j - n_qm, 0, n_mm - 1)

    # Flexible TIP3P excludes all intramolecular nonbonded pairs. Reciprocal
    # exclusions are restored separately by the Ewald exception term.
    mm_pair = (
        within_cutoff
        & ~i_is_qm
        & ~j_is_qm
        & ((i_mm // 3) != (j_mm // 3))
    )
    mm_safe = jnp.where(mm_pair, distance, 1.0)
    mm_real = COULOMB * jnp.sum(
        jnp.where(
            mm_pair,
            mm_charges[i_mm] * mm_charges[j_mm]
            * erfc(alpha * mm_safe) / mm_safe,
            0.0,
        )
    )
    mm_oo = mm_pair & ((i_mm % 3) == 0) & ((j_mm % 3) == 0)
    mm_lj_distance = jnp.where(mm_oo, distance, 1.0)
    mm_lj_x6 = (TIP3P_SIGMA / mm_lj_distance) ** 6
    mm_lj = jnp.sum(
        jnp.where(mm_oo, 4 * TIP3P_EPS * (mm_lj_x6**2 - mm_lj_x6), 0.0)
    )

    cross_pair = within_cutoff & (i_is_qm ^ j_is_qm)
    qm_index = jnp.where(i_is_qm, atom_i, atom_j)
    mm_index = jnp.where(i_is_qm, j_mm, i_mm)
    mm_is_oxygen = (mm_index % 3) == 0

    solute_pair = cross_pair & mm_is_oxygen & (qm_index < 11)
    solute_index = jnp.clip(qm_index, 0, 10)
    solute_distance = jnp.where(solute_pair, distance, 1.0)
    solute_x6 = (cross_sigma[solute_index] / solute_distance) ** 6
    cross_lj = jnp.sum(
        jnp.where(
            solute_pair,
            4 * cross_epsilon[solute_index] * (solute_x6**2 - solute_x6),
            0.0,
        )
    )

    qm_water_oxygen = (qm_index >= 11) & (((qm_index - 11) % 3) == 0)
    water_pair = cross_pair & mm_is_oxygen & qm_water_oxygen
    water_distance = jnp.where(water_pair, distance, 1.0)
    water_x6 = (TIP3P_SIGMA / water_distance) ** 6
    cross_lj += jnp.sum(
        jnp.where(
            water_pair,
            4 * TIP3P_EPS * (water_x6**2 - water_x6),
            0.0,
        )
    )

    mechanical_real = jnp.asarray(0.0, dtype=positions.dtype)
    if mechanical_qm_charges is not None:
        qm_charge_index = jnp.clip(qm_index, 0, n_qm - 1)
        cross_safe = jnp.where(cross_pair, distance, 1.0)
        mechanical_real = COULOMB * jnp.sum(
            jnp.where(
                cross_pair,
                mechanical_qm_charges[qm_charge_index]
                * mm_charges[mm_index]
                * erfc(alpha * cross_safe) / cross_safe,
                0.0,
            )
        )
    return mm_real, mm_lj, cross_lj, mechanical_real


def ordered_pairs_from_dense_neighbors(neighbor_index, n_atoms):
    """Flatten a JAX-MD dense list while retaining each pair exactly once."""
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


def build_energy(atoms, bundle, mode_name, alpha, kmax, qm_water_count,
                 restraint_radius, restraint_k, mechanical_qm_charges=None,
                 nonbonded_cutoff=9.0, neighbor_skin=0.25):
    box = float(atoms.cell[0, 0])
    n_qm = len(atoms) if mode_name == 'polar' else 11 + 3*qm_water_count
    qm_atoms = atoms[:n_qm]
    box, periodic_mode, data, graphdef, params, neighbor_fn, shift_fn = (
        initialize_model(qm_atoms, bundle)
    )
    if mode_name != 'polar' and (len(atoms) - n_qm) % 3:
        raise ValueError('MM region must contain complete water triplets')
    mm_count = (len(atoms) - n_qm) // 3
    mm_charges = jnp.asarray(TIP3P_Q * mm_count, dtype=jnp.float64)
    exception_i = jnp.asarray([3*i+a for i in range(mm_count) for a in (0, 0, 1)])
    exception_j = jnp.asarray([3*i+b for i in range(mm_count) for b in (1, 2, 2)])
    k_axis = np.arange(-kmax, kmax + 1)
    k_int = np.stack(np.meshgrid(k_axis, k_axis, k_axis, indexing='ij'), -1).reshape(-1, 3)
    k_int = k_int[np.any(k_int != 0, axis=1)]
    k_vectors = jnp.asarray(2 * math.pi / box * k_int, dtype=jnp.float64)
    k2 = jnp.sum(k_vectors**2, axis=1)
    k_weights = jnp.exp(-k2 / (4 * alpha**2)) / k2
    if mode_name == 'mechanical':
        if mechanical_qm_charges is None:
            raise ValueError('Mechanical embedding requires fixed QM charges')
        charge_array = np.asarray(mechanical_qm_charges, dtype=float)
        if charge_array.shape != (n_qm,):
            raise ValueError(
                f'Expected {n_qm} mechanical QM charges, got '
                f'{charge_array.shape}'
            )
        if not np.all(np.isfinite(charge_array)) or not np.isclose(
            charge_array.sum(), -1.0, atol=1e-10
        ):
            raise ValueError('Mechanical QM charges must be finite and sum to -1')
        mechanical_qm_charges = jnp.asarray(charge_array, dtype=jnp.float64)
    cross_sigma = jnp.asarray((SOLUTE_SIGMA + TIP3P_SIGMA) / 2)
    cross_epsilon = jnp.asarray(np.sqrt(SOLUTE_EPS * TIP3P_EPS))
    if mode_name == 'polar':
        energy_neighbor_fn = EnergyNeighborFns(neighbor_fn, None, n_qm)
    else:
        if nonbonded_cutoff is None:
            nonbonded_cutoff = min(9.0, 0.49 * box)
        if nonbonded_cutoff <= 0 or neighbor_skin < 0:
            raise ValueError('Nonbonded cutoff must be positive and skin nonnegative')
        displacement, _ = space.periodic(box)
        nonbonded_neighbor_fn = partition.neighbor_list(
            displacement,
            box,
            nonbonded_cutoff,
            dr_threshold=neighbor_skin,
            capacity_multiplier=1.3,
            format=partition.Dense,
        )
        energy_neighbor_fn = EnergyNeighborFns(
            neighbor_fn, nonbonded_neighbor_fn, n_qm
        )

    def mm_internal_terms(mm_r):
        waters = mm_r.reshape(-1, 3, 3)
        oh1 = minimum_image(waters[:, 1] - waters[:, 0], box)
        oh2 = minimum_image(waters[:, 2] - waters[:, 0], box)
        r1, r2 = jnp.linalg.norm(oh1, axis=-1), jnp.linalg.norm(oh2, axis=-1)
        cosine = jnp.sum(oh1 * oh2, axis=-1) / (r1*r2)
        angle = jnp.arccos(jnp.clip(cosine, -1+1e-12, 1-1e-12))
        bonded = 0.5*OH_K*jnp.sum((r1-OH_R0)**2 + (r2-OH_R0)**2)
        bonded += 0.5*HOH_K*jnp.sum((angle-HOH_THETA0)**2)
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
        return bonded + reciprocal + self_energy + background + exceptions

    def mechanical_cross_electrostatics(qm_r, mm_r):
        """Fixed-charge QM--MM Ewald cross term used by mechanical embedding."""
        qm_phase = k_vectors @ qm_r.T
        mm_phase = k_vectors @ mm_r.T
        qm_c = jnp.cos(qm_phase) @ mechanical_qm_charges
        qm_s = jnp.sin(qm_phase) @ mechanical_qm_charges
        mm_c = jnp.cos(mm_phase) @ mm_charges
        mm_s = jnp.sin(mm_phase) @ mm_charges
        reciprocal = (
            COULOMB * 4 * math.pi / box**3
            * jnp.sum(k_weights * (qm_c * mm_c + qm_s * mm_s))
        )
        q_qm = jnp.sum(mechanical_qm_charges)
        q_mm = jnp.sum(mm_charges)
        background = (
            ewald_neutralizing_background_energy(q_qm + q_mm, box**3, alpha)
            - ewald_neutralizing_background_energy(q_qm, box**3, alpha)
            - ewald_neutralizing_background_energy(q_mm, box**3, alpha)
        )
        return reciprocal + background

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
        edge_index, shifts, unit_shifts = graph_edges(
            qm_r, neighbors.model, box
        )
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
        if mode_name == 'polar':
            return polar_energy
        mm_real, mm_lj, cross_lj, mechanical_real = sparse_nonbonded_terms(
            r,
            ordered_pairs_from_dense_neighbors(
                neighbors.nonbonded.idx, r.shape[0]
            ),
            n_qm,
            mm_charges,
            alpha,
            nonbonded_cutoff,
            box,
            cross_sigma,
            cross_epsilon,
            mechanical_qm_charges if mode_name == 'mechanical' else None,
        )
        energy = (
            polar_energy
            + mm_internal_terms(r[n_qm:])
            + mm_real
            + mm_lj
            + cross_lj
            + boundary_restraint(qm_r)
        )
        if mode_name == 'mechanical':
            energy += mechanical_real + mechanical_cross_electrostatics(
                qm_r, r[n_qm:]
            )
        return energy

    return (
        jax.jit(jax.value_and_grad(energy_fn)), params, energy_neighbor_fn,
        shift_fn, n_qm,
    )


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
    parser.add_argument('--mode', choices=['polar', 'mlmm', 'mechanical'], required=True)
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
    parser.add_argument('--nonbonded-cutoff', type=float)
    parser.add_argument('--neighbor-skin', type=float, default=0.25)
    parser.add_argument('--qm-water-count', type=int, default=0)
    parser.add_argument('--restraint-radius', type=float, default=4.2)
    parser.add_argument('--restraint-k', type=float, default=0.2)
    parser.add_argument('--mechanical-qm-charges', type=Path)
    args = parser.parse_args()
    if args.steps % args.chunk or args.chunk % args.save_interval:
        raise ValueError('steps must divide by chunk and chunk by save-interval')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'chunks').mkdir(exist_ok=True)
    atoms = read(args.initial)
    if len(atoms) != 167:
        raise ValueError('Expected 11 hydrogen maleate atoms and 52 waters')
    if args.nonbonded_cutoff is None:
        args.nonbonded_cutoff = min(9.0, 0.49 * float(atoms.cell[0, 0]))
    if args.mode != 'polar':
        if not 1 <= args.qm_water_count < 52:
            raise ValueError('Mixed embedding requires a nonempty first-shell QM water region')
        if int(atoms.info.get('qm_water_count', -1)) != args.qm_water_count:
            raise ValueError('QM water count differs from initial structure metadata')
    mechanical_qm_charges = None
    if args.mode == 'mechanical':
        if args.mechanical_qm_charges is None:
            raise ValueError('--mechanical-qm-charges is required in mechanical mode')
        charge_data = json.loads(args.mechanical_qm_charges.read_text())
        mechanical_qm_charges = np.asarray(charge_data['charges'], dtype=float)
        expected_numbers = np.asarray(atoms.numbers[:11 + 3*args.qm_water_count])
        if not np.array_equal(charge_data.get('atomic_numbers'), expected_numbers):
            raise ValueError('Mechanical charge-file atom order differs from structure')
    initial_h_to_o = np.linalg.norm(atoms.positions[:4]-atoms.positions[10], axis=1)
    if not initial_h_to_o[:2].min() < initial_h_to_o[2:].min():
        raise ValueError('Initial hydrogen maleate proton is not on the left carboxyl')
    if int(atoms.info.get('qm_charge', atoms.info.get('charge', 0))) != -1:
        raise ValueError('Hydrogen maleate QM region must have charge -1')
    write(args.output / 'initial.xyz', atoms, format='extxyz')
    force_eval, params, neighbor_fn, shift_fn, n_qm = build_energy(
        atoms, args.bundle, args.mode, args.ewald_alpha, args.ewald_kmax,
        args.qm_water_count, args.restraint_radius, args.restraint_k,
        mechanical_qm_charges, args.nonbonded_cutoff, args.neighbor_skin,
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
    neighbors = neighbor_fn.allocate(positions)
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
        nbrs = neighbor_fn.update(r, nbrs)
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
            'qm_water_count': args.qm_water_count if args.mode != 'polar' else 52,
            'restraint_radius_angstrom': args.restraint_radius if args.mode != 'polar' else None,
            'restraint_k_eV_per_angstrom2': args.restraint_k if args.mode != 'polar' else None,
            'nonbonded_cutoff_angstrom': (
                args.nonbonded_cutoff if args.mode != 'polar' else None
            ),
            'neighbor_skin_angstrom': (
                args.neighbor_skin if args.mode != 'polar' else None
            ),
            'mechanical_qm_charges': (
                str(args.mechanical_qm_charges) if args.mode == 'mechanical' else None
            ),
        }
        (args.output / 'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print(json.dumps(progress), flush=True)
    final = atoms.copy()
    final.positions = np.asarray(state[0])
    write(args.output / 'final.xyz', final, format='extxyz')
    print(f'Completed {args.steps*args.timestep_fs/1000:.3f} ps', flush=True)


if __name__ == '__main__':
    main()
