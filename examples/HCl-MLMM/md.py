#!/usr/bin/env python3
"""Fixed-partition, periodic MACE-POLAR/AMBER TIP3P NVE trajectory."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase import units
from ase.io import read, write
from flax import nnx
from jax.scipy.special import erf, erfc
from jax_md import partition, space

from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.tools.bundle import load_model_bundle

COULOMB = FIELD_CONSTANT / (4 * math.pi)  # eV Å / e²
KJ_MOL_TO_EV = 1 / 96.48533212331002
TIP3P_Q = (-0.834, 0.417, 0.417)
TIP3P_SIGMA_O = 3.150752406575124  # Å, amber14/tip3p.xml
TIP3P_EPSILON_O = 0.635968 * KJ_MOL_TO_EV  # eV
CL_SIGMA = 4.477656957373345  # Å, Joung-Cheatham Cl- in amber14/tip3p.xml
CL_EPSILON = 0.148912744 * KJ_MOL_TO_EV  # eV
BOND_R0 = 0.9572  # Å
BOND_K = 462750.4 * KJ_MOL_TO_EV / 100  # eV/Å²
ANGLE_R0 = 1.82421813418  # rad
ANGLE_K = 836.8 * KJ_MOL_TO_EV  # eV/rad²


def minimum_image(delta: jax.Array, box: float) -> jax.Array:
    return delta - box * jnp.round(delta / box)


def select_partition(atoms, shell_cutoff: float):
    """Keep complete first-shell waters and the excess-proton hydronium in QM."""
    numbers = np.asarray(atoms.numbers)
    oxygens = np.flatnonzero(numbers == 8)
    hydrogens = np.flatnonzero(numbers == 1)
    chlorides = np.flatnonzero(numbers == 17)
    if len(chlorides) != 1 or len(oxygens) != 63 or len(hydrogens) != 127:
        raise ValueError('Expected the 63-water, one-HCl cell')
    box = float(atoms.cell.lengths()[0])
    h_o = atoms.positions[hydrogens, None, :] - atoms.positions[oxygens][None]
    h_o -= box * np.rint(h_o / box)
    nearest_oxygen = np.linalg.norm(h_o, axis=-1).argmin(axis=1)
    counts = np.bincount(nearest_oxygen, minlength=len(oxygens))
    if sorted(counts.tolist()) != [2] * 62 + [3]:
        raise ValueError('Water/proton assignment is ambiguous')
    hydronium_oxygen = int(oxygens[np.flatnonzero(counts == 3)[0]])
    cl = int(chlorides[0])
    cl_o = atoms.positions[oxygens] - atoms.positions[cl]
    cl_o -= box * np.rint(cl_o / box)
    shell_distances = np.linalg.norm(cl_o, axis=1)
    shell_oxygens = set(oxygens[shell_distances < shell_cutoff].tolist())
    qm_oxygens = shell_oxygens | {hydronium_oxygen}
    qm_atoms = {cl}
    mm_waters = []
    for local_o, oxygen in enumerate(oxygens):
        hs = hydrogens[nearest_oxygen == local_o].tolist()
        if int(oxygen) in qm_oxygens:
            qm_atoms.update([int(oxygen), *hs])
        else:
            mm_waters.append([int(oxygen), *hs])
    qm_index = np.asarray(sorted(qm_atoms), dtype=np.int32)
    mm_waters = np.asarray(mm_waters, dtype=np.int32)
    mm_index = mm_waters.reshape(-1)
    if len(qm_index) + len(mm_index) != len(atoms):
        raise AssertionError('Partition lost or duplicated atoms')
    return (
        qm_index,
        mm_index,
        mm_waters,
        {
            'chloride_index': cl,
            'hydronium_oxygen_index': hydronium_oxygen,
            'shell_cutoff_angstrom': shell_cutoff,
            'first_shell_oxygen_indices': sorted(shell_oxygens),
            'first_shell_water_count': len(shell_oxygens),
            'qm_oxygen_indices': sorted(qm_oxygens),
            'qm_atom_indices': qm_index.tolist(),
            'mm_atom_indices': mm_index.tolist(),
            'mm_water_count': len(mm_waters),
        },
    )


def model_data(atoms, atomic_numbers):
    z_index = {int(z): i for i, z in enumerate(atomic_numbers)}
    species = jnp.asarray([z_index[int(z)] for z in atoms.numbers], dtype=jnp.int32)
    n = len(atoms)
    dtype = jnp.float64
    return {
        'positions': jnp.asarray(atoms.positions, dtype=dtype),
        'node_attrs': jax.nn.one_hot(species, len(z_index), dtype=dtype),
        'node_attrs_index': species,
        'edge_index': jnp.zeros((2, 1), dtype=jnp.int32),
        'shifts': jnp.zeros((1, 3), dtype=dtype),
        'unit_shifts': jnp.zeros((1, 3), dtype=dtype),
        'batch': jnp.zeros(n, dtype=jnp.int32),
        'ptr': jnp.asarray([0, n], dtype=jnp.int32),
        'cell': jnp.asarray(atoms.cell.array, dtype=dtype)[None],
        'pbc': jnp.asarray([[True, True, True]]),
        'head': jnp.asarray([0], dtype=jnp.int32),
        'total_charge': jnp.asarray([0.0], dtype=dtype),
        'total_spin': jnp.asarray([1.0], dtype=dtype),
        'external_field': jnp.zeros((1, 3), dtype=dtype),
    }


def graph_edges(positions, neighbors, box):
    n = positions.shape[0]
    sender, receiver = neighbors.idx
    valid = (sender < n) & (receiver < n)
    sender = jnp.where(valid, sender, 0).astype(jnp.int32)
    receiver = jnp.where(valid, receiver, 0).astype(jnp.int32)
    delta = positions[receiver] - positions[sender]
    unit = -jnp.round(delta / box)
    shifts = unit * box
    shifts = jnp.where(valid[:, None], shifts, jnp.array([box, 0.0, 0.0]))
    unit = jnp.where(valid[:, None], unit, jnp.array([1.0, 0.0, 0.0]))
    return jnp.stack((sender, receiver)), shifts, unit


def tip3p_bonded(mm_positions, box):
    waters = mm_positions.reshape(-1, 3, 3)
    oh1 = minimum_image(waters[:, 1] - waters[:, 0], box)
    oh2 = minimum_image(waters[:, 2] - waters[:, 0], box)
    r1 = jnp.linalg.norm(oh1, axis=-1)
    r2 = jnp.linalg.norm(oh2, axis=-1)
    cosine = jnp.sum(oh1 * oh2, axis=-1) / (r1 * r2)
    angle = jnp.arccos(jnp.clip(cosine, -1 + 1e-12, 1 - 1e-12))
    return 0.5 * BOND_K * jnp.sum(
        (r1 - BOND_R0) ** 2 + (r2 - BOND_R0) ** 2
    ) + 0.5 * ANGLE_K * jnp.sum((angle - ANGLE_R0) ** 2)


def lennard_jones(r, sigma, epsilon):
    x6 = (sigma / r) ** 6
    return 4 * epsilon * (x6 * x6 - x6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shell-cutoff', type=float, default=3.8)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--timestep-fs', type=float, default=0.5)
    parser.add_argument('--temperature-k', type=float, default=300.0)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--save-interval', type=int, default=5)
    parser.add_argument('--ewald-alpha', type=float, default=0.5)
    parser.add_argument('--ewald-kmax', type=int, default=6)
    args = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    atoms = read(args.initial)
    box = float(atoms.cell.array[0, 0])
    if not np.allclose(atoms.cell.array, np.eye(3) * box) or not np.all(atoms.pbc):
        raise ValueError('Expected a cubic periodic cell')
    qm_index_np, mm_index_np, mm_waters, partition_info = select_partition(
        atoms, args.shell_cutoff
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'partition.json').write_text(
        json.dumps(partition_info, indent=2) + '\n'
    )
    write(args.output / 'initial.xyz', atoms, format='extxyz')
    qm_index = jnp.asarray(qm_index_np, dtype=jnp.int32)
    mm_index = jnp.asarray(mm_index_np, dtype=jnp.int32)
    qm_numbers = np.asarray(atoms.numbers)[qm_index_np]
    qm_oxygen_local = jnp.asarray(np.flatnonzero(qm_numbers == 8), dtype=jnp.int32)
    qm_cl_local = int(np.flatnonzero(qm_numbers == 17)[0])
    mm_charges = jnp.asarray(TIP3P_Q * len(mm_waters), dtype=jnp.float64)
    mm_molecule = np.repeat(np.arange(len(mm_waters)), 3)
    real_mask = jnp.asarray(np.triu(mm_molecule[:, None] != mm_molecule[None, :], k=1))
    mm_o_pairs = np.triu_indices(len(mm_waters), k=1)
    exception_i = jnp.asarray(
        [3 * i + a for i in range(len(mm_waters)) for a in (0, 0, 1)],
        dtype=jnp.int32,
    )
    exception_j = jnp.asarray(
        [3 * i + b for i in range(len(mm_waters)) for b in (1, 2, 2)],
        dtype=jnp.int32,
    )
    k_range = np.arange(-args.ewald_kmax, args.ewald_kmax + 1)
    k_integer = np.stack(np.meshgrid(k_range, k_range, k_range, indexing='ij'), -1)
    k_integer = k_integer.reshape(-1, 3)
    k_integer = k_integer[np.any(k_integer != 0, axis=1)]
    k_vectors = jnp.asarray(2 * math.pi / box * k_integer, dtype=jnp.float64)
    k2 = jnp.sum(k_vectors**2, axis=1)
    k_weights = jnp.exp(-k2 / (4 * args.ewald_alpha**2)) / k2

    bundle = load_model_bundle(str(args.bundle), 'float64')
    model = nnx.merge(bundle.graphdef, bundle.params)
    if model.__class__.__name__ != 'PolarMACE':
        raise ValueError('The supplied bundle is not MACE-POLAR')
    qm_atoms = atoms[qm_index_np]
    mode, data = model.prepare_jit_data(
        model_data(qm_atoms, bundle.config['atomic_numbers']), pbc_handling='pbc'
    )
    graphdef, params = nnx.split(model)
    displacement, shift_fn = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        float(bundle.config['r_max']),
        dr_threshold=0.25,
        capacity_multiplier=1.3,
        format=partition.Sparse,
    )
    positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
    neighbors = neighbor_fn.allocate(positions[qm_index])

    def mm_ewald(mm_r):
        delta = minimum_image(mm_r[:, None] - mm_r[None, :], box)
        distance = jnp.sqrt(jnp.sum(delta * delta, axis=-1) + 1e-24)
        safe_r = jnp.where(real_mask, distance, 1.0)
        charge_pairs = mm_charges[:, None] * mm_charges[None, :]
        real = COULOMB * jnp.sum(
            jnp.where(
                real_mask,
                charge_pairs * erfc(args.ewald_alpha * safe_r) / safe_r,
                0.0,
            )
        )
        phase = k_vectors @ mm_r.T
        c = jnp.cos(phase) @ mm_charges
        s = jnp.sin(phase) @ mm_charges
        reciprocal = (
            COULOMB * 2 * math.pi / box**3 * jnp.sum(k_weights * (c * c + s * s))
        )
        self_energy = (
            -COULOMB * args.ewald_alpha / math.sqrt(math.pi) * jnp.sum(mm_charges**2)
        )
        exception_delta = minimum_image(mm_r[exception_i] - mm_r[exception_j], box)
        exception_r = jnp.linalg.norm(exception_delta, axis=-1)
        exceptions = -COULOMB * jnp.sum(
            mm_charges[exception_i]
            * mm_charges[exception_j]
            * erf(args.ewald_alpha * exception_r)
            / exception_r
        )
        return real + reciprocal + self_energy + exceptions

    def short_range(qm_r, mm_r):
        mm_o = mm_r.reshape(-1, 3, 3)[:, 0]
        oo_delta = minimum_image(mm_o[mm_o_pairs[0]] - mm_o[mm_o_pairs[1]], box)
        mm_lj = jnp.sum(
            lennard_jones(
                jnp.linalg.norm(oo_delta, axis=-1),
                TIP3P_SIGMA_O,
                TIP3P_EPSILON_O,
            )
        )
        cross_oo = minimum_image(qm_r[qm_oxygen_local, None] - mm_o[None], box)
        cross_lj = jnp.sum(
            lennard_jones(
                jnp.linalg.norm(cross_oo, axis=-1),
                TIP3P_SIGMA_O,
                TIP3P_EPSILON_O,
            )
        )
        cross_cl_o = minimum_image(qm_r[qm_cl_local] - mm_o, box)
        cross_lj += jnp.sum(
            lennard_jones(
                jnp.linalg.norm(cross_cl_o, axis=-1),
                (TIP3P_SIGMA_O + CL_SIGMA) / 2,
                math.sqrt(TIP3P_EPSILON_O * CL_EPSILON),
            )
        )
        return tip3p_bonded(mm_r, box) + mm_lj + cross_lj

    def total_energy(r, nbrs, model_params):
        qm_r = r[qm_index]
        mm_r = r[mm_index]
        edge_index, shifts, unit_shifts = graph_edges(qm_r, nbrs, box)
        inputs = dict(
            data,
            positions=qm_r,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
            mm_positions=mm_r,
            mm_charges=mm_charges,
        )
        qm_embedded = nnx.merge(graphdef, model_params)(
            inputs, compute_force=False, pbc_handling=mode
        )['energy'][0]
        return qm_embedded + short_range(qm_r, mm_r) + mm_ewald(mm_r)

    force_eval = jax.jit(jax.value_and_grad(total_energy))
    compile_start = time.perf_counter()
    initial_potential, initial_grad = force_eval(positions, neighbors, params)
    initial_potential.block_until_ready()
    force_compile_seconds = time.perf_counter() - compile_start
    if not bool(jnp.isfinite(initial_potential) & jnp.all(jnp.isfinite(initial_grad))):
        raise RuntimeError('Non-finite initial POLAR/MM energy or force')

    masses_np = atoms.get_masses()
    rng = np.random.default_rng(args.seed)
    velocity_np = rng.normal(size=(len(atoms), 3)) * np.sqrt(
        units.kB * args.temperature_k / masses_np[:, None]
    )
    velocity_np -= np.sum(masses_np[:, None] * velocity_np, axis=0) / np.sum(masses_np)
    kinetic_np = 0.5 * np.sum(masses_np[:, None] * velocity_np**2)
    initial_kinetic = 0.5 * (3 * len(atoms) - 3) * units.kB * args.temperature_k
    velocity_np *= np.sqrt(initial_kinetic / kinetic_np)
    velocity = jnp.asarray(velocity_np)
    masses = jnp.asarray(masses_np[:, None])
    dt = args.timestep_fs * units.fs

    def step(carry, _, model_params):
        r, v, nbrs, energy, grad = carry
        half_v = v - 0.5 * dt * grad / masses
        new_r = shift_fn(r, dt * half_v)
        new_nbrs = neighbor_fn.update(new_r[qm_index], nbrs)
        new_energy, new_grad = force_eval(new_r, new_nbrs, model_params)
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
    def run_md(state, model_params):
        return jax.lax.scan(
            lambda carry, i: step(carry, i, model_params),
            state,
            None,
            length=args.steps,
        )

    initial_state = (positions, velocity, neighbors, initial_potential, initial_grad)
    compile_start = time.perf_counter()
    compiled_md = run_md.lower(initial_state, params).compile()
    md_compile_seconds = time.perf_counter() - compile_start
    started = time.perf_counter()
    final_state, samples = compiled_md(initial_state, params)
    samples[2].block_until_ready()
    md_seconds = time.perf_counter() - started

    (
        position_series,
        velocity_series,
        potential_series,
        kinetic_series,
        force_series,
        overflow_series,
    ) = (np.asarray(item) for item in samples)
    if np.any(overflow_series):
        raise RuntimeError('QM neighbor-list capacity overflowed')
    if not np.all(np.isfinite(potential_series)):
        raise RuntimeError('Non-finite energy in POLAR/MM trajectory')
    temperature_series = 2 * kinetic_series / ((3 * len(atoms) - 3) * units.kB)
    frames = []
    for i in range(args.steps):
        if (i + 1) % args.save_interval and i + 1 != args.steps:
            continue
        frame = atoms.copy()
        frame.positions = position_series[i]
        frame.set_velocities(velocity_series[i])
        frame.info['time_fs'] = (i + 1) * args.timestep_fs
        frame.info['potential_energy_eV'] = float(potential_series[i])
        frame.info['kinetic_energy_eV'] = float(kinetic_series[i])
        frame.info['temperature_K'] = float(temperature_series[i])
        frames.append(frame)
    write(args.output / 'trajectory.xyz', frames, format='extxyz')
    write(args.output / 'final.xyz', frames[-1], format='extxyz')

    result = {
        'atoms': len(atoms),
        'qm_atoms': len(qm_index_np),
        'mm_atoms': len(mm_index_np),
        'box_angstrom': box,
        'steps': args.steps,
        'timestep_fs': args.timestep_fs,
        'duration_fs': args.steps * args.timestep_fs,
        'ensemble': 'NVE',
        'seed': args.seed,
        'initial_structure': str(args.initial),
        'initial_temperature_k': args.temperature_k,
        'final_temperature_k': float(temperature_series[-1]),
        'initial_potential_eV': float(initial_potential),
        'final_potential_eV': float(potential_series[-1]),
        'initial_total_eV': float(initial_potential) + initial_kinetic,
        'final_total_eV': float(potential_series[-1] + kinetic_series[-1]),
        'total_energy_drift_meV_per_atom': 1000
        * (
            float(potential_series[-1] + kinetic_series[-1])
            - float(initial_potential)
            - initial_kinetic
        )
        / len(atoms),
        'initial_max_force_mev_per_angstrom': float(
            1000 * jnp.max(jnp.linalg.norm(initial_grad, axis=1))
        ),
        'final_max_force_mev_per_angstrom': float(1000 * force_series[-1]),
        'force_compile_seconds': force_compile_seconds,
        'md_compile_seconds': md_compile_seconds,
        'md_seconds': md_seconds,
        'mean_md_step_seconds': md_seconds / args.steps,
        'neighbor_slots': int(final_state[2].idx.shape[1]),
        'ewald_alpha_per_angstrom': args.ewald_alpha,
        'ewald_kmax': args.ewald_kmax,
        'partition': partition_info,
        'samples': [
            {
                'step': i + 1,
                'time_fs': (i + 1) * args.timestep_fs,
                'potential_eV': float(potential_series[i]),
                'kinetic_eV': float(kinetic_series[i]),
                'temperature_k': float(temperature_series[i]),
                'max_force_mev_per_angstrom': float(1000 * force_series[i]),
            }
            for i in range(args.steps)
        ],
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
