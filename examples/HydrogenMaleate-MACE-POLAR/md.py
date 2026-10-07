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
from simulation_settings import (
    HOH_K_KJ_MOL_RAD2,
    HOH_THETA0,
    KJ_MOL_PER_EV,
    OH_K_KJ_MOL_ANGSTROM2,
    OH_R0,
    SOLUTE_EPS_KJ_MOL,
    SOLUTE_SIGMA,
    TIP3P_EPS_KJ_MOL,
    TIP3P_Q,
    TIP3P_SIGMA,
    generate_velocities,
    resolve_force_settings,
)

from mace_jax.modules.fires import fires_energy
from mace_jax.modules.langevin import kinetic_energy, middle_step
from mace_jax.modules.point_pme import PointPME
from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.modules.polar_periodic import ewald_neutralizing_background_energy
from mace_jax.tools.physical_constants import OPENMM_BOLTZMANN, OPENMM_COULOMB

COULOMB = FIELD_CONSTANT / (4 * math.pi)
KJ_MOL_TO_EV = 1 / KJ_MOL_PER_EV
TIP3P_EPS = TIP3P_EPS_KJ_MOL * KJ_MOL_TO_EV
OH_K = OH_K_KJ_MOL_ANGSTROM2 * KJ_MOL_TO_EV
HOH_K = HOH_K_KJ_MOL_RAD2 * KJ_MOL_TO_EV
SOLUTE_EPS = SOLUTE_EPS_KJ_MOL * KJ_MOL_TO_EV


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
        model_neighbors = self.model.allocate(positions[: self.n_qm])
        nonbonded_neighbors = (
            None if self.nonbonded is None else self.nonbonded.allocate(positions)
        )
        return EnergyNeighbors(model_neighbors, nonbonded_neighbors)

    def update(self, positions, neighbors):
        model_neighbors = self.model.update(positions[: self.n_qm], neighbors.model)
        nonbonded_neighbors = (
            None
            if self.nonbonded is None
            else self.nonbonded.update(positions, neighbors.nonbonded)
        )
        return EnergyNeighbors(model_neighbors, nonbonded_neighbors)


def minimum_image(delta, box):
    return delta - box * jnp.round(delta / box)


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
    coulomb=COULOMB,
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
    mm_pair = within_cutoff & ~i_is_qm & ~j_is_qm & ((i_mm // 3) != (j_mm // 3))
    mm_safe = jnp.where(mm_pair, distance, 1.0)
    mm_real = coulomb * jnp.sum(
        jnp.where(
            mm_pair,
            mm_charges[i_mm] * mm_charges[j_mm] * erfc(alpha * mm_safe) / mm_safe,
            0.0,
        )
    )
    mm_oo = mm_pair & ((i_mm % 3) == 0) & ((j_mm % 3) == 0)
    mm_lj_distance = jnp.where(mm_oo, distance, 1.0)
    mm_lj_x6 = (TIP3P_SIGMA / mm_lj_distance) ** 6
    mm_lj = jnp.sum(jnp.where(mm_oo, 4 * TIP3P_EPS * (mm_lj_x6**2 - mm_lj_x6), 0.0))

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
        mechanical_real = coulomb * jnp.sum(
            jnp.where(
                cross_pair,
                mechanical_qm_charges[qm_charge_index]
                * mm_charges[mm_index]
                * erfc(alpha * cross_safe)
                / cross_safe,
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


def build_energy(
    atoms,
    bundle,
    mode_name,
    alpha,
    kmax,
    qm_water_count,
    restraint_mode,
    restraint_radius,
    restraint_k,
    mechanical_qm_charges=None,
    nonbonded_cutoff=None,
    neighbor_skin=0.25,
    reciprocal_method='pme',
    pme_mesh_spacing=None,
    pme_assignment_order=8,
    *,
    compatibility_profile='generalized',
    ml_reciprocal_method=None,
    point_pme_grid=None,
    pme_error_tolerance=None,
    return_terms=False,
):
    if compatibility_profile == 'openmm' and mode_name == 'mlmm':
        raise ValueError('Use the generalized profile for electrostatic embedding')
    box = float(atoms.cell[0, 0])
    settings = resolve_force_settings(
        compatibility_profile,
        box,
        cutoff=nonbonded_cutoff,
        alpha=alpha,
        grid_shape=point_pme_grid,
        order=pme_assignment_order,
        tolerance=pme_error_tolerance,
    )
    alpha, nonbonded_cutoff = settings.alpha, settings.cutoff
    coulomb = (
        OPENMM_COULOMB
        if compatibility_profile != 'generalized' and mode_name != 'mlmm'
        else COULOMB
    )
    if pme_mesh_spacing is None:
        pme_mesh_spacing = (
            0.4 if compatibility_profile == 'generalized-accurate' else 0.5
        )
    point_solver = (
        PointPME(alpha, settings.grid_shape)
        if compatibility_profile == 'openmm'
        else None
    )
    ml_method = ml_reciprocal_method or (
        'direct'
        if compatibility_profile == 'openmm' and mode_name != 'mlmm'
        else reciprocal_method
    )
    if compatibility_profile == 'generalized-accurate' and (
        reciprocal_method != 'pme' or ml_method != 'pme'
    ):
        raise ValueError(
            'The generalized-accurate profile requires generalized PME for the model'
        )
    if ml_method not in ('pme', 'direct'):
        raise ValueError('Invalid ML reciprocal method')
    n_qm = len(atoms) if mode_name == 'polar' else 11 + 3 * qm_water_count
    qm_atoms = atoms[:n_qm]
    box, periodic_mode, data, graphdef, params, neighbor_fn, shift_fn = (
        initialize_model(
            qm_atoms,
            bundle,
            generalized_pme=ml_method == 'pme',
            pme_mesh_spacing=pme_mesh_spacing,
            pme_assignment_order=pme_assignment_order,
        )
    )
    if reciprocal_method not in ('direct', 'pme'):
        raise ValueError("reciprocal_method must be 'direct' or 'pme'")
    if mode_name != 'polar' and (len(atoms) - n_qm) % 3:
        raise ValueError('MM region must contain complete water triplets')
    mm_count = (len(atoms) - n_qm) // 3
    mm_charges = jnp.asarray(TIP3P_Q * mm_count, dtype=jnp.float64)
    exception_i = jnp.asarray([3 * i + a for i in range(mm_count) for a in (0, 0, 1)])
    exception_j = jnp.asarray([3 * i + b for i in range(mm_count) for b in (1, 2, 2)])
    k_axis = np.arange(-kmax, kmax + 1)
    k_int = np.stack(np.meshgrid(k_axis, k_axis, k_axis, indexing='ij'), -1).reshape(
        -1, 3
    )
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
                f'Expected {n_qm} mechanical QM charges, got {charge_array.shape}'
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
        energy_neighbor_fn = EnergyNeighborFns(neighbor_fn, nonbonded_neighbor_fn, n_qm)

    energy_neighbor_fn.model_mesh_shape = (
        tuple(data['pme_mesh_template'].shape) if 'pme_mesh_template' in data else None
    )

    def background_energy(charge, volume, ewald_alpha):
        return -coulomb * math.pi * charge**2 / (2 * volume * ewald_alpha**2)

    qq_i, qq_j = np.triu_indices(n_qm, k=1)
    qq_i, qq_j = jnp.asarray(qq_i), jnp.asarray(qq_j)

    def openmm_exception_tail(qm_r):
        # OpenMM-ML restores unscreened QQ exceptions, then subtracts a
        # cutoff QM-only PME force. Only its beyond-cutoff erfc tail remains.
        if compatibility_profile == 'generalized' or mode_name != 'mechanical':
            return jnp.asarray(0.0, dtype=qm_r.dtype)
        delta = minimum_image(qm_r[qq_i] - qm_r[qq_j], box)
        distance = jnp.linalg.norm(delta, axis=-1)
        safe = jnp.where(distance >= nonbonded_cutoff, distance, 1.0)
        return coulomb * jnp.sum(
            jnp.where(
                distance >= nonbonded_cutoff,
                mechanical_qm_charges[qq_i]
                * mechanical_qm_charges[qq_j]
                * erfc(alpha * safe)
                / safe,
                0.0,
            )
        )

    def mm_internal_terms(mm_r, reciprocal):
        waters = mm_r.reshape(-1, 3, 3)
        oh1 = minimum_image(waters[:, 1] - waters[:, 0], box)
        oh2 = minimum_image(waters[:, 2] - waters[:, 0], box)
        r1, r2 = jnp.linalg.norm(oh1, axis=-1), jnp.linalg.norm(oh2, axis=-1)
        cosine = jnp.sum(oh1 * oh2, axis=-1) / (r1 * r2)
        angle = jnp.arccos(jnp.clip(cosine, -1 + 1e-12, 1 - 1e-12))
        bond = 0.5 * OH_K * jnp.sum((r1 - OH_R0) ** 2 + (r2 - OH_R0) ** 2)
        bend = 0.5 * HOH_K * jnp.sum((angle - HOH_THETA0) ** 2)
        self_energy = -coulomb * alpha / math.sqrt(math.pi) * jnp.sum(mm_charges**2)
        background = background_energy(jnp.sum(mm_charges), box**3, alpha)
        excluded = minimum_image(mm_r[exception_i] - mm_r[exception_j], box)
        excluded_r = jnp.linalg.norm(excluded, axis=-1)
        exceptions = -coulomb * jnp.sum(
            mm_charges[exception_i]
            * mm_charges[exception_j]
            * erf(alpha * excluded_r)
            / excluded_r
        )
        return {
            'mm_bond': bond,
            'mm_angle': bend,
            'mm_reciprocal': reciprocal,
            'mm_self': self_energy,
            'mm_background': background,
            'mm_exceptions': exceptions,
        }

    def mechanical_cross_electrostatics(qm_r, mm_r, reciprocal):
        """Fixed-charge QM--MM Ewald cross term used by mechanical embedding."""
        q_qm = jnp.sum(mechanical_qm_charges)
        q_mm = jnp.sum(mm_charges)
        background = (
            background_energy(q_qm + q_mm, box**3, alpha)
            - background_energy(q_qm, box**3, alpha)
            - background_energy(q_mm, box**3, alpha)
        )
        return reciprocal + background

    def boundary_restraint(qm_r, mm_r):
        if qm_water_count == 0:
            return jnp.asarray(0.0, dtype=qm_r.dtype)
        if restraint_mode == 'radial':
            radial_delta = minimum_image(qm_r[11::3, None] - qm_r[:4][None], box)
            qm_distance = jnp.linalg.norm(radial_delta, axis=-1).min(axis=-1)
            excess = jnp.maximum(qm_distance - restraint_radius, 0.0)
            return 0.5 * restraint_k * jnp.sum(excess**2)
        if restraint_mode == 'fires':
            # The boundary depends on the outermost QM water and on the solute
            # coordinates, so autodiff supplies the paired FIRES forces on the
            # QM water, intruding MM waters, and solute.
            return fires_energy(qm_r[11::3], mm_r[::3], qm_r[:8], box, restraint_k)
        return jnp.asarray(0.0, dtype=qm_r.dtype)

    def energy_terms_fn(r, neighbors, model_params):
        qm_r = r[:n_qm]
        edge_index, shifts, unit_shifts = graph_edges(qm_r, neighbors.model, box)
        inputs = dict(
            data,
            positions=qm_r,
            edge_index=edge_index,
            shifts=shifts,
            unit_shifts=unit_shifts,
        )
        if mode_name == 'mlmm' or (
            mode_name == 'mechanical' and point_solver is None and ml_method == 'pme'
        ):
            inputs['mm_positions'] = r[n_qm:]
            inputs['mm_charges'] = mm_charges
            inputs['mm_ewald_alpha'] = jnp.asarray(alpha, dtype=r.dtype)
        if mode_name == 'mechanical' and point_solver is None and ml_method == 'pme':
            inputs['mechanical_qm_charges'] = mechanical_qm_charges
        model_result = nnx.merge(graphdef, model_params)(
            inputs, pbc_handling=periodic_mode
        )
        polar_energy = (
            settings.ml_energy_scale if mode_name != 'mlmm' else 1.0
        ) * model_result['energy'][0]
        if mode_name == 'polar':
            return {'ml': polar_energy}
        if point_solver is not None:
            if mode_name == 'mechanical':
                mm_reciprocal, mechanical_reciprocal = point_solver.reciprocal_parts(
                    qm_r,
                    mechanical_qm_charges,
                    r[n_qm:],
                    mm_charges,
                    jnp.eye(3, dtype=r.dtype) * box,
                )
            else:
                mm_reciprocal = point_solver.reciprocal_energy(
                    r[n_qm:], mm_charges, jnp.eye(3, dtype=r.dtype) * box
                )
                mechanical_reciprocal = jnp.asarray(0.0, dtype=r.dtype)
        elif ml_method == 'pme':
            mm_reciprocal = model_result['mm_ewald_reciprocal_energy'][0]
            mechanical_reciprocal = model_result['mechanical_ewald_cross_energy'][0]
            # The Gaussian model keeps its trained electrostatic constant.
            # Fixed-charge mechanical terms use one common classical constant.
            mm_reciprocal *= coulomb / COULOMB
            mechanical_reciprocal *= coulomb / COULOMB
        else:
            mm_r = r[n_qm:]
            phase = k_vectors @ mm_r.T
            c = jnp.cos(phase) @ mm_charges
            s = jnp.sin(phase) @ mm_charges
            mm_reciprocal = (
                coulomb * 2 * math.pi / box**3 * jnp.sum(k_weights * (c * c + s * s))
            )
            mechanical_reciprocal = jnp.asarray(0.0, dtype=r.dtype)
            if mode_name == 'mechanical':
                qm_phase = k_vectors @ qm_r.T
                qm_c = jnp.cos(qm_phase) @ mechanical_qm_charges
                qm_s = jnp.sin(qm_phase) @ mechanical_qm_charges
                mechanical_reciprocal = (
                    coulomb
                    * 4
                    * math.pi
                    / box**3
                    * jnp.sum(k_weights * (qm_c * c + qm_s * s))
                )
        mm_real, mm_lj, cross_lj, mechanical_real = sparse_nonbonded_terms(
            r,
            ordered_pairs_from_dense_neighbors(neighbors.nonbonded.idx, r.shape[0]),
            n_qm,
            mm_charges,
            alpha,
            nonbonded_cutoff,
            box,
            cross_sigma,
            cross_epsilon,
            mechanical_qm_charges if mode_name == 'mechanical' else None,
            coulomb,
        )
        terms = {
            'ml': polar_energy,
            'mm_real': mm_real,
            'mm_lj': mm_lj,
            'cross_lj': cross_lj,
            'restraint': boundary_restraint(qm_r, r[n_qm:]),
        }
        terms.update(mm_internal_terms(r[n_qm:], mm_reciprocal))
        if mode_name == 'mechanical':
            terms['cross_coulomb'] = mechanical_real + mechanical_cross_electrostatics(
                qm_r, r[n_qm:], mechanical_reciprocal
            )
            terms['openmm_qq_tail'] = openmm_exception_tail(qm_r)
        return terms

    def energy_fn(r, neighbors, model_params):
        return sum(energy_terms_fn(r, neighbors, model_params).values())

    result = (
        jax.jit(jax.value_and_grad(energy_fn)),
        params,
        energy_neighbor_fn,
        shift_fn,
        n_qm,
    )
    return result + (jax.jit(energy_terms_fn),) if return_terms else result


def write_chunk(
    args, atoms, position_series, energy_series, kinetic_series, start_step
):
    frames = []
    samples = []
    box = float(atoms.cell[0, 0])
    for i in range(len(energy_series)):
        step = start_step + i + 1
        positions = position_series[i]
        h_to_o = positions[:4] - positions[10]
        h_to_o -= box * np.rint(h_to_o / box)
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
            samples.append(
                {
                    'step': step,
                    'time_ps': step * args.timestep_fs / 1000,
                    'potential_eV': float(energy_series[i]),
                    'temperature_K': float(
                        2
                        * kinetic_series[i]
                        / (
                            (3 * len(atoms) - 3)
                            * (
                                OPENMM_BOLTZMANN
                                if args.compatibility_profile != 'generalized'
                                else units.kB
                            )
                        )
                    ),
                    'left_carboxyl_H_angstrom': left_h,
                    'right_carboxyl_H_angstrom': right_h,
                    'proton_coordinate_angstrom': proton_coordinate,
                }
            )
    if frames:
        tmp = args.output / 'chunks' / f'{start_step + len(energy_series):08d}.tmp.xyz'
        final = args.output / 'chunks' / f'{start_step + len(energy_series):08d}.xyz'
        write(tmp, frames, format='extxyz')
        tmp.replace(final)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=['polar', 'mlmm', 'mechanical'], required=True
    )
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
    parser.add_argument('--ewald-alpha', type=float)
    parser.add_argument('--ewald-kmax', type=int, default=6)
    parser.add_argument('--reciprocal-method', choices=['pme', 'direct'], default='pme')
    parser.add_argument('--pme-mesh-spacing', type=float)
    parser.add_argument('--pme-assignment-order', type=int, default=8)
    parser.add_argument('--nonbonded-cutoff', type=float)
    parser.add_argument('--neighbor-skin', type=float, default=0.25)
    parser.add_argument('--qm-water-count', type=int, default=0)
    parser.add_argument(
        '--restraint-mode', choices=['none', 'radial', 'fires'], default='radial'
    )
    parser.add_argument('--restraint-radius', type=float, default=4.2)
    parser.add_argument('--restraint-k', type=float, default=0.2)
    parser.add_argument('--mechanical-qm-charges', type=Path)
    parser.add_argument(
        '--compatibility-profile',
        choices=['generalized', 'generalized-accurate', 'openmm'],
        default='generalized',
    )
    parser.add_argument('--ml-reciprocal-method', choices=['direct', 'pme'])
    parser.add_argument('--point-pme-grid', type=int, nargs=3)
    parser.add_argument('--pme-error-tolerance', type=float)
    parser.add_argument('--integrator', choices=['baoab', 'langevin-middle'])
    parser.add_argument('--thermostat-seed', type=int)
    parser.add_argument(
        '--noise-file',
        type=Path,
        help='Replay standard normal noise, shape (steps, atoms, 3)',
    )
    args = parser.parse_args()
    if args.timestep_fs <= 0 or args.thermostat_tau_ps <= 0 or args.temperature_k < 0:
        raise ValueError('Invalid dynamics settings')
    args.integrator = args.integrator or (
        'baoab' if args.compatibility_profile == 'generalized' else 'langevin-middle'
    )
    if args.thermostat_seed is None:
        args.thermostat_seed = args.seed + (args.compatibility_profile == 'generalized')
    if (
        args.chunk <= 0
        or args.save_interval <= 0
        or args.report_interval <= 0
        or args.steps % args.chunk
        or args.chunk % args.save_interval
    ):
        raise ValueError('steps must divide by chunk and chunk by save-interval')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'chunks').mkdir(exist_ok=True)
    atoms = read(args.initial)
    if len(atoms) != 167:
        raise ValueError('Expected 11 hydrogen maleate atoms and 52 waters')
    settings = resolve_force_settings(
        args.compatibility_profile,
        float(atoms.cell[0, 0]),
        cutoff=args.nonbonded_cutoff,
        alpha=args.ewald_alpha,
        grid_shape=args.point_pme_grid,
        order=args.pme_assignment_order,
        tolerance=args.pme_error_tolerance,
    )
    args.nonbonded_cutoff, args.ewald_alpha = settings.cutoff, settings.alpha
    if args.pme_mesh_spacing is None:
        args.pme_mesh_spacing = (
            0.4 if args.compatibility_profile == 'generalized-accurate' else 0.5
        )
    if args.pme_error_tolerance is None:
        args.pme_error_tolerance = (
            1e-6 if args.compatibility_profile == 'generalized-accurate' else 5e-5
        )
    restart_settings = dict(
        settings.to_dict(),
        integrator=args.integrator,
        temperature=args.temperature_k,
        timestep_fs=args.timestep_fs,
        thermostat_tau_ps=args.thermostat_tau_ps,
        thermostat_seed=args.thermostat_seed,
        mode=args.mode,
        qm_water_count=args.qm_water_count,
        ml_reciprocal_method=args.ml_reciprocal_method
        or (
            'direct'
            if args.compatibility_profile == 'openmm' and args.mode != 'mlmm'
            else args.reciprocal_method
        ),
        reciprocal_method=args.reciprocal_method,
        restraint_mode=args.restraint_mode,
        restraint_radius=args.restraint_radius,
        restraint_k=args.restraint_k,
        noise_file=str(args.noise_file) if args.noise_file else None,
    )
    restart_json = json.dumps(restart_settings, sort_keys=True)
    replay_noise = np.load(args.noise_file, mmap_mode='r') if args.noise_file else None
    if replay_noise is not None and replay_noise.shape != (args.steps, len(atoms), 3):
        raise ValueError('Noise file must contain finite (steps, atoms, 3) samples')
    if args.mode != 'polar':
        if not 0 <= args.qm_water_count < 52:
            raise ValueError('Mixed embedding requires between 0 and 51 QM waters')
        if int(atoms.info.get('qm_water_count', -1)) != args.qm_water_count:
            raise ValueError('QM water count differs from initial structure metadata')
    mechanical_qm_charges = None
    if args.mode == 'mechanical':
        if args.mechanical_qm_charges is None:
            raise ValueError('--mechanical-qm-charges is required in mechanical mode')
        charge_data = json.loads(args.mechanical_qm_charges.read_text())
        mechanical_qm_charges = np.asarray(charge_data['charges'], dtype=float)
        expected_numbers = np.asarray(atoms.numbers[: 11 + 3 * args.qm_water_count])
        if not np.array_equal(charge_data.get('atomic_numbers'), expected_numbers):
            raise ValueError('Mechanical charge-file atom order differs from structure')
    restart_settings.update(
        bundle=str(args.bundle.resolve()),
        mechanical_qm_charges=mechanical_qm_charges.tolist()
        if mechanical_qm_charges is not None
        else None,
        pme_mesh_spacing=args.pme_mesh_spacing,
        pme_assignment_order=args.pme_assignment_order,
        ewald_kmax=args.ewald_kmax,
        neighbor_skin=args.neighbor_skin,
        seed=args.seed,
    )
    restart_json = json.dumps(restart_settings, sort_keys=True)
    initial_h_to_o = np.linalg.norm(atoms.positions[:4] - atoms.positions[10], axis=1)
    if initial_h_to_o.min() > 1.3:
        raise ValueError(
            'Initial hydrogen maleate proton is not bonded to a carboxyl oxygen'
        )
    if int(atoms.info.get('qm_charge', atoms.info.get('charge', 0))) != -1:
        raise ValueError('Hydrogen maleate QM region must have charge -1')
    force_eval, params, neighbor_fn, shift_fn, n_qm = build_energy(
        atoms,
        args.bundle,
        args.mode,
        args.ewald_alpha,
        args.ewald_kmax,
        args.qm_water_count,
        args.restraint_mode,
        args.restraint_radius,
        args.restraint_k,
        mechanical_qm_charges,
        args.nonbonded_cutoff,
        args.neighbor_skin,
        args.reciprocal_method,
        args.pme_mesh_spacing,
        args.pme_assignment_order,
        compatibility_profile=args.compatibility_profile,
        ml_reciprocal_method=args.ml_reciprocal_method,
        point_pme_grid=settings.grid_shape,
        pme_error_tolerance=args.pme_error_tolerance,
    )
    if args.compatibility_profile == 'generalized-accurate':
        restart_settings.update(
            generalized_mesh_shape=neighbor_fn.model_mesh_shape,
            ewald_error_parameter=args.pme_error_tolerance,
            ml_energy_scale=1.0 if args.mode == 'mlmm' else settings.ml_energy_scale,
            classical_coulomb_eV_angstrom_per_e2=COULOMB
            if args.mode == 'mlmm'
            else OPENMM_COULOMB,
        )
        restart_json = json.dumps(restart_settings, sort_keys=True)
    checkpoint = args.output / 'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if 'settings' in saved:
                if str(saved['settings']) != restart_json:
                    raise ValueError(
                        'Checkpoint physical settings differ from this run'
                    )
            elif (
                args.compatibility_profile != 'generalized'
                or args.integrator != 'baoab'
            ):
                raise ValueError('Legacy checkpoint only supports generalized BAOAB')
            start_step = int(saved['step'])
            positions = jnp.asarray(saved['positions'])
            velocities = jnp.asarray(saved['velocities'])
        print(f'Restarting from step {start_step}', flush=True)
    else:
        start_step = 0
        atoms.info['source_time_fs'] = atoms.info.get('time_fs', 0.0)
        for key in ('potential_energy_eV', 'kinetic_energy_eV', 'temperature_K'):
            atoms.info.pop(key, None)
        atoms.info['time_fs'] = 0.0
        write(args.output / 'initial.xyz', atoms, format='extxyz')
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        velocities = jnp.asarray(
            generate_velocities(
                atoms, args.temperature_k, args.seed, args.compatibility_profile
            )
        )
        if args.integrator == 'langevin-middle':
            velocities = velocities * units.fs * 1000.0
    masses = jnp.asarray(atoms.get_masses()[:, None], dtype=jnp.float64)
    neighbors = neighbor_fn.allocate(positions)
    compile_start = time.perf_counter()
    initial_energy, initial_grad = force_eval(positions, neighbors, params)
    initial_energy.block_until_ready()
    print(
        f'Compiled force in {time.perf_counter() - compile_start:.1f} s; initial energy {float(initial_energy):.6f} eV',
        flush=True,
    )
    dt = args.timestep_fs * units.fs
    c = math.exp(-args.timestep_fs / (1000 * args.thermostat_tau_ps))
    noise_scale = jnp.sqrt(
        (1 - c * c)
        * (
            units.kB
            if args.compatibility_profile == 'generalized'
            else OPENMM_BOLTZMANN
        )
        * args.temperature_k
        / masses
    )

    def step(carry, sample, model_params):
        absolute_step, supplied_noise = sample
        r, v, nbrs, energy, grad = carry
        noise = (
            supplied_noise
            if replay_noise is not None
            else jax.random.normal(
                jax.random.fold_in(
                    jax.random.PRNGKey(args.thermostat_seed), absolute_step
                ),
                r.shape,
                dtype=r.dtype,
            )
        )
        if args.integrator == 'langevin-middle':
            r, v = middle_step(
                r,
                v,
                -grad,
                masses,
                noise,
                timestep_ps=args.timestep_fs / 1000,
                temperature=args.temperature_k,
                friction_ps=1 / args.thermostat_tau_ps,
                shift=shift_fn,
            )
        else:
            v = v - 0.5 * dt * grad / masses
            r = shift_fn(r, 0.5 * dt * v)
            v = c * v + noise_scale * noise
            r = shift_fn(r, 0.5 * dt * v)
        nbrs = neighbor_fn.update(r, nbrs)
        new_energy, new_grad = force_eval(r, nbrs, model_params)
        if args.integrator == 'langevin-middle':
            kinetic = kinetic_energy(v, masses)
        else:
            v = v - 0.5 * dt * new_grad / masses
            kinetic = 0.5 * jnp.sum(masses * v * v)
        return (r, v, nbrs, new_energy, new_grad), (
            r,
            new_energy,
            kinetic,
            nbrs.did_buffer_overflow,
        )

    @jax.jit
    def run_chunk(state, offset, noise, model_params):
        # Keep model weights as runtime inputs, avoiding compiler constant folding
        # of the full network inside the MD loop.
        return jax.lax.scan(
            lambda carry, sample: step(carry, sample, model_params),
            state,
            (offset + jnp.arange(args.chunk), noise),
        )

    empty_noise = jnp.zeros((args.chunk, len(atoms), 3), dtype=jnp.float64)
    state = (positions, velocities, neighbors, initial_energy, initial_grad)
    compile_start = time.perf_counter()
    compiled = run_chunk.lower(
        state, jnp.asarray(start_step, dtype=jnp.int32), empty_noise, params
    ).compile()
    print(
        f'Compiled MD chunk in {time.perf_counter() - compile_start:.1f} s', flush=True
    )
    started = time.perf_counter()
    for offset in range(start_step, args.steps, args.chunk):
        noise = (
            empty_noise
            if replay_noise is None
            else jnp.asarray(replay_noise[offset : offset + args.chunk])
        )
        if replay_noise is not None and not np.all(np.isfinite(np.asarray(noise))):
            raise ValueError('Nonfinite replay noise')
        state, output = compiled(
            state, jnp.asarray(offset, dtype=jnp.int32), noise, params
        )
        positions_np, potential_np, kinetic_np, overflow_np = (
            np.asarray(x) for x in output
        )
        if np.any(overflow_np) or not all(
            np.all(np.isfinite(value))
            for value in (
                positions_np,
                potential_np,
                kinetic_np,
                np.asarray(state[1]),
                np.asarray(state[4]),
            )
        ):
            raise RuntimeError(
                f'Nonfinite energy or neighbor overflow in chunk {offset}'
            )
        samples = write_chunk(
            args, atoms, positions_np, potential_np, kinetic_np, offset
        )
        new_step = offset + args.chunk
        tmp = args.output / 'checkpoint.tmp.npz'
        np.savez(
            tmp,
            step=new_step,
            positions=np.asarray(state[0]),
            velocities=np.asarray(state[1]),
            settings=restart_json,
        )
        tmp.replace(checkpoint)
        progress = {
            'settings': restart_settings,
            'mode': args.mode,
            'step': new_step,
            'time_ps': new_step * args.timestep_fs / 1000,
            'target_time_ps': args.steps * args.timestep_fs / 1000,
            'elapsed_seconds_this_process': time.perf_counter() - started,
            'last_sample': samples[-1] if samples else None,
            'cpu_only': os.environ.get('JAX_PLATFORMS') == 'cpu',
            'qm_water_count': args.qm_water_count if args.mode != 'polar' else 52,
            'restraint_radius_angstrom': (
                args.restraint_radius
                if args.mode != 'polar'
                and args.qm_water_count
                and args.restraint_mode == 'radial'
                else None
            ),
            'restraint_mode': (
                args.restraint_mode
                if args.mode != 'polar' and args.qm_water_count
                else None
            ),
            'restraint_k_eV_per_angstrom2': (
                args.restraint_k
                if args.mode != 'polar' and args.qm_water_count
                else None
            ),
            'nonbonded_cutoff_angstrom': (
                args.nonbonded_cutoff if args.mode != 'polar' else None
            ),
            'neighbor_skin_angstrom': (
                args.neighbor_skin if args.mode != 'polar' else None
            ),
            'reciprocal_method': args.reciprocal_method,
            'pme_mesh_spacing_angstrom': (
                args.pme_mesh_spacing if args.reciprocal_method == 'pme' else None
            ),
            'pme_assignment_order': (
                args.pme_assignment_order if args.reciprocal_method == 'pme' else None
            ),
            'mechanical_qm_charges': (
                str(args.mechanical_qm_charges) if args.mode == 'mechanical' else None
            ),
        }
        (args.output / 'progress.json').write_text(
            json.dumps(progress, indent=2) + '\n'
        )
        print(json.dumps(progress), flush=True)
    final = atoms.copy()
    final.positions = np.asarray(state[0])
    h_delta = final.positions[:4] - final.positions[10]
    box = float(final.cell[0, 0])
    h_delta -= box * np.rint(h_delta / box)
    distances = np.linalg.norm(h_delta, axis=1)
    final.info.update(
        time_fs=args.steps * args.timestep_fs,
        potential_energy_eV=float(state[3]),
        kinetic_energy_eV=float(kinetic_energy(state[1], masses))
        if args.integrator == 'langevin-middle'
        else float(0.5 * jnp.sum(masses * state[1] ** 2)),
        left_carboxyl_H_angstrom=float(distances[:2].min()),
        right_carboxyl_H_angstrom=float(distances[2:].min()),
        proton_coordinate_angstrom=float(distances[2:].min() - distances[:2].min()),
    )
    write(args.output / 'final.xyz', final, format='extxyz')
    print(f'Completed {args.steps * args.timestep_fs / 1000:.3f} ps', flush=True)


if __name__ == '__main__':
    main()
