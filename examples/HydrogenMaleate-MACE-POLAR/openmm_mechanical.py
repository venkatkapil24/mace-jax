#!/usr/bin/env python3
"""Independent OpenMM-ML mechanical-embedding reference for hydrogen maleate.

This reproduces the thermodynamic conditions used by ``md.py`` while using the
original Torch MACE-POLAR model and OpenMM-ML's mechanical embedding. Both
drivers share force parameters, PME settings and initial velocities. The JAX
OpenMM compatibility profile uses the same middle-integrator convention.
Supplied noise permits short, deterministic comparisons across engines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import openmm
from ase import units as ase_units
from ase.io import read, write
from openmm import app, unit
from openmmml import MLPotential
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
    resolve_force_settings,
)
from simulation_settings import (
    generate_velocities as shared_velocities,
)

from mace_jax.tools.physical_constants import OPENMM_BOLTZMANN

TIP3P_SIGMA_NM = TIP3P_SIGMA * 0.1
OH_R0_NM = OH_R0 * 0.1
OH_K_KJ_MOL_NM2 = OH_K_KJ_MOL_ANGSTROM2 * 100
SOLUTE_SIGMA_NM = SOLUTE_SIGMA * 0.1
SOLUTE_BONDS = (
    (0, 6),
    (1, 6),
    (2, 7),
    (3, 7),
    (4, 6),
    (4, 5),
    (5, 7),
    (8, 4),
    (9, 5),
    (10, 0),
)


def generate_velocities(atoms, temperature: float, seed: int, profile='generalized'):
    return shared_velocities(atoms, temperature, seed, profile) * ase_units.fs * 100.0


def build_topology(atoms) -> tuple[app.Topology, list[app.Topology.Atom]]:
    topology = app.Topology()
    chain = topology.addChain('A')
    topology_atoms = []
    solute = topology.addResidue('HMA', chain)
    for i, symbol in enumerate(atoms.get_chemical_symbols()[:11]):
        topology_atoms.append(
            topology.addAtom(
                f'{symbol}{i + 1}', app.Element.getBySymbol(symbol), solute
            )
        )
    for i, j in SOLUTE_BONDS:
        topology.addBond(topology_atoms[i], topology_atoms[j])
    for water in range(52):
        residue = topology.addResidue('HOH', chain)
        base = 11 + 3 * water
        oxygen = topology.addAtom('O', app.Element.getBySymbol('O'), residue)
        h1 = topology.addAtom('H1', app.Element.getBySymbol('H'), residue)
        h2 = topology.addAtom('H2', app.Element.getBySymbol('H'), residue)
        topology_atoms.extend((oxygen, h1, h2))
        topology.addBond(topology_atoms[base], topology_atoms[base + 1])
        topology.addBond(topology_atoms[base], topology_atoms[base + 2])
    cell_nm = np.asarray(atoms.cell.array, dtype=float) * 0.1
    topology.setPeriodicBoxVectors(
        tuple(openmm.Vec3(*vector) * unit.nanometer for vector in cell_nm)
    )
    return topology, topology_atoms


def build_classical_system(atoms, charges: np.ndarray, settings=None) -> openmm.System:
    system = openmm.System()
    cell_nm = np.asarray(atoms.cell.array, dtype=float) * 0.1
    system.setDefaultPeriodicBoxVectors(
        *(openmm.Vec3(*vector) * unit.nanometer for vector in cell_nm)
    )
    for mass in atoms.get_masses():
        system.addParticle(float(mass) * unit.dalton)

    bonds = openmm.HarmonicBondForce()
    angles = openmm.HarmonicAngleForce()
    bonds.setUsesPeriodicBoundaryConditions(True)
    angles.setUsesPeriodicBoundaryConditions(True)
    for water in range(52):
        base = 11 + 3 * water
        bonds.addBond(base, base + 1, OH_R0_NM, OH_K_KJ_MOL_NM2)
        bonds.addBond(base, base + 2, OH_R0_NM, OH_K_KJ_MOL_NM2)
        angles.addAngle(base + 1, base, base + 2, HOH_THETA0, HOH_K_KJ_MOL_RAD2)
    system.addForce(bonds)
    system.addForce(angles)

    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    # The 11.76 A cell permits at most 5.88 A.  This includes essentially all
    # of the alpha=0.5/A real-space Ewald contribution used by md.py.
    nonbonded.setCutoffDistance(
        (settings.cutoff * 0.1 if settings else 0.58) * unit.nanometer
    )
    if settings:
        nonbonded.setPMEParameters(settings.alpha * 10, *settings.grid_shape)
    nonbonded.setEwaldErrorTolerance(5e-5)
    nonbonded.setUseDispersionCorrection(False)
    for i in range(11):
        nonbonded.addParticle(
            float(charges[i]),
            float(SOLUTE_SIGMA_NM[i]),
            float(SOLUTE_EPS_KJ_MOL[i]),
        )
    for atom in range(11, len(atoms)):
        water_atom = (atom - 11) % 3
        sigma = TIP3P_SIGMA_NM if water_atom == 0 else 0.1
        epsilon = TIP3P_EPS_KJ_MOL if water_atom == 0 else 0.0
        nonbonded.addParticle(float(charges[atom]), sigma, epsilon)
    # Flexible TIP3P excludes all intramolecular Coulomb and LJ pairs.
    for water in range(52):
        base = 11 + 3 * water
        for i, j in ((base, base + 1), (base, base + 2), (base + 1, base + 2)):
            nonbonded.addException(i, j, 0.0, 0.1, 0.0)
    system.addForce(nonbonded)
    return system


def add_radial_boundary_restraint(
    system: openmm.System,
    qm_water_count: int,
    radius_angstrom: float,
    k_ev_angstrom2: float,
) -> None:
    if qm_water_count == 0:
        return
    expression = (
        '0.5*k*step(d-r0)*(d-r0)^2; '
        'd=min(min(distance(p1,p2),distance(p1,p3)),'
        'min(distance(p1,p4),distance(p1,p5)))'
    )
    force = openmm.CustomCompoundBondForce(5, expression)
    force.addGlobalParameter('r0', 0.1 * radius_angstrom)
    force.addGlobalParameter('k', k_ev_angstrom2 * KJ_MOL_PER_EV * 100.0)
    force.setUsesPeriodicBoundaryConditions(True)
    for water in range(qm_water_count):
        oxygen = 11 + 3 * water
        force.addBond((oxygen, 0, 1, 2, 3), ())
    system.addForce(force)


def add_fires_restraint(
    system: openmm.System,
    qm_water_count: int,
    k_ev_angstrom2: float,
) -> None:
    """Add FIRES ordering in the nearest-solute-heavy-atom coordinate."""
    if qm_water_count == 0:
        return
    system.addForce(create_fires_force(qm_water_count, k_ev_angstrom2))


def create_fires_force(qm_water_count: int, k_ev_angstrom2: float):
    """Native OpenMM FIRES force: one compiled kernel, no Python callback.

    One compound bond per MM oxygen includes every QM oxygen and the eight
    solute heavy atoms. Each bond contributes its own intrusion energy; OpenMM
    differentiates the shared moving boundary and sums the reaction forces.
    """
    if not 0 < qm_water_count < 52:
        raise ValueError('Native FIRES requires between 1 and 51 QM waters')

    def reduce_expression(operation, terms):
        # A balanced tree reduces symbolic derivative depth during compilation.
        if len(terms) == 1:
            return terms[0]
        middle = len(terms) // 2
        left = reduce_expression(operation, terms[:middle])
        right = reduce_expression(operation, terms[middle:])
        return f'{operation}({left},{right})'

    def nearest_distance(particle):
        return reduce_expression(
            'min',
            [f'distance(p{particle},p{qm_water_count + h + 1})' for h in range(8)],
        )

    expression = '0.5*k*max(b-d,0)^2; '
    expression += (
        'b=' + reduce_expression('max', [f'q{i}' for i in range(qm_water_count)]) + '; '
    )
    expression += 'd=' + nearest_distance(qm_water_count + 9) + '; '
    expression += '; '.join(
        f'q{i}={nearest_distance(i + 1)}' for i in range(qm_water_count)
    )
    force = openmm.CustomCompoundBondForce(qm_water_count + 9, expression)
    force.addGlobalParameter('k', k_ev_angstrom2 * KJ_MOL_PER_EV * 100.0)
    force.setUsesPeriodicBoundaryConditions(True)
    common_particles = list(11 + 3 * np.arange(qm_water_count)) + list(range(8))
    for oxygen in 11 + 3 * np.arange(qm_water_count, 52):
        force.addBond(common_particles + [int(oxygen)], ())
    return force


def _compute_fires(state, qm_water_count: int, k_ev_angstrom2: float):
    """Return exact FIRES energy and forces for OpenMM's PythonForce."""
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    inverse_cell = np.linalg.inv(cell)
    oxygen_indices = 11 + 3 * np.arange(52)
    displacement = positions[oxygen_indices, None] - positions[None, :8]
    fractional = displacement @ inverse_cell
    displacement -= np.rint(fractional) @ cell
    distance_squared = np.einsum('whc,whc->wh', displacement, displacement)
    nearest_heavy = np.argmin(distance_squared, axis=1)
    water = np.arange(52)
    nearest_vector = displacement[water, nearest_heavy]
    distance = np.sqrt(distance_squared[water, nearest_heavy])
    unit_vector = nearest_vector / distance[:, None]

    outer_qm = int(np.argmax(distance[:qm_water_count]))
    boundary = distance[outer_qm]
    intrusion = np.maximum(boundary - distance[qm_water_count:], 0.0)
    energy_ev = 0.5 * k_ev_angstrom2 * np.sum(intrusion**2)

    forces = np.zeros_like(positions)
    boundary_force = k_ev_angstrom2 * np.sum(intrusion) * unit_vector[outer_qm]
    forces[oxygen_indices[outer_qm]] -= boundary_force
    forces[nearest_heavy[outer_qm]] += boundary_force
    # Accumulate repeated heavy-atom indices correctly without a Python loop.
    mm_force = k_ev_angstrom2 * intrusion[:, None] * unit_vector[qm_water_count:]
    forces[oxygen_indices[qm_water_count:]] += mm_force
    np.add.at(forces, nearest_heavy[qm_water_count:], -mm_force)

    return energy_ev * KJ_MOL_PER_EV, forces * KJ_MOL_PER_EV * 10.0


def replay_integrator(dt_ps, temperature, friction_ps):
    """Unconstrained Langevin-middle with caller-supplied standard normal noise."""
    integrator = openmm.CustomIntegrator(dt_ps)
    integrator.addGlobalVariable('decay', np.exp(-dt_ps * friction_ps))
    integrator.addGlobalVariable('kt', OPENMM_BOLTZMANN * KJ_MOL_PER_EV * temperature)
    integrator.addPerDofVariable('eta', 0)
    integrator.addComputePerDof('v', 'v+dt*f/m')
    integrator.addComputePerDof('x', 'x+0.5*dt*v')
    integrator.addComputePerDof('v', 'decay*v+sqrt((1-decay*decay)*kt/m)*eta')
    integrator.addComputePerDof('x', 'x+0.5*dt*v')
    return integrator


def proton_observables(positions_angstrom: np.ndarray, box_angstrom: float):
    delta = positions_angstrom[:4] - positions_angstrom[10]
    delta -= box_angstrom * np.rint(delta / box_angstrom)
    distances = np.linalg.norm(delta, axis=1)
    left = float(np.min(distances[:2]))
    right = float(np.min(distances[2:]))
    return left, right, right - left


def evaluate(simulation: app.Simulation, groups: int = -1):
    state = simulation.context.getState(getEnergy=True, getForces=True, groups=groups)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    return float(energy), np.asarray(forces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--charges', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--timestep-fs', type=float, default=0.5)
    parser.add_argument('--temperature-k', type=float, default=330.0)
    parser.add_argument('--thermostat-tau-ps', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--qm-water-count', type=int, default=13)
    parser.add_argument(
        '--restraint-mode', choices=('none', 'radial', 'fires'), default='radial'
    )
    parser.add_argument('--restraint-radius', type=float, default=4.2)
    parser.add_argument('--restraint-k', type=float, default=0.2)
    parser.add_argument('--report-interval', type=int, default=20)
    parser.add_argument('--checkpoint-interval', type=int, default=2000)
    parser.add_argument(
        '--trajectory-interval',
        type=int,
        default=0,
        help='Save an extxyz coordinate frame every this many steps (0 disables)',
    )
    parser.add_argument('--torch-device', default='cpu')
    parser.add_argument(
        '--platform', choices=['Reference', 'CPU', 'CUDA', 'OpenCL'], default='CPU'
    )
    parser.add_argument(
        '--cpu-threads',
        type=int,
        default=int(os.environ.get('OPENMM_CPU_THREADS', '8')),
    )
    parser.add_argument(
        '--platform-precision', choices=['single', 'mixed', 'double'], default='double'
    )
    parser.add_argument(
        '--compatibility-profile', choices=['generalized', 'openmm'], default='openmm'
    )
    parser.add_argument('--thermostat-seed', type=int)
    parser.add_argument('--nonbonded-cutoff', type=float)
    parser.add_argument('--ewald-alpha', type=float)
    parser.add_argument('--point-pme-grid', type=int, nargs=3)
    parser.add_argument('--pme-error-tolerance', type=float, default=5e-5)
    parser.add_argument(
        '--noise-file',
        type=Path,
        help='Replay standard normal noise, shape (steps, atoms, 3)',
    )
    args = parser.parse_args()
    if args.timestep_fs <= 0 or args.thermostat_tau_ps <= 0 or args.temperature_k < 0:
        raise ValueError('Invalid dynamics settings')
    if args.steps < 0 or args.report_interval <= 0 or args.steps % args.report_interval:
        raise ValueError('steps must be a nonnegative report-interval multiple')
    if args.checkpoint_interval <= 0:
        raise ValueError('checkpoint-interval must be positive')
    if args.trajectory_interval < 0 or (
        args.trajectory_interval and args.trajectory_interval % args.report_interval
    ):
        raise ValueError(
            'trajectory-interval must be zero or a report-interval multiple'
        )

    atoms = read(args.initial)
    replay_noise = np.load(args.noise_file, mmap_mode='r') if args.noise_file else None
    if replay_noise is not None and replay_noise.shape != (args.steps, len(atoms), 3):
        raise ValueError('Noise file must have shape (steps, atoms, 3)')
    if len(atoms) != 167 or not 0 <= args.qm_water_count < 52:
        raise ValueError('Expected 167 atoms and between 0 and 51 QM waters')
    if int(atoms.info.get('qm_water_count', -1)) != args.qm_water_count:
        raise ValueError('QM water count differs from initial structure metadata')
    charge_data = json.loads(args.charges.read_text())
    qm_charges = np.asarray(charge_data['charges'], dtype=float)
    n_qm = 11 + 3 * args.qm_water_count
    if qm_charges.shape != (n_qm,) or not np.isclose(qm_charges.sum(), -1.0):
        raise ValueError(f'Invalid {n_qm}-atom QM charge array')
    charges = np.concatenate((qm_charges, np.tile(TIP3P_Q, 52 - args.qm_water_count)))
    if not np.isclose(charges.sum(), -1.0):
        raise ValueError('The complete periodic cell must have charge -1')

    topology, _ = build_topology(atoms)
    settings = resolve_force_settings(
        'openmm',
        float(atoms.cell[0, 0]),
        cutoff=args.nonbonded_cutoff,
        alpha=args.ewald_alpha,
        grid_shape=args.point_pme_grid,
        tolerance=args.pme_error_tolerance,
    )
    classical = build_classical_system(atoms, charges, settings)
    potential = MLPotential('mace', modelPath=str(args.model))
    mixed = potential.createMixedSystem(
        topology,
        classical,
        list(range(n_qm)),
        embedding='mechanical',
        charge=-1,
        multiplicity=1,
        precision='double',
        returnEnergyType='energy',
        mlLongRange=True,
        device=args.torch_device,
    )
    # Add this after createMixedSystem so OpenMM-ML does not remove the
    # restraint as an intraregion classical bonded term.
    if args.restraint_mode == 'radial':
        add_radial_boundary_restraint(
            mixed, args.qm_water_count, args.restraint_radius, args.restraint_k
        )
    elif args.restraint_mode == 'fires':
        add_fires_restraint(mixed, args.qm_water_count, args.restraint_k)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature_k * unit.kelvin,
        (1.0 / args.thermostat_tau_ps) / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    if replay_noise is None:
        integrator.setRandomNumberSeed(
            args.seed if args.thermostat_seed is None else args.thermostat_seed
        )
    else:
        integrator = replay_integrator(
            args.timestep_fs / 1000, args.temperature_k, 1 / args.thermostat_tau_ps
        )
    restart_settings = dict(
        settings.to_dict(),
        integrator='langevin-middle',
        temperature=args.temperature_k,
        timestep_fs=args.timestep_fs,
        thermostat_tau_ps=args.thermostat_tau_ps,
        seed=args.seed,
        thermostat_seed=args.seed
        if args.thermostat_seed is None
        else args.thermostat_seed,
        velocity_profile=args.compatibility_profile,
        qm_water_count=args.qm_water_count,
        platform=args.platform,
        platform_precision=args.platform_precision,
        cpu_threads=args.cpu_threads,
        restraint_mode=args.restraint_mode,
        restraint_radius=args.restraint_radius,
        restraint_k=args.restraint_k,
        charges=charges.tolist(),
        model=str(args.model.resolve()),
        noise_file=str(args.noise_file.resolve()) if args.noise_file else None,
    )
    restart_settings = json.loads(json.dumps(restart_settings))
    platform = openmm.Platform.getPlatformByName(args.platform)
    properties = (
        {'Threads': str(args.cpu_threads)}
        if args.platform == 'CPU'
        else {'Precision': args.platform_precision}
        if args.platform in ('CUDA', 'OpenCL')
        else {}
    )
    simulation = app.Simulation(topology, mixed, integrator, platform, properties)
    positions_nm = np.asarray(atoms.positions) * 0.1
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setVelocities(
        generate_velocities(
            atoms, args.temperature_k, args.seed, args.compatibility_profile
        )
        * unit.nanometer
        / unit.picosecond
    )

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.chk'
    progress_path = args.output / 'progress.json'
    samples_path = args.output / 'samples.csv'
    chunks = args.output / 'chunks'
    if args.trajectory_interval:
        chunks.mkdir(exist_ok=True)
    start_step = 0
    if checkpoint.exists() and progress_path.exists():
        simulation.loadCheckpoint(str(checkpoint))
        saved_progress = json.loads(progress_path.read_text())
        if saved_progress.get('settings') != restart_settings:
            raise ValueError(
                'Checkpoint settings differ or belong to the historical driver; use a new output directory'
            )
        start_step = int(saved_progress['step'])
        simulation.currentStep = start_step
    elif checkpoint.exists() != progress_path.exists():
        raise RuntimeError(
            'Incomplete restart: checkpoint and progress must both exist'
        )

    initial_start = time.perf_counter()
    energy_kj_mol, forces_kj_mol_nm = evaluate(simulation)
    first_eval_seconds = time.perf_counter() - initial_start
    validation = {
        'settings': restart_settings,
        'force_settings': settings.to_dict(),
        'platform': args.platform,
        'platform_properties': properties,
        'openmm_version': openmm.__version__,
        'openmmml_source': str(Path(__import__('openmmml').__file__).resolve()),
        'model': str(args.model),
        'torch_device': args.torch_device,
        'atom_count': len(atoms),
        'qm_atom_count': n_qm,
        'qm_water_count': args.qm_water_count,
        'total_fixed_charge_e': float(charges.sum()),
        'initial_energy_eV': energy_kj_mol / KJ_MOL_PER_EV,
        'initial_force_rms_eV_per_angstrom': (
            float(np.sqrt(np.mean(forces_kj_mol_nm**2))) / KJ_MOL_PER_EV / 10.0
        ),
        'initial_force_max_eV_per_angstrom': (
            float(np.max(np.linalg.norm(forces_kj_mol_nm, axis=1)))
            / KJ_MOL_PER_EV
            / 10.0
        ),
        'first_evaluation_seconds': first_eval_seconds,
        'conditions': {
            'temperature_K': args.temperature_k,
            'timestep_fs': args.timestep_fs,
            'thermostat_tau_ps': args.thermostat_tau_ps,
            'seed': args.seed,
            'restraint_radius_angstrom': (
                args.restraint_radius
                if args.qm_water_count and args.restraint_mode == 'radial'
                else None
            ),
            'restraint_mode': (args.restraint_mode if args.qm_water_count else None),
            'restraint_k_eV_per_angstrom2': (
                args.restraint_k if args.qm_water_count else None
            ),
            'electrostatics': 'OpenMM PME, error tolerance 5e-5',
        },
    }
    if start_step == 0:
        (args.output / 'validation.json').write_text(
            json.dumps(validation, indent=2) + '\n'
        )
        np.savez(
            args.output / 'initial-evaluation.npz',
            positions_angstrom=np.asarray(atoms.positions),
            energy_eV=energy_kj_mol / KJ_MOL_PER_EV,
            forces_eV_per_angstrom=forces_kj_mol_nm / KJ_MOL_PER_EV / 10.0,
        )
    print(json.dumps(validation, indent=2), flush=True)
    if args.steps == 0:
        return

    if start_step == 0:
        initial_frame = atoms.copy()
        initial_frame.info['source_time_fs'] = initial_frame.info.get('time_fs', 0.0)
        initial_frame.info.update(
            time_fs=0.0, potential_energy_eV=energy_kj_mol / KJ_MOL_PER_EV
        )
        initial_frame.info.pop('kinetic_energy_eV', None)
        write(args.output / 'initial.xyz', initial_frame, format='extxyz')
    append = start_step > 0
    with samples_path.open('a' if append else 'w', newline='') as handle:
        fieldnames = [
            'step',
            'time_ps',
            'potential_eV',
            'kinetic_eV',
            'temperature_K',
            'left_carboxyl_H_angstrom',
            'right_carboxyl_H_angstrom',
            'proton_coordinate_angstrom',
            'elapsed_seconds_this_process',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        started = time.perf_counter()
        box_angstrom = float(atoms.cell[0, 0])
        for step in range(
            start_step + args.report_interval, args.steps + 1, args.report_interval
        ):
            if replay_noise is None:
                simulation.step(args.report_interval)
            else:
                for index in range(step - args.report_interval, step):
                    noise = replay_noise[index]
                    if not np.all(np.isfinite(noise)):
                        raise ValueError('Nonfinite replay noise')
                    integrator.setPerDofVariableByName('eta', noise)
                    simulation.step(1)
            state = simulation.context.getState(
                getPositions=True, getEnergy=True, enforcePeriodicBox=False
            )
            positions = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
            )
            potential_ev = (
                state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                / KJ_MOL_PER_EV
            )
            kinetic_ev = (
                state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                / KJ_MOL_PER_EV
            )
            left, right, coordinate = proton_observables(positions, box_angstrom)
            row = {
                'step': step,
                'time_ps': step * args.timestep_fs / 1000.0,
                'potential_eV': potential_ev,
                'kinetic_eV': kinetic_ev,
                'temperature_K': 2
                * kinetic_ev
                / (
                    (3 * len(atoms) - 3)
                    * (
                        OPENMM_BOLTZMANN
                        if args.compatibility_profile == 'openmm'
                        else ase_units.kB
                    )
                ),
                'left_carboxyl_H_angstrom': left,
                'right_carboxyl_H_angstrom': right,
                'proton_coordinate_angstrom': coordinate,
                'elapsed_seconds_this_process': time.perf_counter() - started,
            }
            writer.writerow(row)
            handle.flush()
            if args.trajectory_interval and step % args.trajectory_interval == 0:
                frame = atoms.copy()
                frame.positions = positions
                frame.info['time_fs'] = step * args.timestep_fs
                frame.info['temperature_K'] = row['temperature_K']
                frame.info.update(
                    potential_energy_eV=potential_ev,
                    kinetic_energy_eV=kinetic_ev,
                    left_carboxyl_H_angstrom=left,
                    right_carboxyl_H_angstrom=right,
                )
                frame.info['proton_coordinate_angstrom'] = coordinate
                write(chunks / f'{step:08d}.xyz', frame, format='extxyz')
            if step % args.checkpoint_interval == 0 or step == args.steps:
                simulation.saveCheckpoint(str(checkpoint))
                elapsed = time.perf_counter() - started
                completed_ps = (step - start_step) * args.timestep_fs / 1000.0
                progress = {
                    'settings': restart_settings,
                    'mode': 'openmm-ml-mechanical',
                    'step': step,
                    'time_ps': step * args.timestep_fs / 1000.0,
                    'target_time_ps': args.steps * args.timestep_fs / 1000.0,
                    'elapsed_seconds_this_process': elapsed,
                    'throughput_ps_per_day_this_process': (
                        completed_ps / elapsed * 86400 if elapsed else None
                    ),
                    'last_sample': row,
                }
                progress_path.write_text(json.dumps(progress, indent=2) + '\n')
                print(json.dumps(progress), flush=True)

    state = simulation.context.getState(
        getPositions=True, getEnergy=True, enforcePeriodicBox=False
    )
    final = atoms.copy()
    final.positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    )
    left, right, delta = proton_observables(final.positions, float(atoms.cell[0, 0]))
    final.info.update(
        time_fs=args.steps * args.timestep_fs,
        potential_energy_eV=state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        / KJ_MOL_PER_EV,
        kinetic_energy_eV=state.getKineticEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        / KJ_MOL_PER_EV,
        left_carboxyl_H_angstrom=left,
        right_carboxyl_H_angstrom=right,
        proton_coordinate_angstrom=delta,
    )
    write(args.output / 'final.xyz', final, format='extxyz')


if __name__ == '__main__':
    main()
