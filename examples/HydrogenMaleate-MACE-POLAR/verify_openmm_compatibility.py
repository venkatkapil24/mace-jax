#!/usr/bin/env python3
"""Same-coordinate and supplied-noise checks against OpenMM-ML, without MD production runs."""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
import jax
import jax.numpy as jnp
import numpy as np
import openmm
import torch
from ase import units
from ase.io import read
from md import build_energy
from openmm import app, unit
from openmm_mechanical import (
    build_classical_system,
    build_topology,
    evaluate,
    replay_integrator,
)
from openmmml import MLPotential
from simulation_settings import (
    KJ_MOL_PER_EV,
    TIP3P_Q,
    generate_velocities,
    resolve_force_settings,
)

from mace_jax.modules.langevin import kinetic_energy, middle_step
from mace_jax.tools.physical_constants import OPENMM_BOLTZMANN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[3]
    example = root / 'Examples/HydrogenMaleate-MoleculeOnly'
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument(
        '--output',
        type=Path,
        default=root
        / 'Examples/HydrogenMaleate-Analysis/openmm-compatibility-verification.json',
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    initial = read(example / 'structure/initial-100ps-full-ml.xyz')
    q = np.asarray(
        json.loads((example / 'structure/molecule-fixed-charges.json').read_text())[
            'charges'
        ]
    )
    charges = np.concatenate((q, np.tile(TIP3P_Q, 52)))
    settings = resolve_force_settings('openmm', float(initial.cell[0, 0]))
    topology, _ = build_topology(initial)
    mixed = MLPotential(
        'mace', modelPath=str(root / 'artifacts/MACE-POLAR-1-M.model')
    ).createMixedSystem(
        topology,
        build_classical_system(initial, charges, settings),
        list(range(11)),
        embedding='mechanical',
        charge=-1,
        multiplicity=1,
        precision='double',
        returnEnergyType='energy',
        mlLongRange=True,
        device='cpu',
        forceGroup=4,
    )
    # The ML adapter owns group 4. The independent force groups expose each
    # retained classical force, including the subtracted QM-only PME force.
    groups = {}
    group = 0
    for force in mixed.getForces():
        if force.getForceGroup() != 4:
            force.setForceGroup(group)
            groups[group] = force.__class__.__name__
            group += 1
    integrator = replay_integrator(0.0005, 330.0, 1.0)
    simulation = app.Simulation(
        topology, mixed, integrator, openmm.Platform.getPlatformByName('Reference')
    )
    force_eval, params, neighbor_fn, shift, n_qm, terms_eval = build_energy(
        initial,
        root / 'artifacts/MACE-POLAR-1-M-jax-fixed.msgpack',
        'mechanical',
        None,
        6,
        0,
        'none',
        4.2,
        0.2,
        q,
        None,
        0.25,
        'pme',
        0.5,
        8,
        compatibility_profile='openmm',
        return_terms=True,
    )
    output = {
        'settings': settings.to_dict(),
        'platform': 'OpenMM Reference / JAX CPU float64',
        'versions': {
            'jax': jax.__version__,
            'torch': torch.__version__,
            'openmm': openmm.__version__,
        },
        'configurations': {},
        'steps_with_supplied_noise': args.steps,
    }
    frames = {
        'initial': initial,
        'openmm_final': read(example / 'openmm-mechanical-100ps/final.xyz'),
        'jax_final': read(example / 'mechanical-100ps/final.xyz'),
    }
    start = time.perf_counter()
    for name, atoms in frames.items():
        r = jnp.asarray(atoms.positions)
        neighbors = neighbor_fn.allocate(r)
        if neighbors.did_buffer_overflow:
            raise RuntimeError('Neighbor buffer overflow in verification')
        energy, grad = force_eval(r, neighbors, params)
        terms = {
            key: float(value) for key, value in terms_eval(r, neighbors, params).items()
        }
        simulation.context.setPositions(atoms.positions * 0.1)
        om_energy, om_force = evaluate(simulation)
        om_energy /= KJ_MOL_PER_EV
        om_force /= KJ_MOL_PER_EV * 10
        om_ml, om_ml_force = evaluate(simulation, 1 << 4)
        classical = {
            f'{key}:{value}': evaluate(simulation, 1 << key)[0] / KJ_MOL_PER_EV
            for key, value in groups.items()
        }
        difference = -np.asarray(grad) - om_force
        row = {
            'energy_difference_eV': float(energy) - om_energy,
            'force_component_rms_eV_per_angstrom': float(
                np.sqrt(np.mean(difference**2))
            ),
            'max_atom_force_difference_eV_per_angstrom': float(
                np.linalg.norm(difference, axis=1).max()
            ),
            'jax_terms_eV': terms,
            'openmm_classical_terms_eV': classical,
            'ml_energy_difference_eV': terms['ml'] - om_ml / KJ_MOL_PER_EV,
            'classical_energy_difference_eV': sum(
                value for key, value in terms.items() if key != 'ml'
            )
            - sum(classical.values()),
        }
        output['configurations'][name] = row
        print(name, json.dumps(row), flush=True)
    r = jnp.asarray(initial.positions)
    v = jnp.asarray(
        generate_velocities(initial, 330.0, 20260922, 'openmm') * units.fs * 1000
    )
    masses = jnp.asarray(initial.get_masses()[:, None])
    neighbors = neighbor_fn.allocate(r)
    energy, grad = force_eval(r, neighbors, params)
    simulation.context.setPositions(np.asarray(r) * 0.1)
    simulation.context.setVelocities(np.asarray(v) * 0.1)

    @jax.jit
    def step(r, v, neighbors, grad, noise):
        r, v = middle_step(
            r,
            v,
            -grad,
            masses,
            noise,
            timestep_ps=0.0005,
            temperature=330.0,
            friction_ps=1.0,
            shift=shift,
        )
        neighbors = neighbor_fn.update(r, neighbors)
        energy, grad = force_eval(r, neighbors, params)
        return r, v, neighbors, energy, grad

    maximum_position = maximum_velocity = maximum_kinetic = 0.0
    rng = np.random.default_rng(31)
    for _ in range(args.steps):
        noise = rng.normal(size=(len(initial), 3))
        integrator.setPerDofVariableByName('eta', noise)
        r, v, neighbors, energy, grad = step(r, v, neighbors, grad, jnp.asarray(noise))
        integrator.step(1)
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        om_r = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        delta = np.asarray(r) - om_r
        box = float(initial.cell[0, 0])
        delta -= box * np.rint(delta / box)
        maximum_position = max(maximum_position, float(np.abs(delta).max()))
        maximum_velocity = max(
            maximum_velocity,
            float(
                np.abs(
                    np.asarray(v)
                    - state.getVelocities(asNumpy=True).value_in_unit(
                        unit.nanometer / unit.picosecond
                    )
                    * 10
                ).max()
            ),
        )
        maximum_kinetic = max(
            maximum_kinetic,
            abs(
                float(kinetic_energy(v, masses))
                - state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                / KJ_MOL_PER_EV
            ),
        )
    output['replayed_integration'] = {
        'maximum_position_difference_angstrom': maximum_position,
        'maximum_velocity_difference_angstrom_per_ps': maximum_velocity,
        'maximum_kinetic_energy_difference_eV': maximum_kinetic,
    }
    output['elapsed_seconds'] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['replayed_integration']), flush=True)
    assert (
        max(
            abs(row['energy_difference_eV'])
            for row in output['configurations'].values()
        )
        < 1e-7
    )
    assert (
        max(
            abs(row['classical_energy_difference_eV'])
            for row in output['configurations'].values()
        )
        < 1e-8
    )
    assert (
        max(
            row['max_atom_force_difference_eV_per_angstrom']
            for row in output['configurations'].values()
        )
        < 1e-7
    )
    assert maximum_position < 1e-7 and maximum_velocity < 1e-5


if __name__ == '__main__':
    main()
