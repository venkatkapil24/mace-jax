#!/usr/bin/env python3
"""Isolate FIRES cost and verify native OpenMM against JAX and analytic forces.

This is a force-kernel benchmark, not physical MD: tiny position changes force
reevaluation without changing the test configuration appreciably. No ML model
is evaluated. Times exclude compilation and include device synchronization.
"""

import argparse
import ast
import gc
import json
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import openmm
from ase.io import read
from openmm import unit

from mace_jax.modules.fires import fires_energy

KJ_MOL_PER_EV = 96.48533212331002


def extract(name):
    path = Path(__file__).with_name('openmm_mechanical.py')
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = dict(np=np, openmm=openmm, unit=unit, KJ_MOL_PER_EV=KJ_MOL_PER_EV)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(path),
            'exec',
        ),
        namespace,
    )
    return namespace[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    jax.config.update('jax_enable_x64', True)
    atoms = read(args.initial)
    box = float(atoms.cell[0, 0])
    nq = 86
    k = 500 * 4.184 / KJ_MOL_PER_EV
    initial = atoms.positions.copy()
    active = initial.copy()
    delta = initial[11:86:3, None] - initial[None, :8]
    delta -= box * np.rint(delta / box)
    distances = np.linalg.norm(delta, axis=-1)
    qm_distance = distances.min(axis=1)
    outer = int(qm_distance.argmax())
    heavy = int(distances[outer].argmin())
    direction = delta[outer, heavy] / qm_distance[outer]
    active[nq] = initial[heavy] + direction * (qm_distance[outer] - 0.2)

    def energy(r):
        return fires_energy(r[11:nq:3], r[nq::3], r[:8], box, k)

    force = jax.jit(jax.value_and_grad(energy))
    start = time.perf_counter()
    jax.block_until_ready(force(jnp.array(active)))
    report = dict(
        qm_water_count=25,
        mm_water_count=27,
        k_eV_per_A2=k,
        jax_force_compile_seconds=time.perf_counter() - start,
        scope='FIRES only; no ML evaluation or physical MD',
        correctness=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    analytic = extract('_compute_fires')
    factory = extract('create_fires_force')
    platform = openmm.Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'double', 'DeviceIndex': '0'}
    frames = {'inactive': initial, 'active': active}
    cases = [
        ('baseline', None),
        ('python', partial(analytic, qm_water_count=25, k_ev_angstrom2=k)),
        ('native', factory(25, k)),
    ]
    for name, restraint in cases:
        system = openmm.System()
        for _ in initial:
            system.addParticle(1)
        system.setDefaultPeriodicBoxVectors(
            *[openmm.Vec3(*v) * unit.angstrom for v in atoms.cell]
        )
        if name == 'python':
            restraint = openmm.PythonForce(restraint)
            restraint.setUsesPeriodicBoundaryConditions(True)
        if restraint is not None:
            system.addForce(restraint)
        # Force a fresh force evaluation per iteration. This is not an MD integrator.
        integrator = openmm.CustomIntegrator(0.0005)
        integrator.addComputePerDof('x', 'x+1e-8')
        integrator.addComputePerDof('v', 'f')
        start = time.perf_counter()
        context = openmm.Context(system, integrator, platform, properties)
        context.setPositions(active * unit.angstrom)
        context.getState(getEnergy=True, getForces=True)
        report[name + '_context_compile_seconds'] = time.perf_counter() - start
        print('CONTEXT', name, report[name + '_context_compile_seconds'], flush=True)
        for label, r in frames.items():
            context.setPositions(r * unit.angstrom)
            state = context.getState(getPositions=True, getEnergy=True, getForces=True)
            if name != 'baseline':
                ej, gj = force(jnp.array(r))
                en = (
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    / KJ_MOL_PER_EV
                )
                fn = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                ) / (10 * KJ_MOL_PER_EV)
                em, fm = analytic(state, 25, k)
                row = dict(
                    method=name,
                    configuration=label,
                    energy_eV=en,
                    energy_error_eV=abs(en - float(ej)),
                    force_error_eV_per_A=float(np.max(abs(fn + np.asarray(gj)))),
                    analytic_force_error_eV_per_A=float(
                        np.max(abs(fn - fm / (10 * KJ_MOL_PER_EV)))
                    ),
                )
                assert (
                    row['energy_error_eV'] < 1e-9 and row['force_error_eV_per_A'] < 1e-8
                ), row
                assert row['analytic_force_error_eV_per_A'] < 1e-8, row
                report['correctness'].append(row)
            integrator.step(50)
            windows = []
            for _ in range(3):
                start = time.perf_counter()
                integrator.step(1000)
                context.getState(getEnergy=True)  # synchronize the GPU
                windows.append((time.perf_counter() - start) / 1000)
            report[name + '_' + label + '_seconds_per_force_iteration'] = windows
            print(name, label, 'seconds/iteration', np.median(windows), flush=True)
            save()
        del context, integrator, system, restraint
        gc.collect()

    @jax.jit
    def scan(r):
        def step(r, unused):
            e, g = force(r)
            return r + 1e-12 * g, e

        return jax.lax.scan(step, r, None, length=1000)

    start = time.perf_counter()
    compiled = scan.lower(jnp.array(active)).compile()
    report['jax_scan_compile_seconds'] = time.perf_counter() - start
    for label, r in frames.items():
        r = jnp.array(r)
        jax.block_until_ready(compiled(r))
        windows = []
        for _ in range(3):
            start = time.perf_counter()
            jax.block_until_ready(compiled(r))
            windows.append((time.perf_counter() - start) / 1000)
        report['jax_' + label + '_seconds_per_force_iteration'] = windows
        print('jax', label, 'seconds/iteration', np.median(windows), flush=True)
    report['passed'] = True
    save()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
