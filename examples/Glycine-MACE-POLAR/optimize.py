#!/usr/bin/env python3
"""Fixed-cell ASE BFGS of neutral aqueous glycine with MACE-POLAR."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixBondLength
from ase.io import read, write
from ase.optimize import BFGS
from flax import nnx

from common import graph_edges, initialize_model


class PolarCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, atoms, bundle_path):
        super().__init__()
        (
            self.box,
            self.mode,
            self.data,
            self.graphdef,
            self.params,
            self.neighbor_fn,
            _,
        ) = initialize_model(atoms, bundle_path)
        self.neighbors = self.neighbor_fn.allocate(jnp.asarray(atoms.positions))

        def energy_fn(r, neighbors, params):
            edge_index, shifts, unit_shifts = graph_edges(r, neighbors, self.box)
            inputs = dict(
                self.data,
                positions=r,
                edge_index=edge_index,
                shifts=shifts,
                unit_shifts=unit_shifts,
            )
            return nnx.merge(self.graphdef, params)(
                inputs, compute_force=False, pbc_handling=self.mode
            )['energy'][0]

        self.value_grad = jax.jit(jax.value_and_grad(energy_fn))

    def calculate(self, atoms=None, properties=('energy', 'forces'), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = jnp.asarray(atoms.positions, dtype=jnp.float64)
        self.neighbors = self.neighbor_fn.update(positions, self.neighbors)
        if bool(self.neighbors.did_buffer_overflow):
            raise RuntimeError('Neighbor-list capacity overflow during BFGS')
        energy, gradient = self.value_grad(positions, self.neighbors, self.params)
        self.results = {
            'energy': float(energy),
            'forces': -np.asarray(gradient),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fmax', type=float, default=0.1)
    parser.add_argument('--max-steps', type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.initial)
    if len(atoms) != 166 or list(atoms.numbers[:10]) != [8, 8, 7, 6, 6, 1, 1, 1, 1, 1]:
        raise ValueError('Expected glycine atom order followed by 52 intact waters')
    # Constrain only the donor O-H length while relaxing every other degree
    # of freedom. Otherwise the requested neutral initial state can disappear
    # during BFGS, before the MD starts.
    atoms.set_constraint(FixBondLength(0, 9))
    atoms.calc = PolarCalculator(atoms, args.bundle)
    started = time.perf_counter()
    history = []
    with (args.output / 'forces.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'elapsed_s', 'energy_eV', 'max_projected_force_eV_per_A'])

        def record():
            energy = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            row = [int(dyn.nsteps), time.perf_counter() - started, energy, fmax]
            writer.writerow(row)
            stream.flush()
            history.append(row)
            print(f'BFGS {row[0]}: projected fmax={fmax:.4f} eV/A', flush=True)

        dyn = BFGS(atoms, logfile=None, trajectory=str(args.output / 'bfgs.traj'))
        record()
        dyn.attach(record, interval=5)
        converged = bool(dyn.run(fmax=args.fmax, steps=args.max_steps))
        if history[-1][0] != dyn.nsteps:
            record()
    atoms.set_constraint()
    oh = float(np.linalg.norm(atoms.positions[0] - atoms.positions[9]))
    nh = float(np.linalg.norm(atoms.positions[2] - atoms.positions[9]))
    if oh >= nh or oh > 1.25:
        raise RuntimeError(f'BFGS starting state is not neutral glycine: O-H={oh}, N-H={nh}')
    atoms.calc = None
    if converged:
        write(args.output / 'optimized-neutral.xyz', atoms, format='extxyz')
    result = {
        'converged': converged,
        'steps': dyn.nsteps,
        'fmax_target_eV_per_angstrom': args.fmax,
        'final_projected_fmax_eV_per_angstrom': history[-1][3],
        'initial_structure': str(args.initial),
        'optimized_structure': str(args.output / 'optimized-neutral.xyz'),
        'neutral_OH_distance_angstrom': oh,
        'neutral_NH_distance_angstrom': nh,
        'seconds': time.perf_counter() - started,
        'note': 'O-H bond constrained during BFGS; constraint removed for MD.',
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not converged:
        raise RuntimeError('BFGS did not meet 0.1 eV/angstrom threshold')


if __name__ == '__main__':
    main()
