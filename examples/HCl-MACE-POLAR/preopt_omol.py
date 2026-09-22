#!/usr/bin/env python3
"""Fixed-cell ASE BFGS relaxation of aqueous HCl with MACE-MH-1 OMOL."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from mace.calculators.foundations_models import mace_mp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--fmax', type=float, default=0.05, help='ASE threshold in eV/Å'
    )
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    atoms = read(args.initial)
    if not np.all(atoms.pbc):
        raise ValueError('Expected a periodic initial structure')
    cell = np.asarray(atoms.cell)
    if not np.allclose(cell, np.eye(3) * cell[0, 0]):
        raise ValueError('Expected a cubic periodic cell')
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1
    atoms.calc = mace_mp(
        model='mh-1', head='omol', device=args.device, default_dtype='float64'
    )

    history = []
    started = time.perf_counter()
    with (args.output / 'forces.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['step', 'elapsed_seconds', 'energy_eV', 'max_force_eV_per_angstrom']
        )

        def record() -> None:
            if history and history[-1][0] == dyn.nsteps:
                return
            energy = float(atoms.get_potential_energy())
            fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
            row = [int(dyn.nsteps), time.perf_counter() - started, energy, fmax]
            writer.writerow(row)
            file.flush()
            history.append(row)
            print(
                f'step {row[0]}: max |F| = {1000 * fmax:.3f} meV/Å ({row[1]:.1f} s)',
                flush=True,
            )

        dyn = BFGS(atoms, logfile=None, trajectory=str(args.output / 'trajectory.traj'))
        record()
        dyn.attach(record, interval=args.log_interval)
        converged = bool(dyn.run(fmax=args.fmax, steps=args.max_steps))
        if history[-1][0] != dyn.nsteps:
            record()

    elapsed = time.perf_counter() - started
    write(args.output / 'preoptimized.xyz', atoms, format='extxyz')
    result = {
        'model': 'MACE-MH-1 (OMOL head)',
        'initial': str(args.initial),
        'atoms': len(atoms),
        'box_angstrom': float(cell[0, 0]),
        'charge': 0,
        'spin': 1,
        'device': args.device,
        'force_threshold_mev_per_angstrom': 1000 * args.fmax,
        'steps': dyn.nsteps,
        'converged': converged,
        'initial_max_force_mev_per_angstrom': 1000 * history[0][3],
        'final_max_force_mev_per_angstrom': 1000 * history[-1][3],
        'elapsed_seconds': elapsed,
        'mean_step_seconds': elapsed / dyn.nsteps if dyn.nsteps else None,
        'force_history': history,
    }
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
