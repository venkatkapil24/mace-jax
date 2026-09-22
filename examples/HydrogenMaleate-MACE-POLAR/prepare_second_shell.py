#!/usr/bin/env python3
"""Extend the fixed first-shell QM partition by one water-neighbor shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from md import generate_velocities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=float, default=3.5)
    parser.add_argument('--temperature-k', type=float, default=330.0)
    parser.add_argument('--seed', type=int, default=20260922)
    args = parser.parse_args()

    atoms = read(args.initial)
    if len(atoms) != 167:
        raise ValueError('Expected hydrogen maleate and 52 complete waters')
    first_count = int(atoms.info['qm_water_count'])
    if not 1 <= first_count < 52:
        raise ValueError('Invalid first-shell water count')
    box = float(atoms.cell[0, 0])
    water_o = atoms.positions[11::3]
    displacement = water_o[first_count:, None] - water_o[None, :first_count]
    displacement -= box * np.rint(displacement / box)
    nearest = np.linalg.norm(displacement, axis=-1).min(axis=-1)
    second = np.flatnonzero(nearest < args.cutoff) + first_count
    remaining = np.flatnonzero(nearest >= args.cutoff) + first_count
    if len(second) == 0 or len(remaining) == 0:
        raise ValueError('Second shell must leave both QM and MM waters')

    order = list(range(11))
    for water in np.concatenate((np.arange(first_count), second, remaining)):
        order.extend(range(11 + 3 * int(water), 14 + 3 * int(water)))
    ordered = atoms[order]
    ordered.info['qm_water_count'] = first_count + len(second)
    ordered.info['first_shell_water_count'] = first_count
    ordered.info['second_shell_water_count'] = len(second)
    ordered.arrays['source_atom_index'] = np.asarray(order, dtype=np.int32)

    # The existing runs drew initial velocities in the first-shell atom order.
    # Permuting that array keeps the same velocity on each physical atom.
    velocities = generate_velocities(atoms, args.temperature_k, args.seed)[order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, ordered, format='extxyz')
    velocity_path = args.output.with_name('initial-velocities.npy')
    np.save(velocity_path, velocities)
    metadata = {
        'initial_structure': str(args.initial),
        'second_shell_cutoff_water_OO_angstrom': args.cutoff,
        'first_shell_water_count': first_count,
        'second_shell_water_count': len(second),
        'qm_water_count': first_count + len(second),
        'mm_water_count': len(remaining),
        'second_shell_water_indices_in_first_shell_structure': second.tolist(),
        'second_shell_nearest_first_shell_OO_angstrom': nearest[second - first_count].tolist(),
        'atom_order_in_first_shell_structure': order,
        'initial_velocities': str(velocity_path),
        'temperature_k': args.temperature_k,
        'seed': args.seed,
    }
    (args.output.parent / 'partition.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: v for k, v in metadata.items() if k not in (
        'atom_order_in_first_shell_structure',
        'second_shell_nearest_first_shell_OO_angstrom',
    )}, indent=2), flush=True)


if __name__ == '__main__':
    main()
