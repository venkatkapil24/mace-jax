#!/usr/bin/env python3
"""Put first-shell waters next to glycine for a fixed POLAR/MM partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimized', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cutoff', type=float, default=3.8)
    args = parser.parse_args()
    atoms = read(args.optimized)
    if len(atoms) != 166:
        raise ValueError('Expected glycine and 52 complete waters')
    box = float(atoms.cell[0, 0])
    # First shell is defined by each water oxygen's closest periodic distance
    # to the three polar glycine atoms: acid O, carbonyl O, and N.
    delta = atoms.positions[10::3, None] - atoms.positions[[0, 1, 2]][None]
    delta -= box * np.rint(delta / box)
    distances = np.linalg.norm(delta, axis=-1).min(axis=-1)
    qm_waters = np.flatnonzero(distances < args.cutoff)
    mm_waters = np.flatnonzero(distances >= args.cutoff)
    if not 1 <= len(qm_waters) < 52:
        raise ValueError(f'Unexpected first-shell count: {len(qm_waters)}')
    order = list(range(10))
    for water in np.concatenate((qm_waters, mm_waters)):
        order.extend(range(10 + 3*int(water), 13 + 3*int(water)))
    ordered = atoms[order]
    ordered.info['qm_water_count'] = int(len(qm_waters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, ordered, format='extxyz')
    metadata = {
        'optimized_source': str(args.optimized),
        'md_initial': str(args.output),
        'first_shell_cutoff_angstrom': args.cutoff,
        'qm_water_count': int(len(qm_waters)),
        'mm_water_count': int(len(mm_waters)),
        'qm_water_indices_in_optimized_source': qm_waters.tolist(),
        'first_shell_distances_angstrom': distances[qm_waters].tolist(),
        'atom_order_in_optimized_source': order,
    }
    (args.output.parent / 'partition.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
