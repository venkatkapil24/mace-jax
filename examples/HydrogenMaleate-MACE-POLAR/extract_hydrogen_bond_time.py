#!/usr/bin/env python3
"""Extract the same proton hydrogen-bond geometry from named saved MD runs."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def extract(source, output, key):
    records = []
    chunks = sorted((source / 'chunks').glob('*.xyz'))
    if not chunks:
        raise ValueError(f'No saved structures: {source}')
    for chunk in chunks:
        with chunk.open() as handle:
            while first := handle.readline():
                count = int(first)
                header = handle.readline()
                cell = np.array([float(x) for x in re.search(
                    r'Lattice="([^"]+)"', header)[1].split()]).reshape(3, 3)
                if not np.allclose(cell, np.eye(3) * cell[0, 0]):
                    raise ValueError('This extraction expects the cubic simulation cell')
                box = float(cell[0, 0])
                raw = [handle.readline().split() for _ in range(count)]
                pos = np.array([[float(v) for v in row[1:4]] for row in raw])
                oxygen = np.array([i for i, row in enumerate(raw) if row[0] == 'O'])
                water = oxygen[oxygen >= 11]
                if (len(water) != 52 or raw[10][0] != 'H'
                        or any(row[0] != 'O' for row in raw[:4])):
                    raise ValueError('Unexpected atom order or solvent size')

                def mic(delta):
                    return delta - box * np.rint(delta / box)

                distances = np.linalg.norm(mic(pos[oxygen] - pos[10]), axis=1)
                solute_r = np.linalg.norm(mic(pos[:4] - pos[10]), axis=1)
                donor = int(np.argmin(solute_r))
                nearest_all = int(oxygen[np.argmin(distances)])
                opposite = np.array([2, 3] if donor < 2 else [0, 1])
                dv = mic(pos[donor] - pos[10])
                donor_r = np.linalg.norm(dv)

                def geometry(indices):
                    av = mic(pos[indices] - pos[10])
                    ha = np.linalg.norm(av, axis=1)
                    oo = np.linalg.norm(mic(pos[indices] - pos[donor]), axis=1)
                    angle = np.degrees(np.arccos(np.clip(
                        av @ dv / (ha * donor_r), -1, 1)))
                    bond = ((ha < 2.5) & (oo < 3.5) & (angle > 150)
                            & (donor_r < 1.3) & (nearest_all < 4))
                    near = int(np.argmin(ha))
                    return float(ha[near]), float(angle[near]), bool(np.any(bond))

                ri, ai, bi = geometry(opposite)
                rw, aw, bw = geometry(water)
                time = float(re.search(r'\btime_fs=([-+\d.eE]+)', header)[1]) / 1000
                records.append((time, donor, donor_r, ri, ai, rw, aw,
                                int(bi), int(bw), nearest_all, int(nearest_all >= 11)))
    data = np.array(records)
    interval = float(np.median(np.diff(data[:, 0])))
    if (not np.all(np.isfinite(data)) or interval <= 0
            or not np.allclose(np.diff(data[:, 0]), interval)
            or not np.isclose(data[-1, 0], 100)):
        raise ValueError(f'Incomplete or invalid time sampling: {source}')
    np.savetxt(output / f'{key}.csv', data, delimiter=',', comments='',
               header='time_ps,solute_donor_index,donor_H_distance_angstrom,'
               'internal_H_acceptor_distance_angstrom,internal_D_H_A_angle_degrees,'
               'water_H_acceptor_distance_angstrom,water_D_H_A_angle_degrees,'
               'internal_hbond,water_hbond,nearest_oxygen_index,nearest_oxygen_is_water')
    metadata = {
        'source': str(source.resolve()), 'frames': len(data),
        'saved_frame_interval_ps': interval, 'end_time_ps': float(data[-1, 0]),
        'internal_hbond_fraction': float(data[:, 7].mean()),
        'water_hbond_fraction': float(data[:, 8].mean()),
        'proton_nearest_water_fraction': float(data[:, -1].mean()),
        'maximum_solute_donor_H_distance_angstrom': float(data[:, 2].max()),
    }
    print(key, json.dumps(metadata), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', nargs=2, required=True,
                        metavar=('KEY', 'DIRECTORY'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if len({key for key, _ in args.run}) != len(args.run):
        parser.error('Run keys must be unique')
    result = {
        'definition': 'Donor: nearest solute O to proton atom 10, required <1.3 Angstrom and closer than every water O. Internal acceptors: both O on the opposite carboxyl group. Water acceptors: all 52 solvent O. H bond: H-acceptor <2.5 Angstrom, donor-acceptor <3.5 Angstrom, donor-H-acceptor angle >150 degrees. Both contact types are evaluated independently. All distances use periodic minimum images.',
        'runs': {},
    }
    for key, directory in args.run:
        result['runs'][key] = extract(Path(directory), args.output, key)
    (args.output / 'sources.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
