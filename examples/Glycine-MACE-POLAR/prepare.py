#!/usr/bin/env python3
"""Build a reproducible neutral IIp-like glycine / 52-water starting cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glycine-sdf', type=Path, required=True)
    parser.add_argument('--water-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--box', type=float, default=11.76)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gly = read(args.glycine_sdf)
    if len(gly) != 10 or sorted(gly.get_chemical_symbols()) != sorted(
        ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'N', 'O', 'O']
    ):
        raise ValueError('Expected neutral C2H5NO2 glycine from PubChem')
    # PubChem CID 750 SDF atom order: OH O, carbonyl O, N, CA, C, five H.
    acid_o, carbonyl_o, nitrogen, ca, carbonyl_c, acid_h = 0, 1, 2, 3, 4, 9
    if np.linalg.norm(gly.positions[acid_o] - gly.positions[acid_h]) > 1.2:
        raise ValueError('PubChem OH assignment changed')
    # Rotate the carboxyl group around CA-C by 180 degrees to obtain the
    # neutral conformer whose O-H points toward N, as in the IIp pathway.
    axis = gly.positions[carbonyl_c] - gly.positions[ca]
    axis /= np.linalg.norm(axis)
    origin = gly.positions[carbonyl_c].copy()
    for atom_index in (acid_o, carbonyl_o, acid_h):
        offset = gly.positions[atom_index] - origin
        gly.positions[atom_index] = origin + 2 * np.dot(offset, axis) * axis - offset
    # Rotate the OH proton about the C-O bond so that it faces the amine.
    oh_axis = gly.positions[acid_o] - gly.positions[carbonyl_c]
    oh_axis /= np.linalg.norm(oh_axis)
    oh_offset = gly.positions[acid_h] - gly.positions[acid_o]
    candidates = []
    for angle in np.linspace(0.0, 2*np.pi, 721):
        rotated = (
            oh_offset*np.cos(angle)
            + np.cross(oh_axis, oh_offset)*np.sin(angle)
            + oh_axis*np.dot(oh_axis, oh_offset)*(1-np.cos(angle))
        )
        candidates.append(gly.positions[acid_o] + rotated)
    gly.positions[acid_h] = min(
        candidates, key=lambda p: np.linalg.norm(p-gly.positions[nitrogen])
    )
    gly.translate(np.full(3, args.box / 2) - gly.positions.mean(axis=0))
    gly.cell = np.eye(3) * args.box
    gly.pbc = True

    source = read(args.water_source)
    source_box = float(source.cell[0, 0])
    oxygens = np.flatnonzero(source.numbers == 8)
    hydrogens = np.flatnonzero(source.numbers == 1)
    if len(oxygens) != 63 or len(hydrogens) != 127:
        raise ValueError('Expected the published 63-water HCl snapshot')
    displacement = minimum_image(
        source.positions[hydrogens, None] - source.positions[oxygens][None],
        source_box,
    )
    parent = np.argmin(np.linalg.norm(displacement, axis=-1), axis=1)
    waters = []
    for local_o, oxygen in enumerate(oxygens):
        hs = hydrogens[parent == local_o]
        if len(hs) == 2:
            waters.append([int(oxygen), *map(int, hs)])
    if len(waters) != 62:
        raise ValueError('Could not identify 62 intact waters')
    scale = args.box / source_box
    center = np.full(3, args.box / 2)
    # Delete the ten waters closest to the glycine insertion site.
    water_dist = [
        np.linalg.norm(minimum_image(source.positions[triplet[0]] * scale - center, args.box))
        for triplet in waters
    ]
    chosen = sorted(np.argsort(water_dist)[-52:])
    water_indices = [index for i in chosen for index in waters[i]]
    water = source[water_indices]
    # Scale oxygen centers, preserving intramolecular O-H vectors.
    water_positions = []
    for i in range(52):
        triplet = water.positions[3 * i:3 * i + 3]
        oxygen = triplet[0] * scale
        oh = minimum_image(triplet[1:] - triplet[0], source_box)
        water_positions.extend([oxygen, oxygen + oh[0], oxygen + oh[1]])
    water.positions = np.asarray(water_positions) % args.box
    water.cell = np.eye(3) * args.box
    water.pbc = True
    atoms = gly + water
    atoms.info['charge'] = 0
    atoms.info['spin'] = 1
    write(args.output / 'initial-neutral.xyz', atoms, format='extxyz')
    info = {
        'glycine_source': 'https://pubchem.ncbi.nlm.nih.gov/compound/750',
        'glycine_conformer': 'PubChem 3D conformer with 180-degree CA-C carboxyl rotation toward IIp',
        'water_source': str(args.water_source),
        'water_source_publication': 'https://www.nature.com/articles/s41467-025-60794-2',
        'water_source_dataset': 'https://doi.org/10.5281/zenodo.15490289',
        'selected_water_source_indices': [waters[i] for i in chosen],
        'box_angstrom': args.box,
        'glycine_atom_indices': list(range(10)),
        'acid_oxygen_index': acid_o,
        'nitrogen_index': nitrogen,
        'acid_hydrogen_index': acid_h,
        'oxygen_hydrogen_angstrom': float(np.linalg.norm(atoms.positions[acid_o] - atoms.positions[acid_h])),
        'nitrogen_hydrogen_angstrom': float(np.linalg.norm(atoms.positions[nitrogen] - atoms.positions[acid_h])),
    }
    (args.output / 'structure.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({k: v for k, v in info.items() if k != 'selected_water_source_indices'}, indent=2))


if __name__ == '__main__':
    main()
