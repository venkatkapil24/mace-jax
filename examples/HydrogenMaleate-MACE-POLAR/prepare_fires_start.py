#!/usr/bin/env python3
"""Build a fixed-count FIRES partition from a full-ML trajectory frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write


SOLUTE_ATOMS = 11


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    return fractional @ cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-chunk", type=Path, required=True)
    parser.add_argument("--water-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    atoms = read(args.trajectory_chunk, index=-1)
    if len(atoms) != 167 or not 1 <= args.water_count < 52:
        raise ValueError("Expected hydrogen maleate, 52 waters, and 1-51 QM waters")
    time_ps = float(atoms.info["time_fs"]) / 1000.0
    if abs(time_ps - 10.0) > 1e-9:
        raise ValueError(f"Expected the 10 ps frame, found {time_ps} ps")

    cell = np.asarray(atoms.cell.array)
    anchor = atoms.positions[5]
    solute = anchor + minimum_image(atoms.positions[:SOLUTE_ATOMS] - anchor, cell)
    water_oxygen = atoms.positions[SOLUTE_ATOMS::3]
    distances = np.linalg.norm(
        minimum_image(water_oxygen[:, None] - solute[None, :8], cell), axis=-1
    ).min(axis=1)
    order_water = np.argsort(distances)
    qm_water = order_water[:args.water_count]
    mm_water = order_water[args.water_count:]
    atom_order = list(range(SOLUTE_ATOMS))
    for water in np.concatenate((qm_water, mm_water)):
        atom_order.extend(range(SOLUTE_ATOMS + 3 * water, SOLUTE_ATOMS + 3 * water + 3))

    ordered = atoms[atom_order]
    ordered.info["qm_water_count"] = args.water_count
    ordered.info["qm_charge"] = -1
    ordered.info["qm_spin"] = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, ordered, format="extxyz")
    metadata = {
        "source": str(args.trajectory_chunk),
        "source_time_ps": time_ps,
        "distance_coordinate": "water O to nearest of 8 solute heavy atoms",
        "qm_water_count": args.water_count,
        "mm_water_count": 52 - args.water_count,
        "qm_water_indices_in_source_zero_based": qm_water.tolist(),
        "outermost_qm_distance_angstrom": float(distances[qm_water[-1]]),
        "nearest_mm_distance_angstrom": float(distances[mm_water[0]]),
        "atom_order_in_source": atom_order,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
