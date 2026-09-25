#!/usr/bin/env python3
"""Prepare conventional fixed charges for OpenMM-style mechanical embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read


# Published classical hydrogen-maleate charges, reordered from the paper's
# O,H,O,C,C,H,C,H,C,O,O topology to this example's O,O,O,O,C,C,C,C,H,H,H
# atom order. The acidic-H value absorbs the 1e-6 e rounding residual so the
# molecular charge is exactly -1.
HYDROGEN_MALEATE_Q = np.asarray(
    [
        -0.823123,
        -0.707777,
        -0.823123,
        -0.707777,
        -0.327958,
        -0.327958,
        0.907584,
        0.907584,
        0.144250,
        0.144250,
        0.614048,
    ]
)
TIP3P_Q = np.asarray([-0.834, 0.417, 0.417])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qm-water-count", type=int, default=13)
    args = parser.parse_args()

    atoms = read(args.initial)
    n_qm = 11 + 3 * args.qm_water_count
    expected_symbols = "OOOOCCCCHHH" + "OHH" * args.qm_water_count
    if "".join(atoms.get_chemical_symbols()[:n_qm]) != expected_symbols:
        raise ValueError("Unexpected QM atom order")
    if int(atoms.info.get("qm_water_count", -1)) != args.qm_water_count:
        raise ValueError("QM water count differs from initial structure metadata")
    solute = np.asarray(atoms.positions[:11])
    expected_bonds = ((0, 6), (1, 6), (2, 7), (3, 7), (4, 6), (4, 5),
                      (5, 7), (8, 4), (9, 5), (10, 0))
    if any(np.linalg.norm(solute[i] - solute[j]) > 1.7 for i, j in expected_bonds):
        raise ValueError("Hydrogen-maleate connectivity differs from charge mapping")

    charges = np.concatenate(
        (HYDROGEN_MALEATE_Q, np.tile(TIP3P_Q, args.qm_water_count))
    )
    if charges.shape != (n_qm,) or not np.isclose(charges.sum(), -1.0, atol=1e-12):
        raise RuntimeError("Invalid fixed-charge array")

    payload = {
        "definition": (
            "Conventional fixed point charges used for QM-MM electrostatics in "
            "OpenMM-style mechanical embedding"
        ),
        "solute_charge_source": (
            "Vener et al., Int. J. Mol. Sci. 23, 6302 (2022), DOI "
            "10.3390/ijms23116302, Supplementary Section S2; reordered to the "
            "example atom order"
        ),
        "solute_charge_note": (
            "Acidic-H charge changed from 0.614049 to 0.614048 e to remove the "
            "1e-6 e decimal-rounding residual in the published table"
        ),
        "water_charge_source": "AMBER14 TIP3P: O=-0.834 e, H=+0.417 e",
        "initial": str(args.initial),
        "qm_water_count": args.qm_water_count,
        "atom_count": n_qm,
        "total_charge": float(np.sum(charges)),
        "atomic_numbers": np.asarray(atoms.numbers[:n_qm]).tolist(),
        "charges": charges.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
