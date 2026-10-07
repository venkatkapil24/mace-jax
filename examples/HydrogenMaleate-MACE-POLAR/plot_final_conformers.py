#!/usr/bin/env python3
"""Compare the final hydrogen-maleate conformers from four 100 ps runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.collections import LineCollection


SOLUTE_ATOMS = 11
HEAVY_ATOMS = 8


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def unwrap_solute(path: Path, box: float) -> tuple[np.ndarray, np.ndarray]:
    atoms = read(path)
    raw = np.asarray(atoms.positions[:SOLUTE_ATOMS])
    anchor = raw[5]
    positions = anchor + minimum_image(raw - anchor, box)
    return positions, np.asarray(atoms.numbers[:SOLUTE_ATOMS])


def reference_frame(initial: Path, box: float) -> tuple[np.ndarray, np.ndarray]:
    positions, _ = unwrap_solute(initial, box)
    heavy = positions[:HEAVY_ATOMS]
    origin = heavy.mean(axis=0)
    x_axis = heavy[2:4].mean(axis=0) - heavy[:2].mean(axis=0)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = heavy[0] - heavy[1]
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    target = (positions - origin) @ basis
    return target, basis


def align_to_reference(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    moving_heavy = positions[:HEAVY_ATOMS] - positions[:HEAVY_ATOMS].mean(axis=0)
    target_heavy = target[:HEAVY_ATOMS] - target[:HEAVY_ATOMS].mean(axis=0)
    u, _, vt = np.linalg.svd(moving_heavy.T @ target_heavy)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    centered = positions - positions[:HEAVY_ATOMS].mean(axis=0)
    return centered @ rotation


def infer_bonds(initial: Path, box: float) -> list[tuple[int, int]]:
    positions, numbers = unwrap_solute(initial, box)
    bonds: list[tuple[int, int]] = []
    # The transferred proton (index 10) is drawn separately against each endpoint.
    for i in range(10):
        for j in range(i + 1, 10):
            if numbers[i] == numbers[j] == 1:
                continue
            cutoff = 1.25 if 1 in (numbers[i], numbers[j]) else 1.80
            if np.linalg.norm(positions[i] - positions[j]) < cutoff:
                bonds.append((i, j))
    return bonds


def dihedral(positions: np.ndarray, indices: tuple[int, int, int, int]) -> float:
    a, b, c, d = (positions[i] for i in indices)
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def draw_molecule(
    axis: plt.Axes,
    positions: np.ndarray,
    numbers: np.ndarray,
    bonds: list[tuple[int, int]],
    coordinates: tuple[int, int],
) -> None:
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#c7ccd0")
        spine.set_linewidth(0.8)

    x, y = coordinates
    segments = [positions[[i, j]][:, [x, y]] for i, j in bonds]
    axis.add_collection(LineCollection(segments, colors="#4b5358", linewidths=3.0, zorder=2))

    proton = positions[10]
    oxygen_distance = np.linalg.norm(positions[:4] - proton, axis=1)
    closest = np.argsort(oxygen_distance)[:2]
    axis.plot(
        positions[[closest[0], 10], x], positions[[closest[0], 10], y],
        color="#70427d", linewidth=3.0, zorder=2,
    )
    axis.plot(
        positions[[closest[1], 10], x], positions[[closest[1], 10], y],
        color="#9b75a5", linewidth=1.5, linestyle=(0, (3, 2)), zorder=1,
    )

    styles = {
        1: ("white", "#596168", 70),
        6: ("#4b5358", "white", 170),
        8: ("#d95145", "white", 180),
    }
    for atomic_number, (face, edge, size) in styles.items():
        indices = np.flatnonzero(numbers == atomic_number)
        if atomic_number == 1:
            indices = indices[indices != 10]
        axis.scatter(
            positions[indices, x], positions[indices, y], s=size,
            c=face, edgecolors=edge, linewidths=0.9, zorder=3,
        )
    axis.scatter(
        positions[10, x], positions[10, y], s=100, c="#ad57bb",
        edgecolors="white", linewidths=0.9, zorder=4,
    )

    shown = positions[:, [x, y]]
    center = shown.mean(axis=0)
    half_span = max(np.ptp(shown[:, 0]), np.ptp(shown[:, 1])) / 2 + 0.65
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = args.structures / "initial.xyz"
    box = float(read(initial).cell[0, 0])
    target, _ = reference_frame(initial, box)
    bonds = infer_bonds(initial, box)
    sources = [
        ("ML", "ml-100ps.xyz"),
        ("ML/MM electrostatic", "electrostatic-100ps.xyz"),
        ("ML/MM mechanical", "mechanical-100ps.xyz"),
        ("ML/MM mechanical (OpenMM)", "openmm-mechanical-100ps.xyz"),
    ]

    conformers = []
    for label, filename in sources:
        positions, numbers = unwrap_solute(args.structures / filename, box)
        aligned = align_to_reference(positions, target)
        left_twist = abs(dihedral(aligned, (0, 6, 4, 5)))
        right_twist = abs(dihedral(aligned, (2, 7, 5, 4)))
        conformers.append((label, aligned, numbers, left_twist, right_twist))

    figure, axes = plt.subplots(2, 4, figsize=(15.2, 7.6), dpi=180)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.07, top=0.90, wspace=0.10, hspace=0.18)
    for column, (label, positions, numbers, left_twist, right_twist) in enumerate(conformers):
        draw_molecule(axes[0, column], positions, numbers, bonds, (0, 1))
        draw_molecule(axes[1, column], positions, numbers, bonds, (0, 2))
        axes[0, column].set_title(label, fontsize=12.5, weight="bold", pad=9)
        axes[1, column].text(
            0.5, -0.10,
            rf"carboxyl twists: $|\phi_L|={left_twist:.0f}^\circ$, $|\phi_R|={right_twist:.0f}^\circ$",
            transform=axes[1, column].transAxes, ha="center", va="top", fontsize=10,
        )

    axes[0, 0].text(-0.11, 0.5, "molecular-plane view", transform=axes[0, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=11, weight="bold")
    axes[1, 0].text(-0.11, 0.5, "edge-on view", transform=axes[1, 0].transAxes,
                    rotation=90, ha="center", va="center", fontsize=11, weight="bold")
    figure.suptitle(
        "Hydrogen maleate conformers after 100 ps at 330 K",
        fontsize=17, weight="bold", y=0.965,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    for label, _, _, left_twist, right_twist in conformers:
        print(f"{label}: left={left_twist:.2f} deg, right={right_twist:.2f} deg")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
