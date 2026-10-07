#!/usr/bin/env python3
"""Compute solute-COM and carbon-site water-oxygen RDFs for hydrogen maleate."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOLUTE_ATOMS = 11
WATER_COUNT = 52
ATOM_COUNT = SOLUTE_ATOMS + 3 * WATER_COUNT
TIME = re.compile(r"\btime_fs=([-+\d.eE]+)")
LATTICE = re.compile(r'Lattice="([^"]+)"')
MASSES = np.array([15.999] * 4 + [12.011] * 4 + [1.008] * 3)


def minimum_image(delta: np.ndarray, cell: np.ndarray) -> np.ndarray:
    fractional = delta @ np.linalg.inv(cell)
    fractional -= np.rint(fractional)
    return fractional @ cell


def iter_frames(directory: Path, stride: int):
    frame_index = 0
    for chunk in sorted((directory / "chunks").glob("*.xyz")):
        with chunk.open() as handle:
            while count_line := handle.readline():
                count = int(count_line)
                header = handle.readline()
                if count != ATOM_COUNT:
                    raise ValueError(f"Unexpected atom count {count} in {chunk}")
                time_match = TIME.search(header)
                lattice_match = LATTICE.search(header)
                if time_match is None or lattice_match is None:
                    raise ValueError(f"Missing time or lattice in {chunk}")
                positions = np.empty((count, 3), dtype=float)
                for atom in range(count):
                    fields = handle.readline().split()
                    positions[atom] = [float(value) for value in fields[1:4]]
                if frame_index % stride == 0:
                    cell = np.asarray(
                        [float(value) for value in lattice_match.group(1).split()]
                    ).reshape(3, 3)
                    yield float(time_match.group(1)) / 1000.0, cell, positions
                frame_index += 1


def smooth(values: np.ndarray, width: int = 9) -> np.ndarray:
    kernel = np.hanning(width)
    kernel /= kernel.sum()
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def first_minimum(radius: np.ndarray, rdf: np.ndarray) -> tuple[float, float | None]:
    filtered = smooth(rdf)
    peak_candidates = np.flatnonzero((radius > 2.0) & (radius < radius.max() - 0.4))
    peak = peak_candidates[np.argmax(filtered[peak_candidates])]
    derivative = np.diff(filtered)
    candidates = np.flatnonzero(
        (np.arange(len(derivative) - 1) > peak)
        & (derivative[:-1] < 0)
        & (derivative[1:] >= 0)
        & (radius[:-2] < radius.max() - 0.20)
    )
    if len(candidates):
        minimum = int(candidates[0] + 1)
        return float(radius[peak]), float(radius[minimum])
    return float(radius[peak]), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.04)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    first_chunk = next(iter(sorted((args.trajectory / "chunks").glob("*.xyz"))))
    with first_chunk.open() as handle:
        handle.readline()
        header = handle.readline()
    lattice_match = LATTICE.search(header)
    if lattice_match is None:
        raise ValueError("Missing cell")
    reference_cell = np.asarray(
        [float(value) for value in lattice_match.group(1).split()]
    ).reshape(3, 3)
    cell_lengths = np.linalg.norm(reference_cell, axis=1)
    r_max = float(cell_lengths.min() / 2)
    edges = np.arange(0.0, r_max + args.bin_width, args.bin_width)
    centers = (edges[:-1] + edges[1:]) / 2
    hist_com = np.zeros(len(centers), dtype=np.int64)
    hist_alkene = np.zeros(len(centers), dtype=np.int64)
    hist_carboxyl = np.zeros(len(centers), dtype=np.int64)
    frame_count = 0
    first_time = None
    last_time = None
    nearest_13_overlap = []
    oxygen_shell_counts = []
    oxygen_shell_com_extent = []

    for time_ps, cell, positions in iter_frames(args.trajectory, args.stride):
        anchor = positions[5]
        solute = anchor + minimum_image(positions[:SOLUTE_ATOMS] - anchor, cell)
        center_of_mass = np.average(solute, axis=0, weights=MASSES)
        water_oxygen = positions[SOLUTE_ATOMS::3]
        com_distance = np.linalg.norm(
            minimum_image(water_oxygen - center_of_mass, cell), axis=1
        )
        solute_oxygen_distance = np.linalg.norm(
            minimum_image(
                water_oxygen[:, None, :] - solute[None, :4, :], cell
            ),
            axis=2,
        ).min(axis=1)
        alkene_distance = np.linalg.norm(
            minimum_image(
                water_oxygen[:, None, :] - solute[None, 4:6, :], cell
            ),
            axis=2,
        ).ravel()
        carboxyl_distance = np.linalg.norm(
            minimum_image(
                water_oxygen[:, None, :] - solute[None, 6:8, :], cell
            ),
            axis=2,
        ).ravel()
        hist_com += np.histogram(com_distance, bins=edges)[0]
        hist_alkene += np.histogram(alkene_distance, bins=edges)[0]
        hist_carboxyl += np.histogram(carboxyl_distance, bins=edges)[0]
        nearest_com = set(np.argsort(com_distance)[:13])
        nearest_oxygen = set(np.argsort(solute_oxygen_distance)[:13])
        nearest_13_overlap.append(len(nearest_com & nearest_oxygen))
        oxygen_shell = solute_oxygen_distance < 3.8
        oxygen_shell_counts.append(int(oxygen_shell.sum()))
        oxygen_shell_com_extent.append(
            float(com_distance[oxygen_shell].max()) if oxygen_shell.any() else 0.0
        )
        frame_count += 1
        first_time = time_ps if first_time is None else first_time
        last_time = time_ps

    shell_volume = 4 * np.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = abs(np.linalg.det(reference_cell))
    oxygen_density = WATER_COUNT / volume
    rdf_com = hist_com / (frame_count * oxygen_density * shell_volume)
    rdf_alkene = hist_alkene / (frame_count * 2 * oxygen_density * shell_volume)
    rdf_carboxyl = hist_carboxyl / (frame_count * 2 * oxygen_density * shell_volume)
    coordination = np.cumsum(hist_com) / frame_count
    peak_radius, minimum_radius = first_minimum(centers, rdf_com)
    shell_count = None
    if minimum_radius is not None:
        minimum_index = int(np.argmin(abs(centers - minimum_radius)))
        shell_count = float(coordination[minimum_index])
    radius_13_waters = float(np.interp(13.0, coordination, centers))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["r_angstrom", "g_com_o", "coordination_com_o", "g_alkene_c_o", "g_carboxyl_c_o"]
        )
        writer.writerows(
            zip(centers, rdf_com, coordination, rdf_alkene, rdf_carboxyl, strict=True)
        )

    summary = {
        "trajectory": str(args.trajectory),
        "frames": frame_count,
        "time_range_ps": [first_time, last_time],
        "water_count": WATER_COUNT,
        "box_volume_angstrom3": volume,
        "oxygen_density_angstrom_minus3": oxygen_density,
        "maximum_reliable_radius_angstrom": r_max,
        "com_rdf_first_peak_angstrom": peak_radius,
        "com_rdf_first_minimum_angstrom": minimum_radius,
        "mean_waters_inside_first_minimum": shell_count,
        "radius_containing_13_waters_on_average_angstrom": radius_13_waters,
        "nearest_13_com_vs_nearest_13_oxygen_mean_overlap": float(
            np.mean(nearest_13_overlap)
        ),
        "waters_within_3p8_angstrom_of_any_solute_oxygen_mean": float(
            np.mean(oxygen_shell_counts)
        ),
        "com_radius_covering_all_3p8_angstrom_oxygen_shell_waters_mean": float(
            np.mean(oxygen_shell_com_extent)
        ),
        "com_radius_covering_all_3p8_angstrom_oxygen_shell_waters_p95": float(
            np.percentile(oxygen_shell_com_extent, 95)
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.25), dpi=180)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.84, wspace=0.30)
    axes[0].plot(centers, smooth(rdf_com), color="#245b8a", linewidth=2.3)
    axes[0].set(xlabel=r"$r_{\mathrm{COM}-O_w}$ (Å)", ylabel=r"$g_{\mathrm{COM}-O_w}(r)$")
    axes[0].text(
        0.97, 0.95, "no first minimum within $L/2$",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=10,
    )

    axes[1].plot(centers, coordination, color="#33806c", linewidth=2.3)
    axes[1].axvline(radius_13_waters, color="#cf7b29", linestyle="--", linewidth=1.6)
    axes[1].scatter([radius_13_waters], [13], color="#cf7b29", s=35, zorder=3)
    axes[1].set(xlabel=r"sphere radius $r$ (Å)", ylabel="mean waters inside sphere")
    axes[1].text(
        0.05, 0.95, f"13 waters at {radius_13_waters:.2f} Å on average",
        transform=axes[1].transAxes, ha="left", va="top", fontsize=10,
    )

    axes[2].plot(centers, smooth(rdf_alkene), color="#7b4f9d", linewidth=2.2, label="alkene C")
    axes[2].plot(centers, smooth(rdf_carboxyl), color="#c2544b", linewidth=2.2, label="carboxyl C")
    axes[2].set(xlabel=r"$r_{C-O_w}$ (Å)", ylabel=r"$g_{C-O_w}(r)$")
    axes[2].legend(frameon=False, fontsize=10)

    for axis in axes:
        axis.set_xlim(0, r_max)
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"Hydrogen maleate–water structure from full ML ({first_time:.1f}–{last_time:.1f} ps)",
        fontsize=15, weight="bold", y=0.96,
    )
    figure.savefig(args.output, bbox_inches="tight", facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
