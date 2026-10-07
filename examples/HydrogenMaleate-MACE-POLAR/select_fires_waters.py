#!/usr/bin/env python3
"""Select a fixed FIRES water region from a short full-ML trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solute_water_rdf import MASSES, SOLUTE_ATOMS, iter_frames, minimum_image


def unwrap_solute(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    anchor = positions[5]
    return anchor + minimum_image(positions[:SOLUTE_ATOMS] - anchor, cell)


def nearest_heavy_distances(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    solute = unwrap_solute(positions, cell)
    water_oxygen = positions[SOLUTE_ATOMS::3]
    displacement = minimum_image(
        water_oxygen[:, None, :] - solute[None, :8, :], cell
    )
    return np.linalg.norm(displacement, axis=2).min(axis=1)


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    radius = max(1, int(np.ceil(4 * sigma_bins)))
    grid = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode="edge"), kernel, mode="valid")


def shell_cutoff(
    distances: np.ndarray,
    edges: np.ndarray,
    smoothing_angstrom: float,
    reference_probability: np.ndarray,
) -> tuple[float, np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = float(edges[1] - edges[0])
    histogram = np.histogram(distances.ravel(), bins=edges)[0]
    expected = distances.shape[0] * distances.shape[1] * reference_probability
    mddf = np.divide(
        histogram,
        expected,
        out=np.zeros_like(expected, dtype=float),
        where=expected > 0,
    )
    smoothed = gaussian_smooth(mddf, smoothing_angstrom / bin_width)

    peak_region = np.flatnonzero((centers >= 2.2) & (centers <= 3.5))
    peak = int(peak_region[np.argmax(smoothed[peak_region])])
    derivative = np.diff(smoothed)
    minima = np.flatnonzero(
        (np.arange(len(derivative) - 1) > peak)
        & (centers[:-2] > centers[peak] + 0.30)
        & (centers[:-2] < 5.2)
        & (derivative[:-1] < 0)
        & (derivative[1:] >= 0)
    )
    if not len(minima):
        raise RuntimeError("No first-shell minimum found after the first peak")
    return float(centers[minima[0] + 1]), smoothed


def ideal_gas_reference(
    solute_heavy_frames: list[np.ndarray],
    cells: list[np.ndarray],
    edges: np.ndarray,
    samples_per_frame: int,
) -> np.ndarray:
    """Reference distance distribution for uniform points around the same solute."""
    rng = np.random.default_rng(20260929)
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    for solute_heavy, cell in zip(solute_heavy_frames, cells, strict=True):
        points = rng.random((samples_per_frame, 3)) @ cell
        displacement = minimum_image(
            points[:, None, :] - solute_heavy[None, :, :], cell
        )
        distances = np.linalg.norm(displacement, axis=2).min(axis=1)
        histogram += np.histogram(distances, bins=edges)[0]
    return histogram / (len(solute_heavy_frames) * samples_per_frame)


def integer_percentile(values: np.ndarray, percentile: float) -> int:
    # "higher" ensures at least the requested fraction of observed occupancies
    # is no larger than the returned fixed count.
    return int(np.percentile(values, percentile, method="higher"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--duration-ps", type=float, default=10.0)
    parser.add_argument("--bin-width", type=float, default=0.04)
    parser.add_argument("--smoothing-angstrom", type=float, default=0.12)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--reference-points-per-frame", type=int, default=2000)
    args = parser.parse_args()

    times: list[float] = []
    distance_frames: list[np.ndarray] = []
    solute_heavy_frames: list[np.ndarray] = []
    cells: list[np.ndarray] = []
    final_positions = None
    final_cell = None
    for time_ps, cell, positions in iter_frames(args.trajectory, 1):
        if time_ps > args.duration_ps + 1e-9:
            break
        times.append(time_ps)
        distance_frames.append(nearest_heavy_distances(positions, cell))
        solute_heavy_frames.append(unwrap_solute(positions, cell)[:8])
        cells.append(cell.copy())
        final_positions = positions.copy()
        final_cell = cell.copy()

    if final_positions is None or final_cell is None:
        raise RuntimeError("No trajectory frames found in the requested interval")
    times_array = np.asarray(times)
    distances = np.asarray(distance_frames)
    if abs(times_array[-1] - args.duration_ps) > 1e-6:
        raise RuntimeError(
            f"Last available frame is {times_array[-1]:.6f} ps, expected "
            f"{args.duration_ps:.6f} ps"
        )

    r_max = float(np.linalg.norm(final_cell, axis=1).min() / 2)
    edges = np.arange(0.0, r_max + args.bin_width, args.bin_width)
    centers = 0.5 * (edges[:-1] + edges[1:])
    reference_probability = ideal_gas_reference(
        solute_heavy_frames,
        cells,
        edges,
        args.reference_points_per_frame,
    )
    cutoff, mddf = shell_cutoff(
        distances, edges, args.smoothing_angstrom, reference_probability
    )
    occupancies = np.sum(distances < cutoff, axis=1)

    block_cutoffs = []
    for indices in np.array_split(np.arange(len(distances)), args.blocks):
        block_cutoff, _ = shell_cutoff(
            distances[indices], edges, args.smoothing_angstrom, reference_probability
        )
        block_cutoffs.append(block_cutoff)

    median_count = int(np.median(occupancies))
    p95_count = integer_percentile(occupancies, 95)
    final_distances = nearest_heavy_distances(final_positions, final_cell)
    order = np.argsort(final_distances)

    selections = {}
    for label, count in (("median_occupancy", median_count), ("p95_occupancy", p95_count)):
        water_indices = order[:count]
        selections[label] = {
            "water_count": count,
            "water_indices_zero_based": water_indices.tolist(),
            "oxygen_atom_indices_zero_based": (SOLUTE_ATOMS + 3 * water_indices).tolist(),
            "outermost_selected_distance_angstrom": float(final_distances[water_indices[-1]]),
            "nearest_unselected_distance_angstrom": float(final_distances[order[count]]),
        }

    unique, frequency = np.unique(occupancies, return_counts=True)
    summary = {
        "trajectory": str(args.trajectory),
        "analysis_window_ps": [float(times_array[0]), float(times_array[-1])],
        "frames": int(len(times_array)),
        "frame_spacing_ps": float(np.median(np.diff(times_array))),
        "distance_coordinate": "minimum periodic distance from water O to any solute heavy atom",
        "distribution": "minimum-distance distribution function normalized by a uniform ideal-gas reference",
        "ideal_gas_reference_points": int(
            len(solute_heavy_frames) * args.reference_points_per_frame
        ),
        "solute_heavy_atom_count": 8,
        "water_count": int(distances.shape[1]),
        "shell_cutoff_angstrom": cutoff,
        "block_cutoff_angstrom": {
            "values": block_cutoffs,
            "mean": float(np.mean(block_cutoffs)),
            "standard_deviation": float(np.std(block_cutoffs, ddof=1)),
        },
        "shell_occupancy": {
            "mean": float(np.mean(occupancies)),
            "median": median_count,
            "p05": integer_percentile(occupancies, 5),
            "p95": p95_count,
            "minimum": int(np.min(occupancies)),
            "maximum": int(np.max(occupancies)),
            "histogram": {str(int(n)): int(f) for n, f in zip(unique, frequency, strict=True)},
        },
        "selection_frame_ps": float(times_array[-1]),
        "selections_at_final_frame": selections,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    npz_path = args.output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        time_ps=times_array,
        nearest_heavy_distances_angstrom=distances,
        shell_occupancy=occupancies,
        distance_centers_angstrom=centers,
        minimum_distance_distribution_function=mddf,
        ideal_gas_reference_probability=reference_probability,
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), dpi=180)
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.80, wspace=0.30)
    axes[0].plot(centers, mddf, color="#39765f", linewidth=2.2)
    axes[0].axvline(cutoff, color="#c3514a", linestyle="--", linewidth=1.6)
    axes[0].set(
        xlabel=r"$d_{\min}(O_w,\mathrm{solute\ heavy})$ (Å)",
        ylabel="minimum-distance distribution $g_{md}(d)$",
        xlim=(2.2, 5.4),
    )
    axes[0].text(
        0.96, 0.94, f"first minimum = {cutoff:.2f} Å",
        transform=axes[0].transAxes, ha="right", va="top",
    )

    axes[1].bar(unique, frequency / len(occupancies), color="#557fa6", width=0.85)
    axes[1].axvline(median_count, color="#c3514a", linestyle="--", linewidth=1.5)
    axes[1].axvline(p95_count, color="#d19535", linestyle=":", linewidth=2.0)
    axes[1].set(xlabel="waters inside first shell", ylabel="fraction of frames")
    axes[1].text(
        0.96, 0.94,
        f"mean = {np.mean(occupancies):.1f}\nmedian = {median_count}\n95th percentile = {p95_count}",
        transform=axes[1].transAxes, ha="right", va="top",
    )

    axes[2].plot(times_array, occupancies, color="#626a70", linewidth=0.9)
    axes[2].axhline(median_count, color="#c3514a", linestyle="--", linewidth=1.5)
    axes[2].axhline(p95_count, color="#d19535", linestyle=":", linewidth=2.0)
    axes[2].set(xlabel="full-ML time (ps)", ylabel="first-shell waters")

    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "FIRES water-region selection from the first 10 ps of full ML",
        fontsize=14, weight="bold", y=0.96,
    )
    figure_path = args.output_prefix.with_suffix(".png")
    figure.savefig(figure_path, bbox_inches="tight", facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}, {npz_path}, and {figure_path}")


if __name__ == "__main__":
    main()
