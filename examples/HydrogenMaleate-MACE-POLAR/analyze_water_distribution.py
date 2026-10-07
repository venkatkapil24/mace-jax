#!/usr/bin/env python3
"""Characterize water around hydrogen maleate for boundary-restraint design."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from solute_water_rdf import (
    LATTICE,
    MASSES,
    SOLUTE_ATOMS,
    WATER_COUNT,
    iter_frames,
    minimum_image,
    smooth,
)


GROUPS = {
    "alkene C": np.asarray([4, 5]),
    "carboxyl C": np.asarray([6, 7]),
    "carboxyl O": np.asarray([0, 1, 2, 3]),
}
COLORS = {
    "COM": "#285f8f",
    "nearest carboxyl O": "#c6534b",
    "nearest heavy atom": "#4c8a68",
}


def unwrap_solute(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    anchor = positions[5]
    return anchor + minimum_image(positions[:SOLUTE_ATOMS] - anchor, cell)


def molecular_reference(solute: np.ndarray) -> np.ndarray:
    center = np.average(solute, axis=0, weights=MASSES)
    centered = solute - center
    x_axis = solute[2:4].mean(axis=0) - solute[:2].mean(axis=0)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = solute[0] - solute[1]
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    return centered @ np.column_stack((x_axis, y_axis, z_axis))


def align_rotation(solute: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.average(solute, axis=0, weights=MASSES)
    moving = solute[:8] - solute[:8].mean(axis=0)
    target_heavy = target[:8] - target[:8].mean(axis=0)
    u, _, vt = np.linalg.svd(moving.T @ target_heavy)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return center, u @ correction @ vt


def infer_bonds(reference: np.ndarray) -> list[tuple[int, int]]:
    atomic_numbers = np.asarray([8] * 4 + [6] * 4 + [1] * 3)
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            if atomic_numbers[i] == atomic_numbers[j] == 1:
                continue
            cutoff = 1.25 if 1 in (atomic_numbers[i], atomic_numbers[j]) else 1.8
            if np.linalg.norm(reference[i] - reference[j]) < cutoff:
                pairs.append((i, j))
    return pairs


def draw_solute(axis: plt.Axes, reference: np.ndarray, coordinates: tuple[int, int]) -> None:
    x, y = coordinates
    bonds = infer_bonds(reference)
    segments = [reference[[i, j]][:, [x, y]] for i, j in bonds]
    axis.add_collection(
        LineCollection(segments, colors="#30383d", linewidths=2.0, zorder=5)
    )
    axis.scatter(reference[4:8, x], reference[4:8, y], s=62, c="#424b50",
                 edgecolors="white", linewidths=0.5, zorder=6)
    axis.scatter(reference[:4, x], reference[:4, y], s=68, c="#dc5549",
                 edgecolors="white", linewidths=0.5, zorder=6)
    axis.scatter(reference[8:10, x], reference[8:10, y], s=25, c="white",
                 edgecolors="#4d5559", linewidths=0.7, zorder=6)
    # The transferring proton is shown against its nearest reference oxygen.
    nearest = int(np.argmin(np.linalg.norm(reference[:4] - reference[10], axis=1)))
    axis.plot(reference[[nearest, 10], x], reference[[nearest, 10], y],
              color="#754a7e", linewidth=2.0, zorder=5)
    axis.scatter(reference[10, x], reference[10, y], s=36, c="#ad57bb",
                 edgecolors="white", linewidths=0.6, zorder=7)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.04)
    parser.add_argument("--spatial-bin-width", type=float, default=0.12)
    parser.add_argument("--spatial-stride", type=int, default=5)
    parser.add_argument("--slab-half-width", type=float, default=1.5)
    args = parser.parse_args()

    first_chunk = next(iter(sorted((args.trajectory / "chunks").glob("*.xyz"))))
    with first_chunk.open() as handle:
        handle.readline()
        header = handle.readline()
    lattice_match = LATTICE.search(header)
    if lattice_match is None:
        raise ValueError("Missing cell")
    cell = np.asarray(
        [float(value) for value in lattice_match.group(1).split()]
    ).reshape(3, 3)
    volume = abs(np.linalg.det(cell))
    oxygen_density = WATER_COUNT / volume
    r_max = float(np.linalg.norm(cell, axis=1).min() / 2)
    distance_edges = np.arange(0, r_max + args.bin_width, args.bin_width)
    distance_centers = (distance_edges[:-1] + distance_edges[1:]) / 2
    shell_volume = 4 * np.pi / 3 * (
        distance_edges[1:] ** 3 - distance_edges[:-1] ** 3
    )

    coordinate_hist = {
        name: np.zeros(len(distance_centers), dtype=np.int64)
        for name in COLORS
    }
    site_hist = {
        name: np.zeros(len(distance_centers), dtype=np.int64)
        for name in GROUPS
    }
    boundary = {name: {"r13": [], "r14": []} for name in COLORS}
    overlap = {name: [] for name in ("COM", "nearest heavy atom")}
    group_counts = {name: [] for name in GROUPS}
    group_only_counts = {name: [] for name in ("alkene C", "carboxyl C")}
    heavy_shell_counts_4p26 = []
    heavy_boundary_radii = {n: [] for n in range(20, 29)}

    spatial_limit = r_max
    spatial_edges = np.arange(
        -spatial_limit, spatial_limit + args.spatial_bin_width,
        args.spatial_bin_width,
    )
    top_hist = np.zeros((len(spatial_edges) - 1, len(spatial_edges) - 1))
    side_hist = np.zeros_like(top_hist)
    spatial_frames = 0
    reference = None
    frame_count = 0
    first_time = None
    last_time = None

    for time_ps, frame_cell, positions in iter_frames(args.trajectory, 1):
        solute = unwrap_solute(positions, frame_cell)
        center = np.average(solute, axis=0, weights=MASSES)
        water = positions[SOLUTE_ATOMS::3]
        water_from_center = minimum_image(water - center, frame_cell)
        distances = {
            "COM": np.linalg.norm(water_from_center, axis=1),
            "nearest carboxyl O": np.linalg.norm(
                minimum_image(water[:, None, :] - solute[None, :4, :], frame_cell),
                axis=2,
            ).min(axis=1),
            "nearest heavy atom": np.linalg.norm(
                minimum_image(water[:, None, :] - solute[None, :8, :], frame_cell),
                axis=2,
            ).min(axis=1),
        }
        for name, values in distances.items():
            coordinate_hist[name] += np.histogram(values, bins=distance_edges)[0]
            ordered = np.sort(values)
            boundary[name]["r13"].append(float(ordered[12]))
            boundary[name]["r14"].append(float(ordered[13]))
        heavy_shell_counts_4p26.append(
            int(np.sum(distances["nearest heavy atom"] < 4.26))
        )
        ordered_heavy = np.sort(distances["nearest heavy atom"])
        for n in heavy_boundary_radii:
            heavy_boundary_radii[n].append(float(ordered_heavy[n - 1]))

        oxygen_nearest = set(np.argsort(distances["nearest carboxyl O"])[:13])
        overlap["COM"].append(
            len(oxygen_nearest & set(np.argsort(distances["COM"])[:13]))
        )
        overlap["nearest heavy atom"].append(
            len(oxygen_nearest & set(np.argsort(distances["nearest heavy atom"])[:13]))
        )

        for name, indices in GROUPS.items():
            pair_distance = np.linalg.norm(
                minimum_image(
                    water[:, None, :] - solute[None, indices, :], frame_cell
                ),
                axis=2,
            )
            site_hist[name] += np.histogram(pair_distance.ravel(), bins=distance_edges)[0]
            nearest = pair_distance.min(axis=1)
            group_counts[name].append(int(np.sum(nearest < 3.8)))
            if name in group_only_counts:
                group_only_counts[name].append(
                    int(np.sum((nearest < 3.8) & (distances["nearest carboxyl O"] >= 3.8)))
                )

        if reference is None:
            reference = molecular_reference(solute)
        if frame_count % args.spatial_stride == 0:
            _, rotation = align_rotation(solute, reference)
            aligned_water = water_from_center @ rotation
            top = aligned_water[np.abs(aligned_water[:, 2]) < args.slab_half_width]
            side = aligned_water[np.abs(aligned_water[:, 1]) < args.slab_half_width]
            top_hist += np.histogram2d(top[:, 0], top[:, 1], bins=spatial_edges)[0]
            side_hist += np.histogram2d(side[:, 0], side[:, 2], bins=spatial_edges)[0]
            spatial_frames += 1

        frame_count += 1
        first_time = time_ps if first_time is None else first_time
        last_time = time_ps

    if reference is None:
        raise RuntimeError("No trajectory frames")

    coordination = {
        name: np.cumsum(hist) / frame_count
        for name, hist in coordinate_hist.items()
    }
    radius_13 = {
        name: float(np.interp(13.0, count, distance_centers))
        for name, count in coordination.items()
    }
    site_rdf = {
        name: hist / (
            frame_count * len(GROUPS[name]) * oxygen_density * shell_volume
        )
        for name, hist in site_hist.items()
    }

    summary = {
        "trajectory": str(args.trajectory),
        "frames": frame_count,
        "time_range_ps": [first_time, last_time],
        "box_length_angstrom": float(np.linalg.norm(cell[0])),
        "maximum_spherical_rdf_radius_angstrom": r_max,
        "radius_containing_13_waters_on_average_angstrom": radius_13,
        "boundary_order_statistics_angstrom": {},
        "nearest_13_overlap_with_carboxyl_oxygen_definition": {
            name: float(np.mean(values)) for name, values in overlap.items()
        },
        "mean_water_counts_within_3p8_angstrom": {
            name: float(np.mean(values)) for name, values in group_counts.items()
        },
        "mean_water_counts_within_3p8_angstrom_of_carbon_but_not_carboxyl_oxygen": {
            name: float(np.mean(values)) for name, values in group_only_counts.items()
        },
        "nearest_heavy_atom_shell_at_4p26_angstrom": {
            "mean_water_count": float(np.mean(heavy_shell_counts_4p26)),
            "median_water_count": float(np.median(heavy_shell_counts_4p26)),
            "p05_p95_water_count": np.percentile(
                heavy_shell_counts_4p26, [5, 95]
            ).tolist(),
        },
        "nearest_heavy_atom_boundary_gaussian_fits": {},
    }
    for name in COLORS:
        r13 = np.asarray(boundary[name]["r13"])
        r14 = np.asarray(boundary[name]["r14"])
        gap = r14 - r13
        summary["boundary_order_statistics_angstrom"][name] = {
            "r13_mean": float(r13.mean()),
            "r13_p05_p95": np.percentile(r13, [5, 95]).tolist(),
            "r14_mean": float(r14.mean()),
            "r14_p05_p95": np.percentile(r14, [5, 95]).tolist(),
            "gap_mean": float(gap.mean()),
            "gap_p05_p95": np.percentile(gap, [5, 95]).tolist(),
        }
    kbt_ev = 8.617333262145e-5 * 330.0
    for n, values in heavy_boundary_radii.items():
        radii = np.asarray(values)
        mean = float(radii.mean())
        standard_deviation = float(radii.std())
        centered = (radii - mean) / standard_deviation
        summary["nearest_heavy_atom_boundary_gaussian_fits"][str(n)] = {
            "mean_angstrom": mean,
            "standard_deviation_angstrom": standard_deviation,
            "skewness": float(np.mean(centered**3)),
            "excess_kurtosis": float(np.mean(centered**4) - 3.0),
            "effective_free_energy_curvature_eV_per_angstrom2": (
                kbt_ev / standard_deviation**2
            ),
        }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r_angstrom",
                "n_com", "n_nearest_carboxyl_o", "n_nearest_heavy",
                "g_alkene_c_o", "g_carboxyl_c_o", "g_carboxyl_o_o",
            ]
        )
        writer.writerows(
            zip(
                distance_centers,
                coordination["COM"],
                coordination["nearest carboxyl O"],
                coordination["nearest heavy atom"],
                site_rdf["alkene C"],
                site_rdf["carboxyl C"],
                site_rdf["carboxyl O"],
                strict=True,
            )
        )

    # Spatial density projections.
    slab_thickness = 2 * args.slab_half_width
    pixel_area = args.spatial_bin_width**2
    normalization = spatial_frames * oxygen_density * slab_thickness * pixel_area
    top_density = top_hist.T / normalization
    side_density = side_hist.T / normalization
    vmax = float(np.percentile(np.concatenate((top_density.ravel(), side_density.ravel())), 99.3))
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), dpi=180)
    figure.subplots_adjust(left=0.065, right=0.91, bottom=0.11, top=0.86, wspace=0.17)
    maps = []
    for axis, density, coordinates, title in (
        (axes[0], top_density, (0, 1), "molecular-plane projection"),
        (axes[1], side_density, (0, 2), "edge-on projection"),
    ):
        image = axis.pcolormesh(
            spatial_edges, spatial_edges, density, shading="auto",
            cmap="Blues", vmin=0, vmax=vmax,
        )
        maps.append(image)
        draw_solute(axis, reference, coordinates)
        axis.set_aspect("equal")
        axis.set_xlim(-5.5, 5.5)
        axis.set_ylim(-5.5, 5.5)
        axis.set_xlabel("molecular-frame x (Å)")
        axis.set_ylabel("molecular-frame y (Å)" if coordinates == (0, 1) else "molecular-frame z (Å)")
        axis.set_title(title, fontsize=13, weight="bold")
    color_axis = figure.add_axes([0.925, 0.16, 0.018, 0.62])
    colorbar = figure.colorbar(maps[0], cax=color_axis)
    colorbar.set_label(r"water-O density / bulk density")
    figure.suptitle(
        f"Water distribution around hydrogen maleate: full ML, {first_time:.1f}–{last_time:.1f} ps",
        fontsize=15, weight="bold", y=0.965,
    )
    spatial_path = args.output_prefix.with_name(args.output_prefix.name + "-spatial.png")
    figure.savefig(spatial_path, bbox_inches="tight", facecolor="white")
    figure.savefig(spatial_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Coordinate comparison for selecting a boundary restraint.
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.5), dpi=180)
    figure.subplots_adjust(left=0.06, right=0.985, bottom=0.17, top=0.82, wspace=0.28)
    for name, values in coordination.items():
        axes[0].plot(distance_centers, values, color=COLORS[name], linewidth=2.1, label=name)
        axes[0].scatter([radius_13[name]], [13], color=COLORS[name], s=24, zorder=3)
    axes[0].set(xlabel="cutoff (Å)", ylabel="mean waters inside boundary")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")

    positions = np.arange(len(COLORS))
    width = 0.34
    r13_data = [boundary[name]["r13"] for name in COLORS]
    r14_data = [boundary[name]["r14"] for name in COLORS]
    box13 = axes[1].boxplot(r13_data, positions=positions - width / 2, widths=0.28,
                           patch_artist=True, showfliers=False)
    box14 = axes[1].boxplot(r14_data, positions=positions + width / 2, widths=0.28,
                           patch_artist=True, showfliers=False)
    for patch in box13["boxes"]:
        patch.set(facecolor="#5b8db8", alpha=0.75)
    for patch in box14["boxes"]:
        patch.set(facecolor="#d68a49", alpha=0.75)
    axes[1].set_xticks(positions, ["COM", "carboxyl O", "heavy atom"], rotation=15)
    axes[1].set_ylabel("ordered water distance (Å)")
    axes[1].legend([box13["boxes"][0], box14["boxes"][0]], ["13th water", "14th water"],
                   frameon=False, fontsize=9)

    for name, values in site_rdf.items():
        axes[2].plot(distance_centers, smooth(values), linewidth=2.0, label=name)
    axes[2].set(xlabel=r"solute-site–$O_w$ distance (Å)", ylabel=r"site–$O_w$ $g(r)$")
    axes[2].legend(frameon=False, fontsize=8.5)

    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(0, r_max)
    axes[2].set_xlim(0, r_max)
    figure.suptitle("Distance-coordinate comparison for a water boundary restraint",
                    fontsize=15, weight="bold", y=0.96)
    coordinate_path = args.output_prefix.with_name(args.output_prefix.name + "-coordinates.png")
    figure.savefig(coordinate_path, bbox_inches="tight", facecolor="white")
    figure.savefig(coordinate_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Focused first-shell criterion from the nearest-heavy-atom coordinate.
    shell_cutoff = 4.26
    radial_count = coordinate_hist["nearest heavy atom"] / (
        frame_count * args.bin_width
    )
    window = np.hanning(15)
    window /= window.sum()
    radial_count_smooth = np.convolve(
        np.pad(radial_count, 7, mode="edge"), window, mode="valid"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.80, wspace=0.28)
    axes[0].plot(distance_centers, radial_count_smooth, color="#4c8a68", linewidth=2.3)
    axes[0].fill_between(
        distance_centers, 0, radial_count_smooth,
        where=distance_centers <= shell_cutoff, color="#79af91", alpha=0.28,
    )
    axes[0].axvline(shell_cutoff, color="#c6534b", linestyle="--", linewidth=1.7)
    axes[0].text(
        shell_cutoff - 0.08, 0.95, f"first minimum ≈ {shell_cutoff:.2f} Å",
        transform=axes[0].get_xaxis_transform(), ha="right", va="top", fontsize=10,
    )
    axes[0].set(xlabel=r"$d_{\min}(O_w,\mathrm{heavy})$ (Å)",
                ylabel="mean water count per Å")
    axes[0].set_xlim(2.2, 5.5)

    count_values, count_frequency = np.unique(
        heavy_shell_counts_4p26, return_counts=True
    )
    axes[1].bar(
        count_values, count_frequency / frame_count,
        color="#5b8db8", edgecolor="white", width=0.88,
    )
    axes[1].axvline(23, color="#c6534b", linestyle="--", linewidth=1.7)
    axes[1].set(xlabel=rf"waters with $d_{{\min}} < {shell_cutoff:.2f}$ Å",
                ylabel="fraction of frames")
    axes[1].set_xlim(min(count_values) - 0.8, max(count_values) + 0.8)
    axes[1].text(
        0.97, 0.95, "median = 23\n90% range = 20–26",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
    )
    for axis in axes:
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("First hydration layer from the nearest-heavy-atom distance",
                    fontsize=15, weight="bold", y=0.96)
    shell_path = args.output_prefix.with_name(args.output_prefix.name + "-first-shell.png")
    figure.savefig(shell_path, bbox_inches="tight", facecolor="white")
    figure.savefig(shell_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # Gaussian model for fluctuations of the fixed-n boundary radius.
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=180)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.80, wspace=0.25)
    for axis, n, color in zip(axes, (23, 26), ("#4c8a68", "#5b8db8"), strict=True):
        values = np.asarray(heavy_boundary_radii[n])
        mean = values.mean()
        standard_deviation = values.std()
        axis.hist(values, bins=45, density=True, color=color, alpha=0.58,
                  edgecolor="white", label="full-ML distribution")
        grid = np.linspace(values.min(), values.max(), 400)
        gaussian = np.exp(-0.5 * ((grid - mean) / standard_deviation) ** 2) / (
            standard_deviation * np.sqrt(2 * np.pi)
        )
        axis.plot(grid, gaussian, color="#252b2f", linewidth=2.1, label="Gaussian fit")
        curvature = kbt_ev / standard_deviation**2
        axis.text(
            0.04, 0.95,
            f"mean = {mean:.3f} Å\nσ = {standard_deviation:.3f} Å\n"
            f"$k_{{\\rm eff}}$ = {curvature:.2f} eV Å$^{{-2}}$",
            transform=axis.transAxes, ha="left", va="top", fontsize=10,
        )
        axis.set(xlabel=rf"$R_{{{n}}}$: distance of {n}th water (Å)",
                 ylabel="probability density")
        axis.set_title(f"fixed inner region: {n} waters", fontsize=12, weight="bold")
        axis.grid(alpha=0.18, linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle("Gaussian model of the fluctuating first-layer boundary",
                    fontsize=15, weight="bold", y=0.96)
    gaussian_path = args.output_prefix.with_name(args.output_prefix.name + "-boundary-gaussians.png")
    figure.savefig(gaussian_path, bbox_inches="tight", facecolor="white")
    figure.savefig(gaussian_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(
        f"Wrote {spatial_path}, {coordinate_path}, {shell_path}, "
        f"and {gaussian_path}"
    )


if __name__ == "__main__":
    main()
