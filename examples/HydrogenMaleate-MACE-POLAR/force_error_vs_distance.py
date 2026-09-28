#!/usr/bin/env python3
"""Compare electrostatic and mechanical embedding forces with full MACE-POLAR.

The two potentials are evaluated on identical saved configurations.  The
current ML/MM definition includes the fixed 13-water QM partition and its
flat-bottom restraint unless the corresponding command-line parameters are
changed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from md import build_energy

CARBOXYL_OXYGENS = np.asarray([0, 1, 2, 3])
SOLUTE_ATOMS = 11


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def reactive_center_distances(positions: np.ndarray, box: float) -> np.ndarray:
    """Minimum-image distances from the center of the four carboxyl oxygens."""
    reactive = positions[CARBOXYL_OXYGENS]
    unwrapped = reactive[0] + minimum_image(reactive - reactive[0], box)
    center = unwrapped.mean(axis=0)
    return np.linalg.norm(minimum_image(positions - center, box), axis=-1)


def choose_chunks(directory: Path, count: int, discard_ps: float) -> list[Path]:
    """Choose approximately uniformly spaced chunk endpoints."""
    candidates = []
    for path in sorted((directory / "chunks").glob("*.xyz")):
        # Chunk names are ending MD steps; the production timestep was 0.5 fs.
        if int(path.stem) * 0.0005 >= discard_ps:
            candidates.append(path)
    if len(candidates) < count:
        raise ValueError(
            f"Requested {count} configurations but only {len(candidates)} chunks "
            f"remain after discarding {discard_ps:g} ps"
        )
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[index] for index in indices]


def atom_groups(atom_count: int, qm_atom_count: int) -> dict[str, np.ndarray]:
    indices = np.arange(atom_count)
    return {
        "Solute": indices < SOLUTE_ATOMS,
        "QM water": (indices >= SOLUTE_ATOMS) & (indices < qm_atom_count),
        "MM water": indices >= qm_atom_count,
    }


def radial_statistics(
    force_difference: np.ndarray,
    distances: np.ndarray,
    groups: dict[str, np.ndarray],
    bin_edges: np.ndarray,
) -> dict[str, dict[str, list[float] | list[int]]]:
    """Calculate vector-force RMSE and between-frame spread in radial bins."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    squared_norm = np.sum(force_difference**2, axis=-1)
    result = {}
    for label, group_mask in groups.items():
        rmse = np.full(len(centers), np.nan)
        lower = np.full(len(centers), np.nan)
        upper = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=int)
        for bin_index, (left, right) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:], strict=True)
        ):
            frame_values = []
            for frame_index in range(len(force_difference)):
                selected = (
                    group_mask
                    & (distances[frame_index] >= left)
                    & (distances[frame_index] < right)
                )
                counts[bin_index] += int(np.sum(selected))
                if np.any(selected):
                    frame_values.append(
                        math.sqrt(float(np.mean(squared_norm[frame_index, selected])))
                    )
            if frame_values:
                values = np.asarray(frame_values)
                rmse[bin_index] = math.sqrt(float(np.mean(values**2)))
                lower[bin_index], upper[bin_index] = np.percentile(values, (16, 84))
        result[label] = {
            "distance_angstrom": centers.tolist(),
            "force_rmse_meV_per_angstrom": (1000 * rmse).tolist(),
            "frame_16_percentile_meV_per_angstrom": (1000 * lower).tolist(),
            "frame_84_percentile_meV_per_angstrom": (1000 * upper).tolist(),
            "atom_frame_samples": counts.tolist(),
        }
    return result


def overall_statistics(
    difference: np.ndarray,
    reference: np.ndarray,
    groups: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    result = {}
    for label, selected in {
        **groups,
        "All atoms": np.ones(difference.shape[1], dtype=bool),
    }.items():
        delta = difference[:, selected]
        ref = reference[:, selected]
        error_norm = np.linalg.norm(delta, axis=-1)
        result[label] = {
            "vector_force_rmse_meV_per_angstrom": 1000
            * float(np.sqrt(np.mean(np.sum(delta**2, axis=-1)))),
            "mean_vector_error_meV_per_angstrom": 1000 * float(np.mean(error_norm)),
            "median_vector_error_meV_per_angstrom": 1000 * float(np.median(error_norm)),
            "reference_force_rms_meV_per_angstrom": 1000
            * float(np.sqrt(np.mean(np.sum(ref**2, axis=-1)))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mechanical-qm-charges", type=Path)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--discard-ps", type=float, default=5.0)
    parser.add_argument("--qm-water-count", type=int, default=13)
    parser.add_argument("--restraint-radius", type=float, default=4.2)
    parser.add_argument("--restraint-k", type=float, default=0.2)
    parser.add_argument("--ewald-alpha", type=float, default=0.5)
    parser.add_argument("--ewald-kmax", type=int, default=6)
    parser.add_argument(
        "--reciprocal-method", choices=["pme", "direct"], default="pme"
    )
    parser.add_argument("--pme-mesh-spacing", type=float, default=0.5)
    parser.add_argument("--pme-assignment-order", type=int, default=8)
    parser.add_argument("--nonbonded-cutoff", type=float)
    parser.add_argument("--neighbor-skin", type=float, default=0.25)
    parser.add_argument("--radial-bin-width", type=float, default=0.5)
    args = parser.parse_args()

    atoms = read(args.initial)
    atoms.info["qm_water_count"] = args.qm_water_count
    box = float(atoms.cell[0, 0])
    if args.nonbonded_cutoff is None:
        args.nonbonded_cutoff = min(9.0, 0.49 * box)
    qm_atom_count = SOLUTE_ATOMS + 3 * args.qm_water_count
    if len(atoms) != 167 or qm_atom_count >= len(atoms):
        raise ValueError("Expected hydrogen maleate, a nonempty QM water shell, and MM water")

    full_eval, full_params, full_neighbor_fn, _, full_count = build_energy(
        atoms,
        args.bundle,
        "polar",
        args.ewald_alpha,
        args.ewald_kmax,
        args.qm_water_count,
        args.restraint_radius,
        args.restraint_k,
        None,
        args.nonbonded_cutoff,
        args.neighbor_skin,
        args.reciprocal_method,
        args.pme_mesh_spacing,
        args.pme_assignment_order,
    )
    mixed_eval, mixed_params, mixed_neighbor_fn, _, mixed_count = build_energy(
        atoms,
        args.bundle,
        "mlmm",
        args.ewald_alpha,
        args.ewald_kmax,
        args.qm_water_count,
        args.restraint_radius,
        args.restraint_k,
        None,
        args.nonbonded_cutoff,
        args.neighbor_skin,
        args.reciprocal_method,
        args.pme_mesh_spacing,
        args.pme_assignment_order,
    )
    mechanical_eval = mechanical_params = mechanical_neighbor_fn = None
    mechanical_charge_metadata = None
    if args.mechanical_qm_charges is not None:
        charge_data = json.loads(args.mechanical_qm_charges.read_text())
        mechanical_charge_metadata = {
            key: charge_data.get(key)
            for key in (
                "definition",
                "solute_charge_source",
                "solute_charge_note",
                "water_charge_source",
                "total_charge",
            )
        }
        mechanical_charges = np.asarray(charge_data["charges"], dtype=float)
        (
            mechanical_eval,
            mechanical_params,
            mechanical_neighbor_fn,
            _,
            mechanical_count,
        ) = build_energy(
            atoms,
            args.bundle,
            "mechanical",
            args.ewald_alpha,
            args.ewald_kmax,
            args.qm_water_count,
            args.restraint_radius,
            args.restraint_k,
            mechanical_charges,
            args.nonbonded_cutoff,
            args.neighbor_skin,
            args.reciprocal_method,
            args.pme_mesh_spacing,
            args.pme_assignment_order,
        )
        if mechanical_count != qm_atom_count:
            raise RuntimeError("Unexpected mechanical model partition")
    if full_count != len(atoms) or mixed_count != qm_atom_count:
        raise RuntimeError("Unexpected full or mixed model partition")

    chunk_paths = choose_chunks(args.trajectory, args.samples, args.discard_ps)
    positions = []
    times_ps = []
    for path in chunk_paths:
        frame = read(path, index=-1)
        positions.append(np.asarray(frame.positions))
        times_ps.append(float(frame.info["time_fs"]) / 1000)

    full_neighbors = full_neighbor_fn.allocate(jnp.asarray(positions[0]))
    mixed_neighbors = mixed_neighbor_fn.allocate(jnp.asarray(positions[0]))
    mechanical_neighbors = (
        mechanical_neighbor_fn.allocate(jnp.asarray(positions[0]))
        if mechanical_neighbor_fn is not None
        else None
    )
    full_forces = []
    mixed_forces = []
    mechanical_forces = []
    full_energies = []
    mixed_energies = []
    mechanical_energies = []
    distances = []
    for sample_index, xyz in enumerate(positions):
        xyz_jax = jnp.asarray(xyz)
        full_neighbors = full_neighbor_fn.update(xyz_jax, full_neighbors)
        mixed_neighbors = mixed_neighbor_fn.update(xyz_jax, mixed_neighbors)
        if mechanical_neighbor_fn is not None:
            mechanical_neighbors = mechanical_neighbor_fn.update(
                xyz_jax, mechanical_neighbors
            )
        if bool(full_neighbors.did_buffer_overflow) or bool(
            mixed_neighbors.did_buffer_overflow
        ):
            raise RuntimeError("Neighbor-list capacity overflow")
        if mechanical_neighbors is not None and bool(
            mechanical_neighbors.did_buffer_overflow
        ):
            raise RuntimeError("Mechanical neighbor-list capacity overflow")
        full_energy, full_gradient = full_eval(
            xyz_jax, full_neighbors, full_params
        )
        mixed_energy, mixed_gradient = mixed_eval(
            xyz_jax, mixed_neighbors, mixed_params
        )
        if mechanical_eval is not None:
            mechanical_energy, mechanical_gradient = mechanical_eval(
                xyz_jax, mechanical_neighbors, mechanical_params
            )
            mechanical_energies.append(float(mechanical_energy))
            mechanical_forces.append(-np.asarray(mechanical_gradient))
        full_energies.append(float(full_energy))
        mixed_energies.append(float(mixed_energy))
        full_forces.append(-np.asarray(full_gradient))
        mixed_forces.append(-np.asarray(mixed_gradient))
        distances.append(reactive_center_distances(xyz, box))
        print(
            f"Evaluated {sample_index + 1}/{len(positions)}: "
            f"{times_ps[sample_index]:.3f} ps",
            flush=True,
        )

    full_forces = np.asarray(full_forces)
    mixed_forces = np.asarray(mixed_forces)
    distances = np.asarray(distances)
    groups = atom_groups(len(atoms), qm_atom_count)
    upper_distance = math.ceil(float(distances.max()) / args.radial_bin_width)
    bin_edges = (
        np.arange(upper_distance + 1, dtype=float) * args.radial_bin_width
    )
    comparisons = {
        "Electrostatic embedding": mixed_forces - full_forces,
    }
    if mechanical_forces:
        mechanical_forces = np.asarray(mechanical_forces)
        comparisons["Mechanical embedding"] = mechanical_forces - full_forces
    radial = {
        label: radial_statistics(difference, distances, groups, bin_edges)
        for label, difference in comparisons.items()
    }
    overall = {
        label: overall_statistics(difference, full_forces, groups)
        for label, difference in comparisons.items()
    }

    summary = {
        "reference": "full periodic MACE-POLAR",
        "approximations": list(comparisons),
        "trajectory": str(args.trajectory),
        "samples": len(times_ps),
        "sample_times_ps": times_ps,
        "qm_water_count": args.qm_water_count,
        "qm_atom_count": qm_atom_count,
        "mm_atom_count": len(atoms) - qm_atom_count,
        "nonbonded_cutoff_angstrom": args.nonbonded_cutoff,
        "neighbor_skin_angstrom": args.neighbor_skin,
        "mechanical_fixed_charges": mechanical_charge_metadata,
        "restraint": {
            "radius_angstrom": args.restraint_radius,
            "k_eV_per_angstrom2": args.restraint_k,
            "applied_to": "selected QM-water oxygens relative to nearest carboxyl oxygen",
        },
        "distance_definition": (
            "minimum-image distance from the instantaneous geometric center "
            "of the four hydrogen-maleate carboxyl oxygens"
        ),
        "force_error_definition": "norm of F_embedding minus F_full for each atom",
        "reciprocal_method": args.reciprocal_method,
        "pme_mesh_spacing_angstrom": (
            args.pme_mesh_spacing if args.reciprocal_method == "pme" else None
        ),
        "pme_assignment_order": (
            args.pme_assignment_order if args.reciprocal_method == "pme" else None
        ),
        "overall": overall,
        "radial": radial,
    }

    colors = {
        "Solute": "#7b398a",
        "QM water": "#254b8d",
        "MM water": "#d06a29",
    }
    figure, axes = plt.subplots(
        1, len(comparisons), figsize=(8.6 * len(comparisons), 5.7),
        dpi=180, sharex=True, sharey=True, squeeze=False,
    )
    rng = np.random.default_rng(20260925)
    jitter = {
        label: rng.normal(0, 0.018, int(np.sum(mask)) * len(distances))
        for label, mask in groups.items()
    }
    for axis, (embedding_label, difference) in zip(
        axes[0], comparisons.items(), strict=True
    ):
        error_norm = 1000 * np.linalg.norm(difference, axis=-1)
        for group_label, group_mask in groups.items():
            x = distances[:, group_mask].ravel()
            y = error_norm[:, group_mask].ravel()
            axis.scatter(
                x + jitter[group_label], y, s=8, alpha=0.09,
                color=colors[group_label], linewidths=0,
            )
            values = radial[embedding_label][group_label]
            radius = np.asarray(values["distance_angstrom"])
            central = np.asarray(values["force_rmse_meV_per_angstrom"])
            lower = np.asarray(values["frame_16_percentile_meV_per_angstrom"])
            upper = np.asarray(values["frame_84_percentile_meV_per_angstrom"])
            count = np.asarray(values["atom_frame_samples"])
            valid = (count > 0) & np.isfinite(central)
            axis.plot(
                radius[valid], central[valid], color=colors[group_label],
                marker="o", markersize=4, linewidth=2.1,
                label=f"{group_label}: radial RMSE",
            )
            axis.fill_between(
                radius[valid], lower[valid], upper[valid],
                color=colors[group_label], alpha=0.15,
            )
        axis.set_yscale("log")
        axis.set_xlabel("Distance from reactive center (Å)")
        axis.set_title(embedding_label, loc="left", weight="bold")
        axis.text(
            0.02, 0.97,
            f"{len(times_ps)} configurations · {times_ps[0]:.1f}–{times_ps[-1]:.1f} ps",
            transform=axis.transAxes, va="top", color="#555d63",
        )
        axis.grid(True, which="both", color="#e2e6e8", linewidth=0.7)
        axis.legend(frameon=False, loc="best")
    axes[0, 0].set_ylabel(
        r"Force error $\|\mathbf{F}_{\mathrm{embedding}}-\mathbf{F}_{\mathrm{full}}\|$ "
        "(meV/Å)"
    )
    figure.suptitle("Force error relative to full MACE-POLAR", weight="bold")
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, facecolor="white")
    figure.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        sample_times_ps=np.asarray(times_ps),
        distances_angstrom=distances,
        full_forces_eV_per_angstrom=full_forces,
        mixed_forces_eV_per_angstrom=mixed_forces,
        mechanical_forces_eV_per_angstrom=(
            mechanical_forces if len(mechanical_forces) else np.empty((0,))
        ),
        atomic_numbers=np.asarray(atoms.numbers),
        solute_mask=groups["Solute"],
        qm_water_mask=groups["QM water"],
        mm_water_mask=groups["MM water"],
        full_energies_eV=np.asarray(full_energies),
        mixed_energies_eV=np.asarray(mixed_energies),
        mechanical_energies_eV=np.asarray(mechanical_energies),
    )
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
