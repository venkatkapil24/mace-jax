#!/usr/bin/env python3
"""Estimate a provisional symmetrized 1D proton free energy with block errors."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


TIME = re.compile(r'\btime_fs=([-+\d.eE]+)')
COORDINATE = re.compile(r'\bproton_coordinate_angstrom=([-+\d.eE]+)')
EDGES = np.arange(-1.3, 1.3 + 0.0125, 0.025)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
WELL = (np.abs(CENTERS) >= 0.2) & (np.abs(CENTERS) <= 0.8)
MIDPOINT = np.abs(CENTERS) < 0.025
KB_EV_PER_K = 8.617333262145e-5


def read_samples(directory: Path) -> np.ndarray:
    samples = []
    for chunk in sorted((directory / 'chunks').glob('*.xyz')):
        with chunk.open() as handle:
            for line in handle:
                time = TIME.search(line)
                coordinate = COORDINATE.search(line)
                if time and coordinate:
                    samples.append((float(time.group(1)) / 1000, float(coordinate.group(1))))
    result = np.asarray(samples, dtype=float).reshape(-1, 2)
    if len(result) and np.any(np.diff(result[:, 0]) <= 0):
        raise ValueError(f'Sample times are not increasing in {directory}')
    return result


def free_energy(q: np.ndarray, bandwidth: float) -> tuple[np.ndarray, float]:
    # q and -q are paired representations of the same frame.
    mirrored = np.concatenate((q, -q))
    counts, _ = np.histogram(mirrored, bins=EDGES)
    density = gaussian_filter1d(counts.astype(float), bandwidth / 0.025, mode='constant')
    density /= np.trapezoid(density, CENTERS)
    well_density = density[WELL].max()
    profile = -np.log(np.maximum(density, 1e-300) / well_density)
    midpoint_density = density[MIDPOINT].mean()
    barrier = math.log(well_density / midpoint_density)
    return profile, barrier


def block_resample(q: np.ndarray, block_frames: int, rng: np.random.Generator) -> np.ndarray:
    n = len(q)
    starts = rng.integers(0, n, size=math.ceil(n / block_frames))
    indices = (starts[:, None] + np.arange(block_frames)[None, :]) % n
    return q[indices.ravel()[:n]]


def bootstrap(
    q: np.ndarray, bandwidth: float, block_frames: int, repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    profiles = []
    barriers = []
    for _ in range(repetitions):
        profile, barrier = free_energy(block_resample(q, block_frames, rng), bandwidth)
        profiles.append(profile)
        barriers.append(barrier)
    return np.asarray(profiles), np.asarray(barriers)


def describe(q: np.ndarray, bandwidth: float, repetitions: int, seed: int) -> dict:
    frame_interval_ps = 0.01
    profile, barrier = free_energy(q, bandwidth)
    blocks = {}
    for block_ps in (0.5, 1.0, 2.0):
        block_frames = round(block_ps / frame_interval_ps)
        samples, barriers = bootstrap(
            q, bandwidth, block_frames, repetitions,
            np.random.default_rng(seed + round(100 * block_ps)),
        )
        blocks[str(block_ps)] = {
            'barrier_95_percent_kBT': np.percentile(barriers, (2.5, 97.5)).tolist(),
            'profile_16_84_percent_kBT': np.percentile(samples, (16, 84), axis=0).tolist()
            if block_ps == 1.0 else None,
            'approximate_number_of_blocks': len(q) / block_frames,
        }
    smoothing = {
        str(width): free_energy(q, width)[1]
        for width in (0.05, 0.075, 0.1)
    }
    return {
        'frames': len(q),
        'barrier_kBT': barrier,
        'profile_kBT': profile.tolist(),
        'block_bootstrap': blocks,
        'barrier_by_bandwidth_kBT': smoothing,
        'barrier_after_first_ps_kBT': free_energy(q[100:], bandwidth)[1],
        'first_half_barrier_kBT': free_energy(q[:len(q) // 2], bandwidth)[1],
        'second_half_barrier_kBT': free_energy(q[len(q) // 2:], bandwidth)[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--temperature-k', type=float, default=330.0)
    parser.add_argument('--bandwidth-angstrom', type=float, default=0.075)
    parser.add_argument('--bootstrap-repetitions', type=int, default=400)
    args = parser.parse_args()

    runs = {
        'Full MACE-POLAR': ('HydrogenMaleate-MACE-POLAR', '#254b8d'),
        'First-shell ML/MM': ('HydrogenMaleate-MLMM', '#d06a29'),
    }
    data = {
        label: read_samples(args.examples / folder / 'md-100ps')
        for label, (folder, _) in runs.items()
    }
    if any(len(array) == 0 for array in data.values()):
        raise ValueError('Both trajectories need saved frames')
    common_end = min(array[-1, 0] for array in data.values())
    subsets = {
        'Matched interval': {
            label: array[array[:, 0] <= common_end + 1e-9, 1]
            for label, array in data.items()
        },
        'All available frames': {label: array[:, 1] for label, array in data.items()},
    }
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7.5), sharex=True, sharey=True)
    summary = {
        'temperature_K': args.temperature_k,
        'kBT_meV': 1000 * KB_EV_PER_K * args.temperature_k,
        'density_smoothing_bandwidth_angstrom': args.bandwidth_angstrom,
        'matched_end_ps': common_end,
        'frame_interval_ps': 0.01,
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'coordinate_grid_angstrom': CENTERS.tolist(),
        'panels': {},
    }
    for axis, (panel_name, panel) in zip(axes, subsets.items(), strict=True):
        summary['panels'][panel_name] = {}
        for index, (label, q) in enumerate(panel.items()):
            result = describe(q, args.bandwidth_angstrom, args.bootstrap_repetitions, 431 + index)
            summary['panels'][panel_name][label] = result
            color = runs[label][1]
            axis.plot(CENTERS, result['profile_kBT'], color=color, linewidth=2,
                      label=f'{label} ({len(q)} frames)')
            lower, upper = result['block_bootstrap']['1.0']['profile_16_84_percent_kBT']
            axis.fill_between(CENTERS, lower, upper, color=color, alpha=0.16)
        polar_q = panel['Full MACE-POLAR']
        mlmm_q = panel['First-shell ML/MM']
        _, polar_barriers = bootstrap(
            polar_q, args.bandwidth_angstrom, 100, args.bootstrap_repetitions,
            np.random.default_rng(1451),
        )
        _, mlmm_barriers = bootstrap(
            mlmm_q, args.bandwidth_angstrom, 100, args.bootstrap_repetitions,
            np.random.default_rng(1452),
        )
        difference = mlmm_barriers - polar_barriers
        summary['panels'][panel_name]['comparison'] = {
            'mlmm_minus_polar_barrier_kBT': (
                summary['panels'][panel_name]['First-shell ML/MM']['barrier_kBT']
                - summary['panels'][panel_name]['Full MACE-POLAR']['barrier_kBT']
            ),
            'difference_95_percent_kBT': np.percentile(difference, (2.5, 97.5)).tolist(),
        }
        axis.set_title(
            f'{panel_name}: 0–{common_end:.2f} ps' if panel_name == 'Matched interval'
            else 'All saved frames (different durations)'
        )
        axis.set_ylabel('Relative free energy / kBT')
        axis.axvline(0, color='0.4', linestyle='--', linewidth=1)
        axis.grid(axis='y', alpha=0.2)
        axis.legend(frameon=False)
        axis.set_xlim(-0.85, 0.85)
        axis.set_ylim(0, 5)
    axes[-1].set_xlabel('Proton coordinate, symmetrized (Å)')
    figure.suptitle('Hydrogen maleate: provisional 1D free energy at 330 K')
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    for panel_name, panel in summary['panels'].items():
        print(panel_name)
        for label, result in panel.items():
            if label == 'comparison':
                print(f'  ML/MM minus POLAR barrier: {result}')
                continue
            print(
                f'  {label}: {result["frames"]} frames; central barrier '
                f'{result["barrier_kBT"]:.2f} kBT; 95% block interval '
                f'{result["block_bootstrap"]["1.0"]["barrier_95_percent_kBT"]}; '
                f'bandwidth sensitivity {result["barrier_by_bandwidth_kBT"]}; '
                f'halves {result["first_half_barrier_kBT"]:.2f}/'
                f'{result["second_half_barrier_kBT"]:.2f}'
            )


if __name__ == '__main__':
    main()
