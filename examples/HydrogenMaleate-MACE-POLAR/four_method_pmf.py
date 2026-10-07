#!/usr/bin/env python3
"""Compare four hydrogen-maleate embeddings using consistent PMF estimates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


TIME = re.compile(r'\btime_fs=([-+\d.eE]+)')
COORDINATE = re.compile(r'\bproton_coordinate_angstrom=([-+\d.eE]+)')
CENTRAL_EDGES = np.arange(-1.3, 1.3 + 0.0125, 0.025)
CENTRAL_CENTERS = (CENTRAL_EDGES[:-1] + CENTRAL_EDGES[1:]) / 2
CENTRAL_WELL = (
    (np.abs(CENTRAL_CENTERS) >= 0.2)
    & (np.abs(CENTRAL_CENTERS) <= 0.8)
)
MIDPOINT = np.abs(CENTRAL_CENTERS) < 0.025
COLORS = {
    'Full MACE-POLAR': '#254b8d',
    'ML/MM electrostatic': '#d06a29',
    'ML/MM mechanical': '#36845b',
    'OpenMM-ML mechanical': '#8b5aa5',
}


def read_xyz_samples(directory: Path) -> np.ndarray:
    values = []
    for chunk in sorted((directory / 'chunks').glob('*.xyz')):
        with chunk.open() as handle:
            for line in handle:
                time = TIME.search(line)
                coordinate = COORDINATE.search(line)
                if time and coordinate:
                    values.append(
                        (float(time.group(1)) / 1000, float(coordinate.group(1)))
                    )
    return validate_samples(np.asarray(values, dtype=float).reshape(-1, 2), directory)


def read_csv_samples(path: Path) -> np.ndarray:
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    values = np.asarray(
        [
            (float(row['time_ps']), float(row['proton_coordinate_angstrom']))
            for row in rows
        ],
        dtype=float,
    )
    return validate_samples(values, path)


def validate_samples(values: np.ndarray, source: Path) -> np.ndarray:
    if not len(values):
        raise ValueError(f'No samples in {source}')
    if not np.all(np.isfinite(values)) or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError(f'Invalid or non-increasing samples in {source}')
    return values


def central_pmf(q: np.ndarray, bandwidth: float = 0.075) -> tuple[np.ndarray, float]:
    mirrored = np.concatenate((q, -q))
    counts, _ = np.histogram(mirrored, bins=CENTRAL_EDGES)
    density = gaussian_filter1d(
        counts.astype(float), bandwidth / 0.025, mode='constant'
    )
    well_density = density[CENTRAL_WELL].max()
    profile = -np.log(np.maximum(density, 1e-300) / well_density)
    barrier = math.log(well_density / density[MIDPOINT].mean())
    return profile, barrier


def folded_pmf(
    q: np.ndarray, edges: np.ndarray, bandwidth: float = 0.10
) -> np.ndarray:
    counts, _ = np.histogram(np.abs(q), bins=edges)
    density = gaussian_filter1d(
        counts.astype(float), bandwidth / np.diff(edges).mean(), mode='constant'
    )
    profile = -np.log(np.maximum(density, 1e-300) / density.max())
    return profile


def block_resample(
    q: np.ndarray, frames: int, rng: np.random.Generator
) -> np.ndarray:
    starts = rng.integers(0, len(q), size=math.ceil(len(q) / frames))
    indices = (starts[:, None] + np.arange(frames)[None, :]) % len(q)
    return q[indices.ravel()[: len(q)]]


def central_bootstrap(
    q: np.ndarray, repetitions: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    profiles = []
    barriers = []
    for _ in range(repetitions):
        profile, barrier = central_pmf(block_resample(q, 100, rng))
        profiles.append(profile)
        barriers.append(barrier)
    return np.asarray(profiles), np.asarray(barriers)


def folded_bootstrap(
    q: np.ndarray, edges: np.ndarray, repetitions: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        [
            folded_pmf(block_resample(q, 100, rng), edges)
            for _ in range(repetitions)
        ]
    )


def timing_from_log(path: Path) -> dict:
    records = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'time_ps' in row and 'elapsed_seconds_this_process' in row:
                records.append(
                    (
                        float(row['step']),
                        float(row['time_ps']),
                        float(row['elapsed_seconds_this_process']),
                    )
                )
    data = np.asarray(records, dtype=float)
    if len(data) < 3:
        raise ValueError(f'Insufficient timing records in {path}')
    differences = np.diff(data, axis=0)
    valid = np.all(differences > 0, axis=1)
    differences = differences[valid]
    rates = differences[:, 1] * 86400 / differences[:, 2]
    total_steps = differences[:, 0].sum()
    total_ps = differences[:, 1].sum()
    total_seconds = differences[:, 2].sum()
    return {
        'progress_records': len(data),
        'timed_steps': int(total_steps),
        'timed_ps': float(total_ps),
        'timed_seconds': float(total_seconds),
        'ps_per_day': float(total_ps * 86400 / total_seconds),
        'milliseconds_per_md_step': float(1000 * total_seconds / total_steps),
        'chunk_10_90_percent_ps_per_day': np.percentile(
            rates, (10, 90)
        ).tolist(),
    }


def draw_central(
    axis,
    trajectories: dict[str, np.ndarray],
    end_ps: float | None,
    repetitions: int,
    title: str,
) -> dict:
    summary = {}
    for index, (label, samples) in enumerate(trajectories.items()):
        subset = (
            samples[samples[:, 0] <= end_ps + 1e-9]
            if end_ps is not None
            else samples
        )
        q = subset[:, 1]
        profile, barrier = central_pmf(q)
        boot_profiles, boot_barriers = central_bootstrap(
            q, repetitions, 20260929 + index
        )
        lower, upper = np.percentile(boot_profiles, (16, 84), axis=0)
        axis.plot(
            CENTRAL_CENTERS,
            profile,
            color=COLORS[label],
            linewidth=2,
            label=f'{label} ({subset[-1, 0]:.1f} ps)',
        )
        axis.fill_between(
            CENTRAL_CENTERS, lower, upper, color=COLORS[label], alpha=0.13
        )
        summary[label] = {
            'frames': len(q),
            'end_ps': float(subset[-1, 0]),
            'barrier_kBT': float(barrier),
            'barrier_95_percent_kBT': np.percentile(
                boot_barriers, (2.5, 97.5)
            ).tolist(),
            'fraction_outside_central_range': float(np.mean(np.abs(q) > 1.3)),
        }
    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(0, 6)
    axis.axvline(0, color='0.5', linestyle='--', linewidth=1)
    axis.set_xlabel(r'$\delta=r_R-r_L$ (Å)')
    axis.set_ylabel(r'$\beta[F(\delta)-F_{\rm well}]$')
    axis.set_title(title, loc='left', weight='bold')
    axis.grid(axis='y', alpha=0.2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=400)
    args = parser.parse_args()

    locations = {
        'Full MACE-POLAR': (
            args.examples / 'HydrogenMaleate-PME/full-500ps',
            args.examples / 'HydrogenMaleate-PME/full-500ps/run.log',
            'xyz',
        ),
        'ML/MM electrostatic': (
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps',
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps/run.log',
            'xyz',
        ),
        'ML/MM mechanical': (
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps',
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps/run.log',
            'xyz',
        ),
        'OpenMM-ML mechanical': (
            args.examples / 'HydrogenMaleate-OpenMM-Mechanical/md-100ps/samples.csv',
            args.examples / 'HydrogenMaleate-OpenMM-Mechanical/run.log',
            'csv',
        ),
    }
    trajectories = {
        label: (
            read_xyz_samples(location)
            if kind == 'xyz'
            else read_csv_samples(location)
        )
        for label, (location, _, kind) in locations.items()
    }
    timings = {
        label: timing_from_log(log) for label, (_, log, _) in locations.items()
    }
    common_end = min(samples[-1, 0] for samples in trajectories.values())

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), dpi=180)
    matched = draw_central(
        axes[0, 0],
        trajectories,
        common_end,
        args.bootstrap_repetitions,
        f'(a) Symmetrized PMF, matched 0–{common_end:.1f} ps',
    )
    available = draw_central(
        axes[0, 1],
        trajectories,
        None,
        args.bootstrap_repetitions,
        '(b) Symmetrized PMF, all available samples',
    )
    handles, labels = axes[0, 1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )

    max_coordinate = max(
        np.abs(samples[:, 1]).max() for samples in trajectories.values()
    )
    folded_edges = np.arange(0, math.ceil(max_coordinate * 10) / 10 + 0.05, 0.05)
    folded_centers = (folded_edges[:-1] + folded_edges[1:]) / 2
    folded_summary = {}
    for index, (label, samples) in enumerate(trajectories.items()):
        q = samples[:, 1]
        profile = folded_pmf(q, folded_edges)
        boot = folded_bootstrap(
            q, folded_edges, args.bootstrap_repetitions, 20261029 + index
        )
        lower, upper = np.percentile(boot, (16, 84), axis=0)
        mask = profile <= 8
        axes[1, 0].plot(
            folded_centers[mask], profile[mask], color=COLORS[label], linewidth=2
        )
        axes[1, 0].fill_between(
            folded_centers[mask],
            lower[mask],
            upper[mask],
            color=COLORS[label],
            alpha=0.13,
        )
        folded_summary[label] = {
            'maximum_abs_delta_angstrom': float(np.abs(q).max()),
            'fraction_abs_delta_above_1p3': float(np.mean(np.abs(q) > 1.3)),
        }
    axes[1, 0].axvline(1.3, color='0.5', linestyle='--', linewidth=1)
    axes[1, 0].text(1.34, 7.5, 'central-range boundary', fontsize=8, color='0.4')
    axes[1, 0].set_xlim(0, folded_edges[-1])
    axes[1, 0].set_ylim(0, 8)
    axes[1, 0].set_xlabel(r'$|\delta|$ (Å)')
    axes[1, 0].set_ylabel(r'$\beta[F(|\delta|)-F_{\min}]$')
    axes[1, 0].set_title(
        '(c) Full coordinate PMF, all available samples', loc='left', weight='bold'
    )
    axes[1, 0].grid(axis='y', alpha=0.2)

    labels = list(timings)
    x = np.arange(len(labels))
    rates = np.asarray([timings[label]['ps_per_day'] for label in labels])
    lower = rates - np.asarray(
        [timings[label]['chunk_10_90_percent_ps_per_day'][0] for label in labels]
    )
    upper = np.asarray(
        [timings[label]['chunk_10_90_percent_ps_per_day'][1] for label in labels]
    ) - rates
    bars = axes[1, 1].bar(
        x,
        rates,
        yerr=[np.maximum(lower, 0), np.maximum(upper, 0)],
        color=[COLORS[label] for label in labels],
        capsize=4,
        width=0.65,
    )
    axes[1, 1].set_xticks(
        x,
        [
            'Full ML',
            'Electrostatic\nML/MM',
            'Mechanical\nML/MM',
            'OpenMM-ML\nmechanical',
        ],
    )
    axes[1, 1].set_ylabel('Throughput (ps/day)')
    axes[1, 1].set_title('(d) Measured GPU MD throughput', loc='left', weight='bold')
    axes[1, 1].grid(axis='y', alpha=0.2)
    axes[1, 1].set_axisbelow(True)
    for bar, label in zip(bars, labels, strict=True):
        rate = timings[label]['ps_per_day']
        milliseconds = timings[label]['milliseconds_per_md_step']
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{rate:.0f}\n{milliseconds:.1f} ms',
            ha='center',
            va='bottom',
            fontsize=9,
        )

    figure.suptitle(
        'Hydrogen maleate at 330 K · 0.5 fs timestep · '
        'PME JAX runs and OpenMM reference',
        y=1.02,
        fontsize=14,
        weight='bold',
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2, w_pad=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches='tight')
    figure.savefig(args.output.with_suffix('.pdf'), bbox_inches='tight')
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'temperature_K': 330.0,
        'timestep_fs': 0.5,
        'coordinate_definition': 'delta = nearest right O-H minus nearest left O-H',
        'symmetrization': 'each observation contributes at delta and -delta',
        'central_histogram_bin_angstrom': 0.025,
        'central_gaussian_bandwidth_angstrom': 0.075,
        'bootstrap_block_ps': 1.0,
        'bootstrap_repetitions': args.bootstrap_repetitions,
        'matched_end_ps': float(common_end),
        'available_end_ps': {
            label: float(samples[-1, 0]) for label, samples in trajectories.items()
        },
        'matched_central_pmf': matched,
        'all_available_central_pmf': available,
        'all_available_folded_coordinate': folded_summary,
        'timings': timings,
    }
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
