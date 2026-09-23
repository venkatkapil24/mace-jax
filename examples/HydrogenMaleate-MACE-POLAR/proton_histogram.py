#!/usr/bin/env python3
"""Plot hydrogen-maleate proton-coordinate histograms from saved MD chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TIME = re.compile(r'\btime_fs=([-+\d.eE]+)')
COORDINATE = re.compile(r'\bproton_coordinate_angstrom=([-+\d.eE]+)')


def samples(directory: Path) -> np.ndarray:
    values = []
    for chunk in sorted((directory / 'chunks').glob('*.xyz')):
        with chunk.open() as handle:
            for line in handle:
                time = TIME.search(line)
                coordinate = COORDINATE.search(line)
                if time and coordinate:
                    values.append((float(time.group(1)) / 1000, float(coordinate.group(1))))
    result = np.asarray(values, dtype=float).reshape(-1, 2)
    if len(result) and (not np.all(np.isfinite(result)) or np.any(np.diff(result[:, 0]) <= 0)):
        raise ValueError(f'Invalid or repeated samples in {directory}')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-second-shell', action='store_true')
    args = parser.parse_args()
    names = {
        'Full MACE-POLAR': 'HydrogenMaleate-MACE-POLAR',
        'First-shell ML/MM': 'HydrogenMaleate-MLMM',
    }
    if args.include_second_shell:
        names['Second-shell ML/MM'] = 'HydrogenMaleate-MLMM-SecondShell'
    data = {
        label: samples(args.examples / folder / 'md-100ps')
        for label, folder in names.items()
    }
    available = {label: array for label, array in data.items() if len(array)}
    if len(available) < 2:
        raise ValueError('Need at least two trajectories with saved frames')
    common_end = min(array[-1, 0] for array in available.values())
    matched = {
        label: array[array[:, 0] <= common_end + 1e-9]
        for label, array in available.items()
    }
    coordinates = np.concatenate([array[:, 1] for array in available.values()])
    width = 0.05
    bins = np.arange(
        np.floor(coordinates.min() / width) * width - width,
        np.ceil(coordinates.max() / width) * width + 2 * width,
        width,
    )
    colors = {
        'Full MACE-POLAR': '#254b8d',
        'First-shell ML/MM': '#d06a29',
        'Second-shell ML/MM': '#329778',
    }
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True, sharey=True)
    for axis, subset, title in (
        (axes[0], matched, f'Matched interval: 0–{common_end:.2f} ps'),
        (axes[1], available, 'All saved frames (different durations)'),
    ):
        for label, array in subset.items():
            axis.hist(
                array[:, 1], bins=bins, density=True, histtype='step',
                linewidth=2, color=colors[label],
                label=f'{label} ({len(array)} frames, {array[-1, 0]:.2f} ps)',
            )
        axis.axvline(0, color='0.35', linestyle='--', linewidth=1)
        axis.set_title(title)
        axis.set_ylabel('Probability density (Å⁻¹)')
        axis.legend(frameon=False, fontsize=9)
        axis.grid(axis='y', alpha=0.2)
    axes[-1].set_xlabel('Proton coordinate: nearest right O–H minus nearest left O–H (Å)')
    fig.suptitle('Hydrogen maleate proton position at 330 K')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    summary = {
        'matched_end_ps': common_end,
        'bin_width_angstrom': width,
        'trajectories': {
            label: {
                'saved_frames': len(array),
                'last_time_ps': float(array[-1, 0]) if len(array) else None,
                'matched_frames': len(matched[label]) if label in matched else 0,
                'matched_fraction_proton_nearer_right': float(np.mean(matched[label][:, 1] < 0)) if label in matched else None,
                'all_fraction_proton_nearer_right': float(np.mean(array[:, 1] < 0)) if len(array) else None,
            }
            for label, array in data.items()
        },
    }
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
