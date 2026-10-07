#!/usr/bin/env python3
"""Plot the four hydrogen-maleate PMFs over the proton-transfer region."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from four_method_pmf import (
    CENTRAL_CENTERS,
    COLORS,
    central_pmf,
    read_csv_samples,
    read_xyz_samples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    trajectories = {
        'Full MACE-POLAR': read_xyz_samples(
            args.examples / 'HydrogenMaleate-PME/full-500ps'
        ),
        'ML/MM electrostatic': read_xyz_samples(
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps'
        ),
        'ML/MM mechanical': read_xyz_samples(
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps'
        ),
        'OpenMM-ML mechanical': read_csv_samples(
            args.examples
            / 'HydrogenMaleate-OpenMM-Mechanical/md-100ps/samples.csv'
        ),
    }
    figure, axis = plt.subplots(figsize=(8.2, 5.7), dpi=180)
    summary = {}
    for label, samples in trajectories.items():
        profile, barrier = central_pmf(samples[:, 1])
        duration = float(samples[-1, 0])
        axis.plot(
            CENTRAL_CENTERS,
            profile,
            color=COLORS[label],
            linewidth=2.4,
            label=f'{label} ({duration:.1f} ps)',
        )
        summary[label] = {
            'duration_ps': duration,
            'frames': len(samples),
            'barrier_kBT': float(barrier),
        }
    axis.axvline(0, color='0.5', linestyle='--', linewidth=1)
    axis.set_xlim(-1.3, 1.3)
    axis.set_ylim(0, 6)
    axis.set_xlabel(r'$\delta=r_R-r_L$ (Å)', fontsize=12)
    axis.set_ylabel(r'$\beta[F(\delta)-F_{\rm well}]$', fontsize=12)
    axis.set_title('Hydrogen maleate PMF at 330 K')
    axis.grid(axis='y', alpha=0.2)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    figure.savefig(args.output.with_suffix('.pdf'))
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
