#!/usr/bin/env python3
"""Three-panel 100 ps PMF comparison with GPU timing controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from four_method_pmf import (
    CENTRAL_CENTERS,
    central_bootstrap,
    central_pmf,
    read_xyz_samples,
    timing_from_log,
)
from three_panel_summary import draw_molecule


COLORS = {
    'ML': '#254b8d',
    'ML/MM electrostatic': '#d06a29',
    'ML/MM mechanical': '#36845b',
    'OpenMM-ML mechanical': '#8b5aa5',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=400)
    args = parser.parse_args()

    sources = {
        'ML': args.examples / 'HydrogenMaleate-MACE-POLAR/md-500ps',
        'ML/MM electrostatic': (
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps'
        ),
        'ML/MM mechanical': (
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps'
        ),
    }
    samples = {
        label: read_xyz_samples(path) for label, path in sources.items()
    }
    samples = {
        label: trajectory[trajectory[:, 0] <= 100.0 + 1e-9]
        for label, trajectory in samples.items()
    }
    if any(len(trajectory) != 10_000 for trajectory in samples.values()):
        raise ValueError('Expected exactly 10,000 saved frames for each 100 ps PMF')

    timing_logs = {
        'ML': args.examples / 'HydrogenMaleate-PME/full-500ps/run.log',
        'ML/MM electrostatic': (
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps/run.log'
        ),
        'ML/MM mechanical': (
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps/run.log'
        ),
        'OpenMM-ML mechanical': (
            args.examples / 'HydrogenMaleate-OpenMM-Mechanical/run.log'
        ),
    }
    timings = {
        label: timing_from_log(path) for label, path in timing_logs.items()
    }

    figure = plt.figure(figsize=(15.2, 5.5), dpi=180, facecolor='white')
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.07, 1.43, 0.92],
        left=0.04,
        right=0.985,
        top=0.86,
        bottom=0.18,
        wspace=0.32,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    draw_molecule(
        axes[0],
        args.examples / 'HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz',
    )

    pmf_summary = {}
    for index, (label, trajectory) in enumerate(samples.items()):
        q = trajectory[:, 1]
        profile, barrier = central_pmf(q)
        profiles, barriers = central_bootstrap(
            q, args.bootstrap_repetitions, 20260930 + index
        )
        profiles = np.where(np.isfinite(profiles), profiles, np.nan)
        barriers = barriers[np.isfinite(barriers)]
        lower, upper = np.nanpercentile(profiles, (16, 84), axis=0)
        axes[1].plot(
            CENTRAL_CENTERS,
            profile,
            color=COLORS[label],
            linewidth=2.4,
            label=label,
        )
        axes[1].fill_between(
            CENTRAL_CENTERS,
            lower,
            upper,
            color=COLORS[label],
            alpha=0.17,
        )
        pmf_summary[label] = {
            'frames': len(q),
            'duration_ps': 100.0,
            'barrier_kBT': float(barrier),
            'barrier_95_percent_kBT': np.percentile(
                barriers, (2.5, 97.5)
            ).tolist(),
            'valid_bootstrap_repetitions': len(barriers),
        }
    axes[1].axvline(0, color='0.5', linestyle='--', linewidth=1)
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_ylim(0, 6)
    axes[1].set_xlabel(r'$\delta=r_R-r_L$ (Å)', fontsize=12)
    axes[1].set_ylabel(r'$\beta[F(\delta)-F_{\rm well}]$', fontsize=12)
    axes[1].set_title(
        '(b) Symmetrized free energy, 0–100 ps',
        loc='left',
        fontsize=14,
        weight='bold',
        pad=12,
    )
    axes[1].text(
        0.5,
        -0.18,
        'Shading: 16–84% from 1 ps block bootstrap',
        transform=axes[1].transAxes,
        ha='center',
        fontsize=9,
        color='#596169',
    )
    axes[1].legend(frameon=False, fontsize=9.5, loc='upper center')
    axes[1].grid(axis='y', alpha=0.2)

    labels = list(timings)
    rates = np.asarray([timings[label]['ps_per_day'] for label in labels])
    low = np.asarray(
        [timings[label]['chunk_10_90_percent_ps_per_day'][0] for label in labels]
    )
    high = np.asarray(
        [timings[label]['chunk_10_90_percent_ps_per_day'][1] for label in labels]
    )
    x = np.arange(len(labels))
    bars = axes[2].bar(
        x,
        rates,
        yerr=[np.maximum(rates - low, 0), np.maximum(high - rates, 0)],
        capsize=4,
        color=[COLORS[label] for label in labels],
        width=0.66,
        error_kw={'elinewidth': 1.3, 'ecolor': '#56616a'},
    )
    axes[2].set_xticks(
        x,
        ['ML', 'Electrostatic\nML/MM', 'Mechanical\nML/MM', 'OpenMM\ncontrol'],
        fontsize=8.5,
    )
    axes[2].set_ylabel('Throughput (ps/day)', fontsize=12)
    axes[2].set_ylim(0, high.max() * 1.18)
    axes[2].set_title(
        '(c) GPU throughput', loc='left', fontsize=14, weight='bold', pad=12
    )
    axes[2].grid(axis='y', alpha=0.2)
    axes[2].set_axisbelow(True)
    for bar, label in zip(bars, labels, strict=True):
        rate = timings[label]['ps_per_day']
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            rate + 0.025 * high.max(),
            f'{rate:.0f}',
            ha='center',
            va='bottom',
            fontsize=9,
            weight='bold',
        )
    axes[2].text(
        0.5,
        -0.18,
        'Whiskers: 10–90% across timing chunks',
        transform=axes[2].transAxes,
        ha='center',
        fontsize=9,
        color='#596169',
    )

    figure.suptitle('Hydrogen maleate at 330 K', fontsize=16, weight='bold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor='white')
    figure.savefig(args.output.with_suffix('.pdf'), facecolor='white')
    summary = {'pmf': pmf_summary, 'gpu_throughput': timings}
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
