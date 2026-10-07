#!/usr/bin/env python3
"""Check population drift and block-length sensitivity in saved trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from four_method_pmf import block_resample
from plot_full_trajectory_comparison import COLORS, LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        'groups': {'A': '|delta| < 1.3 Angstrom',
                   'B': '1.3 <= |delta| < 3.5 Angstrom',
                   'C': '|delta| >= 3.5 Angstrom'},
        'uncertainty': '95 percent circular block bootstrap intervals; 400 resamples; descriptive sampling uncertainty, not a convergence guarantee',
        'methods': {},
    }
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharey=True)
    for i, (key, label) in enumerate(LABELS.items()):
        values = np.loadtxt(args.data / f'{key}.csv', delimiter=',', skiprows=1)
        time, q = values.T
        interval = np.median(np.diff(time))
        if not np.all(np.isfinite(values)) or not np.allclose(np.diff(time), interval):
            raise ValueError(f'Invalid time series: {key}')
        group = np.digitize(np.abs(q), [1.3, 3.5])
        fractions = lambda g: np.bincount(g, minlength=3) / len(g)
        entry = np.flatnonzero(group == 2)
        record = {
            'time_ps': float(time[-1]), 'population_A_B_C': fractions(group).tolist(),
            'first_half_population_A_B_C': fractions(group[:len(q)//2]).tolist(),
            'second_half_population_A_B_C': fractions(group[len(q)//2:]).tolist(),
            'first_C_entry_ps': float(time[entry[0]]) if len(entry) else None,
            'population_95_percent_intervals': {},
        }
        for block_ps in (1, 5):
            rng = np.random.default_rng(20261001 + i * 10 + block_ps)
            estimates = np.array([fractions(block_resample(
                group, round(block_ps / interval), rng)) for _ in range(400)])
            record['population_95_percent_intervals'][f'{block_ps}_ps_blocks'] = (
                np.percentile(estimates, [2.5, 97.5], axis=0).T.tolist())
        result['methods'][key] = record
        axes[i].plot(time, q, color=COLORS[key], linewidth=.5)
        axes[i].set(title=f'{label} — {time[-1]:.3g} ps', ylabel='δ (Å)',
                    xlim=(0, time[-1]), ylim=(-5, 5))
        axes[i].spines[['top', 'right']].set_visible(False)
        axes[i].axhline(0, color='.6', lw=.5)
        axes[i].grid(alpha=.15)
        axes[i].set_xlabel('Time (ps)')
    fig.suptitle('Proton coordinate along each complete simulation', fontsize=14)
    fig.tight_layout(rect=(0, .045, 1, .955))
    fig.text(.5, .012,
             'δ: nearest proton–oxygen distance at the right end minus that at the left end. Each horizontal axis spans its own full simulation.',
             ha='center', fontsize=8)
    fig.savefig(args.output / 'coordinate-time-series.png', dpi=180)
    plt.close(fig)
    (args.output / 'sampling-diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    for key, record in result['methods'].items():
        print(key, json.dumps(record))


if __name__ == '__main__':
    main()
