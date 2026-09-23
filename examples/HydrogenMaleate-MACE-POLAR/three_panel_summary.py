#!/usr/bin/env python3
"""Show the proton coordinate, symmetrized free energy, and CPU throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from render_comparison import align_top_view, centered_positions, solute_bonds, top_view_reference
from symmetrized_free_energy import CENTERS, bootstrap, free_energy, read_samples


COLORS = {'Full MACE-POLAR': '#254b8d', 'First-shell ML/MM': '#d06a29'}
FOLDERS = {'Full MACE-POLAR': 'HydrogenMaleate-MACE-POLAR',
           'First-shell ML/MM': 'HydrogenMaleate-MLMM'}


def draw_molecule(ax, initial: Path) -> None:
    atoms = read(initial)
    box = float(atoms.cell[0, 0])
    xyz = align_top_view(centered_positions(np.asarray(atoms.positions), box),
                         top_view_reference(initial, box))
    left_o = int(np.argmin(np.linalg.norm(xyz[:2] - xyz[10], axis=1)))
    right_o = 2 + int(np.argmin(np.linalg.norm(xyz[2:4] - xyz[10], axis=1)))
    for i, j in solute_bonds(initial, box):
        ax.plot(xyz[[i, j], 0], xyz[[i, j], 1], color='#4b5358', lw=2.2, zorder=1)
    ax.plot(xyz[[left_o, 10], 0], xyz[[left_o, 10], 1],
            color='#4b5358', lw=2.2, zorder=1)
    ax.plot(xyz[[right_o, 10], 0], xyz[[right_o, 10], 1],
            color='#9860a1', lw=1.8, linestyle=(0, (3, 3)), zorder=1)
    for index in range(11):
        symbol = atoms[index].symbol
        face = {'C': '#4b5358', 'O': '#d95449', 'H': 'white'}[symbol]
        edge = '#56616a' if symbol == 'H' else 'white'
        text_color = '#303940' if symbol == 'H' else 'white'
        if index == 10:
            face, edge, text_color = '#9a50af', 'white', 'white'
        size = 350 if symbol != 'H' else 190
        ax.scatter(xyz[index, 0], xyz[index, 1], s=size, c=face,
                   edgecolors=edge, linewidths=1, zorder=3)
        ax.text(xyz[index, 0], xyz[index, 1], symbol, ha='center', va='center',
                fontsize=9, weight='bold', color=text_color, zorder=4)
    ax.annotate(r'$r_L$', xy=(xyz[left_o, :2] + xyz[10, :2]) / 2,
                xytext=(-25, 24), textcoords='offset points', color='#7b398a',
                fontsize=13, arrowprops={'arrowstyle': '-', 'color': '#7b398a'})
    ax.annotate(r'$r_R$', xy=(xyz[right_o, :2] + xyz[10, :2]) / 2,
                xytext=(12, 28), textcoords='offset points', color='#7b398a',
                fontsize=13, arrowprops={'arrowstyle': '-', 'color': '#7b398a'})
    ax.text(0.5, -0.04, r'$\delta=r_R-r_L$', transform=ax.transAxes,
            ha='center', va='top', fontsize=15)
    ax.text(0.5, -0.14, r'$r_{L/R}$: H to nearest left/right carboxyl O',
            transform=ax.transAxes, ha='center', va='top', fontsize=9, color='#596169')
    ax.set_xlim(-3.7, 3.7)
    ax.set_ylim(-2.4, 2.4)
    ax.set_aspect('equal')
    ax.set_anchor('N')
    ax.axis('off')
    ax.set_title('(a) Proton coordinate', loc='left', fontsize=14, weight='bold', pad=12)


def draw_free_energy(ax, samples: dict[str, np.ndarray]) -> dict:
    common_end = min(array[-1, 0] for array in samples.values())
    summary = {'matched_time_ps': common_end, 'runs': {}}
    for number, (label, array) in enumerate(samples.items()):
        q = array[array[:, 0] <= common_end + 1e-9, 1]
        profile, barrier = free_energy(q, 0.075)
        profiles, barriers = bootstrap(q, 0.075, 100, 400,
                                       np.random.default_rng(20260923 + number))
        lower, upper = np.percentile(profiles, (16, 84), axis=0)
        ax.plot(CENTERS, profile, color=COLORS[label], linewidth=2.3, label=label)
        ax.fill_between(CENTERS, lower, upper, color=COLORS[label], alpha=0.18)
        summary['runs'][label] = {
            'frames': len(q), 'barrier_kBT': barrier,
            'barrier_95_percent_kBT': np.percentile(barriers, (2.5, 97.5)).tolist(),
        }
    ax.set_xlim(-0.82, 0.82)
    ax.set_ylim(0, 4.2)
    ax.axvline(0, color='#838b91', linestyle='--', lw=1)
    ax.set_xlabel(r'$\delta=r_R-r_L$ (Å)', fontsize=12)
    ax.set_ylabel(r'$\beta[F(\delta)-F_{\min}]$', fontsize=13)
    ax.set_title('(b) Symmetrized free energy', loc='left', fontsize=14,
                 weight='bold', pad=12)
    ax.text(0.03, 0.94,
            rf'$\beta=(k_{{\mathrm{{B}}}}T)^{{-1}}$, $T=330$ K · matched {common_end:.1f} ps',
            transform=ax.transAxes, va='top', fontsize=9.5, color='#4d555b')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.83),
              fontsize=9.5)
    ax.grid(axis='y', color='#e3e7e9', linewidth=0.8)
    ax.text(0.5, -0.18, 'Shading: 16–84% from 1 ps block bootstrap',
            transform=ax.transAxes, ha='center', fontsize=9, color='#596169')
    return summary


def throughput_from_log(path: Path) -> dict:
    rows = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith('{'):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'time_ps' in row and 'elapsed_seconds_this_process' in row:
                rows.append((row['time_ps'], row['elapsed_seconds_this_process']))
    array = np.asarray(rows, dtype=float)
    if len(array) < 3:
        raise ValueError(f'Insufficient progress records in {path}')
    dt_ps, dt_sec = np.diff(array, axis=0).T
    valid = (dt_ps > 0) & (dt_sec > 0)
    chunks = dt_ps[valid] * 86400 / dt_sec[valid]
    rate = (array[-1, 0] - array[0, 0]) * 86400 / (array[-1, 1] - array[0, 1])
    return {
        'ps_per_day': float(rate),
        'chunk_10_90_percent_ps_per_day': np.percentile(chunks, (10, 90)).tolist(),
        'chunks': len(chunks),
        'time_ps': float(array[-1, 0]),
    }


def draw_throughput(ax, examples: Path) -> dict:
    result = {
        label: throughput_from_log(examples / folder / 'md-100ps/run.log')
        for label, folder in FOLDERS.items()
    }
    labels = list(result)
    heights = [result[label]['ps_per_day'] for label in labels]
    lower = [heights[i] - result[label]['chunk_10_90_percent_ps_per_day'][0]
             for i, label in enumerate(labels)]
    upper = [result[label]['chunk_10_90_percent_ps_per_day'][1] - heights[i]
             for i, label in enumerate(labels)]
    # The chunk quantiles can be asymmetric around the overall time-weighted rate.
    lower = np.maximum(lower, 0)
    upper = np.maximum(upper, 0)
    bars = ax.bar(np.arange(2), heights, yerr=[lower, upper], capsize=4,
                  color=[COLORS[label] for label in labels], width=0.58,
                  error_kw={'elinewidth': 1.3, 'ecolor': '#56616a'})
    ax.set_xticks(np.arange(2), ['Full\nMACE-POLAR', 'First-shell\nML/MM'])
    ax.set_ylabel('Throughput (ps/day)', fontsize=12)
    ax.set_ylim(0, max(result[label]['chunk_10_90_percent_ps_per_day'][1]
                       for label in labels) * 1.15)
    ax.set_title('(c) CPU throughput', loc='left', fontsize=14, weight='bold', pad=12)
    ax.text(0.5, 0.94, 'Boltzmann · 8 cores per run', transform=ax.transAxes,
            ha='center', va='top', fontsize=9.5, color='#4d555b')
    ax.grid(axis='y', color='#e3e7e9', linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, height in zip(bars, heights, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, height * 0.91,
                f'{height:.1f}', ha='center', va='top', fontsize=11,
                weight='bold', color='white')
    ax.text(0.5, -0.18, 'Whiskers: 10–90% across 0.1 ps chunks',
            transform=ax.transAxes, ha='center', fontsize=9, color='#596169')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    samples = {
        label: read_samples(args.examples / folder / 'md-100ps')
        for label, folder in FOLDERS.items()
    }
    figure = plt.figure(figsize=(15.2, 5.5), dpi=180, facecolor='white')
    grid = figure.add_gridspec(1, 3, width_ratios=[1.07, 1.43, 0.8],
                              left=0.04, right=0.98, top=0.86, bottom=0.17,
                              wspace=0.32)
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    draw_molecule(axes[0], args.examples / 'HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz')
    energy = draw_free_energy(axes[1], samples)
    throughput = draw_throughput(axes[2], args.examples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor='white')
    figure.savefig(args.output.with_suffix('.pdf'), facecolor='white')
    args.output.with_suffix('.json').write_text(
        json.dumps({'free_energy': energy, 'throughput': throughput}, indent=2) + '\n'
    )
    print(json.dumps({'free_energy': energy, 'throughput': throughput}, indent=2))


if __name__ == '__main__':
    main()
