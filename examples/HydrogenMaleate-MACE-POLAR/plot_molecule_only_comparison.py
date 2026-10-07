#!/usr/bin/env python3
"""Compare completed molecule-only runs with the saved 13-water figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from four_method_pmf import CENTRAL_CENTERS, block_resample, central_pmf, timing_from_log
from three_panel_summary import draw_molecule

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Examples/HydrogenMaleate-Analysis'
DATA = OUT / 'molecule-only-comparison-data'
LABELS = {
    'ml': 'ML reference',
    'electrostatic': 'ML/MM electrostatic',
    'mechanical': 'ML/MM mechanical',
    'openmm': 'OpenMM mechanical',
}
COLORS = dict(zip(LABELS, ['#254b8d', '#d06a29', '#36845b', '#8b5aa5']))


def main():
    old = json.loads((OUT / 'three-panel-pme-100ps.json').read_text())
    samples = {key: np.loadtxt(DATA / f'{key}.csv', delimiter=',', skiprows=1)
               for key in LABELS}
    timings = {'ml': old['gpu_throughput']['ML']}
    for key, filename in (
        ('electrostatic', 'electrostatic-100ps.log'),
        ('mechanical', 'mechanical-100ps.log'),
        ('openmm', 'openmm-mechanical-100ps.log'),
    ):
        timings[key] = timing_from_log(DATA / filename)

    figure = plt.figure(figsize=(15.2, 6.0), dpi=180, facecolor='white')
    grid = figure.add_gridspec(1, 3, width_ratios=[1.07, 1.43, 0.92],
                              left=0.04, right=0.985, top=0.84, bottom=0.28, wspace=0.32)
    axes = [figure.add_subplot(grid[0, i]) for i in range(3)]
    draw_molecule(axes[0], DATA / 'initial.xyz')
    summary = {
        'temperature_K': 330,
        'duration_ps': 100,
        'new_runs_ML_atoms': 11,
        'new_runs_MM_atoms': 156,
        'new_runs_restraint': None,
        'reference': 'Earlier full ML, first 100 ps, same reference used in upper figure',
        'new_runs_start': 'Full ML frame at 100 ps (not the 10 ps FIRES start)',
        'coordinate': 'delta = nearest H-to-right-carboxyl-O distance minus nearest H-to-left-carboxyl-O distance; Angstrom',
        'method': 'Symmetrized histogram, 0.025 Angstrom bins, Gaussian smoothing sigma 0.075 Angstrom; 1 ps circular block bootstrap, 400 draws',
        'central_curves': 'Conditional on |delta| <= 1.3 Angstrom; offset to maximum density in 0.2 <= |delta| <= 0.8 Angstrom',
        'beta': '1/(k_B T), where k_B is the Boltzmann constant and T is temperature',
        'timing': 'Production intervals between progress records; compilation excluded; includes integration and output',
        'runs': {},
    }
    for index, (key, array) in enumerate(samples.items()):
        if len(array) != 10000 or not np.isclose(array[-1, 0], 100):
            raise ValueError(f'Incomplete 100 ps samples: {key}')
        q = array[:, 1]
        fraction = float(np.mean(np.abs(q) <= 1.3))
        profile, barrier = central_pmf(q)
        rng = np.random.default_rng(20261001 + index)
        profiles, barriers = [], []
        for _ in range(400):
            resampled = block_resample(q, 100, rng)
            if not np.any(np.abs(resampled) <= 1.3):
                continue
            p, b = central_pmf(resampled)
            profiles.append(p)
            barriers.append(b)
        lower, upper = np.percentile(profiles, (16, 84), axis=0)
        label = LABELS[key] + (f' ({100*fraction:.1f}% retained)' if fraction < .999 else '')
        axes[1].plot(CENTRAL_CENTERS, profile, color=COLORS[key], lw=2.2, label=label)
        axes[1].fill_between(CENTRAL_CENTERS, lower, upper, color=COLORS[key], alpha=.15)
        summary['runs'][key] = {
            'samples': len(q), 'fraction_inside_central_range': fraction,
            'central_barrier_kBT': float(barrier),
            'central_barrier_95_percent_kBT': np.percentile(barriers, (2.5, 97.5)).tolist(),
            'half_run_central_barriers_kBT': [float(central_pmf(part)[1])
                                             if np.any(np.abs(part) <= 1.3) else None
                                             for part in np.array_split(q, 2)],
            'profile_kBT': profile.tolist(),
            'profile_16_84_percent_kBT': [lower.tolist(), upper.tolist()],
            'timing': timings[key],
        }
    axes[1].set(xlim=(-1.3, 1.3), ylim=(0, 6), xlabel=r'$\delta=r_R-r_L$ (Å)',
                ylabel=r'$\beta[F(\delta)-F_{\rm well}]$')
    axes[1].set_title('(b) Symmetrized free energy, 0–100 ps', loc='left', weight='bold', fontsize=14, pad=12)
    axes[1].axvline(0, color='.5', ls='--', lw=1)
    axes[1].legend(frameon=False, fontsize=8.5, loc='upper center')
    axes[1].text(.5, -.20, 'Shading: 16–84% from 1 ps block bootstrap',
                 transform=axes[1].transAxes, ha='center', fontsize=9, color='#596169')

    rates = np.array([timings[key]['ps_per_day'] for key in LABELS])
    low = np.array([timings[key]['chunk_10_90_percent_ps_per_day'][0] for key in LABELS])
    high = np.array([timings[key]['chunk_10_90_percent_ps_per_day'][1] for key in LABELS])
    x = np.arange(4)
    bars = axes[2].bar(x, rates, yerr=[np.maximum(rates-low, 0), np.maximum(high-rates, 0)],
                       capsize=4, color=list(COLORS.values()), width=.66)
    axes[2].set_xticks(x, ['ML', 'Electrostatic\nML/MM', 'Mechanical\nML/MM', 'OpenMM\ncontrol'], fontsize=8.5)
    axes[2].set(ylabel='Throughput (ps/day)', ylim=(0, high.max()*1.18))
    axes[2].set_title('(c) GPU throughput', loc='left', weight='bold', fontsize=14, pad=12)
    for bar, rate in zip(bars, rates):
        axes[2].text(bar.get_x()+bar.get_width()/2, rate+.025*high.max(),
                     f'{rate:.0f}', ha='center', fontsize=9, weight='bold')
    axes[2].text(.5, -.20, 'Whiskers: 10–90% across timing chunks',
                 transform=axes[2].transAxes, ha='center', fontsize=9, color='#596169')
    for axis in axes[1:]:
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    figure.suptitle('Molecule-only ML/MM · 330 K · no restraint', fontsize=16, weight='bold')
    figure.text(.5, .035,
                r'$F$: free energy; $\beta=1/(k_{\rm B}T)$; $k_{\rm B}$: Boltzmann constant; $T$: temperature.'
                '\nOpenMM: 71.9% of frames lie outside ±1.3 Å; its displayed curve is conditional on the central conformations.',
                ha='center', fontsize=10, color='#465159')
    new_path = OUT / 'molecule-only-fes-throughput.png'
    figure.savefig(new_path, facecolor='white')
    figure.savefig(new_path.with_suffix('.pdf'), facecolor='white')
    plt.close(figure)
    (new_path.with_suffix('.json')).write_text(json.dumps(summary, indent=2)+'\n')

    # Preserve the previously delivered top figure exactly.
    images = [plt.imread(OUT / 'three-panel-pme-100ps.png'), plt.imread(new_path)]
    stacked, stacked_axes = plt.subplots(2, 1, figsize=(15.2, 12.0), dpi=180,
                                         gridspec_kw={'height_ratios': [im.shape[0]/im.shape[1] for im in images]})
    stacked.subplots_adjust(left=0, right=1, bottom=0, top=.975, hspace=.08)
    for axis, im, title in zip(stacked_axes, images,
                              ['Earlier: 13 ML waters, fixed soft restraint', 'New: molecule-only, no restraint']):
        axis.imshow(im)
        axis.axis('off')
        axis.set_title(title, fontsize=13, weight='bold', pad=4)
    stacked.savefig(OUT / 'earlier-vs-molecule-only-fes-throughput.png', facecolor='white')
    stacked.savefig(OUT / 'earlier-vs-molecule-only-fes-throughput.pdf', facecolor='white')
    plt.close(stacked)

    diagnostic, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    for key, array in samples.items():
        axis.hist(np.abs(array[:, 1]), bins=np.arange(0, 6.1, .1), density=True,
                  histtype='step', lw=2, color=COLORS[key], label=LABELS[key])
    axis.axvline(1.3, color='.4', ls='--', label='central plot limit')
    axis.set(xlabel=r'$|\delta|$ (Å)', ylabel='Probability density (Å$^{-1}$)',
             title='All 100 ps frames: wider proton coordinate distribution')
    axis.legend(frameon=False)
    diagnostic.tight_layout()
    diagnostic.savefig(OUT / 'molecule-only-wide-coordinate.png')
    plt.close(diagnostic)
    for key, result in summary['runs'].items():
        print(key, 'barrier', result['central_barrier_kBT'], 'CI', result['central_barrier_95_percent_kBT'],
              'retained', result['fraction_inside_central_range'],
              'ps/day', result['timing']['ps_per_day'],
              'ms/step', result['timing']['milliseconds_per_md_step'],
              'half barriers', result['half_run_central_barriers_kBT'])


if __name__ == '__main__':
    main()
