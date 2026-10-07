#!/usr/bin/env python3
"""Compare proton hydrogen-bond partners for all completed current MD methods."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def interval_fractions(path):
    data = np.genfromtxt(path, delimiter=',', names=True)
    time = data['time_ps']
    spacing = float(np.median(np.diff(time)))
    count = round(1 / spacing)
    duration = round(time[-1])
    if (not np.allclose(np.diff(time), spacing) or len(time) != count * duration
            or not np.allclose(time.reshape(-1, count)[:, -1],
                               np.arange(1, duration + 1))):
        raise ValueError(f'Invalid saved frame sampling: {path}')
    centers = np.arange(.5, duration, 1)
    internal = data['internal_hbond'].reshape(-1, count).mean(axis=1)
    water = data['water_hbond'].reshape(-1, count).mean(axis=1)
    water_dominant = np.flatnonzero((water > .5) & (internal < .5))
    first_water = float(centers[water_dominant[0]]) if len(water_dominant) else None
    return_indices = (np.flatnonzero((centers > first_water) & (internal > .5))
                      if first_water is not None else np.array([], dtype=int))
    last = time > time[-1] - 3
    summary = {
        'frames': len(time), 'last_time_ps': float(time[-1]),
        'saved_frame_interval_ps': spacing, 'frames_per_1_ps_interval': count,
        'internal_fraction': float(data['internal_hbond'].mean()),
        'water_fraction': float(data['water_hbond'].mean()),
        'last_3_ps_internal_fraction': float(data['internal_hbond'][last].mean()),
        'last_3_ps_water_fraction': float(data['water_hbond'][last].mean()),
        'first_water_dominant_interval_center_ps': first_water,
        'first_internal_dominant_interval_after_water_ps': (
            float(centers[return_indices[0]]) if len(return_indices) else None),
        'proton_nearest_water_fraction': float(data['nearest_oxygen_is_water'].mean()),
        'maximum_solute_O_H_distance_angstrom': float(data['donor_H_distance_angstrom'].max()),
    }
    return centers, internal, water, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--full-ml-data', type=Path, required=True)
    args = parser.parse_args()
    out = args.directory
    fig = plt.figure(figsize=(13, 11), dpi=180)
    grid = fig.add_gridspec(4, 2, left=.085, right=.98, top=.88, bottom=.17,
                           height_ratios=[.9, 1, 1, 1], hspace=.62, wspace=.15)
    runs = [('ml', args.full_ml_data, 'Full ML reference · 500 ps', grid[0, :])]
    methods = [('electrostatic', 'ML/MM electrostatic'),
               ('mechanical', 'ML/MM mechanical'),
               ('openmm', 'ML/MM mechanical (OpenMM)')]
    for row, (method, label) in enumerate(methods, start=1):
        for column, (region, description) in enumerate([
                ('molecule', 'molecule-only'), ('fires', '25-water FIRES')]):
            key = f'{region}-{method}'
            runs.append((key, out / 'data' / f'{key}.csv',
                         f'{label} · {description}', grid[row, column]))
    summary = {'definition': json.loads((out / 'data' / 'sources.json').read_text())['definition'],
               'time_markers_definition': 'First 1 ps interval with water fraction >0.5 and internal fraction <0.5; first later interval with internal fraction >0.5. These are descriptive interval centers, not reaction rates.',
               'runs': {}}
    for key, path, title, position in runs:
        t, internal, water, record = interval_fractions(path)
        summary['runs'][key] = record
        axis = fig.add_subplot(position)
        axis.plot(t, internal, color='#254b8d', lw=1.5)
        axis.plot(t, water, color='#d06a29', lw=1.5)
        axis.set(xlim=(0, record['last_time_ps']), ylim=(-.04, 1.04),
                 xlabel='Time (ps)', yticks=[0, .5, 1])
        axis.set_title(title, fontsize=10.5, loc='left')
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.14)
    fig.suptitle('Hydrogen-bond partners · hydrogen maleate in water · 330 K',
                 fontsize=15, weight='bold', y=.98)
    fig.legend(handles=[Line2D([], [], color='#254b8d', label='Internal hydrogen bond'),
                        Line2D([], [], color='#d06a29', label='Hydrogen bond to water')],
               loc='upper center', bbox_to_anchor=(.5, .955), ncol=2, frameon=False)
    fig.supylabel('Fraction of frames in each 1 ps interval', x=.025, fontsize=11)
    fig.text(.5, .018,
             'Top: all 500 ps of full ML. Other panels: all 100 ps of each run; molecule-only on the left, 25-water FIRES on the right.\n'
             'Internal acceptor: an oxygen on the opposite carboxyl group. Water acceptor: any solvent oxygen.\n'
             'A hydrogen bond requires proton–acceptor distance <2.5 Å, donor–acceptor distance <3.5 Å and donor–proton–acceptor angle >150°.\n'
             'Donor: nearest solute oxygen, required within 1.3 Å of the proton and closer than every water oxygen. Periodic distances are used.\n'
             'JAX: 100 saved frames per 1 ps interval; OpenMM: 5. FIRES strength: 500 kcal mol⁻¹ Å⁻². Both contact types can coexist.\n'
             'These plots describe the saved trajectories; they do not establish equilibrium probabilities or transition rates.',
             ha='center', va='bottom', fontsize=8, color='#465159')
    for suffix in ('.png', '.pdf'):
        fig.savefig(out / f'all-hydrogen-bond-partners{suffix}')
    plt.close(fig)
    (out / 'hydrogen-bond-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for key, record in summary['runs'].items():
        print(key, json.dumps(record))


if __name__ == '__main__':
    main()
