#!/usr/bin/env python3
"""Plot internal and water hydrogen-bond contacts along the full-ML trajectory."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    data = np.genfromtxt(args.directory / 'hydrogen-bond-time-series.csv',
                         delimiter=',', names=True)
    t = data['time_ps']
    if len(t) != 50000 or not np.allclose(np.diff(t), .01):
        raise ValueError('Expected the complete, uniformly saved 500 ps trajectory')
    # Nonoverlapping 1 ps pieces: each point is the fraction of the 100 saved
    # frames satisfying the geometric criterion, not a time-averaged distance.
    if not np.allclose(t.reshape(-1, 100)[:, -1], np.arange(1, 501)):
        raise ValueError('Time intervals do not align with whole picoseconds')
    times = np.arange(.5, 500, 1)
    internal = data['internal_hbond'].reshape(-1, 100).mean(axis=1)
    water = data['water_hbond'].reshape(-1, 100).mean(axis=1)
    loss = np.flatnonzero((times > 400) & (internal < .5) & (water > .5))
    regain = np.flatnonzero((times > 475) & (internal > .5))
    if not len(loss) or not len(regain):
        raise ValueError('No detected solvent switch and subsequent internal return')
    switch_time = float(times[loss[0]])
    return_time = float(times[regain[0]])
    report = {
        'bin_width_ps': 1,
        'switch_to_water_bin_center_ps': switch_time,
        'internal_return_bin_center_ps': return_time,
        'time_markers_definition': 'First 1 ps interval after 400 ps with internal fraction <0.5 and water fraction >0.5; first interval after 475 ps with internal fraction >0.5. Markers are approximate, based on this trajectory, not kinetic rates.',
        'proton_nearest_water_fraction': float(data['nearest_oxygen_is_water'].mean()),
        'last_3_ps_internal_fraction': float(data['internal_hbond'][t > 497].mean()),
        'last_3_ps_water_fraction': float(data['water_hbond'][t > 497].mean()),
        'intervals': [],
    }
    for lo, hi in ((0, 430), (440, 495), (497, 500)):
        selected = (t > lo) & (t <= hi)
        report['intervals'].append({
            'start_ps': lo, 'end_ps': hi,
            'internal_fraction': float(data['internal_hbond'][selected].mean()),
            'water_fraction': float(data['water_hbond'][selected].mean()),
        })
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharey=True,
                             gridspec_kw={'height_ratios': [1, 1.15]})
    fig.subplots_adjust(left=.10, right=.98, top=.86, bottom=.20, hspace=.50)
    for axis, limits, title in zip(
            axes, [(0, 500), (425, 500)],
            ['Complete 500 ps trajectory', 'Last 75 ps: opening and return']):
        axis.plot(times, internal, color='#254b8d', linewidth=1.6,
                  label='Internal hydrogen bond')
        axis.plot(times, water, color='#d06a29', linewidth=1.6,
                  label='Hydrogen bond to water')
        axis.set(xlim=limits, ylim=(-.03, 1.05), xlabel='Time (ps)',
                 ylabel='Fraction of frames\nin each 1 ps interval')
        axis.set_title(title, loc='left', fontsize=11)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(alpha=.15)
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    for tmark, text, offset in ((switch_time, 'To water', 7),
                                 (return_time, 'Internal bond reforms', -20)):
        axes[1].axvline(tmark, color='.5', linestyle='--', linewidth=.7)
        axes[1].annotate(f'{text}\n≈{tmark:.3g} ps',
                         xy=(tmark, .45), xytext=(tmark + offset, .56),
                         fontsize=9, ha='center',
                         arrowprops={'arrowstyle': '-', 'color': '.5'})
    fig.suptitle('Full ML · hydrogen maleate in water · 330 K',
                 fontsize=14, weight='bold', y=.98)
    fig.text(.5, .91, 'The proton stays attached to the solute; its hydrogen-bond partner changes.',
             ha='center', fontsize=10)
    fig.text(.5, .025,
             'Internal: proton bonds to an oxygen on the opposite carboxyl group. Water: proton bonds to any solvent oxygen.\n'
             'A contact is counted when proton–acceptor distance <2.5 Å, donor–acceptor distance <3.5 Å,\n'
             'and donor–proton–acceptor angle >150°. Donor: nearest solute oxygen, required within 1.3 Å of the proton\n'
             'and nearer than any water oxygen. Distances account for periodic boundaries. Both types can coexist.\n'
             'These curves describe this trajectory; they do not establish equilibrium probabilities.',
             ha='center', va='bottom', fontsize=8, color='#465159')
    for suffix in ('.png', '.pdf'):
        fig.savefig(args.directory / f'full-ml-hydrogen-bond-switch-return{suffix}', dpi=180)
    plt.close(fig)
    (args.directory / 'switch-return-summary.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
