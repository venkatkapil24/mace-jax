#!/usr/bin/env python3
"""Show complete coordinate distributions for two ML/MM partitions."""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from four_method_pmf import block_resample, timing_from_log
from plot_final_conformers import infer_bonds
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'Examples/HydrogenMaleate-Analysis'
DATA = OUT / 'molecule-only-comparison-data'
LABELS = {'ml': 'ML', 'electrostatic': 'ML/MM electrostatic',
          'mechanical': 'ML/MM mechanical', 'openmm': 'ML/MM mechanical (OpenMM)'}
COLORS = dict(zip(LABELS, ['#254b8d', '#d06a29', '#36845b', '#8b5aa5']))


def three_significant(value, signed=False):
    """Preserve trailing decimal zeros, e.g. 1.20 and 0.390."""
    text = format(value, '+#.3g' if signed else '#.3g')
    # Integer annotations such as 476 ps already contain three digits.
    return text[:-1] if text.endswith('.') else text


def draw_snapshot(axis, label, source_label, bonds):
    atoms = read(DATA / 'conformer-snapshots' / f'{source_label}.xyz')
    raw = np.asarray(atoms.positions)
    box = float(atoms.cell[0, 0])
    xyz = raw - raw[5]
    xyz -= box*np.rint(xyz/box)
    xyz -= xyz[:8].mean(axis=0)
    # Use each actual structure's best-fit heavy-atom plane for its top view.
    _, _, vt = np.linalg.svd(xyz[:8], full_matrices=False)
    xy = xyz @ vt[:2].T
    if (xy[2:4].mean(axis=0)-xy[:2].mean(axis=0))[0] < 0:
        xy[:, 0] *= -1
    if xy[4:6, 1].mean() > xy[:4, 1].mean():
        xy[:, 1] *= -1
    for i, j in bonds:
        axis.plot(xy[[i,j],0], xy[[i,j],1], color='#4b5358', lw=2.0, zorder=1)
    distances = np.linalg.norm(xyz[:4]-xyz[10],axis=1)
    nearest = int(np.argmin(distances))
    if distances[nearest] < 1.3:
        axis.plot(xy[[nearest,10],0],xy[[nearest,10],1],color='#855092',lw=2.1,zorder=1)
    opposite = np.arange(2,4) if nearest < 2 else np.arange(0,2)
    other = int(opposite[np.argmin(distances[opposite])])
    if distances[other] < 2.4:
        axis.plot(xy[[other,10],0],xy[[other,10],1],color='#aa83b2',ls='--',lw=1.1,zorder=1)
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        face = '#ad57bb' if i==10 else {'C':'#4b5358','O':'#d95145','H':'white'}[symbol]
        size = 100 if symbol=='H' else 190
        axis.scatter(xy[i,0],xy[i,1],s=size,c=face,edgecolors='#647079' if symbol=='H' and i!=10 else 'white',lw=.7,zorder=3)
        axis.text(xy[i,0],xy[i,1],symbol,ha='center',va='center',fontsize=7.5,
                  color='#303940' if symbol=='H' and i!=10 else 'white',weight='bold',zorder=4)
    center = (xy.min(axis=0)+xy.max(axis=0))/2
    span = max(np.ptp(xy[:,0]),np.ptp(xy[:,1]))/2+.5
    axis.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span))
    axis.set_aspect('equal')
    axis.axis('off')
    delta = float(atoms.info['delta_angstrom'])
    axis.text(.5,-.025,f'{label}   $\\delta={three_significant(delta, signed=True)}$ Å',
              transform=axis.transAxes,ha='center',va='top',fontsize=11,weight='bold')
    method = '13-water mechanical' if atoms.info['source_run'].endswith('mechanical-100ps') else 'ML'
    axis.text(.5,-.16,f'{method}, {three_significant(atoms.info["source_time_ps"])} ps',transform=axis.transAxes,
              ha='center',va='top',fontsize=8,color='#596169')
    metadata = {key: value.item() if isinstance(value, np.generic) else value
                for key, value in atoms.info.items()}
    metadata.update(label=label, source_snapshot_label=source_label)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-data', type=Path)
    parser.add_argument('--fires-data', type=Path)
    parser.add_argument('--molecule-data', type=Path,
                        help='Optional molecule-only row above the FIRES row')
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--new-only', action='store_true')
    args = parser.parse_args()
    if args.matched_data:
        args.matched_data = args.matched_data.resolve()
    if args.fires_data:
        args.fires_data = args.fires_data.resolve()
    if args.molecule_data:
        args.molecule_data = args.molecule_data.resolve()
        if not args.fires_data:
            parser.error('--molecule-data requires --fires-data')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        '13 ML waters; fixed soft restraint': {
            'ml': 'full-ml-all.csv', 'electrostatic': 'shell-electrostatic-all.csv',
            'mechanical': 'shell-mechanical-all.csv', 'openmm': 'shell-openmm-all.csv'},
        'Molecule-only; no restraint': {
            'ml': 'full-ml-all.csv', 'electrostatic': 'electrostatic.csv',
            'mechanical': 'mechanical.csv', 'openmm': 'openmm.csv'},
    }
    if args.matched_data:
        previous = {key: DATA / f'{key}.csv' for key in LABELS}
        previous['ml'] = args.matched_data / 'ml.csv'
        matched = {key: args.matched_data / f'{key}.csv' for key in LABELS}
        paths = {'Earlier molecule-only': previous,
                 'Matched 24³ + cuEquivariance; molecule-only': matched}
        if args.new_only:
            paths = {'Matched 24³ + cuEquivariance; molecule-only': matched}
    if args.fires_data:
        paths = {}
        if args.molecule_data:
            paths['Molecule-only; no restraint'] = {
                key: args.molecule_data / f'{key}.csv' for key in LABELS}
        paths['Molecule + 25 ML waters; FIRES, 500 kcal mol⁻¹ Å⁻²'] = {
            key: args.fires_data / f'{key}.csv' for key in LABELS}
    data = {row: {key: np.loadtxt(DATA / filename, delimiter=',', skiprows=1)
                  for key, filename in files.items()} for row, files in paths.items()}
    maximum = max(np.max(np.abs(a[:, 1])) for row in data.values() for a in row.values())
    extent = float(np.ceil((maximum + .2)*2)/2)
    edges = np.arange(-extent, extent + .0125, .025)
    centers = (edges[:-1]+edges[1:])/2

    def profile(q):
        counts = np.histogram(np.concatenate((q, -q)), bins=edges)[0]
        smoothed = gaussian_filter1d(counts.astype(float), .075/.025, mode='constant')
        energy = -np.log(np.maximum(smoothed, 1e-300)/smoothed.max())
        # Leave insufficiently sampled bins blank instead of assigning finite
        # free energies to unvisited portions of the coordinate.
        energy[smoothed < 1] = np.nan
        a_region = energy[np.abs(centers) < 1.3]
        if not np.any(np.isfinite(a_region)):
            # A resampled trajectory may contain no visits to A. Its relative
            # curve is undefined; count it as unsupported, rather than giving
            # it another reference or an artificial finite value.
            return np.full_like(energy,np.nan)
        # Apply the same reference choice separately to every uncertainty
        # estimate, retaining correlations between the A minimum and other bins.
        energy -= np.nanmin(a_region)
        return energy

    if args.fires_data:
        directories = ([args.molecule_data] if args.molecule_data else []) + [args.fires_data]
        timing_rows = [{key: timing_from_log(directory / f'{key}.log')
                        for key in LABELS} for directory in directories]
    elif args.matched_data:
        old_timings = {'ml': timing_from_log(args.matched_data / 'ml.log')}
        for key in ('electrostatic', 'mechanical', 'openmm'):
            filename = 'openmm-mechanical-100ps.log' if key == 'openmm' else f'{key}-100ps.log'
            old_timings[key] = timing_from_log(DATA / filename)
        new_timings = {key: timing_from_log(args.matched_data / f'{key}.log')
                       for key in LABELS}
        timing_rows = [new_timings] if args.new_only else [old_timings, new_timings]
    else:
        old_timings = json.loads((DATA / 'shell-all-timings.json').read_text())
        new_timings = {'ml': old_timings['ml']}
        for key in ('electrostatic', 'mechanical', 'openmm'):
            filename = 'openmm-mechanical-100ps.log' if key == 'openmm' else f'{key}-100ps.log'
            new_timings[key] = timing_from_log(DATA / filename)
        timing_rows = [old_timings, new_timings]
    all_high = max(t['chunk_10_90_percent_ps_per_day'][1]
                   for row in timing_rows for t in row.values()) / 1000

    fig = plt.figure(figsize=(18, 9 if len(data) == 1 else 13.5), dpi=180, facecolor='white')
    grid = fig.add_gridspec(len(data), 3, width_ratios=[1.05, 1.85, 1.05],
                           left=.035, right=.985, top=.85,
                           bottom=.25 if len(data) == 1 else .18,
                           wspace=.22, hspace=.58)
    summary = {
        'x_range_angstrom': [-extent, extent],
        'temperature_K': 330,
        'definition': 'Symmetrized -ln(probability density), offset to the A minimum of each curve (absolute delta <1.3 Angstrom); the same reference is applied independently to each bootstrap estimate',
        'coordinate': 'delta = nearest H-to-right-carboxyl-O distance minus nearest H-to-left-carboxyl-O distance; Angstrom',
        'smoothing_sigma_angstrom': .075, 'bin_width_angstrom': .025,
        'uncertainty': '16-84 percent circular block bootstrap; 1 ps blocks, 400 resamples of original frames before symmetrization',
        'sampling_mask': 'Bins with fewer than 1 Gaussian-smoothed count are blank',
        'timing': 'Production progress intervals, includes integration and output, excludes compilation',
        'plotted_throughput_unit': 'ns/day',
        'plotted_throughput_significant_digits': 3,
        'rows': {},
        'minima_labels': 'A-C group mirrored minima by increasing absolute delta: A <1.3 Angstrom, B 1.3-3.5 Angstrom, C >3.5 Angstrom; these are coordinate groups, not independently established conformer assignments. Curve minima detected with prominence >=0.3 kBT and separation >=0.4 Angstrom',
        'snapshots': {},
        'source_files': {title: {key: str(DATA / filename) for key, filename in paths_for_row.items()}
                         for title, paths_for_row in paths.items()},
    }
    snapshot_grid = grid[:,0].subgridspec(4,1,height_ratios=[.04,1,1,1],hspace=.22)
    bonds = infer_bonds(DATA/'initial.xyz',float(read(DATA/'initial.xyz').cell[0,0]))
    # Use positive-delta frames from the same full ML trajectory for all three
    # groups; the mirrored wells are not independent conformer assignments.
    for index, (letter, source_letter) in enumerate(zip('ABC','DEF')):
        axis = fig.add_subplot(snapshot_grid[index+1,0])
        summary['snapshots'][letter] = draw_snapshot(axis,letter,source_letter,bonds)
    fig.text(.04,.87,'Representative snapshots (A–C)',fontsize=12,weight='bold')
    cached_ml = None
    free_energy_axes = []
    free_energy_low, free_energy_high = 0., 0.
    for row_index, ((row_title, samples), timings) in enumerate(zip(data.items(), timing_rows)):
        axes = [None, fig.add_subplot(grid[row_index,1]),fig.add_subplot(grid[row_index,2])]
        free_energy_axes.append(axes[1])
        summary['rows'][row_title] = {}
        for key, a in samples.items():
            q = a[:, 1]
            if np.any(np.diff(a[:, 0]) <= 0):
                raise ValueError(f'Non-monotonic times: {row_title} {key}')
            interval = float(np.median(np.diff(a[:, 0])))
            if not np.allclose(np.diff(a[:, 0]), interval, atol=1e-6):
                raise ValueError('Uneven time sampling')
            block_frames = round(1/interval)
            p = profile(q)
            if not np.any(np.isfinite(p)):
                raise ValueError(f'No observations to define the A minimum: {row_title} {key}')
            if key == 'ml' and cached_ml is not None:
                lower, upper, bootstrap_reference_fraction = cached_ml
            else:
                rng = np.random.default_rng(20261001 + row_index*10 + list(LABELS).index(key))
                boot = np.asarray([profile(block_resample(q, block_frames, rng)) for _ in range(400)])
                bootstrap_reference_fraction = float(np.mean(np.any(np.isfinite(boot),axis=1)))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    lower, upper = np.nanpercentile(boot, (16, 84), axis=0)
                support = np.mean(np.isfinite(boot), axis=0) >= .84
                lower[~support] = np.nan
                upper[~support] = np.nan
                if key == 'ml':
                    cached_ml = lower, upper, bootstrap_reference_fraction
            axes[1].plot(centers, p, lw=2, color=COLORS[key],
                         label=f'{LABELS[key]} ({three_significant(a[-1,0])} ps)')
            axes[1].fill_between(centers, lower, upper, color=COLORS[key], alpha=.14)
            finite_values = np.concatenate((p,lower,upper))
            finite_values = finite_values[np.isfinite(finite_values)]
            free_energy_low = min(free_energy_low,float(finite_values.min()))
            free_energy_high = max(free_energy_high,float(finite_values.max()))
            minimum_indices, _ = find_peaks(-np.nan_to_num(p,nan=50),prominence=.3,distance=16)
            labeled_minima = []
            for i in minimum_indices:
                coordinate = centers[i]
                basin = int(np.digitize(abs(coordinate),[1.3,3.5]))
                letter = 'ABC'[basin]
                dx, dy = {'ml':(0,-.45),'electrostatic':(-.14,-.90),
                          'mechanical':(.13,-1.35),'openmm':(-.10,-1.80)}[key]
                axes[1].scatter([coordinate],[p[i]],s=17,c=COLORS[key],zorder=5)
                axes[1].annotate(letter,xy=(coordinate,p[i]),xytext=(coordinate+dx,p[i]+dy),
                                 fontsize=9,weight='bold',color=COLORS[key],ha='center',
                                 arrowprops={'arrowstyle':'-','color':COLORS[key],'lw':.55},
                                 bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.3},zorder=6)
                labeled_minima.append({'label':letter,'delta_angstrom':float(coordinate),'relative_free_energy_kBT':float(p[i])})
                free_energy_low = min(free_energy_low,float(p[i]+dy-.2))
            summary['rows'][row_title][key] = {
                'label': LABELS[key], 'frames': len(q), 'last_time_ps': float(a[-1,0]),
                'fraction_abs_delta_above_1p3': float(np.mean(np.abs(q)>1.3)),
                'fraction_abs_delta_above_1p3_first_half': float(np.mean(np.abs(q[:len(q)//2])>1.3)),
                'fraction_abs_delta_above_1p3_second_half': float(np.mean(np.abs(q[len(q)//2:])>1.3)),
                'timing': timings[key],
                'A_minimum_relative_free_energy_kBT': float(np.nanmin(p[np.abs(centers)<1.3])),
                'bootstrap_fraction_with_observations_at_A': bootstrap_reference_fraction,
                'labeled_minima': labeled_minima,
                'profile_kBT': [float(v) if np.isfinite(v) else None for v in p],
                'lower_68_percent_kBT': [float(v) if np.isfinite(v) else None for v in lower],
                'upper_68_percent_kBT': [float(v) if np.isfinite(v) else None for v in upper],
            }
        axes[1].set(xlim=(-extent, extent),
                    xlabel=r'$\delta=r_R-r_L$ (Å)',
                    ylabel=r'$\beta[F(\delta)-F_A]$')
        axes[1].set_title(row_title, loc='left', fontsize=12, weight='bold',pad=14)
        axes[1].legend(frameon=False, fontsize=8.2, loc='upper center', ncol=2)
        axes[1].axvline(0, color='.6', lw=.8, ls='--')
        axes[1].axhline(0, color='.6', lw=.8, ls='--')
        rates = np.array([timings[k]['ps_per_day'] for k in LABELS]) / 1000
        lo = np.array([timings[k]['chunk_10_90_percent_ps_per_day'][0] for k in LABELS]) / 1000
        hi = np.array([timings[k]['chunk_10_90_percent_ps_per_day'][1] for k in LABELS]) / 1000
        bars = axes[2].bar(np.arange(4), rates, color=list(COLORS.values()),
                           yerr=[np.maximum(rates-lo,0), np.maximum(hi-rates,0)], capsize=3)
        axes[2].set_xticks(np.arange(4), ['ML', 'ML/MM\nelectrostatic', 'ML/MM\nmechanical', 'ML/MM mechanical\n(OpenMM)'], fontsize=8)
        axes[2].set(ylabel='Throughput (ns/day)', ylim=(0, all_high*1.17))
        axes[2].set_title('GPU throughput', loc='left', fontsize=12, weight='bold')
        for bar, rate in zip(bars, rates):
            axes[2].text(bar.get_x()+bar.get_width()/2, rate+.025*all_high,
                         three_significant(rate), ha='center', fontsize=9, weight='bold')
        for axis in axes[1:]:
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
            axis.spines[['top','right']].set_visible(False)

    y_limits = (float(np.floor(free_energy_low-.15)), float(np.ceil(free_energy_high)+1.5))
    for axis in free_energy_axes:
        axis.set_ylim(y_limits)
    summary['free_energy_y_range_kBT'] = list(y_limits)
    fig.suptitle('Hydrogen maleate · 330 K · all saved frames', fontsize=17, weight='bold', y=.975)
    fig.text(.5, .936, ('Full ML: 500 ps; each ML/MM method: 100 ps; molecule + 25 ML waters and 27 MM waters.'
                       if args.fires_data and not args.molecule_data else
                       'Full ML: 500 ps; molecule-only ML/MM: 100 ps each; no restraint.'
                       if args.new_only and not args.fires_data else
                       'Same coordinate range, labels, colors, and axis scales in both rows; the same full ML reference is repeated.'),
             ha='center', fontsize=11)
    fig.text(.5, .025,
             '$\\delta=r_R-r_L$; $r_L$ and $r_R$: distances from the purple proton to the nearest oxygen at the left and right ends of the molecule.\n'
             '$F$: free energy; $F_A$: its value at the A minimum of each curve; $\\beta=1/(k_B T)$; $k_B$: Boltzmann constant; $T$: temperature.\n'
             'Results at +δ and −δ are averaged. A, B and C label paired minima from the centre outward; drawings show example structures from ML.\n'
             'ML: MACE-POLAR; MM: classical water force field. Shading covers the middle 68% of estimates made by repeatedly selecting 1 ps pieces of each simulation.\n'
             'Gaps mean too few observations. Bars show simulated nanoseconds per day; error bars cover the middle 80% of measured speeds. Setup time is excluded.\n'
             'Longer simulations may change these free energy curves; the shaded ranges do not include all uncertainty from slow changes of molecular shape.'
             + ('\nAll three FIRES ML/MM runs use cuEquivariance; the retained full ML reference does not. JAX FIRES runs shared a GPU; OpenMM used another GPU.'
                if args.fires_data else
                '\nOnly the new JAX ML/MM runs use cuEquivariance; the full ML and OpenMM reference runs use their earlier implementations.'
                if args.matched_data else ''),
             ha='center', va='bottom', fontsize=9, color='#465159')
    filename = ('molecule-only-vs-fires-25w-fes-throughput.png' if args.molecule_data else
                'fires-25w-fes-throughput.png') if args.fires_data else (
                'matched-molecule-only-fes-throughput.png' if args.new_only else
                'earlier-vs-matched-molecule-only-fes-throughput.png') if args.matched_data else 'full-trajectories-labeled-minima-conformers.png'
    destination = args.output_dir / filename
    fig.savefig(destination, facecolor='white')
    fig.savefig(destination.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    summary['coordinate_centers_angstrom'] = centers.tolist()
    destination.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    for title, row in summary['rows'].items():
        print(title)
        for key, result in row.items():
            print(key, result['last_time_ps'], result['fraction_abs_delta_above_1p3'], result['timing']['ps_per_day'])


if __name__ == '__main__':
    main()
