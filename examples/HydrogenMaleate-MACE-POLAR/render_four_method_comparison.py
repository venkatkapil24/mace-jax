#!/usr/bin/env python3
"""Render synchronized top-view 100 ps trajectories in a four-panel MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

from render_comparison import (
    draw_panel,
    load_frames,
    solute_bonds,
    top_view_reference,
)


def main() -> None:
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-ps', type=float, default=100.0)
    parser.add_argument('--stride-fs', type=int, default=200)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()

    all_times = list(
        range(args.stride_fs, round(args.duration_ps * 1000) + 1, args.stride_fs)
    )
    times = all_times[:1] if args.preview else all_times
    requested = set(times)
    sources = {
        'ML': args.examples / 'HydrogenMaleate-MACE-POLAR/md-500ps',
        'ML/MM electrostatic': (
            args.examples / 'HydrogenMaleate-PME/electrostatic-500ps'
        ),
        'ML/MM mechanical': (
            args.examples / 'HydrogenMaleate-PME/mechanical-100ps'
        ),
        'ML/MM mechanical (OpenMM)': (
            args.examples / 'HydrogenMaleate-OpenMM-Mechanical/video-rerun-100ps'
        ),
    }
    trajectories = {
        label: load_frames(directory, requested)
        for label, directory in sources.items()
    }

    initial = args.examples / 'HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz'
    box = float(read(initial).cell[0, 0])
    bonds = solute_bonds(initial, box)
    reference_heavy = top_view_reference(initial, box)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 11.5), dpi=100)
    figure.subplots_adjust(
        left=0.035, right=0.985, bottom=0.035, top=0.90, wspace=0.06, hspace=0.13
    )
    flat_axes = axes.ravel()

    def render(time_fs: int) -> np.ndarray:
        for axis, (label, trajectory) in zip(
            flat_axes, trajectories.items(), strict=True
        ):
            draw_panel(
                axis,
                trajectory[time_fs],
                box,
                bonds,
                reference_heavy,
                mlmm=label != 'ML',
                time_ps=time_fs / 1000,
            )
            axis.set_title(label, fontsize=15, pad=9)
        figure.suptitle(
            'Hydrogen maleate at 330 K · aligned top view',
            fontsize=17,
            weight='bold',
            y=0.975,
        )
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        imageio.imwrite(args.output.with_suffix('.preview.png'), render(times[0]))
        return

    with imageio.get_writer(
        args.output,
        fps=args.fps,
        codec='libx264',
        quality=8,
        macro_block_size=8,
    ) as writer:
        for index, time_fs in enumerate(times, 1):
            writer.append_data(render(time_fs))
            if index % 25 == 0 or index == len(times):
                print(
                    f'Rendered {index}/{len(times)} frames '
                    f'({time_fs / 1000:.1f} ps)',
                    flush=True,
                )
    plt.close(figure)
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
