#!/usr/bin/env python3
"""Render a top-view side-by-side hydrogen-maleate MP4 (requires imageio-ffmpeg)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.collections import LineCollection


TIME = re.compile(r'\btime_fs=([-+\d.eE]+)')
QM_WATERS = 13
WATERS = 52
SOLUTE_ATOMS = 11


def minimum_image(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def load_frames(directory: Path, times_fs: set[int]) -> dict[int, np.ndarray]:
    selected = {}
    for chunk in sorted((directory / 'chunks').glob('*.xyz')):
        with chunk.open() as handle:
            while atom_count := handle.readline():
                n = int(atom_count)
                header = handle.readline()
                time_match = TIME.search(header)
                if time_match is None:
                    raise ValueError(f'Missing time in {chunk}')
                time_fs = round(float(time_match.group(1)))
                if time_fs in times_fs:
                    if n != SOLUTE_ATOMS + 3 * WATERS:
                        raise ValueError(f'Wrong atom count in {chunk}')
                    xyz = np.empty((n, 3), dtype=float)
                    for i in range(n):
                        xyz[i] = [float(x) for x in handle.readline().split()[1:4]]
                    selected[time_fs] = xyz
                else:
                    for _ in range(n):
                        handle.readline()
        if len(selected) == len(times_fs):
            break
    missing = times_fs - selected.keys()
    if missing:
        raise ValueError(f'{directory}: missing {len(missing)} requested frames')
    return selected


def centered_positions(raw: np.ndarray, box: float) -> np.ndarray:
    result = np.empty_like(raw)
    anchor = raw[5]
    solute = anchor + minimum_image(raw[:SOLUTE_ATOMS] - anchor, box)
    center = solute[:8].mean(axis=0)
    result[:SOLUTE_ATOMS] = solute - center
    for water in range(WATERS):
        start = SOLUTE_ATOMS + 3 * water
        oxygen = raw[start]
        wrapped_oxygen = center + minimum_image(oxygen - center, box)
        result[start] = wrapped_oxygen - center
        result[start + 1:start + 3] = (
            wrapped_oxygen
            + minimum_image(raw[start + 1:start + 3] - oxygen, box)
            - center
        )
    return result


def top_view_reference(initial: Path, box: float) -> np.ndarray:
    heavy = centered_positions(np.asarray(read(initial).positions), box)[:8]
    x_axis = heavy[2:4].mean(axis=0) - heavy[:2].mean(axis=0)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = heavy[0] - heavy[1]
    y_axis -= np.dot(y_axis, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return heavy @ basis


def align_top_view(positions: np.ndarray, reference_heavy: np.ndarray) -> np.ndarray:
    moving = positions[:8] - positions[:8].mean(axis=0)
    target = reference_heavy - reference_heavy.mean(axis=0)
    u, _, vt = np.linalg.svd(moving.T @ target)
    reflection = np.eye(3)
    reflection[2, 2] = np.linalg.det(u @ vt)
    return positions @ (u @ reflection @ vt)


def solute_bonds(initial: Path, box: float) -> list[tuple[int, int]]:
    atoms = read(initial)
    positions = np.asarray(atoms.positions)
    pairs = []
    for i in range(10):
        for j in range(i + 1, 10):
            if atoms.numbers[i] == 1 and atoms.numbers[j] == 1:
                continue
            distance = np.linalg.norm(minimum_image(positions[i] - positions[j], box))
            if distance < (1.25 if 1 in (atoms.numbers[i], atoms.numbers[j]) else 1.8):
                pairs.append((i, j))
    return pairs


def draw_halo(ax, positions: np.ndarray, unrotated: np.ndarray, box: float) -> None:
    # Orthographic projection of the 3D zero-force region. Actual restraint
    # forces below are based on the full 3D minimum-image O–O distance.
    grid = np.linspace(-box / 2, box / 2, 140)
    x, y = np.meshgrid(grid, grid)
    distance = np.full_like(x, np.inf)
    for oxygen in positions[:4]:
        dx = x - oxygen[0]
        dy = y - oxygen[1]
        distance = np.minimum(distance, np.hypot(dx, dy))
    ax.contourf(x, y, distance, levels=[0, 4.2], colors=['#ffe056'], alpha=0.14)
    ax.contour(x, y, distance, levels=[4.2], colors=['#eab800'], linewidths=1.4)
    first_shell_o = positions[SOLUTE_ATOMS:SOLUTE_ATOMS + 3 * QM_WATERS:3]
    original_o = unrotated[SOLUTE_ATOMS:SOLUTE_ATOMS + 3 * QM_WATERS:3]
    delta = minimum_image(original_o[:, None] - unrotated[None, :4], box)
    distance_3d = np.linalg.norm(delta, axis=-1).min(axis=-1)
    active = first_shell_o[distance_3d > 4.2]
    if len(active):
        ax.scatter(active[:, 0], active[:, 1], s=155, facecolors='none',
                   edgecolors='#d8a300', linewidths=2.2, zorder=7)


def draw_panel(
    ax, raw: np.ndarray, box: float, bonds: list[tuple[int, int]],
    reference_heavy: np.ndarray,
    *, mlmm: bool, time_ps: float,
) -> None:
    ax.clear()
    ax.set_facecolor('white')
    unrotated = centered_positions(raw, box)
    xyz = align_top_view(unrotated, reference_heavy)
    ax.set_xlim(-box / 2, box / 2)
    ax.set_ylim(-box / 2, box / 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#b9c0c6')
        spine.set_linewidth(1)

    if mlmm:
        draw_halo(ax, xyz, unrotated, box)

    water_o = xyz[SOLUTE_ATOMS::3]
    water_segments = [
        np.stack((xyz[SOLUTE_ATOMS + 3 * i, :2], xyz[SOLUTE_ATOMS + 3 * i + h, :2]))
        for i in range(WATERS) for h in (1, 2)
    ]
    first_segments = water_segments[:2 * QM_WATERS]
    mm_segments = water_segments[2 * QM_WATERS:]
    if mlmm:
        ax.add_collection(LineCollection(mm_segments, colors='#b9c0c6',
                                         linewidths=0.7, alpha=0.45, zorder=1))
        mm_o = water_o[QM_WATERS:]
        mm_h = np.concatenate([
            xyz[SOLUTE_ATOMS + 3 * i + 1:SOLUTE_ATOMS + 3 * i + 3, :2]
            for i in range(QM_WATERS, WATERS)
        ])
        ax.scatter(mm_h[:, 0], mm_h[:, 1], s=11, c='#d8dcdf', alpha=0.5, zorder=2)
        ax.scatter(mm_o[:, 0], mm_o[:, 1], s=33, c='#aab3ba', alpha=0.65, zorder=3)
    ax.add_collection(LineCollection(first_segments if mlmm else water_segments,
                                     colors='#79b8cb', linewidths=0.9,
                                     alpha=0.75, zorder=2))
    ml_count = QM_WATERS if mlmm else WATERS
    ml_o = water_o[:ml_count]
    ml_h = np.concatenate([
        xyz[SOLUTE_ATOMS + 3 * i + 1:SOLUTE_ATOMS + 3 * i + 3, :2]
        for i in range(ml_count)
    ])
    ax.scatter(ml_h[:, 0], ml_h[:, 1], s=14, c='#c9ecf3',
               edgecolors='#91c6d4', linewidths=0.25, alpha=0.9, zorder=3)
    ax.scatter(ml_o[:, 0], ml_o[:, 1], s=52, c='#238eaf',
               edgecolors='white', linewidths=0.4, zorder=4)

    solute_segments = [xyz[[i, j], :2] for i, j in bonds]
    proton_o = np.argmin(np.linalg.norm(xyz[:4] - xyz[10], axis=1))
    solute_segments.append(xyz[[proton_o, 10], :2])
    ax.add_collection(LineCollection(solute_segments, colors='#414a50',
                                     linewidths=2.3, zorder=5))
    ax.scatter(xyz[4:8, 0], xyz[4:8, 1], s=115, c='#424b50',
               edgecolors='white', linewidths=0.5, zorder=6)
    ax.scatter(xyz[:4, 0], xyz[:4, 1], s=120, c='#db5548',
               edgecolors='white', linewidths=0.5, zorder=6)
    ax.scatter(xyz[8:10, 0], xyz[8:10, 1], s=41, c='white',
               edgecolors='#4d5559', linewidths=0.8, zorder=6)
    ax.scatter(xyz[10, 0], xyz[10, 1], s=58, c='#ad57bb',
               edgecolors='white', linewidths=0.7, zorder=7)

    title = 'Full MACE-POLAR' if not mlmm else 'First-shell ML/MM'
    ax.set_title(title, fontsize=16, pad=11)
    ax.text(0.02, 0.025, f'{time_ps:.2f} ps', transform=ax.transAxes,
            fontsize=11, color='#333b40',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})
    if mlmm:
        ax.text(0.98, 0.025, 'yellow: 4.2 Å projected restraint boundary',
                transform=ax.transAxes, ha='right', fontsize=8.5, color='#7b6300',
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8})


def main() -> None:
    import imageio.v2 as imageio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration-ps', type=float, default=8.0)
    parser.add_argument('--stride-fs', type=int, default=50)
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    times = list(range(args.stride_fs, round(args.duration_ps * 1000) + 1,
                       args.stride_fs))
    initial = args.examples / 'HydrogenMaleate-MACE-POLAR/structure/md-initial.xyz'
    box = float(read(initial).cell[0, 0])
    bonds = solute_bonds(initial, box)
    reference_heavy = top_view_reference(initial, box)
    polar = load_frames(args.examples / 'HydrogenMaleate-MACE-POLAR/md-100ps', set(times))
    mlmm = load_frames(args.examples / 'HydrogenMaleate-MLMM/md-100ps', set(times))
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.045, top=0.91, wspace=0.06)

    def render(time_fs: int) -> np.ndarray:
        draw_panel(axes[0], polar[time_fs], box, bonds, reference_heavy,
                   mlmm=False,
                   time_ps=time_fs / 1000)
        draw_panel(axes[1], mlmm[time_fs], box, bonds, reference_heavy,
                   mlmm=True,
                   time_ps=time_fs / 1000)
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()

    if args.preview:
        imageio.imwrite(args.output / 'preview.png', render(times[0]))
        return

    pair_path = args.output / 'side-by-side.mp4'
    with imageio.get_writer(pair_path, fps=args.fps, codec='libx264', quality=8,
                            macro_block_size=8) as pair:
        for i, time_fs in enumerate(times, 1):
            frame = render(time_fs)
            pair.append_data(frame)
            if i % 20 == 0 or i == len(times):
                print(f'Rendered {i}/{len(times)} frames ({time_fs / 1000:.2f} ps)', flush=True)
    plt.close(figure)
    print(f'Wrote {pair_path}', flush=True)


if __name__ == '__main__':
    main()
