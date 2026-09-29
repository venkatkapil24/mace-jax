#!/usr/bin/env python3
"""Run the six enzyme-scale QM/MM memory cases in isolated processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


SYSTEMS = (
    ('ndpk', 32_339, 130, 180),
    ('chorismate-mutase', 46_323, 43, 72),
    ('petase-acylation', 53_816, 104, 146),
)
MM_LEVELS = (1_000, 5_000, 10_000, 25_000)


def write_summary(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--peak-limit-gib', type=float, default=40.0)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()

    benchmark = Path(__file__).with_name('qmmm_memory_sweep.py')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'matrix-summary.json'
    records: list[dict] = []

    for system, total_atoms, small_qm, large_qm in SYSTEMS:
        cases = [
            (f'{system}-small-{mm_atoms}mm', small_qm, mm_atoms)
            for mm_atoms in MM_LEVELS
            if small_qm + mm_atoms <= total_atoms
        ]
        cases.extend(
            (
                (f'{system}-small-full', small_qm, total_atoms - small_qm),
                (f'{system}-large-full', large_qm, total_atoms - large_qm),
            )
        )
        system_points: list[tuple[int, float]] = []
        system_blocked = False
        for case_name, qm_atoms, mm_atoms in cases:
            record = {
                'case': case_name,
                'system': system,
                'qm_atoms': qm_atoms,
                'mm_atoms': mm_atoms,
                'full_system_atoms': total_atoms,
            }
            if system_blocked:
                record['status'] = 'skipped_after_memory_gate'
                records.append(record)
                write_summary(summary_path, records)
                continue

            if len(system_points) >= 2:
                (n0, m0), (n1, m1) = system_points[-2:]
                slope = (m1 - m0) / (n1 - n0)
                projected = m1 + max(slope, 0.0) * (mm_atoms - n1)
                record['projected_peak_gib'] = projected
                if projected >= args.peak_limit_gib:
                    record['status'] = 'skipped_projected_over_limit'
                    records.append(record)
                    write_summary(summary_path, records)
                    system_blocked = True
                    continue

            output = args.output_dir / f'{case_name}.json'
            log = args.output_dir / f'{case_name}.log'
            command = [
                sys.executable,
                '-u',
                str(benchmark),
                '--bundle',
                str(args.bundle),
                '--system',
                case_name,
                '--qm-atoms',
                str(qm_atoms),
                '--mm-atoms',
                str(mm_atoms),
                '--full-system-atoms',
                str(total_atoms),
                '--repeats',
                str(args.repeats),
                '--output',
                str(output),
            ]
            record['status'] = 'running'
            record['started_unix'] = time.time()
            records.append(record)
            write_summary(summary_path, records)
            with log.open('w') as stream:
                completed = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            record['finished_unix'] = time.time()
            record['returncode'] = completed.returncode
            if completed.returncode != 0 or not output.exists():
                record['status'] = 'failed'
                record['log'] = str(log)
                system_blocked = True
                write_summary(summary_path, records)
                continue

            measurement = json.loads(output.read_text())
            peak_gib = measurement['memory_after_evaluation'][
                'peak_bytes_in_use_gib'
            ]
            record['status'] = 'completed'
            record['peak_gib'] = peak_gib
            record['compile_seconds'] = measurement['compile_seconds']
            record['median_evaluation_seconds'] = measurement[
                'median_evaluation_seconds'
            ]
            record['mesh_shape'] = measurement['mesh_shape']
            system_points.append((mm_atoms, peak_gib))
            if peak_gib >= args.peak_limit_gib:
                system_blocked = True
            write_summary(summary_path, records)

    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
