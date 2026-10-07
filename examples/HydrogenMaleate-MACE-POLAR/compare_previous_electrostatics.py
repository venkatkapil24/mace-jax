#!/usr/bin/env python3
"""Compare previous point-charge electrostatics with converged direct Ewald.

No MD is run. All methods use the same charges, configurations and Coulomb
constant to isolate numerical electrostatic errors. Gaussian ML electrostatics
is a separate comparison and is not included in the point-charge errors here.
"""

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ase.io import read
from jax.scipy.special import erf, erfc
from simulation_settings import TIP3P_Q, resolve_force_settings

from mace_jax.modules.point_pme import PointPME
from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.modules.polar_periodic import PeriodicPolarElectrostatics
from mace_jax.tools.physical_constants import OPENMM_COULOMB

jax.config.update('jax_enable_x64', True)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--verify-model', action='store_true')
    args = parser.parse_args()
    example = root / 'Examples/HydrogenMaleate-MoleculeOnly'
    frames = {
        'initial': read(example / 'structure/initial-100ps-full-ml.xyz'),
        'openmm_final': read(example / 'openmm-mechanical-100ps/final.xyz'),
        'jax_final': read(example / 'mechanical-100ps/final.xyz'),
    }
    atoms = frames['initial']
    box = float(atoms.cell[0, 0])
    cell = jnp.eye(3) * box
    q_qm = np.asarray(
        json.loads((example / 'structure/molecule-fixed-charges.json').read_text())[
            'charges'
        ]
    )
    q = jnp.asarray(np.concatenate((q_qm, np.tile(TIP3P_Q, 52))))
    n = len(q)
    nq = len(q_qm)
    i, j = np.triu_indices(n, 1)
    allowed = (j >= nq) & ~((i >= nq) & ((i - nq) // 3 == (j - nq) // 3))
    i, j = jnp.asarray(i[allowed]), jnp.asarray(j[allowed])
    ex_i = jnp.asarray([nq + 3 * w + a for w in range(52) for a in (0, 0, 1)])
    ex_j = jnp.asarray([nq + 3 * w + b for w in range(52) for b in (1, 2, 2)])
    shifts = jnp.asarray(
        np.stack(
            np.meshgrid(*([np.arange(-1, 2)] * 3), indexing='ij'), axis=-1
        ).reshape(-1, 3)
        * box
    )

    def corrections(r, alpha):
        d = r[ex_i] - r[ex_j]
        d -= box * jnp.round(d / box)
        distance = jnp.linalg.norm(d, axis=-1)
        exceptions = -OPENMM_COULOMB * jnp.sum(
            q[ex_i] * q[ex_j] * erf(alpha * distance) / distance
        )
        self_term = -OPENMM_COULOMB * alpha / math.sqrt(math.pi) * jnp.sum(q[nq:] ** 2)
        background = (
            -OPENMM_COULOMB
            * math.pi
            * (jnp.sum(q) ** 2 - jnp.sum(q[:nq]) ** 2)
            / (2 * alpha**2 * box**3)
        )
        return exceptions + self_term + background

    def real(r, alpha, cutoff, images=False):
        d = r[i] - r[j]
        d -= box * jnp.round(d / box)
        if images:
            d = d[:, None, :] + shifts[None, :, :]
        distance = jnp.linalg.norm(d, axis=-1)
        charge = q[i] * q[j]
        if images:
            charge = charge[:, None]
        return OPENMM_COULOMB * jnp.sum(
            jnp.where(
                distance < cutoff, charge * erfc(alpha * distance) / distance, 0.0
            )
        )

    model = PeriodicPolarElectrostatics(
        0, 1.5, 0, (1.5,), kspace_cutoff=1.8467166423797607
    )

    def generalized(alpha, cutoff, spacing):
        mesh, assignment = model.prepare_mesh_template(
            cell[None], mesh_spacing=spacing, assignment_order=8
        )

        def energy(r):
            cache = model.precompute_mesh(
                r[:nq], jnp.zeros(nq, dtype=jnp.int32), cell[None], mesh, assignment
            )
            qm = model.mesh_point_charge_density(r[:nq], q[:nq], cache)
            mm = model.mesh_point_charge_density(r[nq:], q[nq:], cache)
            reciprocal = model.mesh_point_charge_ewald_reciprocal_energy(
                mm, cache, alpha
            ) + model.mesh_point_charge_ewald_cross_energy(qm, mm, cache, alpha)
            reciprocal *= OPENMM_COULOMB / (FIELD_CONSTANT / (4 * math.pi))
            return real(r, alpha, cutoff) + corrections(r, alpha) + reciprocal

        return energy, mesh.shape

    old_jax, old_shape = generalized(0.5, 0.49 * box, 0.5)
    settings = resolve_force_settings('openmm', box)
    pme = PointPME(settings.alpha, settings.grid_shape)

    def old_openmm(r):
        mm, cross = pme.reciprocal_parts(r[:nq], q[:nq], r[nq:], q[nq:], cell)
        # All QQ pairs in these three frames are shorter than the cutoff;
        # verify this explicitly below so the OpenMM exception tail is zero.
        return (
            real(r, settings.alpha, settings.cutoff)
            + corrections(r, settings.alpha)
            + mm
            + cross
        )

    def direct(alpha, kmax, cutoff=8.0, images=True):
        ints = np.stack(
            np.meshgrid(*([np.arange(-kmax, kmax + 1)] * 3), indexing='ij'), axis=-1
        ).reshape(-1, 3)
        ints = ints[np.any(ints != 0, axis=1)]
        k = jnp.asarray(ints * 2 * np.pi / box)
        k2 = jnp.sum(k * k, axis=-1)
        weight = jnp.exp(-k2 / (4 * alpha**2)) / k2

        def energy(r):
            phase = k @ r.T
            c = jnp.cos(phase)
            s = jnp.sin(phase)
            cm, sm = c[:, nq:] @ q[nq:], s[:, nq:] @ q[nq:]
            cq, sq = c[:, :nq] @ q[:nq], s[:, :nq] @ q[:nq]
            reciprocal = (
                OPENMM_COULOMB
                * 2
                * np.pi
                / box**3
                * jnp.sum(weight * (cm * cm + sm * sm + 2 * (cm * cq + sm * sq)))
            )
            return (
                real(r, alpha, cutoff, images=images)
                + corrections(r, alpha)
                + reciprocal
            )

        return energy

    methods = [
        ('previous_jax', old_jax),
        ('previous_openmm_reference', old_openmm),
        ('direct_ewald', direct(0.8, 16)),
        ('direct_convergence_check', direct(0.7, 14)),
    ]
    candidates = {}
    if args.tune:
        accurate_alpha = math.sqrt(-math.log(2e-6)) / settings.cutoff
        for name, alpha, cutoff, spacing in [
            ('generalized_matched_24', settings.alpha, settings.cutoff, 0.5),
            ('generalized_matched_30', settings.alpha, settings.cutoff, 0.4),
            ('generalized_accurate_30', accurate_alpha, settings.cutoff, 0.4),
            ('generalized_accurate_40', accurate_alpha, settings.cutoff, 0.3),
        ]:
            fn, shape = generalized(alpha, cutoff, spacing)
            methods.append((name, fn))
            candidates[name] = {
                'alpha': alpha,
                'cutoff_angstrom': cutoff,
                'mesh': list(shape),
                'order': 8,
            }
        methods.extend(
            [
                ('legacy_without_mesh_error', direct(0.5, 12, 0.49 * box, False)),
                (
                    'matched_without_mesh_error',
                    direct(settings.alpha, 12, settings.cutoff, False),
                ),
                (
                    'accurate_without_mesh_error',
                    direct(accurate_alpha, 12, settings.cutoff, False),
                ),
            ]
        )
    evaluators = {name: jax.jit(jax.value_and_grad(fn)) for name, fn in methods}
    full_evaluators = {}
    if args.verify_model:
        from md import build_energy  # noqa: PLC0415 - optional model verification

        bundle = root / 'artifacts/MACE-POLAR-1-M-jax-fixed.msgpack'
        for name, profile in [
            ('generalized-accurate', 'generalized-accurate'),
            ('openmm', 'openmm'),
        ]:
            fn, params, neighbors, _, _, terms = build_energy(
                atoms,
                bundle,
                'mechanical',
                None,
                6,
                0,
                'none',
                4.2,
                0.2,
                q_qm,
                compatibility_profile=profile,
                return_terms=True,
            )

            def ml_energy(pos, nbrs, model_params, term_fn=terms):
                return term_fn(pos, nbrs, model_params)['ml']

            full_evaluators[name] = (
                fn,
                params,
                neighbors,
                jax.jit(jax.value_and_grad(ml_energy)),
            )
    report = {
        'definition': 'MM-MM plus fixed-charge ML-MM electrostatic forces, excluding Gaussian ML-ML forces',
        'normalization': 'All methods use identical OpenMM Coulomb constant; double precision',
        'full_model_reference': 'JAX direct Gaussian ML (previously checked against Torch/OpenMM), identical bonded and LJ terms, converged direct point-charge Ewald'
        if args.verify_model
        else None,
        'previous_jax': {
            'mesh': list(old_shape),
            'order': 8,
            'alpha': 0.5,
            'cutoff_angstrom': 0.49 * box,
        },
        'previous_openmm_reference': settings.to_dict(),
        'candidates': candidates,
        'configurations': {},
    }
    for name, atoms in frames.items():
        r = jnp.asarray(atoms.positions)
        qi, qj = np.triu_indices(nq, 1)
        d = atoms.positions[qi] - atoms.positions[qj]
        d -= box * np.rint(d / box)
        assert np.linalg.norm(d, axis=-1).max() < settings.cutoff
        values = {method: fn(r) for method, fn in evaluators.items()}
        e, g = values['direct_ewald']
        row = {}
        for method, (em, gm) in values.items():
            difference = np.asarray(gm - g)
            row[method] = {
                'energy_error_meV': float(1000 * (em - e)),
                'force_component_rms_meV_per_angstrom': float(
                    1000 * np.sqrt(np.mean(difference**2))
                ),
                'max_atom_force_error_meV_per_angstrom': float(
                    1000 * np.linalg.norm(difference, axis=1).max()
                ),
            }
        if args.verify_model:
            full_values = {}
            for profile, (fn, params, nbr_fn, ml_fn) in full_evaluators.items():
                nbr = nbr_fn.allocate(r)
                if nbr.did_buffer_overflow:
                    raise RuntimeError('Verification neighbor overflow')
                total, total_grad = fn(r, nbr, params)
                ml, ml_grad = ml_fn(r, nbr, params)
                full_values[profile] = (total, total_grad, ml, ml_grad)
            om_e, om_g, om_ml, om_ml_g = full_values['openmm']
            acc_e, acc_g, acc_ml, acc_ml_g = full_values['generalized-accurate']
            # Keep the exact same short-range and Gaussian-model definition;
            # replace only the point-charge numerical electrostatics by Ewald.
            om_point_e, om_point_g = values['previous_openmm_reference']
            total_reference = om_e - om_point_e + e
            gradient_reference = om_g - om_point_g + g

            def errors(total, gradient):
                difference = np.asarray(gradient - gradient_reference)
                return {
                    'energy_error_meV': float(1000 * (total - total_reference)),
                    'force_component_rms_meV_per_angstrom': float(
                        1000 * np.sqrt(np.mean(difference**2))
                    ),
                    'max_atom_force_error_meV_per_angstrom': float(
                        1000 * np.linalg.norm(difference, axis=1).max()
                    ),
                }

            ml_difference = np.asarray(acc_ml_g - om_ml_g)
            row['complete_generalized_accurate'] = errors(acc_e, acc_g)
            row['complete_openmm_settings'] = errors(om_e, om_g)
            row['gaussian_ml_mesh_vs_direct'] = {
                'energy_difference_meV': float(1000 * (acc_ml - om_ml)),
                'force_component_rms_meV_per_angstrom': float(
                    1000 * np.sqrt(np.mean(ml_difference**2))
                ),
                'max_atom_force_difference_meV_per_angstrom': float(
                    1000 * np.linalg.norm(ml_difference, axis=1).max()
                ),
            }
            assert (
                row['complete_generalized_accurate'][
                    'force_component_rms_meV_per_angstrom'
                ]
                < row['complete_openmm_settings'][
                    'force_component_rms_meV_per_angstrom'
                ]
            )
        report['configurations'][name] = row
        print(name, json.dumps(row), flush=True)
    target = (
        root
        / 'Examples/HydrogenMaleate-Analysis'
        / (
            'generalized-accurate-model-verification.json'
            if args.verify_model
            else 'generalized-electrostatics-tuning.json'
            if args.tune
            else 'previous-electrostatics-accuracy.json'
        )
    )
    target.write_text(json.dumps(report, indent=2) + '\n')
    assert (
        max(
            row['direct_convergence_check']['max_atom_force_error_meV_per_angstrom']
            for row in report['configurations'].values()
        )
        < 1e-5
    )


if __name__ == '__main__':
    main()
