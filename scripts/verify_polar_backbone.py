#!/usr/bin/env python3
"""Compare eager native JAX POLAR outputs with released POLAR-1-M Torch."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import jax
import numpy as np
import torch
from ase.build import molecule
from ase.collections import s22
from ase.io import read
from flax import nnx
from graph_longrange.kspace import compute_k_vectors_flat
from mace.calculators.mace import MACECalculator
from mace.tools.scripts_utils import extract_config_mace_model

from mace_jax.modules.models import ScaleShiftMACE
from mace_jax.tools.bundle import load_model_bundle
from mace_jax.tools.import_from_torch import import_from_torch
from mace_jax.tools.model_builder import _build_jax_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    structure = parser.add_mutually_exclusive_group()
    structure.add_argument('--structure', type=Path, help='ASE-readable geometry')
    structure.add_argument('--s22', choices=s22.names, help='ASE S22 dimer geometry')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--compute-force', action='store_true')
    parser.add_argument('--compute-stress', action='store_true')
    parser.add_argument('--charge', type=float, default=0.0)
    parser.add_argument('--spin', type=float, default=1.0)
    parser.add_argument('--field', nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument('--fermi-level', type=float, default=0.0)
    parser.add_argument('--jax-bundle', type=Path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--pbc-handling',
        choices=(
            'realspace',
            'pbc',
            'slab',
            'molecule_in_box',
            'mixed_periodic',
            'auto',
        ),
        default='realspace',
    )
    parser.add_argument(
        '--torch-backbone',
        action='store_true',
        help='Diagnose the POLAR head using identical Torch backbone features',
    )
    args = parser.parse_args()

    jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch_model = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if torch_model.__class__.__name__ != 'PolarMACE':
        parser.error('Expected a PolarMACE checkpoint')
    torch_model = torch_model.eval().double()
    torch_model.set_electrostatic_pbcs(args.pbc_handling)

    config = extract_config_mace_model(torch_model)
    config['kspace_cutoff'] = float(torch_model.kspace_cutoff)
    if args.jax_bundle is None:
        jax_model = _build_jax_model(config, rngs=nnx.Rngs(0))
        _, state = nnx.split(jax_model)
        import_from_torch(jax_model, torch_model, state)
    else:
        bundle = load_model_bundle(str(args.jax_bundle), 'float64')
        jax_model = nnx.merge(bundle.graphdef, bundle.params)
    jax_model.pbc_handling = args.pbc_handling

    if args.structure is not None:
        atoms = read(args.structure)
    elif args.s22 is not None:
        atoms = s22[args.s22]
    else:
        atoms = molecule('H2O')
    if args.structure is None and args.s22 is None and args.pbc_handling != 'realspace':
        atoms.cell = [8.0, 8.0, 8.0]
        if args.pbc_handling in ('slab', 'mixed_periodic'):
            atoms.pbc = [True, True, False]
        elif args.pbc_handling == 'molecule_in_box':
            atoms.pbc = [False, False, False]
        else:
            atoms.pbc = True
    atoms.info['charge'] = args.charge
    atoms.info['spin'] = args.spin
    atoms.info['external_field'] = np.asarray(args.field)
    atoms.info['fermi_level'] = args.fermi_level
    kwargs = dict(
        models=torch_model,
        model_type='PolarMACE',
        device='cpu',
        default_dtype='float64',
    )
    if 'pbc_handling' in inspect.signature(MACECalculator).parameters:
        kwargs['pbc_handling'] = args.pbc_handling
    if 'compute_stress' in inspect.signature(MACECalculator).parameters:
        kwargs['compute_stress'] = False
    calculator = MACECalculator(**kwargs)
    data = {
        key: value
        for key, value in calculator._atoms_to_batch(atoms).to_dict().items()
        if isinstance(value, torch.Tensor)
    }
    data['fermi_level'] = torch.full(
        (int(data['ptr'].numel()) - 1,),
        args.fermi_level,
        dtype=data['positions'].dtype,
    )

    torch_products = []
    torch_debug = {
        'sources': [],
        'mixed_features': [],
        'fukui_raw': [],
        'field_raw': [],
        'field_source': [],
        'update_inputs': [],
        'update_potential_embedding': [],
        'update_potential_linear': [],
        'update_node_feats_linear': [],
        'update_charge_embedding': [],
        'update_dot_products': [],
        'update_source_embedding': [],
        'update_nonlinearity': [],
        'update_tp_out': [],
        'update_readout': [],
        'updates': [],
        'local_node_energy': [],
    }
    original_forward_dynamic = None
    handles = [
        product.register_forward_hook(
            lambda _module, _inputs, output: torch_products.append(
                output.detach().cpu().numpy()
            )
        )
        for product in torch_model.products
    ]
    if args.debug:

        def capture(key, transform=lambda value: value):
            def hook(_module, _inputs, output):
                torch_debug[key].append(transform(output).detach().cpu().numpy())

            return hook

        handles.extend(
            [
                module.register_forward_hook(capture('sources', lambda x: x[:, 0, :]))
                for module in torch_model.lr_source_maps
            ]
        )
        handles.append(
            torch_model.layer_feature_mixer.register_forward_hook(
                capture('mixed_features')
            )
        )
        handles.append(
            torch_model.fukui_source_map.register_forward_hook(capture('fukui_raw'))
        )
        if args.pbc_handling == 'realspace':
            handles.append(
                torch_model.electric_potential_descriptor.realspace_features.register_forward_hook(
                    capture('field_raw', lambda x: x[0])
                )
            )
        else:
            descriptor = torch_model.electric_potential_descriptor
            original_forward_dynamic = descriptor.forward_dynamic

            def capture_periodic_field(*field_args, **field_kwargs):
                source = field_kwargs.get('source_feats')
                if source is None:
                    source = field_args[1]
                torch_debug['field_source'].append(source.detach().cpu().numpy())
                value = original_forward_dynamic(*field_args, **field_kwargs)
                torch_debug['field_raw'].append(value.detach().cpu().numpy())
                return value

            descriptor.forward_dynamic = capture_periodic_field
        handles.extend(
            [
                module.register_forward_hook(capture('updates'))
                for module in torch_model.field_dependent_charges_maps
            ]
        )
        if args.torch_backbone:
            first_update = torch_model.field_dependent_charges_maps[0]
            handles.append(
                first_update.register_forward_pre_hook(
                    lambda _module, _args, kwargs: torch_debug['update_inputs'].append(
                        kwargs
                    ),
                    with_kwargs=True,
                )
            )
            for stage in (
                'potential_embedding',
                'dot_products',
                'source_embedding',
                'nonlinearity',
                'tp_out',
                'readout',
            ):
                handles.append(
                    getattr(first_update, stage).register_forward_hook(
                        capture(f'update_{stage}')
                    )
                )
            for stage in ('potential_linear', 'node_feats_linear', 'charge_embedding'):
                handles.append(
                    getattr(
                        first_update.potential_embedding, stage
                    ).register_forward_hook(capture(f'update_{stage}'))
                )
        handles.append(
            torch_model.local_electron_energy.register_forward_hook(
                capture('local_node_energy')
            )
        )
    try:
        if args.compute_force or args.compute_stress:
            reference = torch_model(
                dict(data),
                compute_force=True,
                compute_stress=args.compute_stress,
            )
        else:
            with torch.no_grad():
                reference = torch_model(dict(data), compute_force=False)
    finally:
        for handle in handles:
            handle.remove()
        if original_forward_dynamic is not None:
            torch_model.electric_potential_descriptor.forward_dynamic = (
                original_forward_dynamic
            )
    jax_data = {
        key: jax.numpy.asarray(value.detach().cpu().numpy())
        for key, value in data.items()
    }
    if args.torch_backbone and (args.compute_force or args.compute_stress):
        parser.error('--torch-backbone cannot be combined with derivatives')
    original_backbone = ScaleShiftMACE._energy_fn
    if args.torch_backbone:
        reference_backbone = {
            key: jax.numpy.asarray(reference[key].detach().cpu().numpy())
            for key in ('interaction_energy', 'node_energy', 'node_feats')
        }
        reference_backbone['energy'] = jax.numpy.asarray(
            (
                reference['energy']
                - reference['electron_energy']
                - reference['electrostatic_energy']
            )
            .detach()
            .cpu()
            .numpy()
        )
        ScaleShiftMACE._energy_fn = lambda _self, _data, **_kwargs: reference_backbone
    try:
        candidate = jax_model(
            jax_data,
            compute_force=args.compute_force or args.compute_stress,
            compute_stress=args.compute_stress,
            debug=args.debug,
        )
    finally:
        ScaleShiftMACE._energy_fn = original_backbone
    checks = {
        'energy': (reference['energy'], candidate['energy']),
        'interaction_energy': (
            reference['interaction_energy'],
            candidate['interaction_energy'],
        ),
        'node_energy': (reference['node_energy'], candidate['node_energy']),
        'node_feats': (reference['node_feats'], candidate['node_feats']),
        'electrostatic_energy': (
            reference['electrostatic_energy'],
            candidate['electrostatic_energy'],
        ),
        'electron_energy': (reference['electron_energy'], candidate['electron_energy']),
        'density_coefficients': (
            reference['density_coefficients'],
            candidate['density_coefficients'],
        ),
        'dipole': (reference['dipole'], candidate['dipole']),
        'charges': (reference['charges'], candidate['charges']),
        'spins': (reference['spins'], candidate['spins']),
        'total_charge': (reference['total_charge'], candidate['total_charge']),
        'spin_density': (reference['spin_density'], candidate['spin_density']),
        'spin_charge_density': (
            reference['spin_charge_density'],
            candidate['spin_charge_density'],
        ),
        'charges_history': (reference['charges_history'], candidate['charges_history']),
        'fukui_functions': (reference['fukui_functions'], candidate['fukui_functions']),
        'external_field': (reference['external_field'], candidate['external_field']),
        'fermi_level': (reference['fermi_level'], candidate['fermi_level']),
    }
    if args.compute_force or args.compute_stress:
        checks['forces'] = (reference['forces'], candidate['forces'])
    if args.compute_stress:
        checks['stress'] = (reference['stress'], candidate['stress'])
    feature_width = torch_products[0].shape[-1]
    for index, output in enumerate(torch_products):
        checks[f'product_{index}'] = (
            torch.as_tensor(output),
            candidate['node_feats'][
                :, index * feature_width : (index + 1) * feature_width
            ],
        )
    if args.debug:
        if args.pbc_handling == 'pbc':
            print(
                'kspace_cutoff: torch=',
                float(torch_model.kspace_cutoff),
                'jax=',
                jax_model.periodic_electrostatics.kspace_cutoff,
            )
            ref_k, _, _, _ = compute_k_vectors_flat(
                float(torch_model.kspace_cutoff),
                data['cell'].reshape(-1, 3, 3),
                data['rcell'].reshape(-1, 3, 3),
            )
            jax_cache = jax_model.periodic_electrostatics.precompute(
                jax_data['positions'],
                jax_data['batch'],
                jax_data['cell'],
            )
            actual_k = np.asarray(jax_cache['k_vectors'])
            print(
                'k vectors:',
                len(ref_k),
                len(actual_k),
                'max diff=',
                np.max(np.abs(ref_k.detach().numpy() - actual_k))
                if len(ref_k) == len(actual_k)
                else None,
            )
        debug = candidate['_debug']
        checks['sources'] = (
            torch.as_tensor(np.stack(torch_debug['sources'])),
            debug['sources'],
        )
        checks['mixed_features'] = (
            torch.as_tensor(torch_debug['mixed_features'][0]),
            debug['mixed_features'],
        )
        checks['fukui_raw'] = (
            torch.as_tensor(torch_debug['fukui_raw'][0]),
            debug['fukui_raw'],
        )
        if args.pbc_handling == 'pbc':
            checks['initial_density'] = (
                torch.as_tensor(np.stack(torch_debug['field_source'][:2], axis=1)),
                debug['initial_density'],
            )
        raw_fields = torch_debug['field_raw']
        norms = np.asarray(jax_model.field_feature_norms)
        fields = np.stack(
            [
                np.concatenate((raw_fields[2 * i], raw_fields[2 * i + 1]), axis=-1)
                / np.tile(norms, 2)
                for i in range(len(torch_debug['updates']))
            ]
        )
        checks['fields'] = (torch.as_tensor(fields), debug['fields'])
        if args.pbc_handling == 'pbc':
            for step in range(len(fields)):
                print(
                    'field step',
                    step,
                    'max diff=',
                    np.max(np.abs(fields[step] - np.asarray(debug['fields'][step]))),
                )
        checks['updates'] = (
            torch.as_tensor(np.stack(torch_debug['updates'])),
            debug['updates'],
        )
        if args.pbc_handling == 'pbc':
            for step, expected_update in enumerate(torch_debug['updates']):
                print(
                    'update step',
                    step,
                    'max diff=',
                    np.max(
                        np.abs(expected_update - np.asarray(debug['updates'][step]))
                    ),
                )
        if args.torch_backbone:
            inputs = torch_debug['update_inputs'][0]

            def to_jax(name):
                return jax.numpy.asarray(inputs[name].detach().cpu().numpy())

            update = jax_model.field_dependent_charges_maps[0]
            attrs = to_jax('node_attrs')
            node = to_jax('node_feats')
            potential = to_jax('potential_features')
            charges = to_jax('local_charges')
            embedded = update.potential_embedding(potential, node, attrs, charges)
            dot = update.dot_products(node, embedded)
            source = update.source_embedding(attrs)
            nonlinear = update.nonlinearity(
                jax.numpy.concatenate((dot, source), axis=-1)
            )
            product = update.tp_out(node, nonlinear)
            readout = update.readout(product)
            for stage, actual in (
                ('potential_embedding', embedded),
                ('dot_products', dot),
                ('source_embedding', source),
                ('nonlinearity', nonlinear),
                ('tp_out', product),
                ('readout', readout),
                (
                    'potential_linear',
                    update.potential_embedding.potential_linear(potential),
                ),
                (
                    'node_feats_linear',
                    update.potential_embedding.node_feats_linear(node),
                ),
                (
                    'charge_embedding',
                    update.potential_embedding.charge_embedding(charges),
                ),
            ):
                expected = torch_debug[f'update_{stage}'][0]
                print(
                    'update stage',
                    stage,
                    'max diff=',
                    np.max(np.abs(expected - np.asarray(actual))),
                )
        checks['local_node_energy'] = (
            torch.as_tensor(torch_debug['local_node_energy'][0]),
            debug['local_node_energy'],
        )
    passed = True
    for name, (torch_value, jax_value) in checks.items():
        expected = torch_value.detach().cpu().numpy()
        actual = np.asarray(jax_value)
        diff = np.abs(expected - actual)
        good = np.allclose(expected, actual, rtol=args.rtol, atol=args.atol)
        passed &= good
        print(
            f'{name}: {"PASS" if good else "FAIL"}; '
            f'max abs={diff.max(initial=0):.6e}; shape={actual.shape}'
        )
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
