#!/usr/bin/env python3
"""Compare a released MACE or PolarMACE checkpoint on native Torch and TorchAX.

The verifier deliberately runs both forwards without torch.compile or jax.jit.
TorchAX is an experimental second execution path, not the reference result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import traceback
from collections import defaultdict
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

# Keep imports for the optional, expensive runtimes inside main(). This also
# allows the comparison code to be checked without a Torch installation.
DEFAULT_MODULES = (
    'node_embedding',
    'interactions',
    'products',
    'lr_source_maps',
    'layer_feature_mixer',
    'fukui_source_map',
    'electric_potential_descriptor',
    'field_dependent_charges_maps',
    'local_electron_energy',
    'coulomb_energy',
)


def _array(value: Any) -> np.ndarray:
    """Copy a native Torch or TorchAX tensor to a NumPy array."""
    if hasattr(value, 'jax'):
        return np.asarray(value.jax())
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _flatten_tensors(value: Any, prefix: str = '') -> dict[str, np.ndarray]:
    """Flatten tensor outputs, including tuples returned by interaction blocks."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten_tensors(child, f'{prefix}.{key}' if prefix else str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, child in enumerate(value):
            result.update(_flatten_tensors(child, f'{prefix}.{index}' if prefix else str(index)))
        return result
    if value is None or not hasattr(value, 'shape'):
        return {}
    try:
        array = _array(value)
    except (TypeError, ValueError, RuntimeError):
        return {}
    if array.dtype.kind not in 'biufc':
        return {}
    return {prefix: array}


def compare_arrays(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Return per-tensor comparison results; missing values are failures."""
    comparisons = {}
    for key in sorted(reference.keys() | candidate.keys()):
        if key not in reference or key not in candidate:
            comparisons[key] = {
                'status': 'missing',
                'reference_present': key in reference,
                'torchax_present': key in candidate,
            }
            continue
        ref = np.asarray(reference[key])
        got = np.asarray(candidate[key])
        if ref.shape != got.shape:
            comparisons[key] = {
                'status': 'shape_mismatch',
                'reference_shape': list(ref.shape),
                'torchax_shape': list(got.shape),
            }
            continue
        finite = bool(np.isfinite(ref).all() and np.isfinite(got).all())
        diff = np.abs(ref - got)
        max_abs = float(diff.max(initial=0)) if finite else None
        scale = atol + rtol * np.abs(ref)
        max_scaled = float(np.max(diff / np.maximum(scale, np.finfo(float).tiny), initial=0)) if finite else None
        passed = finite and bool(np.allclose(ref, got, rtol=rtol, atol=atol))
        bitwise_equal = bool(
            ref.dtype == got.dtype
            and np.array_equal(np.ascontiguousarray(ref).view(np.uint8), np.ascontiguousarray(got).view(np.uint8))
        )
        comparisons[key] = {
            'status': 'pass' if passed else 'mismatch',
            'bitwise_equal': bitwise_equal,
            'shape': list(ref.shape),
            'dtype_reference': str(ref.dtype),
            'dtype_torchax': str(got.dtype),
            'max_abs_error': max_abs,
            'max_tolerance_ratio': max_scaled,
            'finite': finite,
        }
    return comparisons


def _capture_modules(model: Any, prefixes: tuple[str, ...], input_names: tuple[str, ...] = ()):
    captured: dict[str, np.ndarray] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    handles = []

    def hook(name: str):
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            index = counts[name]
            counts[name] += 1
            if name in input_names:
                captured.update(_flatten_tensors(_inputs, f'{name}#{index}.input'))
            captured.update(_flatten_tensors(output, f'{name}#{index}'))

        return record

    for name, module in model.named_modules():
        if not name:
            continue
        if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            # Hook only the named parents and list members; child linear layers
            # would produce a very large and less useful report.
            if name in prefixes or any(name.startswith(p + '.') and name.count('.') == p.count('.') + 1 for p in prefixes):
                handles.append(module.register_forward_hook(hook(name)))
    return captured, handles


def _remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return 'not installed'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Local trusted MACE .model checkpoint')
    parser.add_argument('--xyz', type=Path, help='Optional ASE-readable structure; default is a water molecule')
    parser.add_argument('--charge', type=int, default=0)
    parser.add_argument('--spin', type=int, default=1, help='Spin multiplicity, as used by the Torch calculator')
    parser.add_argument('--field', type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument('--pbc-handling', default='realspace', choices=('realspace', 'pbc', 'slab', 'molecule_in_box', 'mixed_periodic'))
    parser.add_argument('--dtype', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--compute-force', action='store_true', help='Compare coordinate gradients as well as forward outputs')
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--require-bitwise', action='store_true', help='Fail unless every compared tensor has identical bytes')
    parser.add_argument('--workaround-view-getitem', action='store_true', help='Apply a TorchAX 0.0.13 View indexing workaround for diagnosis')
    parser.add_argument('--workaround-e3nn-extract', action='store_true', help='Replace e3nn Gate extraction with functional slices for diagnosis')
    parser.add_argument('--capture-prefix', action='append', default=[], help='Capture immediate child modules under this additional module path')
    parser.add_argument('--capture-input', action='append', default=[], help='Capture inputs to this named module as well as outputs')
    parser.add_argument('--report', type=Path, help='Write a machine-readable JSON report')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {args.checkpoint}')

    try:
        import jax  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchax  # noqa: PLC0415
        from ase.build import molecule  # noqa: PLC0415
        from ase.io import read  # noqa: PLC0415
        from mace.calculators.mace import MACECalculator  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(f'Missing verifier dependency: {exc}') from exc

    if args.dtype == 'float64':
        jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(getattr(torch, args.dtype))
    torch.set_num_threads(1)

    # The checkpoint is a trusted, published whole-model pickle. Never load an
    # untrusted checkpoint with weights_only=False.
    model = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_class = model.__class__.__name__
    if model_class not in {'MACE', 'ScaleShiftMACE', 'PolarMACE'}:
        raise SystemExit(f'Unsupported model class: {model_class}')
    model = model.eval().to(dtype=getattr(torch, args.dtype))
    if model_class == 'PolarMACE' and hasattr(model, 'set_electrostatic_pbcs'):
        model.set_electrostatic_pbcs(args.pbc_handling)
    elif model_class == 'PolarMACE' and args.pbc_handling != 'realspace':
        raise SystemExit('This MACE release requires a newer PolarMACE for explicit PBC modes')
    elif model_class != 'PolarMACE' and args.pbc_handling != 'realspace':
        raise SystemExit('--pbc-handling is only supported for PolarMACE')

    atoms = read(args.xyz) if args.xyz else molecule('H2O')
    atoms.info['charge'] = args.charge
    atoms.info['spin'] = args.spin
    atoms.info['external_field'] = np.asarray(args.field, dtype=float)
    calculator_kwargs = dict(
        models=model,
        model_type='PolarMACE' if model_class == 'PolarMACE' else 'MACE',
        device='cpu',
        default_dtype=args.dtype,
    )
    if model_class == 'PolarMACE' and 'pbc_handling' in inspect.signature(MACECalculator).parameters:
        calculator_kwargs['pbc_handling'] = args.pbc_handling
    if 'compute_stress' in inspect.signature(MACECalculator).parameters:
        calculator_kwargs['compute_stress'] = False
    calculator = MACECalculator(**calculator_kwargs)
    data = calculator._atoms_to_batch(atoms).to_dict()
    data = {key: value for key, value in data.items() if isinstance(value, torch.Tensor)}

    report: dict[str, Any] = {
        'checkpoint': str(args.checkpoint.resolve()),
        'checkpoint_sha256': _sha256(args.checkpoint),
        'model_class': model_class,
        'structure': str(args.xyz.resolve()) if args.xyz else 'ase.build.molecule("H2O")',
        'charge': args.charge,
        'spin_multiplicity': args.spin,
        'external_field': args.field,
        'pbc_handling': args.pbc_handling,
        'dtype': args.dtype,
        'compute_force': args.compute_force,
        'torchax_force_method': 'jax.grad' if args.compute_force else None,
        'rtol': args.rtol,
        'atol': args.atol,
        'require_bitwise': args.require_bitwise,
        'workaround_view_getitem': args.workaround_view_getitem,
        'workaround_e3nn_extract': args.workaround_e3nn_extract,
        'capture_prefixes': args.capture_prefix,
        'capture_inputs': args.capture_input,
        'versions': {
            'torch': torch.__version__,
            'torchax': _version('torchax'),
            'jax': jax.__version__,
            'mace-torch': _version('mace-torch'),
            'graph_longrange': _version('graph_longrange'),
        },
        'input_shapes': {key: list(value.shape) for key, value in data.items()},
    }

    capture_prefixes = DEFAULT_MODULES + tuple(args.capture_prefix) + tuple(args.capture_input)
    native_capture, handles = _capture_modules(model, capture_prefixes, tuple(args.capture_input))
    try:
        native_output = model(dict(data), compute_force=args.compute_force, compute_stress=False)
        native_capture.update(_flatten_tensors(native_output, 'output'))
    finally:
        _remove_hooks(handles)
    report['native'] = {'status': 'pass', 'tensor_count': len(native_capture)}

    stage = 'enable_torchax'
    torchax_capture: dict[str, np.ndarray] = {}
    try:
        torchax.enable_globally()
        if args.workaround_view_getitem:
            from torchax.ops.ops_registry import all_torch_functions  # noqa: PLC0415
            from torchax.view import View  # noqa: PLC0415

            getitem_op = all_torch_functions[torch.Tensor.__getitem__]
            original_getitem = getitem_op.func

            def view_getitem(tensor, indexes):
                return original_getitem(tensor.torch() if isinstance(tensor, View) else tensor, indexes)

            getitem_op.func = view_getitem
        if args.workaround_e3nn_extract:
            def functional_extract(extract, features):
                outputs = []
                for instructions in extract.instructions:
                    pieces = [
                        features.narrow(-1, extract.irreps_in[:index].dim, extract.irreps_in[index].dim)
                        for index in instructions
                    ]
                    outputs.append(torch.cat(pieces, dim=-1) if pieces else features.new_zeros((*features.shape[:-1], 0)))
                return tuple(outputs)

            patched = 0
            for module in model.modules():
                if module.__class__.__name__ == 'Gate' and hasattr(module, 'sc') and hasattr(module.sc, 'cut'):
                    module.sc.cut.forward = MethodType(functional_extract, module.sc.cut)
                    patched += 1
            report['patched_e3nn_gate_extracts'] = patched
        stage = 'move_model_to_jax'
        model = model.to('jax')
        stage = 'move_inputs_to_jax'
        jax_data = {key: value.to('jax') for key, value in data.items()}
        stage = 'torchax_forward'
        torchax_capture, handles = _capture_modules(model, capture_prefixes, tuple(args.capture_input))
        try:
            torchax_output = model(dict(jax_data), compute_force=False, compute_stress=False)
            torchax_capture.update(_flatten_tensors(torchax_output, 'output'))
        finally:
            _remove_hooks(handles)
        if args.compute_force:
            stage = 'jax_grad_forces'
            from torchax.tensor import Tensor as TorchAXTensor  # noqa: PLC0415

            environment = jax_data['positions']._env

            def total_energy(positions):
                gradient_data = dict(jax_data)
                gradient_data['positions'] = TorchAXTensor(positions, environment)
                return model(gradient_data, compute_force=False, compute_stress=False)['energy'].jax().sum()

            torchax_capture['output.forces'] = np.asarray(-jax.grad(total_energy)(jax_data['positions'].jax()))
        stage = 'compare'
        comparisons = compare_arrays(native_capture, torchax_capture, rtol=args.rtol, atol=args.atol)
        passed = all(item['status'] == 'pass' and (not args.require_bitwise or item['bitwise_equal']) for item in comparisons.values())
        report['torchax'] = {
            'status': 'pass' if passed else 'mismatch',
            'tensor_count': len(torchax_capture),
            'bitwise_equal_count': sum(item.get('bitwise_equal', False) for item in comparisons.values()),
        }
        report['comparisons'] = comparisons
    except Exception as exc:  # Capture the first unsupported TorchAX operation.
        if torchax_capture:
            report['partial_comparisons'] = compare_arrays(
                {key: native_capture[key] for key in torchax_capture if key in native_capture},
                {key: value for key, value in torchax_capture.items() if key in native_capture},
                rtol=args.rtol,
                atol=args.atol,
            )
        report['torchax'] = {
            'status': 'error',
            'stage': stage,
            'tensor_count_before_error': len(torchax_capture),
            'exception': f'{type(exc).__name__}: {exc}',
            'traceback': traceback.format_exc(),
        }

    rendered = json.dumps(report, indent=2, sort_keys=True, default=str)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + '\n')
    print(rendered)
    return 0 if report['torchax']['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
