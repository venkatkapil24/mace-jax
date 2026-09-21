"""Tests for the numerical report produced by the MACE TorchAX probe."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parents[1] / 'scripts' / 'verify_mace_torchax.py'
_SPEC = spec_from_file_location('verify_mace_torchax', _PATH)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_compare_arrays_reports_tolerance_and_missing_values():
    reference = {
        'close': np.array([1.0, 2.0]),
        'far': np.array([0.0, 1.0]),
        'only_native': np.array([1.0]),
    }
    candidate = {
        'close': np.array([1.0, 2.0 + 1e-8]),
        'far': np.array([0.0, 1.1]),
        'only_torchax': np.array([1.0]),
    }

    result = _MODULE.compare_arrays(reference, candidate, rtol=1e-6, atol=1e-7)

    assert result['close']['status'] == 'pass'
    assert not result['close']['bitwise_equal']
    assert result['far']['status'] == 'mismatch'
    assert result['far']['max_abs_error'] > 0.09
    assert result['only_native']['status'] == 'missing'
    assert result['only_native']['reference_present']
    assert result['only_torchax']['status'] == 'missing'
    assert result['only_torchax']['torchax_present']


def test_compare_arrays_rejects_shape_and_nonfinite_values():
    reference = {
        'shape': np.zeros((2, 3)),
        'nan': np.array([np.nan]),
    }
    candidate = {
        'shape': np.zeros((3, 2)),
        'nan': np.array([np.nan]),
    }

    result = _MODULE.compare_arrays(reference, candidate, rtol=1e-6, atol=1e-7)

    assert result['shape']['status'] == 'shape_mismatch'
    assert result['nan']['status'] == 'mismatch'
    assert not result['nan']['finite']
    assert result['nan']['max_abs_error'] is None


def test_compare_arrays_distinguishes_bitwise_from_numerical_equality():
    result = _MODULE.compare_arrays(
        {'zero': np.array([0.0])},
        {'zero': np.array([-0.0])},
        rtol=0,
        atol=0,
    )

    assert result['zero']['status'] == 'pass'
    assert not result['zero']['bitwise_equal']
