"""CUDA contraction must preserve the polynomial and its input derivatives."""

import re

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.cuequivariance.symmetric_contraction import SymmetricContraction
from mace_jax.adapters.cuequivariance.utility import ir_mul_to_mul_ir


@pytest.mark.skipif(jax.default_backend() != 'gpu', reason='CUDA kernel requires GPU')
def test_symmetric_contraction_cuda_outputs_and_derivatives():
    jax.config.update('jax_enable_x64', True)
    module = SymmetricContraction(
        '8x0e+8x1o+8x2e+8x3o', '8x0e+8x1o', 3, 3, rngs=nnx.Rngs(7)
    )
    x = jax.random.normal(jax.random.PRNGKey(2), (5, 8, 16), dtype=jnp.float64)
    indices = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)

    def native(x):
        return module(x, indices)

    def reference(x):
        array = module._ensure_mul_ir_layout(x)
        result = cuex.equivariant_polynomial(
            module.descriptor,
            [
                module._weight_rep_from_indices(indices, array.dtype),
                module._features_to_rep(array, array.dtype),
            ],
            math_dtype=array.dtype,
            method='naive',
        )
        return ir_mul_to_mul_ir(
            result.change_layout(cue.ir_mul).array, Irreps(module.irreps_out_o3_str)
        )

    native = jax.jit(native)
    reference = jax.jit(reference)
    np.testing.assert_allclose(native(x), reference(x), atol=1e-9, rtol=1e-11)
    native_grad = jax.jit(jax.grad(lambda a: jnp.sum(native(a) ** 2)))(x)
    reference_grad = jax.jit(jax.grad(lambda a: jnp.sum(reference(a) ** 2)))(x)
    np.testing.assert_allclose(native_grad, reference_grad, atol=1e-7, rtol=1e-11)
    hlo = native.lower(x).compiler_ir('hlo').as_hlo_text()
    targets = set(re.findall(r'custom_call_target="([^"]+)"', hlo))
    assert 'uniform_1d_cuda' in targets
    print(
        'max output error:',
        float(jnp.max(jnp.abs(native(x) - reference(x)))),
        'max gradient error:',
        float(jnp.max(jnp.abs(native_grad - reference_grad))),
        'CUDA targets:',
        sorted(targets),
    )
