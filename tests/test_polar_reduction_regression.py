"""Check a scalar/vector contraction together with its downstream concatenation."""
import math

import jax
import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.modules.polar_blocks import SparseUvuTensorProduct


def test_compiled_scalar_vector_contraction_and_derivatives():
    # These dimensions exercise the GPU reduction/concatenation fusion that
    # silently miscompiled the previous reduction on JAX/jaxlib 0.10.2.
    n, channels = 999, 512
    rng = np.random.default_rng(612)
    first, second = rng.normal(size=(2, n, 4 * channels))
    source = rng.normal(size=(n, 7 * channels))
    mask = rng.normal(size=channels)
    probe = rng.normal(size=(n, 2 * channels))
    irreps = Irreps(f'{channels}x0e+{channels}x1o')
    model = SparseUvuTensorProduct(
        irreps, irreps, Irreps(f'{channels}x0e'),
        [(0, 0, 0, 'uvu', True), (1, 1, 0, 'uvu', True)],
    )
    identity = np.eye(channels).reshape(-1)
    model.weight = nnx.Param(jnp.asarray(np.concatenate((identity, identity))))
    model.output_mask = nnx.Param(jnp.asarray(mask))
    graphdef, state = nnx.split(model)

    def combined(params, a, b, context):
        contraction = nnx.merge(graphdef, params)(a, b)
        return jnp.concatenate((contraction, context[:, 5*channels:6*channels]), axis=-1)

    scale = 1 / math.sqrt(2 * channels)
    expected_dot = (first[:, :channels] * second[:, :channels] * scale
                    + np.sum(first[:, channels:].reshape(n, channels, 3)
                             * second[:, channels:].reshape(n, channels, 3), axis=-1)
                    * (scale / math.sqrt(3))) * mask
    expected = np.concatenate((expected_dot, source[:, 5*channels:6*channels]), axis=-1)
    args = tuple(map(jnp.asarray, (first, second, source)))
    actual = jax.jit(combined)(state, *args)
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)

    def objective(params, a, b, context):
        return jnp.sum(combined(params, a, b, context) * jnp.asarray(probe))

    gradients = jax.jit(jax.grad(objective, argnums=(1, 2, 3)))(state, *args)
    factors = probe[:, :channels] * mask * scale
    expected_gradients = []
    for other in (second, first):
        expected_gradients.append(np.concatenate((
            other[:, :channels] * factors,
            (other[:, channels:].reshape(n, channels, 3)
             * factors[..., None] / math.sqrt(3)).reshape(n, 3*channels),
        ), axis=-1))
    source_gradient = np.zeros_like(source)
    source_gradient[:, 5*channels:6*channels] = probe[:, channels:]
    expected_gradients.append(source_gradient)
    for actual, expected in zip(gradients, expected_gradients):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
