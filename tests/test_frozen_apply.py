"""Fixed inference weights leave coordinate and cell derivatives live."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from mace_jax.tools.bundle import ModelBundle


class Quadratic(nnx.Module):
    def __init__(self, dtype):
        self.weights = nnx.Param(jnp.asarray([2., 3., 4.], dtype=dtype))

    def __call__(self, data, *, compute_force=False):
        r, h = data['positions'], data['cell']
        output = {'energy': jnp.sum(self.weights * r**2) + jnp.sum(h**2)}
        if compute_force:
            output['forces'] = -2 * self.weights * r
        return output


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_frozen_apply_keeps_positions_cell_and_keyword_arguments_live(dtype):
    graphdef, parameters = nnx.split(Quadratic(dtype))
    bundle = ModelBundle(config={}, params=parameters, graphdef=graphdef)
    apply = bundle.make_frozen_apply()
    evaluate = jax.jit(lambda r, h: apply(
        {'positions': r, 'cell': h}, compute_force=True))
    r = jnp.asarray([[.2, -.3, .5], [.9, .8, -.4]], dtype=dtype)
    h = jnp.eye(3, dtype=dtype) * 2
    for positions, cell in [(r, h), (r + .13, h * 1.02)]:
        actual = evaluate(positions, cell)
        expected = Quadratic(dtype)({'positions': positions, 'cell': cell},
                                     compute_force=True)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=2e-6, atol=2e-6)
        derivatives = jax.jit(jax.grad(
            lambda r, h: evaluate(r, h)['energy'], argnums=(0, 1)))(positions, cell)
        np.testing.assert_allclose(derivatives[0], -actual['forces'], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(derivatives[1], 2 * cell, rtol=2e-6, atol=2e-6)
        assert actual['energy'].dtype == dtype


@pytest.mark.parametrize('storage', ['state', 'numpy'])
def test_frozen_apply_snapshots_mutable_parameter_storage(storage):
    graphdef, parameters = nnx.split(Quadratic(jnp.float64))
    values = np.array([2., 3., 4.])
    if storage == 'numpy':
        parameters['weights'].value = values
    # State containers can be updated even though their JAX arrays are immutable.
    bundle = ModelBundle(config={}, params=parameters, graphdef=graphdef)
    apply = bundle.make_frozen_apply()
    data = {'positions': jnp.ones((1, 3)), 'cell': jnp.zeros((3, 3))}
    if storage == 'numpy':
        values[:] = 5.
    else:
        parameters['weights'].value = jnp.full(3, 5.)
    np.testing.assert_allclose(jax.jit(apply)(data)['energy'], 9.)
    np.testing.assert_allclose(jax.jit(bundle.make_frozen_apply())(data)['energy'], 15.)
