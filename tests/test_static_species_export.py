"""Fixed species can be validated before exporting a compiled model."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.adapters.cuequivariance.symmetric_contraction import _select_weights


def test_fixed_species_export_and_gradient():
    species = jnp.array([2, 0, 2], dtype=jnp.int32)
    weights = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    select = lambda w: _select_weights(w, species, dtype=w.dtype, num_elements=3)
    exported = jax.export.export(jax.jit(select), platforms=('cpu',))(
        jax.ShapeDtypeStruct(weights.shape, weights.dtype))
    np.testing.assert_array_equal(exported.call(weights), weights[np.array([2, 0, 2])])
    np.testing.assert_array_equal(jax.grad(lambda w: select(w).sum())(weights),
                                  [[1, 1], [0, 0], [2, 2]])


@pytest.mark.parametrize('indices', [[-1], [3]])
def test_fixed_species_out_of_range(indices):
    with pytest.raises(ValueError, match='indices out of range'):
        _select_weights(jnp.ones((3, 2)), jnp.array(indices), dtype=jnp.float32,
                        num_elements=3)


def test_dynamic_species_keeps_runtime_validation():
    select = jax.jit(lambda i: _select_weights(jnp.ones((3, 2)), i,
                                             dtype=jnp.float32, num_elements=3))
    np.testing.assert_array_equal(select(jnp.array([0, 2])), np.ones((2, 2)))
    with pytest.raises(jax.errors.JaxRuntimeError, match='indices out of range'):
        select(jnp.array([3])).block_until_ready()
