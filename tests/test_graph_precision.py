"""Graph geometry keeps model precision independent of the MD precision."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.utils import prepare_graph


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_prepare_graph_preserves_coordinate_dtype_with_x64_enabled(dtype):
    with jax.enable_x64(True):
        positions = jnp.array([[0., 0., 0.], [1., 2., 0.]], dtype=dtype)
        data = dict(positions=positions, cell=jnp.eye(3, dtype=dtype)[None]*10,
                    shifts=jnp.zeros((2, 3), dtype=dtype),
                    edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32),
                    batch=jnp.zeros(2, dtype=jnp.int32),
                    ptr=jnp.array([0, 2], dtype=jnp.int32))
        graph = jax.jit(prepare_graph)(data)
        for value in (graph.positions, graph.cell, graph.vectors, graph.lengths):
            assert value.dtype == dtype
        np.testing.assert_array_equal(graph.vectors, [[1., 2., 0.], [-1., -2., 0.]])

        def energy(r):
            return jnp.sum(prepare_graph({**data, 'positions': r}).vectors**2)

        derivative = jax.jit(jax.grad(energy))(positions)
        assert derivative.dtype == dtype
        np.testing.assert_array_equal(derivative, [[-4., -8., 0.], [4., 8., 0.]])
