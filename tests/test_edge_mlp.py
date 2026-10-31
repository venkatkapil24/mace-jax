"""Factored edge/node MLP evaluation retains values and all input gradients."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from mace_jax.modules.radial import RadialMLP


@pytest.mark.parametrize('scaled', [False, True])
@pytest.mark.parametrize('dtype', ['float32', 'float64', 'mixed'])
@pytest.mark.parametrize('hidden', [(), (16,), (16, 8)])
def test_factored_edge_projection_matches_concatenated_mlp(dtype, hidden, scaled):
    rng = np.random.default_rng(946)
    parameter_dtype = 'float32' if dtype == 'mixed' else dtype
    edge_dtype = 'float64' if dtype == 'mixed' else dtype
    nodes, edges = 7, 19
    edge_dim, source_dim, target_dim, outputs = 8, 5, 7, 6
    edge = rng.normal(size=(edges, edge_dim)).astype(edge_dtype)
    source = rng.normal(size=(nodes, source_dim)).astype(parameter_dtype)
    target = rng.normal(size=(nodes, target_dim)).astype(parameter_dtype)
    # Repeated neighbours and one unused node test gathering and accumulation.
    indices = jnp.asarray(rng.integers(0, nodes-1, size=(2, edges)), dtype=jnp.int32)
    probe = jnp.asarray(rng.normal(size=(edges, outputs)).astype(edge_dtype))
    scale = jnp.asarray(np.linspace(0., 1., edges, dtype=edge_dtype)[:, None])
    model = RadialMLP(
        [edge_dim+source_dim+target_dim, *hidden, outputs],
        output_permutation=tuple(reversed(range(outputs))), rngs=nnx.Rngs(9),
    )
    graphdef, params = nnx.split(model)
    params = jax.tree.map(lambda x: x.astype(parameter_dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)

    def evaluate(state, edge, source, target, scale, *, factored):
        mlp = nnx.merge(graphdef, state)
        if factored:
            return mlp.with_node_embeddings(edge, source, target, indices, output_scale=scale if scaled else None)
        inputs = jnp.concatenate((edge, source[indices[0]], target[indices[1]]), axis=-1)
        output = mlp(inputs)
        return output * scale if scaled else output

    args = (params, jnp.asarray(edge), jnp.asarray(source), jnp.asarray(target), scale)
    results = []
    for factored in (False, True):
        def objective(*args):
            output = evaluate(*args, factored=factored)
            return jnp.sum(output*probe), output
        results.append(jax.jit(jax.value_and_grad(
            objective, argnums=(0, 1, 2, 3, 4), has_aux=True,
        ))(*args))
    expected, actual = map(jax.tree.leaves, results)
    assert len(expected) == len(actual)
    for a, b in zip(actual, expected):
        # Mixed inputs still return float32 parameter/node-feature gradients.
        # Their accumulation may differ by a float32 rounding step on GPU.
        assert a.dtype == b.dtype
        tolerance = 3e-5 if a.dtype == jnp.float32 else 2e-11
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)


def test_edge_output_scale_is_scalar_per_edge():
    model = RadialMLP([3, 4, 2], rngs=nnx.Rngs(4))
    edge = jnp.ones((3, 1), dtype=jnp.float64)
    nodes = jnp.ones((2, 1), dtype=jnp.float64)
    indices = jnp.asarray([[0, 1, 0], [1, 0, 1]])
    column = model.with_node_embeddings(
        edge, nodes, nodes, indices, output_scale=jnp.full((3, 1), .3),
    )
    scalar = model.with_node_embeddings(edge, nodes, nodes, indices, output_scale=.3)
    np.testing.assert_allclose(scalar, column)
    with pytest.raises(ValueError):
        model.with_node_embeddings(
            edge, nodes, nodes, indices, output_scale=jnp.ones((3, 2)),
        )
