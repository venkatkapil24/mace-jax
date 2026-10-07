"""The reduced distance calculation must preserve moving-boundary forces."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.fires import fires_energy


@pytest.mark.parametrize('n_qm', [0, 1, 25, 51, 52])
def test_fires_energy_and_boundary_derivatives(n_qm):
    with jax.experimental.enable_x64():
        box = 11.76
        k = 500 * 4.184 / 96.48533212331002
        rng = np.random.default_rng(31)
        positions = jnp.array(rng.uniform(0, box, (60, 3)))

        def evaluate(r):
            return fires_energy(r[8 : 8 + n_qm], r[8 + n_qm :], r[:8], box, k)

        def reference(r):
            if n_qm in (0, 52):
                return jnp.asarray(0.0)
            delta = r[8:, None] - r[None, :8]
            delta -= box * jnp.round(delta / box)
            distance = jnp.linalg.norm(delta, axis=-1).min(axis=-1)
            intrusion = jnp.maximum(distance[:n_qm].max() - distance[n_qm:], 0)
            return 0.5 * k * jnp.sum(intrusion**2)

        e, g = jax.jit(jax.value_and_grad(evaluate))(positions)
        er, gr = jax.jit(jax.value_and_grad(reference))(positions)
        np.testing.assert_allclose(e, er, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(g, gr, atol=1e-10, rtol=1e-12)
        np.testing.assert_allclose(g.sum(axis=0), 0, atol=1e-10)
        direction = jnp.array(rng.normal(size=positions.shape))
        eps = 1e-5
        fd = (
            evaluate(positions + eps * direction)
            - evaluate(positions - eps * direction)
        ) / (2 * eps)
        np.testing.assert_allclose(fd, jnp.sum(g * direction), atol=1e-5, rtol=1e-7)
