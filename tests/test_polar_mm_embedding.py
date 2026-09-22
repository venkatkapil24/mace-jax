"""Small numerical checks for the periodic point-charge embedding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.polar_periodic import PeriodicPolarElectrostatics


@pytest.mark.parametrize('include_self', [False, True])
def test_zero_mm_charges_recover_periodic_polar_energy_and_field(include_self):
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(
        1, 1.5, 1, (1.5, 3.0), include_energy_self_interaction=include_self
    )
    qm_positions = jnp.asarray([[2.0, 3.0, 4.0], [2.8, 3.1, 4.0]])
    qm_density = jnp.asarray([[0.3, 0.1, -0.2, 0.05], [-0.3, 0.03, 0.04, -0.01]])
    mm_positions = jnp.asarray([[5.0, 4.0, 3.0], [5.9, 4.0, 3.0]])
    cache = model.precompute(
        qm_positions,
        jnp.zeros(2, dtype=jnp.int32),
        jnp.eye(3)[None] * 9.0,
    )
    zero_charges = jnp.zeros(2)
    mixed, cross = model.mixed_coulomb_energy(
        qm_density, mm_positions, zero_charges, cache
    )
    expected = model.coulomb_energy(qm_density, cache)
    np.testing.assert_allclose(mixed, expected, atol=1e-12)
    np.testing.assert_allclose(cross, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        model.point_charge_field_features(mm_positions, zero_charges, cache),
        0.0,
        atol=1e-12,
    )


def test_mixed_cross_is_bilinear_and_has_finite_position_forces():
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(1, 1.5, 1, (1.5, 3.0))
    qm_positions = jnp.asarray([[2.0, 3.0, 4.0], [2.8, 3.1, 4.0]])
    qm_density = jnp.asarray([[0.3, 0.1, -0.2, 0.05], [-0.3, 0.03, 0.04, -0.01]])
    mm_positions = jnp.asarray([[5.0, 4.0, 3.0], [5.9, 4.0, 3.0]])
    charges = jnp.asarray([-0.4, 0.4])
    cell = jnp.eye(3)[None] * 9.0

    def cross_at(qm_pos, mm_pos, mm_charge):
        cache = model.precompute(qm_pos, jnp.zeros(2, dtype=jnp.int32), cell)
        _, cross = model.mixed_coulomb_energy(qm_density, mm_pos, mm_charge, cache)
        return cross[0]

    cross = cross_at(qm_positions, mm_positions, charges)
    doubled = cross_at(qm_positions, mm_positions, 2 * charges)
    np.testing.assert_allclose(doubled, 2 * cross, rtol=1e-10, atol=1e-12)
    compiled = jax.jit(cross_at)(qm_positions, mm_positions, charges)
    np.testing.assert_allclose(compiled, cross, rtol=1e-10, atol=1e-12)
    grads = jax.grad(cross_at, argnums=(0, 1))(qm_positions, mm_positions, charges)
    assert all(bool(jnp.all(jnp.isfinite(gradient))) for gradient in grads)
    eps = 1e-4
    delta = jnp.zeros_like(mm_positions).at[0, 0].set(eps)
    finite_difference = (
        cross_at(qm_positions, mm_positions + delta, charges)
        - cross_at(qm_positions, mm_positions - delta, charges)
    ) / (2 * eps)
    np.testing.assert_allclose(grads[1][0, 0], finite_difference, rtol=1e-5, atol=1e-7)
