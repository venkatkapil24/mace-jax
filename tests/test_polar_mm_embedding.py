"""Small numerical checks for the periodic point-charge embedding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erfc

from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.modules.polar_periodic import (
    PeriodicPolarElectrostatics,
    ewald_neutralizing_background_energy,
)


def _single_charge_ewald(alpha, *, charge=0.7, box=9.0, nmax=3, kmax=12):
    """Converged one-charge cubic Ewald sum with a uniform background."""
    coulomb = FIELD_CONSTANT / (4 * np.pi)
    axis = np.arange(-nmax, nmax + 1)
    lattice = np.stack(
        np.meshgrid(axis, axis, axis, indexing='ij'), -1
    ).reshape(-1, 3)
    lattice = lattice[np.any(lattice != 0, axis=1)]
    distances = jnp.linalg.norm(
        jnp.asarray(lattice, dtype=jnp.float64) * box, axis=1
    )
    real = (
        0.5
        * coulomb
        * charge**2
        * jnp.sum(erfc(alpha * distances) / distances)
    )

    k_axis = np.arange(-kmax, kmax + 1)
    k_int = np.stack(
        np.meshgrid(k_axis, k_axis, k_axis, indexing='ij'), -1
    ).reshape(-1, 3)
    k_int = k_int[np.any(k_int != 0, axis=1)]
    k_vectors = 2 * np.pi / box * jnp.asarray(k_int, dtype=jnp.float64)
    k2 = jnp.sum(k_vectors**2, axis=1)
    reciprocal = (
        coulomb
        * 2
        * np.pi
        / box**3
        * charge**2
        * jnp.sum(jnp.exp(-k2 / (4 * alpha**2)) / k2)
    )
    self_energy = -coulomb * alpha / np.sqrt(np.pi) * charge**2
    background = ewald_neutralizing_background_energy(charge, box**3, alpha)
    return real + reciprocal + self_energy + background, background


def test_charged_ewald_background_restores_alpha_independence():
    jax.config.update('jax_enable_x64', True)
    low_alpha, low_background = _single_charge_ewald(0.35)
    high_alpha, high_background = _single_charge_ewald(0.65)
    np.testing.assert_allclose(low_alpha, high_alpha, atol=2e-11, rtol=0)
    # Without the background, the same charged Ewald sum depends on the
    # arbitrary real/reciprocal splitting parameter.
    assert abs(
        float((low_alpha - low_background) - (high_alpha - high_background))
    ) > 1e-3


def test_charged_mixed_fourier_replacement_is_alpha_independent():
    """Combined-minus-MM Fourier plus restored MM Ewald has one convention."""
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(1, 1.5, 1, (1.5, 3.0))
    box = 9.0
    qm_positions = jnp.asarray([[2.0, 3.0, 4.0], [2.8, 3.1, 4.0]])
    qm_density = jnp.asarray([[0.4, 0.1, -0.2, 0.05], [-0.1, 0.03, 0.04, -0.01]])
    mm_positions = jnp.asarray([[5.0, 4.0, 3.0]])
    mm_charges = jnp.asarray([0.7])
    cache = model.precompute(
        qm_positions,
        jnp.zeros(2, dtype=jnp.int32),
        jnp.eye(3)[None] * box,
    )
    qm_and_cross, _ = model.mixed_coulomb_energy(
        qm_density, mm_positions, mm_charges, cache
    )
    mm_low, _ = _single_charge_ewald(0.35, charge=0.7, box=box)
    mm_high, _ = _single_charge_ewald(0.65, charge=0.7, box=box)
    np.testing.assert_allclose(
        qm_and_cross[0] + mm_low,
        qm_and_cross[0] + mm_high,
        atol=2e-11,
        rtol=0,
    )


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
