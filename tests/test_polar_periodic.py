"""Reference parity for eager Fourier POLAR electrostatics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat

from mace_jax.modules.polar_electrostatics import FIELD_CONSTANT
from mace_jax.modules.polar_periodic import PeriodicPolarElectrostatics


@pytest.mark.parametrize(
    ('mode', 'flags'),
    [
        ('pbc', [True, True, True]),
        ('slab', [True, True, False]),
        ('molecule_in_box', [False, False, False]),
    ],
)
@pytest.mark.parametrize('include_energy_self_interaction', [True, False])
def test_periodic_fixed_coefficients_jit_energy_fields_and_gradients(
    mode, flags, include_energy_self_interaction
):
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(
        1,
        1.5,
        1,
        (1.5, 3.0),
        include_energy_self_interaction=include_energy_self_interaction,
    )
    positions = jnp.asarray([[0.3, 0.5, 0.8], [1.4, 1.2, 1.7]])
    cell = jnp.diag(jnp.asarray([7.0, 8.0, 9.0]))[None]
    density = jnp.asarray([[0.4, -0.1, 0.2, 0.3], [-0.3, 0.2, 0.1, -0.1]])
    batch = jnp.zeros(2, dtype=jnp.int32)
    pbc = jnp.asarray([flags])
    coefficients = model.prepare_coefficients(cell)

    def outputs(pos, current_cell):
        cache = model.precompute(pos, batch, current_cell, coefficients=coefficients)
        return (
            model.coulomb_energy(density, cache, mode=mode, pbc=pbc),
            model.field_features(density, cache, mode=mode, pbc=pbc),
        )

    eager = outputs(positions, cell)
    compiled = jax.jit(outputs)(positions, cell)
    for expected, actual in zip(eager, compiled):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)

    energy = lambda pos, current_cell: outputs(pos, current_cell)[0].sum()
    eager_grads = jax.grad(energy, argnums=(0, 1))(positions, cell)
    compiled_grads = jax.jit(jax.grad(energy, argnums=(0, 1)))(positions, cell)
    for expected, actual in zip(eager_grads, compiled_grads):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-10)


def test_periodic_field_energy_and_position_gradient():
    jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(torch.float64)
    positions = np.asarray([[0.3, 0.5, 0.8], [1.4, 1.2, 1.7], [2.0, 0.8, 2.4]])
    cell = np.diag([7.0, 8.0, 9.0])[None]
    density = np.asarray(
        [[0.4, -0.1, 0.2, 0.3], [-0.3, 0.2, 0.1, -0.1], [0.2, 0.1, -0.2, 0.1]]
    )
    widths = (1.5, 3.0)
    model = PeriodicPolarElectrostatics(1, 1.5, 1, widths)
    rcell = 2 * np.pi * np.linalg.inv(cell.swapaxes(-1, -2))
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        model.kspace_cutoff, torch.tensor(cell), torch.tensor(rcell)
    )
    cache = model.precompute(
        jnp.asarray(positions), jnp.zeros(3, dtype=jnp.int32), jnp.asarray(cell)
    )
    np.testing.assert_allclose(
        np.asarray(cache['k_vectors']), k_vectors.numpy(), atol=1e-12
    )

    torch_field = GTOElectrostaticFeatures(
        1, 1.5, 1, list(widths), False, model.kspace_cutoff, pbc_handling='pbc'
    )
    torch_energy = GTOElectrostaticEnergy(
        1, 1.5, model.kspace_cutoff, include_self_interaction=True, pbc_handling='pbc'
    )
    torch_pos = torch.tensor(positions, requires_grad=True)
    torch_density = torch.tensor(density)
    batch = torch.zeros(3, dtype=torch.long)
    volume = torch.tensor([np.linalg.det(cell[0])])
    pbc = torch.ones(1, 3, dtype=torch.bool)
    torch_fields = torch_field(
        k_vectors,
        k_norm2,
        k_batch,
        k0_mask,
        torch_density,
        torch_pos,
        batch,
        volume,
        pbc,
    )
    torch_coulomb = torch_energy(
        k_vectors,
        k_norm2,
        k_batch,
        k0_mask,
        torch_density,
        torch_pos,
        batch,
        volume,
        pbc,
    )
    torch_force = torch.autograd.grad(torch_coulomb.sum(), torch_pos)[0]
    jax_fields = model.field_features(jnp.asarray(density), cache)
    jax_coulomb = model.coulomb_energy(jnp.asarray(density), cache)
    jax_force = jax.grad(
        lambda p: model.coulomb_energy(
            jnp.asarray(density),
            model.precompute(p, jnp.zeros(3, dtype=jnp.int32), jnp.asarray(cell)),
        ).sum()
    )(jnp.asarray(positions))
    np.testing.assert_allclose(
        np.asarray(jax_fields), torch_fields.detach().numpy(), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(jax_coulomb), torch_coulomb.detach().numpy(), atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(jax_force), torch_force.detach().numpy(), atol=1e-8, rtol=1e-8
    )

    torch_cell = torch.tensor(cell, requires_grad=True)
    torch_rcell = 2 * torch.pi * torch.linalg.inv(torch_cell.transpose(-1, -2))
    strain_k, strain_k2, strain_k_batch, strain_k0 = compute_k_vectors_flat(
        model.kspace_cutoff, torch_cell, torch_rcell
    )
    strain_volume = torch.linalg.det(torch_cell)
    strain_energy = torch_energy(
        strain_k,
        strain_k2,
        strain_k_batch,
        strain_k0,
        torch_density,
        torch_pos.detach(),
        batch,
        strain_volume,
        pbc,
    )
    torch_cell_gradient = torch.autograd.grad(strain_energy.sum(), torch_cell)[0]
    jax_cell_gradient = jax.grad(
        lambda deformed_cell: model.coulomb_energy(
            jnp.asarray(density),
            model.precompute(
                jnp.asarray(positions),
                jnp.zeros(3, dtype=jnp.int32),
                deformed_cell,
                jnp.asarray(cell),
            ),
        ).sum()
    )(jnp.asarray(cell))
    np.testing.assert_allclose(
        np.asarray(jax_cell_gradient),
        torch_cell_gradient.detach().numpy(),
        atol=1e-8,
        rtol=1e-8,
    )

    for mode, flags in (
        ('slab', [True, True, False]),
        ('molecule_in_box', [False, False, False]),
        ('mixed_periodic', [True, True, False]),
    ):
        torch_field.set_pbc_handling(mode)
        torch_energy.set_pbc_handling(mode)
        torch_pbc = torch.tensor([flags])
        jax_pbc = jnp.asarray([flags])
        expected_field = torch_field(
            k_vectors,
            k_norm2,
            k_batch,
            k0_mask,
            torch_density,
            torch_pos,
            batch,
            volume,
            torch_pbc,
        )
        expected_energy = torch_energy(
            k_vectors,
            k_norm2,
            k_batch,
            k0_mask,
            torch_density,
            torch_pos,
            batch,
            volume,
            torch_pbc,
        )
        actual_field = model.field_features(
            jnp.asarray(density), cache, mode=mode, pbc=jax_pbc
        )
        actual_energy = model.coulomb_energy(
            jnp.asarray(density), cache, mode=mode, pbc=jax_pbc
        )
        np.testing.assert_allclose(
            np.asarray(actual_field),
            expected_field.detach().numpy(),
            atol=1e-8,
            rtol=1e-8,
        )
        np.testing.assert_allclose(
            np.asarray(actual_energy),
            expected_energy.detach().numpy(),
            atol=1e-8,
            rtol=1e-8,
        )


def test_mixed_periodic_batch_matches_torch():
    jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(torch.float64)
    positions = np.asarray(
        [
            [0.2, 0.4, 0.8],
            [1.2, 1.0, 1.8],
            [0.3, 0.7, 0.5],
            [1.5, 0.5, 1.4],
        ]
    )
    batch = np.asarray([0, 0, 1, 1])
    cell = np.asarray([np.diag([7.0, 8.0, 9.0]), np.diag([8.0, 9.0, 10.0])])
    pbc = np.asarray([[True, True, True], [True, True, False]])
    density = np.asarray(
        [
            [0.2, 0.1, -0.1, 0.2],
            [-0.3, 0.2, 0.1, -0.1],
            [0.3, -0.1, 0.2, 0.1],
            [-0.2, 0.1, -0.2, 0.2],
        ]
    )
    model = PeriodicPolarElectrostatics(1, 1.5, 1, (1.5, 3.0))
    rcell = 2 * np.pi * np.linalg.inv(cell.swapaxes(-1, -2))
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(
        model.kspace_cutoff, torch.tensor(cell), torch.tensor(rcell)
    )
    torch_field = GTOElectrostaticFeatures(
        1,
        1.5,
        1,
        [1.5, 3.0],
        False,
        model.kspace_cutoff,
        pbc_handling='mixed_periodic',
    )
    torch_energy = GTOElectrostaticEnergy(
        1,
        1.5,
        model.kspace_cutoff,
        include_self_interaction=True,
        pbc_handling='mixed_periodic',
    )
    args = (
        k_vectors,
        k_norm2,
        k_batch,
        k0_mask,
        torch.tensor(density),
        torch.tensor(positions),
        torch.tensor(batch),
        torch.tensor(np.linalg.det(cell)),
        torch.tensor(pbc),
    )
    expected_field = torch_field(*args).detach().numpy()
    expected_energy = torch_energy(*args).detach().numpy()
    cache = model.precompute(
        jnp.asarray(positions), jnp.asarray(batch), jnp.asarray(cell)
    )
    actual_field = model.field_features(
        jnp.asarray(density), cache, mode='mixed_periodic', pbc=jnp.asarray(pbc)
    )
    actual_energy = model.coulomb_energy(
        jnp.asarray(density), cache, mode='mixed_periodic', pbc=jnp.asarray(pbc)
    )
    np.testing.assert_allclose(
        np.asarray(actual_field), expected_field, atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        np.asarray(actual_energy), expected_energy, atol=1e-8, rtol=1e-8
    )


def test_generalized_pme_matches_direct_gaussian_multipoles_and_jit():
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(1, 1.5, 1, (1.5, 3.0))
    positions = jnp.asarray(
        [[0.3, 0.5, 0.8], [1.4, 1.2, 1.7], [2.0, 0.8, 2.4]]
    )
    density = jnp.asarray(
        [[0.4, -0.1, 0.2, 0.3], [-0.3, 0.2, 0.1, -0.1], [0.2, 0.1, -0.2, 0.1]]
    )
    cell = jnp.diag(jnp.asarray([7.0, 8.0, 9.0]))[None]
    batch = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    mesh_template, assignment_template = model.prepare_mesh_template(
        cell, mesh_spacing=0.4, assignment_order=8
    )

    def mesh_outputs(pos):
        cache = model.precompute_mesh(
            pos, batch, cell, mesh_template, assignment_template
        )
        return (
            model.mesh_coulomb_energy(density, cache),
            model.mesh_field_features(density, cache),
        )

    direct_cache = model.precompute(positions, batch, cell)
    direct_energy = model.coulomb_energy(density, direct_cache)
    direct_field = model.field_features(density, direct_cache)
    mesh_energy, mesh_field = jax.jit(mesh_outputs)(positions)
    np.testing.assert_allclose(mesh_energy, direct_energy, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(mesh_field, direct_field, atol=2e-9, rtol=2e-9)

    direct_gradient = jax.grad(
        lambda pos: model.coulomb_energy(
            density, model.precompute(pos, batch, cell)
        ).sum()
    )(positions)
    mesh_gradient = jax.jit(jax.grad(lambda pos: mesh_outputs(pos)[0].sum()))(
        positions
    )
    np.testing.assert_allclose(
        mesh_gradient, direct_gradient, atol=2e-8, rtol=2e-8
    )


def test_generalized_pme_matches_direct_point_charge_embedding_and_ewald():
    jax.config.update('jax_enable_x64', True)
    model = PeriodicPolarElectrostatics(1, 1.5, 1, (1.5, 3.0))
    positions = jnp.asarray([[0.3, 0.5, 0.8], [1.4, 1.2, 1.7]])
    density = jnp.asarray([[0.4, -0.1, 0.2, 0.3], [-0.3, 0.2, 0.1, -0.1]])
    mm_positions = jnp.asarray([[2.3, 1.5, 0.7], [3.1, 2.2, 1.9], [2.8, 3.0, 2.4]])
    mm_charges = jnp.asarray([-0.834, 0.417, 0.417])
    qm_point_charges = jnp.asarray([-0.7, -0.3])
    cell = jnp.diag(jnp.asarray([7.0, 8.0, 9.0]))[None]
    batch = jnp.zeros(positions.shape[0], dtype=jnp.int32)
    direct_cache = model.precompute(positions, batch, cell)
    mesh_template, assignment_template = model.prepare_mesh_template(
        cell, mesh_spacing=0.4, assignment_order=8
    )
    mesh_cache = model.precompute_mesh(
        positions, batch, cell, mesh_template, assignment_template
    )
    mm_density = model.mesh_point_charge_density(
        mm_positions, mm_charges, mesh_cache
    )

    direct_field = model.point_charge_field_features(
        mm_positions, mm_charges, direct_cache
    )
    direct_mixed, direct_cross = model.mixed_coulomb_energy(
        density, mm_positions, mm_charges, direct_cache
    )
    mesh_field = model.mesh_point_charge_field_features(mm_density, mesh_cache)
    mesh_mixed, mesh_cross = model.mesh_mixed_coulomb_energy(
        density, mm_density, mesh_cache
    )
    np.testing.assert_allclose(mesh_field, direct_field, atol=3e-8, rtol=3e-8)
    np.testing.assert_allclose(mesh_mixed, direct_mixed, atol=2e-9, rtol=2e-9)
    np.testing.assert_allclose(mesh_cross, direct_cross, atol=2e-9, rtol=2e-9)

    alpha = jnp.asarray(0.5)
    k_vectors = mesh_cache['k_vectors_mesh']
    k2 = mesh_cache['k_norm2_mesh']
    nonzero = k2 > 0
    kernel = jnp.where(nonzero, jnp.exp(-k2 / (4 * alpha**2)) / k2, 0.0)
    mm_phase = jnp.einsum('...d,nd->...n', k_vectors, mm_positions)
    qm_phase = jnp.einsum('...d,nd->...n', k_vectors, positions)
    mm_structure = jnp.sum(mm_charges * jnp.exp(-1j * mm_phase), axis=-1)
    qm_structure = jnp.sum(qm_point_charges * jnp.exp(-1j * qm_phase), axis=-1)
    volume = jnp.linalg.det(cell[0])
    coulomb = FIELD_CONSTANT / (4 * np.pi)
    direct_mm_ewald = (
        coulomb * 2 * np.pi / volume
        * jnp.sum(jnp.abs(mm_structure) ** 2 * kernel)
    )
    direct_cross_ewald = (
        coulomb * 4 * np.pi / volume
        * jnp.sum(jnp.real(qm_structure * jnp.conj(mm_structure)) * kernel)
    )
    mesh_mm_ewald = model.mesh_point_charge_ewald_reciprocal_energy(
        mm_density, mesh_cache, alpha
    )
    qm_density = model.mesh_point_charge_density(
        positions, qm_point_charges, mesh_cache
    )
    mesh_cross_ewald = model.mesh_point_charge_ewald_cross_energy(
        qm_density, mm_density, mesh_cache, alpha
    )
    np.testing.assert_allclose(
        mesh_mm_ewald, direct_mm_ewald, atol=3e-8, rtol=3e-8
    )
    np.testing.assert_allclose(
        mesh_cross_ewald, direct_cross_ewald, atol=3e-8, rtol=3e-8
    )

    def direct_mixed_energy(qm_pos, mm_pos):
        cache = model.precompute(qm_pos, batch, cell)
        return model.mixed_coulomb_energy(
            density, mm_pos, mm_charges, cache
        )[0].sum()

    def mesh_mixed_energy(qm_pos, mm_pos):
        cache = model.precompute_mesh(
            qm_pos, batch, cell, mesh_template, assignment_template
        )
        current_mm_density = model.mesh_point_charge_density(
            mm_pos, mm_charges, cache
        )
        return model.mesh_mixed_coulomb_energy(
            density, current_mm_density, cache
        )[0].sum()

    direct_gradients = jax.grad(direct_mixed_energy, argnums=(0, 1))(
        positions, mm_positions
    )
    mesh_gradients = jax.jit(jax.grad(mesh_mixed_energy, argnums=(0, 1)))(
        positions, mm_positions
    )
    for actual, expected in zip(mesh_gradients, direct_gradients, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-7)
