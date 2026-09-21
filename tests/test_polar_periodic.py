"""Reference parity for eager Fourier POLAR electrostatics."""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.kspace import compute_k_vectors_flat

from mace_jax.modules.polar_periodic import PeriodicPolarElectrostatics


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
