"""Numerical parity of open-boundary POLAR electrostatics with graph_longrange."""

import jax
import numpy as np
import torch
from graph_longrange.gto_utils import (
    DisplacedGTOExternalFieldBlock as TorchExternalField,
)
from graph_longrange.realspace_electrostatics import (
    RealSpaceFiniteDifferenceElectrostaticFeatures,
    RealSpaceFiniteDiffereneEnergy,
)

from mace_jax.modules.polar_electrostatics import (
    DisplacedGTOExternalFieldBlock,
    RealSpacePolarElectrostatics,
)


def test_realspace_field_and_energy_match_torch():
    jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(4, 3)) * 1.5
    source = rng.normal(size=(4, 4))
    batch = np.array([0, 0, 1, 1], dtype=np.int64)
    widths = (1.5, 3.0)
    torch_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
        density_max_l=1,
        density_smearing_width=1.5,
        projection_max_l=1,
        projection_smearing_widths=list(widths),
        include_self_interaction=False,
        integral_normalization='receiver',
    )
    torch_energy = RealSpaceFiniteDiffereneEnergy(
        density_max_l=1,
        density_smearing_width=1.5,
        include_self_interaction=True,
    )
    jax_electrostatics = RealSpacePolarElectrostatics(
        density_max_l=1,
        density_width=1.5,
        feature_max_l=1,
        feature_widths=widths,
        include_field_self_interaction=False,
        include_energy_self_interaction=True,
    )
    source_t = torch.as_tensor(source)
    positions_t = torch.as_tensor(positions).requires_grad_(True)
    batch_t = torch.as_tensor(batch)
    expected_features = torch_features(source_t, positions_t, batch_t)[0]
    expected_energy = torch_energy(source_t, positions_t, batch_t)
    actual_features = jax_electrostatics.field_features(
        jax.numpy.asarray(source),
        jax.numpy.asarray(positions),
        jax.numpy.asarray(batch),
    )
    actual_energy = jax_electrostatics.coulomb_energy(
        jax.numpy.asarray(source),
        jax.numpy.asarray(positions),
        jax.numpy.asarray(batch),
        num_graphs=2,
    )
    np.testing.assert_allclose(
        np.asarray(actual_features),
        expected_features.detach().numpy(),
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual_energy),
        expected_energy.detach().numpy(),
        atol=1e-9,
        rtol=1e-9,
    )
    torch_gradient = torch.autograd.grad(expected_energy.sum(), positions_t)[0]
    jax_gradient = jax.grad(
        lambda pos: jax_electrostatics.coulomb_energy(
            jax.numpy.asarray(source), pos, jax.numpy.asarray(batch), num_graphs=2
        ).sum()
    )(jax.numpy.asarray(positions))
    np.testing.assert_allclose(
        np.asarray(jax_gradient),
        torch_gradient.detach().numpy(),
        atol=1e-9,
        rtol=1e-9,
    )


def test_external_field_projection_matches_torch():
    jax.config.update('jax_enable_x64', True)
    torch.set_default_dtype(torch.float64)
    positions = np.asarray([[0.2, -0.1, 0.3], [-0.4, 0.5, 0.1]])
    batch = np.asarray([0, 0], dtype=np.int64)
    field = np.asarray([[0.0, 0.01, -0.02, 0.03]])
    reference = TorchExternalField(1, [1.5, 3.0], 'receiver')
    candidate = DisplacedGTOExternalFieldBlock(1, (1.5, 3.0))
    expected = reference(
        torch.as_tensor(batch), torch.as_tensor(positions), torch.as_tensor(field)
    )
    actual = candidate(*map(jax.numpy.asarray, (batch, positions, field)))
    np.testing.assert_allclose(
        np.asarray(actual), expected.detach().numpy(), atol=1e-12, rtol=1e-12
    )
