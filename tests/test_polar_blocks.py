"""Parity checks for the native POLAR feature blocks."""

import jax
import numpy as np
import torch
from e3nn import o3
from e3nn.math._normalize_activation import normalize2mom as torch_normalize2mom
from e3nn_jax import Irreps
from flax import nnx
from mace.modules.blocks import GeneralNonLinearBiasReadoutBlock as TorchGeneralReadout
from mace.modules.blocks import NonLinearBiasReadoutBlock as TorchBiasReadout
from mace.modules.field_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding as TorchPotentialEmbedding,
)
from mace.modules.field_blocks import (
    AgnosticEmbeddedOneBodyVariableUpdate as TorchFieldUpdate,
)
from mace.modules.field_blocks import (
    EnvironmentDependentSpinSourceBlock as TorchSpinSource,
)
from mace.modules.field_blocks import MultiLayerFeatureMixer as TorchMixer
from mace.modules.field_blocks import OneBodyMLPFieldReadout as TorchFieldReadout
from mace.modules.field_blocks import SparseUvuTensorProduct as TorchSparseUvu
from mace.modules.field_blocks import instructions_for_sparse_tp

from mace_jax.adapters.cuequivariance.linear import Linear
from mace_jax.adapters.e3nn.math import register_normalize2mom_const
from mace_jax.adapters.nnx.torch import init_from_torch
from mace_jax.modules.blocks import NonLinearBiasReadoutBlock
from mace_jax.modules.polar_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding,
    AgnosticEmbeddedOneBodyVariableUpdate,
    EnvironmentDependentSpinSourceBlock,
    GeneralNonLinearBiasReadoutBlock,
    MultiLayerFeatureMixer,
    OneBodyMLPFieldReadout,
    SparseUvuTensorProduct,
)


def test_repeated_field_irreps_linear_matches_torch():
    jax.config.update('jax_enable_x64', True)
    irreps_in = '2x0e+2x1o+2x0e+2x1o'
    irreps_out = '4x0e+4x1o'
    source = o3.Linear(irreps_in, irreps_out).double()
    target = Linear(Irreps(irreps_in), Irreps(irreps_out), rngs=nnx.Rngs(0))
    target, _ = init_from_torch(target, source)
    values = np.random.default_rng(42).normal(size=(3, Irreps(irreps_in).dim))
    actual = np.asarray(target(jax.numpy.asarray(values)))
    expected = source(torch.as_tensor(values)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_spin_source_matches_torch():
    jax.config.update('jax_enable_x64', True)
    irreps = '4x0e+3x1o'
    source = TorchSpinSource(o3.Irreps(irreps), max_l=1).double()
    target = EnvironmentDependentSpinSourceBlock(
        Irreps(irreps), max_l=1, rngs=nnx.Rngs(0)
    )
    target, _ = init_from_torch(target, source)
    features = np.random.default_rng(1).normal(size=(3, 13))
    actual = np.asarray(target(jax.numpy.asarray(features)))
    expected = source(torch.as_tensor(features)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_feature_mixer_matches_torch():
    jax.config.update('jax_enable_x64', True)
    irreps = '4x0e+3x1o'
    source = TorchMixer(o3.Irreps(irreps), num_interactions=2).double()
    target = MultiLayerFeatureMixer(
        Irreps(irreps), num_interactions=2, rngs=nnx.Rngs(0)
    )
    target, _ = init_from_torch(target, source)
    features = np.random.default_rng(2).normal(size=(2, 3, 13))
    actual = np.asarray(target(jax.numpy.asarray(features)))
    expected = source(torch.as_tensor(features)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_sparse_uvu_matches_torch():
    jax.config.update('jax_enable_x64', True)
    in_irreps = o3.Irreps('3x0e+3x1o')
    out_irreps = o3.Irreps('3x0e')
    instructions = instructions_for_sparse_tp(in_irreps, in_irreps, out_irreps)
    source = TorchSparseUvu(in_irreps, in_irreps, out_irreps, instructions).double()
    target = SparseUvuTensorProduct(
        Irreps(str(in_irreps)),
        Irreps(str(in_irreps)),
        Irreps(str(out_irreps)),
        instructions,
    )
    target, _ = init_from_torch(target, source)
    rng = np.random.default_rng(3)
    first = rng.normal(size=(3, in_irreps.dim))
    second = rng.normal(size=(3, in_irreps.dim))
    actual = np.asarray(target(jax.numpy.asarray(first), jax.numpy.asarray(second)))
    expected = source(torch.as_tensor(first), torch.as_tensor(second)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_potential_embedding_matches_torch():
    jax.config.update('jax_enable_x64', True)
    potential_irreps = '2x0e+1x1o'
    node_irreps = '3x0e+2x1o'
    charges_irreps = '1x0e+1x1o+1x0e+1x1o'
    source = TorchPotentialEmbedding(
        o3.Irreps(potential_irreps),
        o3.Irreps(node_irreps),
        o3.Irreps('2x0e'),
        charges_irreps=o3.Irreps(charges_irreps),
    ).double()
    target = AgnosticChargeBiasedLinearPotentialEmbedding(
        Irreps(potential_irreps),
        Irreps(node_irreps),
        Irreps(charges_irreps),
        rngs=nnx.Rngs(0),
    )
    target, _ = init_from_torch(target, source)
    rng = np.random.default_rng(4)
    potential = rng.normal(size=(3, 5))
    node = rng.normal(size=(3, 9))
    attrs = rng.normal(size=(3, 2))
    charges = rng.normal(size=(3, 8))
    actual = np.asarray(
        target(*[jax.numpy.asarray(x) for x in (potential, node, attrs, charges)])
    )
    expected = source(*[torch.as_tensor(x) for x in (potential, node, attrs, charges)])
    np.testing.assert_allclose(
        actual, expected.detach().numpy(), atol=1e-10, rtol=1e-10
    )


def test_general_readout_matches_torch():
    jax.config.update('jax_enable_x64', True)
    register_normalize2mom_const(
        'silu', float(torch_normalize2mom(torch.nn.functional.silu).cst)
    )
    register_normalize2mom_const(
        'sigmoid', float(torch_normalize2mom(torch.sigmoid).cst)
    )
    input_irreps = '5x0e+4x1o'
    mlp_irreps = '3x0e+3x1o'
    output_irreps = '1x0e+1x1o+2x0e'
    source = TorchGeneralReadout(
        o3.Irreps(input_irreps),
        o3.Irreps(mlp_irreps),
        torch.nn.functional.silu,
        irreps_out=o3.Irreps(output_irreps),
    ).double()
    target = GeneralNonLinearBiasReadoutBlock(
        Irreps(input_irreps),
        Irreps(mlp_irreps),
        Irreps(output_irreps),
        rngs=nnx.Rngs(0),
    )
    target, _ = init_from_torch(target, source)
    features = np.random.default_rng(5).normal(size=(3, 17))
    actual = np.asarray(target(jax.numpy.asarray(features)))
    expected = source(torch.as_tensor(features)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def test_fukui_readout_matches_torch():
    jax.config.update('jax_enable_x64', True)
    register_normalize2mom_const(
        'silu', float(torch_normalize2mom(torch.nn.functional.silu).cst)
    )
    irreps = '5x0e+4x1o'
    source = TorchBiasReadout(
        o3.Irreps(irreps),
        o3.Irreps('16x0e'),
        torch.nn.functional.silu,
        o3.Irreps('2x0e'),
    ).double()
    target = NonLinearBiasReadoutBlock(
        Irreps(irreps),
        Irreps('16x0e'),
        jax.nn.silu,
        Irreps('2x0e'),
        rngs=nnx.Rngs(0),
    )
    target, _ = init_from_torch(target, source)
    features = np.random.default_rng(42).normal(size=(3, 17))
    actual = np.asarray(target(jax.numpy.asarray(features)))
    expected = source(torch.as_tensor(features)).detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


def _field_block_config():
    return dict(
        node_attrs_irreps=o3.Irreps('2x0e'),
        node_feats_irreps=o3.Irreps('3x0e+3x1o'),
        edge_attrs_irreps=o3.Irreps.spherical_harmonics(2),
        edge_feats_irreps=o3.Irreps('4x0e'),
        target_irreps=o3.Irreps('3x0e'),
        hidden_irreps=o3.Irreps('3x0e+3x1o'),
        avg_num_neighbors=1.0,
        potential_irreps=o3.Irreps('2x0e+2x1o'),
        charges_irreps=o3.Irreps('1x0e+1x1o+1x0e+1x1o'),
    )


def test_field_update_matches_torch():
    jax.config.update('jax_enable_x64', True)
    register_normalize2mom_const(
        'silu', float(torch_normalize2mom(torch.nn.functional.silu).cst)
    )
    register_normalize2mom_const(
        'sigmoid', float(torch_normalize2mom(torch.sigmoid).cst)
    )
    config = _field_block_config()
    source = TorchFieldUpdate(**config, field_norm_factor=1.0).double()
    target = AgnosticEmbeddedOneBodyVariableUpdate(
        Irreps(str(config['node_attrs_irreps'])),
        Irreps(str(config['node_feats_irreps'])),
        Irreps(str(config['potential_irreps'])),
        Irreps(str(config['charges_irreps'])),
        rngs=nnx.Rngs(0),
    )
    target, _ = init_from_torch(target, source)
    rng = np.random.default_rng(6)
    attrs = rng.normal(size=(3, 2))
    node = rng.normal(size=(3, 12))
    potential = rng.normal(size=(3, 8))
    charges = rng.normal(size=(3, 8))
    actual = np.asarray(
        target(*map(jax.numpy.asarray, (attrs, node, potential, charges)))
    )
    edge_attrs = torch.zeros((0, 9), dtype=torch.float64)
    edge_feats = torch.zeros((0, 4), dtype=torch.float64)
    edge_index = torch.zeros((2, 0), dtype=torch.int64)
    expected = (
        source(
            torch.as_tensor(attrs),
            torch.as_tensor(node),
            edge_attrs,
            edge_feats,
            edge_index,
            torch.as_tensor(potential),
            torch.as_tensor(charges),
        )
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)


def test_field_readout_matches_torch():
    jax.config.update('jax_enable_x64', True)
    config = _field_block_config()
    source = TorchFieldReadout(**config).double()
    target = OneBodyMLPFieldReadout(
        Irreps(str(config['node_feats_irreps'])),
        Irreps(str(config['potential_irreps'])),
        Irreps(str(config['charges_irreps'])),
        rngs=nnx.Rngs(0),
    )
    target, _ = init_from_torch(target, source)
    rng = np.random.default_rng(8)
    attrs = rng.normal(size=(3, 2))
    node = rng.normal(size=(3, 12))
    potential = rng.normal(size=(3, 8))
    charges_0 = rng.normal(size=(3, 8))
    charges_induced = rng.normal(size=(3, 8))
    actual = np.asarray(
        target(
            *map(
                jax.numpy.asarray,
                (
                    node,
                    potential,
                    charges_0,
                    charges_induced,
                ),
            )
        )
    )
    expected = (
        source(
            torch.as_tensor(attrs),
            torch.as_tensor(node),
            torch.zeros((0, 9), dtype=torch.float64),
            torch.zeros((0, 4), dtype=torch.float64),
            torch.zeros((2, 0), dtype=torch.int64),
            torch.as_tensor(potential),
            torch.as_tensor(charges_0),
            torch.as_tensor(charges_induced),
        )
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)
