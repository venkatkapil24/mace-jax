"""Native JAX building blocks for the MACE-POLAR response model."""

from __future__ import annotations

import math

import jax.nn as jnn
import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.e3nn import nn
from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.modules.irreps_tools import tp_out_irreps_with_instructions
from mace_jax.modules.radial import RadialMLP
from mace_jax.modules.wrapper_ops import Linear


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class MultiLayerFeatureMixer(nnx.Module):
    """Learned equivariant sum of the interaction layer features."""

    def __init__(
        self,
        node_feats_irreps: Irreps,
        num_interactions: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.linears = nnx.List(
            [
                Linear(node_feats_irreps, node_feats_irreps, rngs=rngs)
                for _ in range(num_interactions)
            ]
        )

    def __call__(self, all_node_feats: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(all_node_feats[0])
        for i, linear in enumerate(self.linears):
            out = out + linear(all_node_feats[i])
        return out


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class EnvironmentDependentSpinSourceBlock(nnx.Module):
    """Predict spin resolved Gaussian multipoles from local features."""

    def __init__(
        self,
        irreps_in: Irreps,
        max_l: int,
        *,
        zero_charges: bool = False,
        rngs: nnx.Rngs,
    ) -> None:
        self.zero_charges = zero_charges
        # Torch's ``2 * irreps`` repeats the list. e3nn-jax instead doubles
        # each multiplicity, which changes the checkpoint's channel order.
        multipoles = Irreps.spherical_harmonics(max_l)
        self.irreps_out = Irreps(list(multipoles) * 2)
        self.linear = Linear(irreps_in, self.irreps_out, rngs=rngs)

    def __call__(self, node_feats: jnp.ndarray) -> jnp.ndarray:
        multipoles = self.linear(node_feats)
        if self.zero_charges:
            multipoles = multipoles.at[:, 0].set(0)
        return multipoles[:, None, :]


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class SparseUvuTensorProduct(nnx.Module):
    """Sparse POLAR tensor product with Torch's flattened weight order.

    The released model uses only invariant contractions and scalar modulation.
    Both paths have one output channel per first-input channel.
    """

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        instructions: list[tuple[int, int, int, str, bool]],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del rngs
        self.irreps_in1 = Irreps(irreps_in1)
        self.irreps_in2 = Irreps(irreps_in2)
        self.irreps_out = Irreps(irreps_out)
        self.instructions = tuple(instructions)
        in1_slices = self.irreps_in1.slices()
        in2_slices = self.irreps_in2.slices()
        out_slices = self.irreps_out.slices()

        path_counts = {}
        for i1, i2, io, mode, trainable in instructions:
            if mode != 'uvu' or not trainable:
                raise ValueError(
                    'POLAR sparse tensor product requires weighted uvu paths'
                )
            path_counts[io] = path_counts.get(io, 0) + self.irreps_in2[i2].mul

        paths = []
        offset = 0
        for i1, i2, io, _, _ in instructions:
            mul1, ir1 = self.irreps_in1[i1]
            mul2, ir2 = self.irreps_in2[i2]
            mul_out, ir_out = self.irreps_out[io]
            if mul_out != mul1:
                raise ValueError('uvu output multiplicity must match the first input')
            if ir_out.dim == 1 and ir1.dim == ir2.dim:
                mode = 'dot'
            elif ir2.dim == 1 and ir_out.dim == ir1.dim:
                mode = 'scale'
            else:
                raise ValueError('Unsupported POLAR sparse tensor product path')
            size = mul1 * mul2
            # e3nn TensorProduct defaults to component/element normalization.
            path_weight = math.sqrt(ir_out.dim / path_counts[io])
            paths.append(
                (
                    in1_slices[i1],
                    in2_slices[i2],
                    out_slices[io],
                    offset,
                    offset + size,
                    path_weight,
                    mode,
                    mul1,
                    mul2,
                    ir1.dim,
                )
            )
            offset += size
        self._paths = tuple(paths)
        self.weight = nnx.Param(jnp.zeros((offset,)))
        self.output_mask = nnx.Param(jnp.ones((self.irreps_out.dim,)))

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        if x1.ndim != 2 or x2.ndim != 2:
            raise ValueError('Expected flattened [batch, irreps.dim] inputs')
        batch = x1.shape[0]
        out = jnp.zeros((batch, self.irreps_out.dim), dtype=x1.dtype)
        for (
            slice1,
            slice2,
            slice_out,
            start,
            stop,
            path_weight,
            mode,
            mul1,
            mul2,
            dim1,
        ) in self._paths:
            first = x1[:, slice1].reshape(batch, mul1, dim1)
            weights = self.weight[start:stop].reshape(mul1, mul2)
            if mode == 'dot':
                second = x2[:, slice2].reshape(batch, mul2, dim1)
                mixed = jnp.einsum('uv,bvd->bud', weights, second)
                contribution = jnp.sum(first * mixed, axis=-1) / math.sqrt(dim1)
            else:
                second = x2[:, slice2].reshape(batch, mul2)
                mixed = jnp.einsum('bv,uv->bu', second, weights)
                contribution = (first * mixed[:, :, None] / math.sqrt(dim1)).reshape(
                    batch, -1
                )
            out = out.at[:, slice_out].add(path_weight * contribution)
        return out * self.output_mask


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticChargeBiasedLinearPotentialEmbedding(nnx.Module):
    """Combine the local feature, electrostatic potential, and multipoles."""

    def __init__(
        self,
        potential_irreps: Irreps,
        node_feats_irreps: Irreps,
        charges_irreps: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.potential_linear = Linear(potential_irreps, node_feats_irreps, rngs=rngs)
        self.node_feats_linear = Linear(node_feats_irreps, node_feats_irreps, rngs=rngs)
        self.charge_embedding = Linear(charges_irreps, node_feats_irreps, rngs=rngs)

    def __call__(
        self,
        potential_feats: jnp.ndarray,
        node_feats: jnp.ndarray,
        node_attrs: jnp.ndarray,
        local_charges: jnp.ndarray,
    ) -> jnp.ndarray:
        del node_attrs
        return (
            self.potential_linear(potential_feats)
            + self.node_feats_linear(node_feats)
            + self.charge_embedding(local_charges)
        )


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class GeneralNonLinearBiasReadoutBlock(nnx.Module):
    """Gated multipole readout with scalar biases at its last two maps."""

    def __init__(
        self,
        irreps_in: Irreps,
        MLP_irreps: Irreps,
        irreps_out: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_irreps = Irreps(MLP_irreps)
        self.irreps_out = Irreps(irreps_out)
        scalars = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l == 0 and ir in self.irreps_out
            ]
        )
        gated = Irreps(
            [
                (mul, ir)
                for mul, ir in self.hidden_irreps
                if ir.l > 0 and ir in self.irreps_out
            ]
        )
        gates = Irreps([(mul, '0e') for mul, _ in gated])
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=scalars,
            act_scalars=[jnn.silu for _ in scalars],
            irreps_gates=gates,
            act_gates=[jnn.sigmoid for _ in gates],
            irreps_gated=gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(irreps_in, self.irreps_nonlin, rngs=rngs)
        self.linear_mid = Linear(
            self.hidden_irreps, self.irreps_nonlin, biases=True, rngs=rngs
        )
        self.linear_2 = Linear(
            self.hidden_irreps, self.irreps_out, biases=True, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.equivariant_nonlin(self.linear_1(x))
        x = self.equivariant_nonlin(self.linear_mid(x))
        out = self.linear_2(x)
        return getattr(out, 'array', out)


def _sparse_dot_instructions(
    first: Irreps, second: Irreps, output: Irreps
) -> list[tuple[int, int, int, str, bool]]:
    _, instructions = tp_out_irreps_with_instructions(first, second, output)
    return [
        (i1, i2, 0, mode, trainable) for i1, i2, _io, mode, trainable in instructions
    ]


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class AgnosticEmbeddedOneBodyVariableUpdate(nnx.Module):
    """One POLAR fixed-point multipole and Fukui update."""

    def __init__(
        self,
        node_attrs_irreps: Irreps,
        node_feats_irreps: Irreps,
        potential_irreps: Irreps,
        charges_irreps: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.node_feats_irreps = Irreps(node_feats_irreps)
        self.charges_irreps = Irreps(charges_irreps)
        invar = Irreps(f'{self.node_feats_irreps.count("0e")}x0e')
        self.potential_embedding = AgnosticChargeBiasedLinearPotentialEmbedding(
            potential_irreps,
            self.node_feats_irreps,
            self.charges_irreps,
            rngs=rngs,
        )
        self.source_embedding = Linear(node_attrs_irreps, invar, rngs=rngs)
        self.dot_products = SparseUvuTensorProduct(
            self.node_feats_irreps,
            self.node_feats_irreps,
            invar,
            _sparse_dot_instructions(
                self.node_feats_irreps, self.node_feats_irreps, invar
            ),
        )
        self.nonlinearity = RadialMLP([2 * invar.dim, 64, 64, 64, invar.dim], rngs=rngs)
        _, output_instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, invar, self.node_feats_irreps
        )
        self.tp_out = SparseUvuTensorProduct(
            self.node_feats_irreps,
            invar,
            self.node_feats_irreps,
            output_instructions,
        )
        basis = Irreps.spherical_harmonics(self.charges_irreps.lmax)
        mlp_irreps = Irreps(list(basis) * 32).sort()[0].simplify()
        self.readout = GeneralNonLinearBiasReadoutBlock(
            self.node_feats_irreps,
            mlp_irreps,
            self.charges_irreps + Irreps('2x0e'),
            rngs=rngs,
        )

    def __call__(
        self,
        node_attrs: jnp.ndarray,
        node_feats: jnp.ndarray,
        potential_features: jnp.ndarray,
        local_charges: jnp.ndarray,
    ) -> jnp.ndarray:
        mixed = self.potential_embedding(
            potential_features, node_feats, node_attrs, local_charges
        )
        invariants = self.dot_products(node_feats, mixed)
        source = self.source_embedding(node_attrs)
        nonlin = self.nonlinearity(jnp.concatenate((invariants, source), axis=-1))
        new_feats = self.tp_out(node_feats, nonlin)
        return self.readout(new_feats)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
class OneBodyMLPFieldReadout(nnx.Module):
    """Post-response local electron energy readout."""

    def __init__(
        self,
        node_feats_irreps: Irreps,
        potential_irreps: Irreps,
        charges_irreps: Irreps,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        node_feats_irreps = Irreps(node_feats_irreps)
        invar = Irreps(f'{node_feats_irreps.count("0e")}x0e')
        self.linear_up_q = Linear(
            charges_irreps, node_feats_irreps, biases=True, rngs=rngs
        )
        self.linear_up_v = Linear(
            potential_irreps, node_feats_irreps, biases=True, rngs=rngs
        )
        dot_instructions = _sparse_dot_instructions(
            node_feats_irreps, node_feats_irreps, invar
        )
        self.dot_products_q = SparseUvuTensorProduct(
            node_feats_irreps, node_feats_irreps, invar, dot_instructions
        )
        self.dot_products_v = SparseUvuTensorProduct(
            node_feats_irreps, node_feats_irreps, invar, dot_instructions
        )
        self.mlp = RadialMLP([2 * invar.dim, 128, 128, 128, 1], rngs=rngs)

    def __call__(
        self,
        node_feats: jnp.ndarray,
        field_feats: jnp.ndarray,
        charges_0: jnp.ndarray,
        charges_induced: jnp.ndarray,
    ) -> jnp.ndarray:
        q_up = self.linear_up_q(charges_0 + charges_induced)
        v_up = self.linear_up_v(field_feats)
        invar = jnp.concatenate(
            (
                self.dot_products_q(node_feats, q_up),
                self.dot_products_v(node_feats, v_up),
            ),
            axis=-1,
        )
        return self.mlp(invar).squeeze(-1)
