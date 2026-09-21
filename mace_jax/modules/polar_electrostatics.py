"""Differentiable eager JAX real-space POLAR electrostatics.

The finite-difference multipole representation and Gaussian smoothing follow
graph_longrange's open-boundary evaluator used by POLAR-1-M.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf
from scipy import integrate, special

from mace_jax.tools.scatter import scatter_sum

FIELD_CONSTANT = 1.0 / (5.526349406e-3)


def _cl_sigma(ell: int, sigma: float, normalization: str) -> float:
    if normalization == 'multipoles':
        denominator = (
            math.sqrt(4 * math.pi / (2 * ell + 1))
            * 2 ** ((2 * ell + 1) / 2)
            * special.gamma((2 * ell + 3) / 2)
            * sigma ** (2 * ell + 3)
        )
    elif normalization == 'receiver':
        denominator = (
            2 ** ((ell + 1) / 2) * special.gamma((ell + 3) / 2) * sigma ** (ell + 3)
        )
    else:
        raise ValueError(f'Unsupported GTO normalization {normalization!r}')
    return 1.0 / denominator


def _self_interaction_constants(
    source_max_l: int,
    source_width: float,
    feature_max_l: int,
    feature_widths: tuple[float, ...],
    *,
    source_normalization: str = 'multipoles',
    feature_normalization: str = 'receiver',
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute fixed GTO overlap factors by the reference quadrature."""
    indices = []
    values = []
    for ell in range(min(source_max_l, feature_max_l) + 1):
        for width in feature_widths:
            radius = np.linspace(0.0001, 10 * max(width, source_width), 10000)
            lower_gamma = special.gammainc(
                (2 * ell + 3) / 2, 0.5 * radius**2 / width**2
            ) * special.gamma((2 * ell + 3) / 2)
            f1 = (
                2 ** (ell + 0.5)
                * width ** (2 * ell + 3)
                * lower_gamma
                * radius ** (-(ell + 1))
            )
            f2 = width**2 * radius**ell * np.exp(-0.5 * radius**2 / width**2)
            integrand = (
                radius ** (ell + 2)
                * np.exp(-0.5 * radius**2 / source_width**2)
                * (f1 + f2)
            )
            value = (
                FIELD_CONSTANT
                / (2 * ell + 1)
                * _cl_sigma(ell, source_width, source_normalization)
                * _cl_sigma(ell, width, feature_normalization)
                * integrate.trapezoid(integrand, x=radius)
            )
            for m in range(2 * ell + 1):
                indices.append(ell**2 + m)
                values.append(value)
    return jnp.asarray(indices, dtype=jnp.int32), jnp.asarray(values)


def _displaced_charges(
    multipoles: jnp.ndarray,
    positions: jnp.ndarray,
    offset: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    vector_charges = multipoles[:, jnp.asarray([3, 1, 2])] / offset
    central_charge = multipoles[:, 0] - jnp.sum(vector_charges, axis=-1)
    charges = jnp.concatenate((central_charge[:, None], vector_charges), axis=-1)
    shifts = jnp.asarray(
        [[0.0, 0.0, 0.0], [offset, 0.0, 0.0], [0.0, offset, 0.0], [0.0, 0.0, offset]],
        dtype=positions.dtype,
    )
    displaced = positions[:, None, :] + shifts[None, :, :]
    return charges.reshape(-1), displaced.reshape(-1, 3)


def _pair_kernel(
    positions: jnp.ndarray,
    batch: jnp.ndarray,
    widths: jnp.ndarray,
    duplicate_count: int,
) -> jnp.ndarray:
    n_total = positions.shape[0]
    ids = jnp.arange(n_total) // duplicate_count
    displacement = positions[None, :, :] - positions[:, None, :]
    valid = (ids[:, None] != ids[None, :]) & (batch[:, None] == batch[None, :])
    squared = jnp.sum(displacement * displacement, axis=-1)
    # Keep the excluded self pairs away from sqrt(0), including in reverse AD.
    distances = jnp.sqrt(jnp.where(valid & (squared > 0), squared, 1.0))
    radial = erf(0.5 * distances[..., None] / widths) / (distances[..., None] + 1e-6)
    return jnp.where(valid[..., None], radial, 0.0)


class RealSpacePolarElectrostatics:
    """Open-boundary field features and Coulomb energy for l<=1 multipoles."""

    def __init__(
        self,
        density_max_l: int,
        density_width: float,
        feature_max_l: int,
        feature_widths: tuple[float, ...],
        *,
        include_field_self_interaction: bool = False,
        include_energy_self_interaction: bool = True,
    ) -> None:
        if density_max_l not in (0, 1) or feature_max_l not in (0, 1):
            raise ValueError('Real-space POLAR evaluator supports l<=1')
        self.density_max_l = density_max_l
        self.feature_max_l = feature_max_l
        self.density_width = density_width
        self.feature_widths = tuple(feature_widths)
        self.include_field_self_interaction = include_field_self_interaction
        self.include_energy_self_interaction = include_energy_self_interaction
        self.feature_offset = 0.1
        self.energy_offset = 0.02
        self.total_widths = jnp.asarray(
            [
                math.sqrt((density_width**2 + width**2) / 2)
                for width in self.feature_widths
            ]
        )
        self.l0_factors = jnp.asarray(
            [
                _cl_sigma(0, width, 'receiver') / _cl_sigma(0, width, 'multipoles')
                for width in self.feature_widths
            ]
        )
        self.l1_factors = jnp.asarray(
            [
                math.sqrt(3)
                * width**2
                * _cl_sigma(1, width, 'receiver')
                / _cl_sigma(0, width, 'multipoles')
                / self.feature_offset
                for width in self.feature_widths
            ]
        )
        if include_field_self_interaction:
            self.field_self_indices, self.field_self_values = (
                _self_interaction_constants(
                    density_max_l,
                    density_width,
                    feature_max_l,
                    self.feature_widths,
                )
            )
        if include_energy_self_interaction:
            self.energy_self_indices, self.energy_self_values = (
                _self_interaction_constants(
                    density_max_l,
                    density_width,
                    density_max_l,
                    (density_width,),
                    feature_normalization='multipoles',
                )
            )

    def field_features(
        self,
        source_feats: jnp.ndarray,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
    ) -> jnp.ndarray:
        n_nodes = positions.shape[0]
        radial_count = len(self.feature_widths)
        if self.density_max_l == 0 and self.feature_max_l == 0:
            kernel = _pair_kernel(positions, batch, self.total_widths, 1)
            scalar = jnp.einsum('rsw,s->rw', kernel, source_feats[:, 0])
            features = scalar * self.l0_factors
        else:
            if self.density_max_l == 0:
                source_feats = jnp.pad(source_feats, ((0, 0), (0, 3)))
            charges, displaced = _displaced_charges(
                source_feats, positions, self.feature_offset
            )
            expanded_batch = jnp.repeat(batch, 4)
            kernel = _pair_kernel(displaced, expanded_batch, self.total_widths, 4)
            scalar = jnp.einsum('rsw,s->rw', kernel, charges)
            scalar = FIELD_CONSTANT * scalar / (4 * math.pi)
            scalar = scalar.reshape(n_nodes, 4, radial_count)
            l0 = scalar[:, 0, :] * self.l0_factors
            if self.feature_max_l == 0:
                features = l0
            else:
                l1 = (
                    jnp.stack(
                        (
                            scalar[:, 2, :] - scalar[:, 0, :],
                            scalar[:, 3, :] - scalar[:, 0, :],
                            scalar[:, 1, :] - scalar[:, 0, :],
                        ),
                        axis=-1,
                    )
                    * self.l1_factors[None, :, None]
                )
                features = jnp.concatenate((l0, l1.reshape(n_nodes, -1)), axis=-1)
        if self.density_max_l == 0 and self.feature_max_l == 0:
            features = features * (FIELD_CONSTANT / (4 * math.pi))
        if self.include_field_self_interaction:
            self_terms = (
                source_feats[:, self.field_self_indices] * self.field_self_values
            )
            features = features.at[:, : self_terms.shape[1]].add(self_terms)
        return features

    def coulomb_energy(
        self,
        source_feats: jnp.ndarray,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
        num_graphs: int,
    ) -> jnp.ndarray:
        if self.density_max_l == 0:
            charges = source_feats[:, 0]
            displaced = positions
            expanded_batch = batch
            duplicate_count = 1
        else:
            charges, displaced = _displaced_charges(
                source_feats, positions, self.energy_offset
            )
            expanded_batch = jnp.repeat(batch, 4)
            duplicate_count = 4
        kernel = _pair_kernel(
            displaced,
            expanded_batch,
            jnp.asarray([self.density_width]),
            duplicate_count,
        )[..., 0]
        node_energies = (
            0.5 * FIELD_CONSTANT / (4 * math.pi) * charges * (kernel @ charges)
        )
        energy = scatter_sum(node_energies, expanded_batch, dim=0, dim_size=num_graphs)
        if self.include_energy_self_interaction:
            self_terms = (
                source_feats[:, self.energy_self_indices] * self.energy_self_values
            )
            self_energy = 0.5 * jnp.sum(
                source_feats[:, : self_terms.shape[1]] * self_terms, axis=-1
            )
            energy = energy + scatter_sum(
                self_energy, batch, dim=0, dim_size=num_graphs
            )
        return energy


class DisplacedGTOExternalFieldBlock:
    """Project a uniform external electric field onto POLAR GTO features."""

    def __init__(self, max_l: int, widths: tuple[float, ...]) -> None:
        output_dim = (max_l + 1) ** 2 * len(widths)
        matrix = np.zeros((output_dim, 4))
        for index, width in enumerate(widths):
            matrix[index, 0] = (
                _cl_sigma(0, width, 'receiver')
                * math.sqrt(8 * math.pi)
                * special.gamma(1.5)
                * width**3
            )
            if max_l >= 1:
                magnitude = (
                    _cl_sigma(1, width, 'receiver')
                    * math.sqrt(1.5)
                    * width**5
                    * 2
                    * math.pi
                )
                for m in range(3):
                    matrix[len(widths) + index * 3 + m, 1 + m] = magnitude
        permutation = np.asarray(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        )
        self.matrix = jnp.asarray(matrix @ permutation)

    def __call__(
        self,
        batch: jnp.ndarray,
        centered_positions: jnp.ndarray,
        external_potential: jnp.ndarray,
    ) -> jnp.ndarray:
        node_fields = external_potential[batch]
        potential = node_fields[:, 0] + jnp.sum(
            centered_positions * node_fields[:, 1:], axis=-1
        )
        permuted = jnp.stack(
            (potential, node_fields[:, 3], node_fields[:, 1], node_fields[:, 2]),
            axis=-1,
        )
        return permuted @ self.matrix.T
