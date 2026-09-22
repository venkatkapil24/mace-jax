"""JAX Fourier GTO electrostatics and finite-size corrections for POLAR."""

from __future__ import annotations

import itertools
import math

import jax.numpy as jnp
import numpy as np
from e3nn_jax import Irreps

from mace_jax.adapters.e3nn.o3 import SphericalHarmonics
from mace_jax.tools.scatter import scatter_sum

from .polar_electrostatics import (
    FIELD_CONSTANT,
    DisplacedGTOExternalFieldBlock,
    _cl_sigma,
    _self_interaction_constants,
)


def _half_space_coefficients(
    cutoff: float, cell: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reproduce graph_longrange's reciprocal half-sphere enumeration."""
    cell_np = np.asarray(cell)
    reciprocal_np = 2 * math.pi * np.linalg.inv(np.swapaxes(cell_np, -1, -2))
    norms = np.linalg.norm(cell_np, axis=-1)
    normalized = cell_np / norms[..., None]
    dot_products = np.sum(reciprocal_np * normalized, axis=-1)
    max_ns = np.ceil(cutoff / dot_products).astype(np.int64)
    n1, n2, n3 = np.max(max_ns, axis=0).tolist()
    coefficients = [(0, 0, 0)]
    coefficients.extend((0, 0, k) for k in range(1, n3))
    coefficients.extend(
        (0, j, k) for j, k in itertools.product(range(1, n2), range(-n3, n3))
    )
    coefficients.extend(
        (i, j, k)
        for i, j, k in itertools.product(range(1, n1), range(-n2, n2), range(-n3, n3))
    )
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    selected = []
    graph_ids = []
    origin_flags = []
    for graph_index, reciprocal in enumerate(reciprocal_np):
        k_vectors = coefficient_array @ reciprocal
        mask = np.sum(k_vectors * k_vectors, axis=-1) <= cutoff**2
        kept = coefficient_array[mask]
        selected.append(kept)
        graph_ids.extend([graph_index] * len(kept))
        origin_flags.extend([1.0] + [0.0] * (len(kept) - 1))
    return (
        jnp.asarray(np.concatenate(selected, axis=0), dtype=cell.dtype),
        jnp.asarray(graph_ids, dtype=jnp.int32),
        jnp.asarray(origin_flags, dtype=cell.dtype),
    )


def _gather_self_terms(
    source_feats: jnp.ndarray, indices: tuple[int, ...]
) -> jnp.ndarray:
    # Repeated advanced indices changed the field result when fused under JIT.
    # Fixed slices preserve the eager calculation and its derivatives.
    return jnp.concatenate(
        [source_feats[:, index : index + 1] for index in indices], axis=-1
    )


class _GTOFourierBasis:
    def __init__(
        self, max_l: int, widths: tuple[float, ...], normalization: str
    ) -> None:
        if max_l > 1:
            raise ValueError('The direct Fourier GTO basis supports l<=1')
        self.max_l = max_l
        self.widths = tuple(widths)
        self.spherical_harmonics = SphericalHarmonics(
            Irreps.spherical_harmonics(max_l),
            normalize=True,
            normalization='integral',
        )
        self.cl_scale = jnp.asarray(
            [
                [_cl_sigma(ell, width, normalization) for ell in range(max_l + 1)]
                for width in widths
            ]
        )
        self.expanded_ell = jnp.asarray(
            [ell for ell in range(max_l + 1) for _ in range(2 * ell + 1)],
            dtype=jnp.int32,
        )
        self.real_phases = jnp.asarray(
            [
                (-1) ** (ell // 2) if ell % 2 == 0 else 0
                for ell in self.expanded_ell.tolist()
            ]
        )
        self.imag_phases = jnp.asarray(
            [
                -((-1) ** ((ell - 1) // 2)) if ell % 2 else 0
                for ell in self.expanded_ell.tolist()
            ]
        )

    def __call__(
        self,
        k_vectors: jnp.ndarray,
        k_norm2: jnp.ndarray,
        k0_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        k_moduli = jnp.sqrt(jnp.where(k0_mask > 0, 0.0, k_norm2))
        replacement = jnp.asarray([1.0, 0.0, 0.0], dtype=k_vectors.dtype)
        safe_vectors = jnp.where(k0_mask[:, None] > 0, replacement, k_vectors)
        harmonics = self.spherical_harmonics(safe_vectors[:, jnp.asarray([1, 2, 0])])
        widths = jnp.asarray(self.widths, dtype=k_vectors.dtype)
        exponential = jnp.exp(-0.5 * k_moduli[:, None] ** 2 * widths[None, :] ** 2)
        prefactor = 4 * math.pi * math.sqrt(math.pi / 2)
        radial = [prefactor * widths[None, :] ** 3 * exponential]
        if self.max_l >= 1:
            radial.append(
                prefactor * widths[None, :] ** 5 * k_moduli[:, None] * exponential
            )
        radial = jnp.stack(radial, axis=-1) * self.cl_scale[None, :, :]
        basis = radial[..., self.expanded_ell] * harmonics[:, None, :]
        return jnp.stack((basis * self.real_phases, basis * self.imag_phases), axis=-1)


class PeriodicPolarElectrostatics:
    """Fourier GTO fields and energy with bulk, slab, and cluster corrections."""

    def __init__(
        self,
        density_max_l: int,
        density_width: float,
        feature_max_l: int,
        feature_widths: tuple[float, ...],
        *,
        kspace_cutoff_factor: float = 1.5,
        kspace_cutoff: float | None = None,
        include_field_self_interaction: bool = False,
        include_energy_self_interaction: bool = True,
    ) -> None:
        widths = (density_width, *feature_widths)
        self.kspace_cutoff = (
            kspace_cutoff_factor
            * 0.75
            / min(widths)
            * (max(density_max_l, feature_max_l) + 1) ** 0.3
            * 3.0
        )
        if kspace_cutoff is not None:
            self.kspace_cutoff = float(kspace_cutoff)
        self.density_basis = _GTOFourierBasis(
            density_max_l, (density_width,), 'multipoles'
        )
        self.feature_basis = _GTOFourierBasis(
            feature_max_l, tuple(feature_widths), 'receiver'
        )
        self.include_field_self_interaction = include_field_self_interaction
        self.include_energy_self_interaction = include_energy_self_interaction
        self.field_self_indices, self.field_self_values = _self_interaction_constants(
            density_max_l,
            density_width,
            feature_max_l,
            tuple(feature_widths),
        )
        self.energy_self_indices, self.energy_self_values = _self_interaction_constants(
            density_max_l,
            density_width,
            density_max_l,
            (density_width,),
            feature_normalization='multipoles',
        )
        self.field_self_index_tuple = tuple(
            np.asarray(self.field_self_indices).tolist()
        )
        self.energy_self_index_tuple = tuple(
            np.asarray(self.energy_self_indices).tolist()
        )
        self.output_permutation = jnp.asarray(
            [
                radial * (feature_max_l + 1) ** 2 + ell**2 + m
                for ell in range(feature_max_l + 1)
                for radial in range(len(feature_widths))
                for m in range(2 * ell + 1)
            ],
            dtype=jnp.int32,
        )
        self.correction_projection = DisplacedGTOExternalFieldBlock(
            feature_max_l, tuple(feature_widths)
        )

    def prepare_coefficients(
        self, reference_cell: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Enumerate the fixed reciprocal indices outside a JIT-compiled call."""
        return _half_space_coefficients(
            self.kspace_cutoff, jnp.asarray(reference_cell).reshape(-1, 3, 3)
        )

    def precompute(
        self,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
        cell: jnp.ndarray,
        reference_cell: jnp.ndarray | None = None,
        coefficients: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    ) -> dict[str, jnp.ndarray]:
        cell = cell.reshape(-1, 3, 3)
        if coefficients is None:
            coefficients = self.prepare_coefficients(
                cell if reference_cell is None else reference_cell
            )
        coefficients, k_batch, k0_mask = coefficients
        reciprocal = 2 * math.pi * jnp.linalg.inv(jnp.swapaxes(cell, -1, -2))
        k_vectors = jnp.einsum('ki,kij->kj', coefficients, reciprocal[k_batch])
        k_norm2 = jnp.sum(k_vectors * k_vectors, axis=-1)
        volume = jnp.abs(jnp.linalg.det(cell))
        phase = k_vectors @ positions.T
        graph_mask = (k_batch[:, None] == batch[None, :]).astype(phase.dtype)
        cosines = jnp.cos(phase) * graph_mask
        sines = jnp.sin(phase) * graph_mask
        return {
            'k_vectors': k_vectors,
            'k_norm2': k_norm2,
            'k_batch': k_batch,
            'k0_mask': k0_mask,
            'volume': volume,
            'volume_per_k': volume[k_batch],
            'cosines': cosines,
            'sines': sines,
            'density_basis': self.density_basis(k_vectors, k_norm2, k0_mask),
            'feature_basis': self.feature_basis(k_vectors, k_norm2, k0_mask),
            'batch': batch,
            'positions': positions,
        }

    @staticmethod
    def _moments(
        source_feats: jnp.ndarray, cache: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        positions = cache['positions']
        batch = cache['batch']
        n_graphs = cache['volume'].shape[0]
        charge = scatter_sum(source_feats[:, 0], batch, dim=0, dim_size=n_graphs)
        dipole = scatter_sum(
            positions * source_feats[:, :1], batch, dim=0, dim_size=n_graphs
        )
        quadrupole = scatter_sum(
            jnp.sum(positions**2, axis=-1) * source_feats[:, 0],
            batch,
            dim=0,
            dim_size=n_graphs,
        )
        if source_feats.shape[-1] > 1:
            local_dipole = source_feats[:, jnp.asarray([3, 1, 2])]
            dipole = dipole + scatter_sum(local_dipole, batch, dim=0, dim_size=n_graphs)
            quadrupole = quadrupole + 2 * scatter_sum(
                jnp.sum(positions * local_dipole, axis=-1),
                batch,
                dim=0,
                dim_size=n_graphs,
            )
        return charge, dipole, quadrupole

    @staticmethod
    def _correction_masks(
        mode: str, pbc: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pbc = pbc.reshape(-1, 3).astype(jnp.bool_)
        all_graphs = jnp.ones((pbc.shape[0],), dtype=jnp.bool_)
        none = jnp.zeros_like(all_graphs)
        if mode == 'slab':
            return none, all_graphs
        if mode == 'molecule_in_box':
            return all_graphs, none
        if mode in ('mixed_periodic', 'auto'):
            return ~jnp.any(pbc, axis=-1), pbc[:, 0] & pbc[:, 1] & ~pbc[:, 2]
        return none, none

    def _correction_features(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        mode: str,
        pbc: jnp.ndarray,
    ) -> jnp.ndarray:
        molecule_mask, slab_mask = self._correction_masks(mode, pbc)
        charge, dipole, quadrupole = self._moments(source_feats, cache)
        positions = cache['positions']
        batch = cache['batch']
        volume = cache['volume']
        spread_q = charge[batch]
        spread_p = dipole[batch]
        spread_quad = quadrupole[batch]
        spread_v = volume[batch]
        r2 = jnp.sum(positions**2, axis=-1)
        const = FIELD_CONSTANT / (4 * math.pi)
        molecule_v = (
            2.837297 * const * (charge / volume**0.333333)[batch]
            - const * 2 * math.pi * spread_q * r2 / (3 * spread_v)
            + const
            * 4
            * math.pi
            * jnp.sum(spread_p * positions, axis=-1)
            / (3 * spread_v)
            - const * 2 * math.pi * spread_quad / (3 * spread_v)
        )
        molecule_field = (
            FIELD_CONSTANT
            * (spread_p - spread_q[:, None] * positions)
            / (3 * spread_v[:, None])
        )
        slab_field_z = FIELD_CONSTANT * dipole[:, 2] / volume
        slab_v = slab_field_z[batch] * positions[:, 2]
        slab_field = jnp.stack(
            (jnp.zeros_like(slab_v), jnp.zeros_like(slab_v), slab_field_z[batch]),
            axis=-1,
        )
        is_molecule = molecule_mask[batch]
        is_slab = slab_mask[batch]
        potential = jnp.where(is_molecule, molecule_v, 0.0) + jnp.where(
            is_slab, slab_v, 0.0
        )
        field = jnp.where(is_molecule[:, None], molecule_field, 0.0) + jnp.where(
            is_slab[:, None], slab_field, 0.0
        )
        node_fields = jnp.concatenate((potential[:, None], field), axis=-1)
        return (
            node_fields[:, jnp.asarray([0, 3, 1, 2])]
            @ self.correction_projection.matrix.T
        )

    def _correction_energy(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        mode: str,
        pbc: jnp.ndarray,
    ) -> jnp.ndarray:
        molecule_mask, slab_mask = self._correction_masks(mode, pbc)
        charge, dipole, quadrupole = self._moments(source_feats, cache)
        volume = cache['volume']
        const = FIELD_CONSTANT / (4 * math.pi)
        molecule = (
            0.5 * 2.837297 * const * charge**2 / volume**0.3333
            + 2 * const * math.pi * jnp.sum(dipole**2, axis=-1) / (3 * volume)
            - 2 * const * math.pi * charge * quadrupole / (3 * volume)
        )
        slab = 0.5 * FIELD_CONSTANT * dipole[:, 2] ** 2 / volume
        return jnp.where(molecule_mask, molecule, 0.0) + jnp.where(slab_mask, slab, 0.0)

    @staticmethod
    def _density_and_potential(
        source_feats: jnp.ndarray, cache: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        cosines = cache['cosines']
        sines = cache['sines']
        basis = cache['density_basis']
        basis_real = basis[..., 0].reshape(basis.shape[0], -1)
        basis_imag = basis[..., 1].reshape(basis.shape[0], -1)
        coeff_cos = cosines @ source_feats
        coeff_sin = sines @ source_feats
        density = jnp.stack(
            (
                jnp.sum(basis_real * coeff_cos + basis_imag * coeff_sin, axis=-1),
                jnp.sum(basis_imag * coeff_cos - basis_real * coeff_sin, axis=-1),
            ),
            axis=-1,
        )
        density = (2 * math.pi) ** 3 * density / cache['volume_per_k'][:, None]
        k0 = cache['k0_mask'] > 0
        coulomb_factor = jnp.where(k0, 0.0, 1.0 / jnp.where(k0, 1.0, cache['k_norm2']))
        potential = density * coulomb_factor[:, None] * FIELD_CONSTANT
        return density, potential

    def field_features(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        *,
        mode: str = 'pbc',
        pbc: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        _density, potential = self._density_and_potential(source_feats, cache)
        basis = cache['feature_basis']
        basis_real = basis[..., 0].reshape(basis.shape[0], -1)
        basis_imag = basis[..., 1].reshape(basis.shape[0], -1)
        a = potential[:, :1] * basis_real + potential[:, 1:] * basis_imag
        b = potential[:, :1] * basis_imag - potential[:, 1:] * basis_real
        projection_factor = jnp.where(cache['k0_mask'] > 0, 0.5, 1.0)
        a = a * projection_factor[:, None]
        b = b * projection_factor[:, None]
        projected = (
            2 * (a.T @ cache['cosines'] + b.T @ cache['sines']).T / (2 * math.pi) ** 3
        )
        features = projected[:, self.output_permutation]
        if not self.include_field_self_interaction:
            self_terms = (
                _gather_self_terms(source_feats, self.field_self_index_tuple)
                * self.field_self_values
            )
            self_terms = jnp.pad(
                self_terms,
                ((0, 0), (0, features.shape[-1] - self_terms.shape[-1])),
            )
            features = features - self_terms
        if mode != 'pbc':
            if pbc is None:
                raise ValueError('pbc is required for correction modes')
            features = features + self._correction_features(
                source_feats, cache, mode, pbc
            )
        return features

    def coulomb_energy(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        *,
        mode: str = 'pbc',
        pbc: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        density, potential = self._density_and_potential(source_feats, cache)
        per_k = 2 * jnp.sum(density * potential, axis=-1)
        n_graphs = cache['volume'].shape[0]
        energy_k = scatter_sum(per_k, cache['k_batch'], dim=0, dim_size=n_graphs)
        energy = 0.5 * cache['volume'] * energy_k / (2 * math.pi) ** 6
        if not self.include_energy_self_interaction:
            self_terms = (
                _gather_self_terms(source_feats, self.energy_self_index_tuple)
                * self.energy_self_values
            )
            node_energy = 0.5 * jnp.sum(
                source_feats[:, : self_terms.shape[-1]] * self_terms, axis=-1
            )
            energy = energy - scatter_sum(
                node_energy, cache['batch'], dim=0, dim_size=n_graphs
            )
        if mode != 'pbc':
            if pbc is None:
                raise ValueError('pbc is required for correction modes')
            energy = energy + self._correction_energy(source_feats, cache, mode, pbc)
        return energy
