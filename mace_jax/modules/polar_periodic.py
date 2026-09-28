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


def _cardinal_bspline_weights(
    fractional: jnp.ndarray, order: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Centered cardinal B-spline offsets and weights for one coordinate."""
    if order < 2 or order % 2:
        raise ValueError('PME B-spline order must be an even integer >= 2')
    offsets = jnp.arange(-order // 2 + 1, order // 2 + 1, dtype=jnp.int32)
    argument = fractional[..., None] - offsets + order / 2
    weights = jnp.zeros_like(argument)
    for index in range(order + 1):
        weights = weights + (
            (-1) ** index
            * math.comb(order, index)
            * jnp.maximum(argument - index, 0.0) ** (order - 1)
        )
    return offsets, weights / math.factorial(order - 1)


def _mesh_indices_and_weights(
    positions: jnp.ndarray,
    cell: jnp.ndarray,
    grid_shape: tuple[int, int, int],
    order: int,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Periodic tensor-product B-spline stencil for each particle."""
    fractional = positions @ jnp.linalg.inv(cell)
    grid_coordinate = fractional * jnp.asarray(grid_shape, dtype=positions.dtype)
    base = jnp.floor(grid_coordinate).astype(jnp.int32)
    remainder = grid_coordinate - base
    offsets, wx = _cardinal_bspline_weights(remainder[:, 0], order)
    _, wy = _cardinal_bspline_weights(remainder[:, 1], order)
    _, wz = _cardinal_bspline_weights(remainder[:, 2], order)
    ix = (base[:, 0, None] + offsets) % grid_shape[0]
    iy = (base[:, 1, None] + offsets) % grid_shape[1]
    iz = (base[:, 2, None] + offsets) % grid_shape[2]
    weights = (
        wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
    )
    shape = weights.shape
    return (
        (
            jnp.broadcast_to(ix[:, :, None, None], shape),
            jnp.broadcast_to(iy[:, None, :, None], shape),
            jnp.broadcast_to(iz[:, None, None, :], shape),
        ),
        weights,
    )


def _spread_to_mesh(
    values: jnp.ndarray,
    indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    weights: jnp.ndarray,
    grid_shape: tuple[int, int, int],
) -> jnp.ndarray:
    """Spread scalar or vector particle values onto a periodic mesh."""
    if values.ndim == 1:
        values = values[:, None]
    contribution = weights[..., None] * values[:, None, None, None, :]
    ix, iy, iz = indices
    mesh = jnp.zeros((*grid_shape, values.shape[-1]), dtype=values.dtype)
    return mesh.at[ix.reshape(-1), iy.reshape(-1), iz.reshape(-1)].add(
        contribution.reshape(-1, values.shape[-1])
    )


def _gather_from_mesh(
    mesh: jnp.ndarray,
    indices: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate mesh channels to particles with the spreading stencil."""
    ix, iy, iz = indices
    samples = mesh[ix, iy, iz]
    return jnp.sum(samples * weights[..., None], axis=(1, 2, 3))


def ewald_neutralizing_background_energy(
    total_charge: jnp.ndarray | float,
    volume: jnp.ndarray | float,
    alpha: jnp.ndarray | float,
) -> jnp.ndarray:
    """Energy of the uniform background used for a charged Ewald sum.

    The reciprocal ``G=0`` component is undefined when the explicit point
    charges have a nonzero total charge. Omitting that component is equivalent
    to adding a uniform compensating charge density. In the conventional
    real/reciprocal Ewald decomposition its energy is

        -k_e * pi * Q**2 / (2 * alpha**2 * V).

    ``total_charge`` is in units of the elementary charge, ``volume`` in
    Angstrom cubed, and ``alpha`` in inverse Angstrom; the result is in eV.
    The term has no position forces for a fixed cell, but must be included in
    energy and cell derivatives whenever a non-neutral MM Ewald term replaces
    the MM-only part of the zero-mean POLAR Fourier energy.
    """
    charge = jnp.asarray(total_charge)
    volume = jnp.asarray(volume, dtype=charge.dtype)
    alpha = jnp.asarray(alpha, dtype=charge.dtype)
    coulomb = FIELD_CONSTANT / (4 * math.pi)
    return -coulomb * math.pi * charge**2 / (2 * alpha**2 * volume)


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

    def prepare_mesh_template(
        self,
        reference_cell: jnp.ndarray,
        *,
        mesh_spacing: float = 0.5,
        assignment_order: int = 8,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create static mesh and assignment templates outside compiled calls."""
        if mesh_spacing <= 0:
            raise ValueError('PME mesh spacing must be positive')
        if assignment_order < 2 or assignment_order % 2:
            raise ValueError('PME assignment order must be an even integer >= 2')
        cell = np.asarray(reference_cell).reshape(-1, 3, 3)
        if cell.shape[0] != 1:
            raise ValueError('Generalized PME currently supports one periodic graph')
        lengths = np.linalg.norm(cell[0], axis=-1)
        reciprocal = 2 * math.pi * np.linalg.inv(cell[0].T)
        reciprocal_spacing = np.linalg.norm(reciprocal, axis=-1)
        cutoff_shape = 2 * np.ceil(
            self.kspace_cutoff / reciprocal_spacing
        ).astype(int) + 2
        spacing_shape = np.ceil(lengths / mesh_spacing).astype(int)
        shape = np.maximum(cutoff_shape, spacing_shape)
        shape = shape + shape % 2
        return (
            jnp.zeros(tuple(shape.tolist()), dtype=jnp.bool_),
            jnp.zeros((assignment_order,), dtype=jnp.bool_),
        )

    def precompute_mesh(
        self,
        positions: jnp.ndarray,
        batch: jnp.ndarray,
        cell: jnp.ndarray,
        mesh_template: jnp.ndarray,
        assignment_template: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Build full reciprocal mesh and QM interpolation stencil."""
        cell = cell.reshape(-1, 3, 3)
        if cell.shape[0] != 1:
            raise ValueError('Generalized PME currently supports one periodic graph')
        grid_shape = mesh_template.shape
        assignment_order = assignment_template.shape[0]
        integer_axes = [jnp.fft.fftfreq(size) * size for size in grid_shape]
        integer_grid = jnp.stack(
            jnp.meshgrid(*integer_axes, indexing='ij'), axis=-1
        )
        reciprocal = 2 * math.pi * jnp.linalg.inv(cell[0].T)
        k_vectors = integer_grid @ reciprocal
        k_norm2 = jnp.sum(k_vectors * k_vectors, axis=-1)
        k0_mask = k_norm2 == 0
        cutoff_mask = (k_norm2 <= self.kspace_cutoff**2) & ~k0_mask
        assignment_window = jnp.prod(
            jnp.sinc(
                integer_grid
                / jnp.asarray(grid_shape, dtype=positions.dtype)
            )
            ** assignment_order,
            axis=-1,
        )
        qm_indices, qm_weights = _mesh_indices_and_weights(
            positions, cell[0], grid_shape, assignment_order
        )
        density_basis = self.density_basis(
            k_vectors.reshape(-1, 3),
            k_norm2.reshape(-1),
            k0_mask.reshape(-1).astype(positions.dtype),
        ).reshape(*grid_shape, -1, 2)
        feature_basis = self.feature_basis(
            k_vectors.reshape(-1, 3),
            k_norm2.reshape(-1),
            k0_mask.reshape(-1).astype(positions.dtype),
        ).reshape(*grid_shape, len(self.feature_basis.widths), -1, 2)
        return {
            'mesh_template': mesh_template,
            'assignment_template': assignment_template,
            'grid_shape_array': jnp.asarray(grid_shape, dtype=jnp.int32),
            'assignment_window': assignment_window,
            'k_vectors_mesh': k_vectors,
            'k_norm2_mesh': k_norm2,
            'k0_mask_mesh': k0_mask,
            'cutoff_mask_mesh': cutoff_mask,
            'density_basis_mesh': density_basis,
            'feature_basis_mesh': feature_basis,
            'qm_mesh_indices': qm_indices,
            'qm_mesh_weights': qm_weights,
            'volume': jnp.abs(jnp.linalg.det(cell)),
            'batch': batch,
            'positions': positions,
            'cell': cell,
        }

    @staticmethod
    def _mesh_structure_factors(
        positions: jnp.ndarray,
        values: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        grid_shape = cache['mesh_template'].shape
        order = cache['assignment_template'].shape[0]
        indices, weights = _mesh_indices_and_weights(
            positions, cache['cell'][0], grid_shape, order
        )
        mesh = _spread_to_mesh(values, indices, weights, grid_shape)
        transformed = jnp.fft.fftn(mesh, axes=(0, 1, 2))
        return transformed / cache['assignment_window'][..., None]

    def _mesh_density(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        coefficients = self._mesh_structure_factors(
            cache['positions'], source_feats, cache
        )
        basis = cache['density_basis_mesh']
        basis_complex = basis[..., 0] + 1j * basis[..., 1]
        density = jnp.sum(basis_complex * coefficients, axis=-1)
        return (2 * math.pi) ** 3 * density / cache['volume'][0]

    def mesh_point_charge_density(
        self,
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        coefficients = self._mesh_structure_factors(positions, charges, cache)[..., 0]
        return (2 * math.pi) ** 3 * coefficients / cache['volume'][0]

    def _mesh_project_density(
        self,
        density: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        inverse_k2 = jnp.where(
            cache['cutoff_mask_mesh'],
            1.0 / jnp.where(cache['k0_mask_mesh'], 1.0, cache['k_norm2_mesh']),
            0.0,
        )
        potential = density * inverse_k2 * FIELD_CONSTANT
        basis = cache['feature_basis_mesh']
        basis_complex = basis[..., 0] + 1j * basis[..., 1]
        spectral = jnp.conj(potential)[..., None, None] * basis_complex
        spectral = spectral.reshape(*cache['mesh_template'].shape, -1)
        # Spreading and gathering each apply the cardinal B-spline window.
        # Dividing here removes the gathering window; the density construction
        # above already removed the spreading window.
        spatial = jnp.fft.fftn(
            spectral / cache['assignment_window'][..., None],
            axes=(0, 1, 2),
        ).real / (2 * math.pi) ** 3
        projected = _gather_from_mesh(
            spatial, cache['qm_mesh_indices'], cache['qm_mesh_weights']
        )
        return projected[:, self.output_permutation]

    def mesh_field_features(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        features = self._mesh_project_density(
            self._mesh_density(source_feats, cache), cache
        )
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
        return features

    def mesh_point_charge_field_features(
        self,
        mm_density: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        return self._mesh_project_density(mm_density, cache)

    def mesh_coulomb_energy(
        self,
        source_feats: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        density = self._mesh_density(source_feats, cache)
        inverse_k2 = jnp.where(
            cache['cutoff_mask_mesh'],
            1.0 / jnp.where(cache['k0_mask_mesh'], 1.0, cache['k_norm2_mesh']),
            0.0,
        )
        energy = (
            0.5
            * cache['volume'][0]
            * FIELD_CONSTANT
            * jnp.sum(jnp.abs(density) ** 2 * inverse_k2)
            / (2 * math.pi) ** 6
        )
        if not self.include_energy_self_interaction:
            self_terms = (
                _gather_self_terms(source_feats, self.energy_self_index_tuple)
                * self.energy_self_values
            )
            energy = energy - 0.5 * jnp.sum(
                source_feats[:, : self_terms.shape[-1]] * self_terms
            )
        return jnp.asarray([energy])

    def mesh_mixed_coulomb_energy(
        self,
        source_feats: jnp.ndarray,
        mm_density: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        qm_density = self._mesh_density(source_feats, cache)
        combined_density = qm_density + mm_density
        inverse_k2 = jnp.where(
            cache['cutoff_mask_mesh'],
            1.0 / jnp.where(cache['k0_mask_mesh'], 1.0, cache['k_norm2_mesh']),
            0.0,
        )
        prefactor = (
            0.5 * cache['volume'][0] * FIELD_CONSTANT / (2 * math.pi) ** 6
        )
        combined = prefactor * jnp.sum(
            jnp.abs(combined_density) ** 2 * inverse_k2
        )
        mm_only = prefactor * jnp.sum(jnp.abs(mm_density) ** 2 * inverse_k2)
        qm_only = prefactor * jnp.sum(jnp.abs(qm_density) ** 2 * inverse_k2)
        qm_and_cross = combined - mm_only
        if not self.include_energy_self_interaction:
            self_terms = (
                _gather_self_terms(source_feats, self.energy_self_index_tuple)
                * self.energy_self_values
            )
            qm_and_cross = qm_and_cross - 0.5 * jnp.sum(
                source_feats[:, : self_terms.shape[-1]] * self_terms
            )
        return jnp.asarray([qm_and_cross]), jnp.asarray(
            [combined - mm_only - qm_only]
        )

    def mesh_point_charge_ewald_reciprocal_energy(
        self,
        mm_density: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        alpha: jnp.ndarray | float,
    ) -> jnp.ndarray:
        """Conventional point-charge Ewald reciprocal energy from the mesh."""
        k2 = cache['k_norm2_mesh']
        nonzero = ~cache['k0_mask_mesh']
        kernel = jnp.where(
            nonzero,
            jnp.exp(-k2 / (4 * jnp.asarray(alpha) ** 2))
            / jnp.where(nonzero, k2, 1.0),
            0.0,
        )
        # Convert the normalized POLAR density back to the point-charge
        # structure factor used by the conventional Ewald expression.
        structure = mm_density * cache['volume'][0] / (2 * math.pi) ** 3
        coulomb = FIELD_CONSTANT / (4 * math.pi)
        return (
            coulomb
            * 2
            * math.pi
            / cache['volume'][0]
            * jnp.sum(jnp.abs(structure) ** 2 * kernel)
        )

    def mesh_point_charge_ewald_cross_energy(
        self,
        first_density: jnp.ndarray,
        second_density: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
        alpha: jnp.ndarray | float,
    ) -> jnp.ndarray:
        """Conventional Ewald reciprocal cross energy for two charge sets."""
        k2 = cache['k_norm2_mesh']
        nonzero = ~cache['k0_mask_mesh']
        alpha = jnp.asarray(alpha).reshape(())
        kernel = jnp.where(
            nonzero,
            jnp.exp(-k2 / (4 * alpha**2)) / jnp.where(nonzero, k2, 1.0),
            0.0,
        )
        normalization = cache['volume'][0] / (2 * math.pi) ** 3
        first_structure = first_density * normalization
        second_structure = second_density * normalization
        coulomb = FIELD_CONSTANT / (4 * math.pi)
        return (
            coulomb
            * 4
            * math.pi
            / cache['volume'][0]
            * jnp.sum(
                jnp.real(first_structure * jnp.conj(second_structure)) * kernel
            )
        )

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

    @staticmethod
    def point_charge_density(
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Fourier density of MM point charges on the POLAR reciprocal grid."""
        phase = cache['k_vectors'] @ positions.T
        density = jnp.stack(
            (jnp.cos(phase) @ charges, -(jnp.sin(phase) @ charges)), axis=-1
        )
        return (2 * math.pi) ** 3 * density / cache['volume_per_k'][:, None]

    def point_charge_field_features(
        self,
        positions: jnp.ndarray,
        charges: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        """Project an MM point-charge potential onto the QM GTO receivers."""
        density = self.point_charge_density(positions, charges, cache)
        k0 = cache['k0_mask'] > 0
        inverse_k2 = jnp.where(k0, 0.0, 1.0 / jnp.where(k0, 1.0, cache['k_norm2']))
        potential = density * inverse_k2[:, None] * FIELD_CONSTANT
        basis = cache['feature_basis']
        basis_real = basis[..., 0].reshape(basis.shape[0], -1)
        basis_imag = basis[..., 1].reshape(basis.shape[0], -1)
        a = potential[:, :1] * basis_real + potential[:, 1:] * basis_imag
        b = potential[:, :1] * basis_imag - potential[:, 1:] * basis_real
        factor = jnp.where(k0, 0.5, 1.0)
        projected = (
            2
            * (
                (a * factor[:, None]).T @ cache['cosines']
                + (b * factor[:, None]).T @ cache['sines']
            ).T
            / (2 * math.pi) ** 3
        )
        return projected[:, self.output_permutation]

    def mixed_coulomb_energy(
        self,
        source_feats: jnp.ndarray,
        mm_positions: jnp.ndarray,
        mm_charges: jnp.ndarray,
        cache: dict[str, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """QM self plus QM/MM cross from one combined Fourier density.

        The point-charge MM/MM Fourier term is removed here. A conventional
        point-charge Ewald solver supplies it with its own self and exclusion
        corrections; the MM density still contributes to the shared potential.
        All Fourier terms omit ``G=0``. If the MM charges are not neutral, the
        restored Ewald energy must therefore include
        :func:`ewald_neutralizing_background_energy` to use the same uniform
        background convention. This proof of concept handles one periodic
        graph.
        """
        qm_density, _ = self._density_and_potential(source_feats, cache)
        mm_density = self.point_charge_density(mm_positions, mm_charges, cache)
        combined_density = qm_density + mm_density
        k0 = cache['k0_mask'] > 0
        inverse_k2 = jnp.where(k0, 0.0, 1.0 / jnp.where(k0, 1.0, cache['k_norm2']))
        prefactor = cache['volume'][0] * FIELD_CONSTANT / (2 * math.pi) ** 6
        combined = prefactor * jnp.sum(
            jnp.sum(combined_density**2, axis=-1) * inverse_k2
        )
        mm_only = prefactor * jnp.sum(jnp.sum(mm_density**2, axis=-1) * inverse_k2)
        qm_only = prefactor * jnp.sum(jnp.sum(qm_density**2, axis=-1) * inverse_k2)
        qm_and_cross = combined - mm_only
        if not self.include_energy_self_interaction:
            self_terms = (
                _gather_self_terms(source_feats, self.energy_self_index_tuple)
                * self.energy_self_values
            )
            qm_and_cross = qm_and_cross - 0.5 * jnp.sum(
                source_feats[:, : self_terms.shape[-1]] * self_terms
            )
        return jnp.asarray([qm_and_cross]), jnp.asarray([combined - mm_only - qm_only])

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
