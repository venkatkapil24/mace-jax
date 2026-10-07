"""Differentiable smooth PME with OpenMM-compatible point-charge conventions.

Coordinates and cell vectors are in Angstrom; charges are in elementary-charge
units and energies are in eV. The discrete cardinal-spline Fourier correction
follows smooth PME (Essmann et al., JCP 103, 8577, 1995), as used by OpenMM's
ReferencePME. This is independent of the Gaussian-multipole mesh solver.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from mace_jax.tools.physical_constants import OPENMM_COULOMB

from .polar_periodic import _spread_to_mesh


def forward_spline_weights(fraction, order):
    """Cardinal splines on the forward grid stencil used by smooth PME."""
    if order < 2:
        raise ValueError('Spline order must be at least 2')
    x = jnp.arange(order) + 1 - fraction[..., None]
    result = jnp.zeros_like(x)
    for k in range(order + 1):
        result += (
            (-1) ** k * math.comb(order, k) * jnp.maximum(x - k, 0.0) ** (order - 1)
        )
    return result / math.factorial(order - 1)


def discrete_spline_modulus(size, order):
    """Squared Fourier transform of spline values at integer grid points."""
    # Integer samples of the cardinal spline; a common grid translation has
    # unit Fourier magnitude and therefore does not affect this correction.
    x = np.arange(1, order + 1, dtype=float)
    values = sum(
        (-1) ** k * math.comb(order, k) * np.maximum(x - k, 0.0) ** (order - 1)
        for k in range(order + 1)
    ) / math.factorial(order - 1)
    phase = 2 * np.pi * np.arange(size)[:, None] * np.arange(order)[None, :] / size
    modulus = np.abs(np.exp(1j * phase) @ values) ** 2
    # Smooth PME replaces almost-zero moduli with adjacent-grid averages.
    for i in range(size):
        if modulus[i] < 1e-7:
            modulus[i] = 0.5 * (modulus[(i - 1) % size] + modulus[(i + 1) % size])
    return modulus


@dataclass(frozen=True)
class PointPME:
    """Static grid settings; all position/charge calculations remain JIT-safe."""

    alpha: float
    grid_shape: tuple[int, int, int]
    order: int = 5
    coulomb: float = OPENMM_COULOMB

    def __post_init__(self):
        if (
            self.alpha <= 0
            or self.order < 2
            or any(n <= self.order for n in self.grid_shape)
        ):
            raise ValueError(
                'Require positive alpha and a grid larger than the spline order'
            )

    def charge_mesh(self, positions, charges, cell):
        fractional = positions @ jnp.linalg.inv(cell)
        coordinate = (fractional - jnp.floor(fractional)) * jnp.asarray(self.grid_shape)
        base = jnp.floor(coordinate).astype(jnp.int32)
        fraction = coordinate - base
        wx, wy, wz = (
            forward_spline_weights(fraction[:, d], self.order) for d in range(3)
        )
        weights = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        indices = []
        for d, shape in enumerate(
            (
                (positions.shape[0], self.order, 1, 1),
                (positions.shape[0], 1, self.order, 1),
                (positions.shape[0], 1, 1, self.order),
            )
        ):
            index = (base[:, d, None] + jnp.arange(self.order)) % self.grid_shape[d]
            indices.append(jnp.broadcast_to(index.reshape(shape), weights.shape))
        return _spread_to_mesh(charges, tuple(indices), weights, self.grid_shape)[
            ..., 0
        ]

    def reciprocal_kernel(self, cell):
        integers = jnp.stack(
            jnp.meshgrid(
                *(jnp.fft.fftfreq(n) * n for n in self.grid_shape), indexing='ij'
            ),
            axis=-1,
        )
        k = integers @ (2 * jnp.pi * jnp.linalg.inv(cell.T))
        k2 = jnp.sum(k * k, axis=-1)
        safe = jnp.where(k2 > 0, k2, 1.0)
        moduli = [
            jnp.asarray(discrete_spline_modulus(n, self.order), dtype=cell.dtype)
            for n in self.grid_shape
        ]
        denominator = (
            moduli[0][:, None, None]
            * moduli[1][None, :, None]
            * moduli[2][None, None, :]
        )
        return jnp.where(
            k2 > 0, jnp.exp(-safe / (4 * self.alpha**2)) / safe / denominator, 0.0
        )

    def reciprocal_energy(self, positions, charges, cell):
        spectrum = jnp.fft.fftn(self.charge_mesh(positions, charges, cell))
        return (
            self.coulomb
            * 2
            * jnp.pi
            / jnp.abs(jnp.linalg.det(cell))
            * jnp.sum(jnp.abs(spectrum) ** 2 * self.reciprocal_kernel(cell))
        )

    def reciprocal_parts(
        self, qm_positions, qm_charges, mm_positions, mm_charges, cell
    ):
        """MM-only and QM–MM energies with the same grid and kernel."""
        qm = jnp.fft.fftn(self.charge_mesh(qm_positions, qm_charges, cell))
        mm = jnp.fft.fftn(self.charge_mesh(mm_positions, mm_charges, cell))
        kernel = self.reciprocal_kernel(cell)
        prefactor = self.coulomb * 2 * jnp.pi / jnp.abs(jnp.linalg.det(cell))
        return (
            prefactor * jnp.sum(jnp.abs(mm) ** 2 * kernel),
            2 * prefactor * jnp.sum(jnp.real(qm * jnp.conj(mm)) * kernel),
        )

    def self_energy(self, charges):
        return -self.coulomb * self.alpha / math.sqrt(math.pi) * jnp.sum(charges**2)

    def background_energy(self, charges, cell):
        return (
            -self.coulomb
            * jnp.pi
            * jnp.sum(charges) ** 2
            / (2 * self.alpha**2 * jnp.abs(jnp.linalg.det(cell)))
        )
