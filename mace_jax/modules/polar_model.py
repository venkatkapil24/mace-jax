"""Native eager JAX implementation of the MACE-POLAR response architecture."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from e3nn_jax import Irreps
from flax import nnx

from mace_jax.adapters.nnx.torch import nxx_auto_import_from_torch
from mace_jax.tools.scatter import scatter_sum

from .blocks import NonLinearBiasReadoutBlock
from .models import ScaleShiftMACE
from .polar_blocks import (
    AgnosticEmbeddedOneBodyVariableUpdate,
    EnvironmentDependentSpinSourceBlock,
    MultiLayerFeatureMixer,
    OneBodyMLPFieldReadout,
)
from .polar_electrostatics import (
    DisplacedGTOExternalFieldBlock,
    RealSpacePolarElectrostatics,
)
from .polar_periodic import PeriodicPolarElectrostatics
from .utils import add_output_interface


def _array(value):
    return getattr(value, 'array', value)


@nxx_auto_import_from_torch(allow_missing_mapper=True)
@add_output_interface
class PolarMACE(ScaleShiftMACE):
    """POLAR-1-M local backbone, spin response, and Coulomb terms."""

    def __init__(
        self,
        *,
        atomic_multipoles_max_l: int = 1,
        atomic_multipoles_smearing_width: float = 1.5,
        field_feature_max_l: int = 1,
        field_feature_widths: tuple[float, ...] = (1.5, 3.0),
        field_feature_norms: tuple[float, ...] | None = None,
        num_recursion_steps: int = 2,
        field_norm_factor: float = 1.0,
        field_si: bool = False,
        include_electrostatic_self_interaction: bool = True,
        add_local_electron_energy: bool = True,
        fixedpoint_update_config: dict[str, Any] | None = None,
        field_readout_config: dict[str, Any] | None = None,
        pbc_handling: str = 'auto',
        kspace_cutoff_factor: float = 1.5,
        kspace_cutoff: float | None = None,
        rngs: nnx.Rngs,
        **kwargs,
    ) -> None:
        del field_norm_factor
        if atomic_multipoles_max_l > 1 or field_feature_max_l > 1:
            raise ValueError('This POLAR port currently supports l<=1')
        update_type = (fixedpoint_update_config or {}).get(
            'type', 'AgnosticEmbeddedOneBodyVariableUpdate'
        )
        readout_type = (field_readout_config or {}).get(
            'type', 'OneBodyMLPFieldReadout'
        )
        if update_type != 'AgnosticEmbeddedOneBodyVariableUpdate':
            raise ValueError(f'Unsupported POLAR field update {update_type!r}')
        if readout_type != 'OneBodyMLPFieldReadout':
            raise ValueError(f'Unsupported POLAR field readout {readout_type!r}')

        kwargs['keep_last_layer_irreps'] = True
        kwargs['spherical_harmonics_permutation'] = (1, 2, 0)
        super().__init__(rngs=rngs, **kwargs)
        self.atomic_multipoles_max_l = atomic_multipoles_max_l
        self.field_feature_max_l = field_feature_max_l
        self.field_feature_widths = tuple(field_feature_widths)
        self.num_recursion_steps = num_recursion_steps
        self.add_local_electron_energy = add_local_electron_energy
        self.pbc_handling = pbc_handling

        hidden = self._hidden_irreps
        basis = Irreps.spherical_harmonics(atomic_multipoles_max_l)
        # Torch's left multiplication repeats the entire irreps list.
        self.charges_irreps = Irreps(list(basis) * 2)
        field_basis = Irreps.spherical_harmonics(field_feature_max_l)
        self.field_irreps = (
            (field_basis * len(self.field_feature_widths)).sort()[0].simplify()
        )
        # Torch repeats the irreps list for the two spin channels. e3nn-jax's
        # integer multiplication scales multiplicities instead.
        self.potential_irreps = Irreps(list(self.field_irreps) * 2)

        self.lr_source_maps = nnx.List(
            [
                EnvironmentDependentSpinSourceBlock(
                    hidden, atomic_multipoles_max_l, rngs=rngs
                )
                for _ in range(self.num_interactions)
            ]
        )
        self.layer_feature_mixer = MultiLayerFeatureMixer(
            hidden, self.num_interactions, rngs=rngs
        )
        self.fukui_source_map = NonLinearBiasReadoutBlock(
            hidden,
            self._mlp_irreps.simplify(),
            self.gate,
            Irreps('2x0e'),
            1,
            self.equivariance_config,
            rngs=rngs,
        )
        node_attrs_irreps = Irreps(f'{self.num_elements}x0e')
        self.field_dependent_charges_maps = nnx.List(
            [
                AgnosticEmbeddedOneBodyVariableUpdate(
                    node_attrs_irreps,
                    hidden,
                    self.potential_irreps,
                    self.charges_irreps,
                    rngs=rngs,
                )
                for _ in range(num_recursion_steps)
            ]
        )
        self.local_electron_energy = OneBodyMLPFieldReadout(
            hidden, self.potential_irreps, self.charges_irreps, rngs=rngs
        )
        self.electrostatics = RealSpacePolarElectrostatics(
            atomic_multipoles_max_l,
            atomic_multipoles_smearing_width,
            field_feature_max_l,
            self.field_feature_widths,
            include_field_self_interaction=field_si,
            include_energy_self_interaction=include_electrostatic_self_interaction,
        )
        self.periodic_electrostatics = PeriodicPolarElectrostatics(
            atomic_multipoles_max_l,
            atomic_multipoles_smearing_width,
            field_feature_max_l,
            self.field_feature_widths,
            kspace_cutoff_factor=kspace_cutoff_factor,
            kspace_cutoff=kspace_cutoff,
            include_field_self_interaction=field_si,
            include_energy_self_interaction=include_electrostatic_self_interaction,
        )
        self.external_field_contribution = DisplacedGTOExternalFieldBlock(
            field_feature_max_l, self.field_feature_widths
        )
        if field_feature_norms is None:
            field_feature_norms = (1.0,) * (
                len(self.field_feature_widths) * (field_feature_max_l + 1)
            )
        if len(field_feature_norms) != len(self.field_feature_widths) * (
            field_feature_max_l + 1
        ):
            raise ValueError('field_feature_norms has the wrong length')
        expanded_norms = []
        for ell in range(field_feature_max_l + 1):
            for width_index in range(len(self.field_feature_widths)):
                expanded_norms.extend(
                    [
                        field_feature_norms[
                            ell * len(self.field_feature_widths) + width_index
                        ]
                    ]
                    * (2 * ell + 1)
                )
        self.field_feature_norms = jnp.asarray(expanded_norms)

    def __call__(
        self, data: dict[str, jnp.ndarray], *, debug: bool = False
    ) -> dict[str, Any]:
        pbc = data.get('pbc')
        periodic = self.pbc_handling in (
            'pbc',
            'slab',
            'molecule_in_box',
            'mixed_periodic',
        ) or (self.pbc_handling == 'auto' and pbc is not None and bool(jnp.any(pbc)))
        if self.pbc_handling not in (
            'realspace',
            'auto',
            'pbc',
            'slab',
            'molecule_in_box',
            'mixed_periodic',
        ):
            raise NotImplementedError(
                f'POLAR electrostatics mode {self.pbc_handling!r} is not ported yet'
            )
        periodic_mode = (
            'mixed_periodic' if self.pbc_handling == 'auto' else self.pbc_handling
        )
        if periodic and pbc is None:
            raise ValueError('Periodic POLAR requires per-graph pbc data')

        backbone = ScaleShiftMACE._energy_fn(self, data, compute_node_feats=True)
        batch = data['batch']
        positions = data['positions']
        periodic_cache = (
            self.periodic_electrostatics.precompute(
                positions, batch, data['cell'], data.get('reference_cell')
            )
            if periodic
            else None
        )
        node_attrs = data['node_attrs']
        num_graphs = int(data['ptr'].shape[0] - 1)
        node_feats_list = jnp.split(
            backbone['node_feats'], self.num_interactions, axis=-1
        )
        final_node_feats = node_feats_list[-1]
        n_nodes = positions.shape[0]
        multipole_dim = Irreps.spherical_harmonics(self.atomic_multipoles_max_l).dim

        spin_charge_density = jnp.zeros(
            (n_nodes, self.charges_irreps.dim), dtype=positions.dtype
        )
        source_outputs = []
        for node_feats, source_map in zip(node_feats_list, self.lr_source_maps):
            source_output = _array(source_map(node_feats))[:, 0, :]
            source_outputs.append(source_output)
            spin_charge_density = spin_charge_density + source_output
        spin_charge_density = spin_charge_density.reshape(n_nodes, 2, multipole_dim)
        mixed_features = self.layer_feature_mixer(jnp.stack(node_feats_list, axis=0))
        fukui = _array(self.fukui_source_map(final_node_feats))
        fukui_raw = fukui
        fukui_norm = scatter_sum(fukui, batch, dim=0, dim_size=num_graphs)[batch]
        fukui = fukui / jnp.where(fukui_norm == 0, 1.0, fukui_norm)

        total_charge_input = jnp.asarray(data['total_charge']).reshape(-1)
        total_spin_input = jnp.asarray(data['total_spin']).reshape(-1)
        targets = jnp.stack(
            (
                (total_charge_input + total_spin_input - 1) / 2,
                (total_charge_input - total_spin_input + 1) / 2,
            ),
            axis=-1,
        )

        def constrain(density: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
            prediction = scatter_sum(
                density[:, :, 0], batch, dim=0, dim_size=num_graphs
            )[batch]
            corrections = weights * (targets[batch] - prediction)
            return density.at[:, :, 0].add(corrections)

        spin_charge_density = constrain(spin_charge_density, fukui)
        initial_density = spin_charge_density
        field_independent_density = spin_charge_density
        external_field = jnp.asarray(data['external_field']).reshape(num_graphs, 3)
        # The released Torch forward uses a zero scalar potential even when
        # data includes a nonzero fermi_level.
        external_potential = jnp.concatenate(
            (
                jnp.zeros((num_graphs, 1), dtype=positions.dtype),
                external_field,
            ),
            axis=-1,
        )
        counts = scatter_sum(
            jnp.ones((n_nodes,), dtype=positions.dtype),
            batch,
            dim=0,
            dim_size=num_graphs,
        )
        barycenter = (
            scatter_sum(positions, batch, dim=0, dim_size=num_graphs) / counts[:, None]
        )
        half_external = 0.5 * self.external_field_contribution(
            batch, positions - barycenter[batch], external_potential
        )

        potential_features = jnp.zeros(
            (n_nodes, self.potential_irreps.dim), dtype=positions.dtype
        )
        final_fukui = fukui
        field_steps = []
        update_steps = []
        density_steps = []
        for update in self.field_dependent_charges_maps:
            if periodic_cache is None:
                alpha = self.electrostatics.field_features(
                    spin_charge_density[:, 0, :], positions, batch
                )
                beta = self.electrostatics.field_features(
                    spin_charge_density[:, 1, :], positions, batch
                )
            else:
                alpha = self.periodic_electrostatics.field_features(
                    spin_charge_density[:, 0, :],
                    periodic_cache,
                    mode=periodic_mode,
                    pbc=pbc,
                )
                beta = self.periodic_electrostatics.field_features(
                    spin_charge_density[:, 1, :],
                    periodic_cache,
                    mode=periodic_mode,
                    pbc=pbc,
                )
            alpha = (alpha + half_external) / self.field_feature_norms
            beta = (beta + half_external) / self.field_feature_norms
            potential_features = jnp.concatenate((alpha, beta), axis=-1)
            field_steps.append(potential_features)
            output = update(
                node_attrs,
                mixed_features,
                potential_features,
                spin_charge_density.reshape(n_nodes, -1),
            )
            update_steps.append(output)
            final_fukui = output[:, -2:]
            sources = output[:, :-2].reshape(n_nodes, 2, multipole_dim)
            spin_charge_density = spin_charge_density + sources
            norm = scatter_sum(final_fukui, batch, dim=0, dim_size=num_graphs)[batch]
            final_fukui = final_fukui / jnp.where(norm == 0, 1.0, norm)
            spin_charge_density = constrain(spin_charge_density, final_fukui)
            density_steps.append(spin_charge_density)

        local_node_energy = self.local_electron_energy(
            final_node_feats,
            potential_features,
            field_independent_density.reshape(n_nodes, -1),
            spin_charge_density.reshape(n_nodes, -1),
        )
        electron_energy = scatter_sum(
            local_node_energy, batch, dim=0, dim_size=num_graphs
        )
        if not self.add_local_electron_energy:
            electron_energy = jnp.zeros_like(electron_energy)

        density = jnp.sum(spin_charge_density, axis=1)
        spin_density = spin_charge_density[:, 0, :] - spin_charge_density[:, 1, :]
        total_charge = scatter_sum(density[:, 0], batch, dim=0, dim_size=num_graphs)
        dipole = scatter_sum(
            positions * density[:, :1], batch, dim=0, dim_size=num_graphs
        )
        if multipole_dim > 1:
            dipole = dipole + scatter_sum(
                density[:, 1:4][:, jnp.asarray([2, 0, 1])],
                batch,
                dim=0,
                dim_size=num_graphs,
            )
        if periodic_cache is None:
            electrostatic_energy = self.electrostatics.coulomb_energy(
                density, positions, batch, num_graphs
            )
        else:
            electrostatic_energy = self.periodic_electrostatics.coulomb_energy(
                density,
                periodic_cache,
                mode=periodic_mode,
                pbc=pbc,
            )
        total_energy = (
            backbone['energy']
            + electron_energy
            + electrostatic_energy
            + jnp.sum(external_field * dipole, axis=-1)
        )
        result = {
            **backbone,
            'energy': total_energy,
            'electron_energy': electron_energy,
            'electrostatic_energy': electrostatic_energy,
            'density_coefficients': density,
            'spin_density': spin_density,
            'spin_charge_density': spin_charge_density,
            'charges_history': spin_charge_density[..., None],
            'charges': density[:, 0],
            'spins': spin_density[:, 0],
            'dipole': dipole,
            'total_charge': total_charge,
            'fukui_functions': final_fukui,
            'external_field': external_field,
            'fermi_level': external_potential[:, 0],
            'electrostatic_potentials': None,
        }
        if debug:
            result['_debug'] = {
                'sources': jnp.stack(source_outputs),
                'mixed_features': mixed_features,
                'fukui_raw': fukui_raw,
                'initial_density': initial_density,
                'fields': jnp.stack(field_steps),
                'updates': jnp.stack(update_steps),
                'densities': jnp.stack(density_steps),
                'local_node_energy': local_node_energy,
            }
        return result
