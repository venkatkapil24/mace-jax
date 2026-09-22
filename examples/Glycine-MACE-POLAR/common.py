"""Shared periodic MACE-POLAR helpers for the glycine examples."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_md import partition, space

from mace_jax.tools.bundle import load_model_bundle


def model_data(atoms, config):
    numbers = np.asarray(atoms.numbers)
    z_to_index = {int(z): i for i, z in enumerate(config['atomic_numbers'])}
    species = jnp.asarray([z_to_index[int(z)] for z in numbers], dtype=jnp.int32)
    n = len(atoms)
    box = float(atoms.cell[0, 0])
    return {
        'positions': jnp.asarray(atoms.positions, dtype=jnp.float64),
        'node_attrs': jax.nn.one_hot(species, len(z_to_index), dtype=jnp.float64),
        'node_attrs_index': species,
        'edge_index': jnp.zeros((2, 1), dtype=jnp.int32),
        'shifts': jnp.zeros((1, 3), dtype=jnp.float64),
        'unit_shifts': jnp.zeros((1, 3), dtype=jnp.float64),
        'batch': jnp.zeros(n, dtype=jnp.int32),
        'ptr': jnp.asarray([0, n], dtype=jnp.int32),
        'cell': jnp.eye(3, dtype=jnp.float64)[None] * box,
        'pbc': jnp.asarray([[True, True, True]]),
        'head': jnp.asarray([0], dtype=jnp.int32),
        'total_charge': jnp.asarray([0.0], dtype=jnp.float64),
        'total_spin': jnp.asarray([1.0], dtype=jnp.float64),
        'external_field': jnp.zeros((1, 3), dtype=jnp.float64),
    }


def graph_edges(positions, neighbors, box):
    n = positions.shape[0]
    sender, receiver = neighbors.idx
    valid = (sender < n) & (receiver < n)
    sender = jnp.where(valid, sender, 0).astype(jnp.int32)
    receiver = jnp.where(valid, receiver, 0).astype(jnp.int32)
    delta = positions[receiver] - positions[sender]
    unit = -jnp.round(delta / box)
    shifts = unit * box
    shifts = jnp.where(valid[:, None], shifts, jnp.array([box, 0.0, 0.0]))
    unit = jnp.where(valid[:, None], unit, jnp.array([1.0, 0.0, 0.0]))
    return jnp.stack((sender, receiver)), shifts, unit


def initialize_model(atoms, bundle_path, capacity_multiplier=1.3):
    jax.config.update('jax_enable_x64', True)
    box = float(atoms.cell[0, 0])
    if not np.all(atoms.pbc) or not np.allclose(atoms.cell.array, np.eye(3) * box):
        raise ValueError('Expected a cubic periodic cell')
    bundle = load_model_bundle(str(bundle_path), 'float64')
    model = nnx.merge(bundle.graphdef, bundle.params)
    if model.__class__.__name__ != 'PolarMACE':
        raise ValueError('Expected MACE-POLAR bundle')
    mode, data = model.prepare_jit_data(model_data(atoms, bundle.config), pbc_handling='pbc')
    graphdef, params = nnx.split(model)
    displacement, shift = space.periodic(box)
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        float(bundle.config['r_max']),
        dr_threshold=0.25,
        capacity_multiplier=capacity_multiplier,
        format=partition.Sparse,
    )
    return box, mode, data, graphdef, params, neighbor_fn, shift
