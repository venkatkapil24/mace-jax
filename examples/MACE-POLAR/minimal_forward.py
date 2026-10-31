"""Full-JIT MACE-POLAR forward pass. Set the two file paths below.

Requires the float64/cuEq bundle and validated periodic water input.
"""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax_md import partition, space

from mace_jax.tools.bundle import load_model_bundle

MODEL = "polar-cueq.msgpack"
INPUT = "input-graph.npz"
MAX_NEIGHBORS = 120
EDGE_CAPACITY = 93_000

jax.config.update('jax_enable_x64', True)
jax.config.update('jax_default_matmul_precision', 'highest')

def as_array(value):
    value = jnp.asarray(value)
    return value.astype(jnp.float64) if jnp.issubdtype(value.dtype, jnp.floating) else value

bundle = load_model_bundle(MODEL, 'float64')
bundle = replace(bundle, params=jax.tree.map(as_array, bundle.params))
model = nnx.merge(bundle.graphdef, bundle.params)
with np.load(INPUT) as archive:
    data = {key: as_array(archive[key]) for key in archive.files}
mode, data = model.prepare_jit_data(
    data, pbc_handling='pbc', generalized_pme=True,
    pme_mesh_spacing=1.2, pme_assignment_order=4,
)
apply = bundle.make_frozen_apply()
positions, cell = data['positions'], data['cell'].reshape(3, 3)
cutoff = float(bundle.config['r_max'])
box = np.asarray(cell)
assert np.allclose(box, np.diag(np.diag(box))) and np.min(np.diag(box)) > 2*cutoff
assert np.all(np.asarray(data['pbc'])) and len(data['ptr']) == 2

displacement, _ = space.periodic_general(cell.T, fractional_coordinates=True)
neighbor_fn = partition.neighbor_list(
    displacement, cell.T, cutoff, dr_threshold=0., capacity_multiplier=1.,
    format=partition.Dense, fractional_coordinates=True,
)
fractional = positions @ jnp.linalg.inv(cell)
neighbors = neighbor_fn.allocate(fractional)
extra = MAX_NEIGHBORS-neighbors.idx.shape[1]
assert extra >= 0, 'Increase MAX_NEIGHBORS'
neighbors = neighbor_fn.allocate(fractional, extra_capacity=extra)

@jax.jit
def forward(r, h, neighbors):
    neighbors = neighbor_fn.update(r @ jnp.linalg.inv(h), neighbors, box=h.T)
    n = len(r)
    receivers = neighbors.idx.reshape(-1)
    senders = jnp.repeat(jnp.arange(n, dtype=jnp.int32), neighbors.idx.shape[1])
    valid = (receivers >= 0) & (receivers < n)
    count = jnp.sum(valid)
    selected = jnp.nonzero(valid, size=EDGE_CAPACITY, fill_value=0)[0]
    occupied = jnp.arange(EDGE_CAPACITY) < count
    senders = jnp.where(occupied, senders[selected], 0)
    receivers = jnp.where(occupied, receivers[selected], 0).astype(jnp.int32)
    units = -jnp.round((r[receivers]-r[senders]) @ jnp.linalg.inv(h))
    # Unused edges lie beyond cutoff, with a nonzero distance for safe AD.
    units = jnp.where(occupied[:, None], units, jnp.asarray([2., 0., 0.]))
    payload = dict(
        data, positions=r, cell=h.reshape(data['cell'].shape),
        edge_index=jnp.stack((senders, receivers)), unit_shifts=units,
        shifts=units @ h, volume=jnp.linalg.det(h).reshape(data['volume'].shape),
        rcell=(2*jnp.pi*jnp.linalg.inv(h.T)).reshape(data['rcell'].shape),
    )
    output = apply(payload, compute_force=True, compute_stress=True, pbc_handling=mode)
    output = {key: output[key] for key in ['energy', 'forces', 'stress']}
    overflow = neighbors.did_buffer_overflow | (count > EDGE_CAPACITY)
    return output, neighbors, overflow, count

output, neighbors, overflow, count = jax.block_until_ready(forward(positions, cell, neighbors))
assert not bool(overflow), 'Increase graph/neighbor capacity and recompile'
print('Energy (eV):', np.asarray(output['energy']))
