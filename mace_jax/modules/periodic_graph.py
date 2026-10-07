"""Image-aware cubic periodic graphs on top of a JAX-MD neighbour list."""

import itertools
import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class ImageGraph(NamedTuple):
    base: object
    edge_index: object
    unit_shifts: object
    image_overflow: object

    @property
    def idx(self):
        return self.edge_index

    @property
    def did_buffer_overflow(self):
        return self.base.did_buffer_overflow | self.image_overflow


class ImageGraphNeighborFns:
    """Keep valid copies of each local edge, with fixed-capacity JIT outputs.

    Only needed when the ML cutoff exceeds half the box side. The expensive
    atomic search still uses JAX-MD; candidate image copies are compacted before
    they reach the neural network. Capacity overflow is reported to the caller.
    """

    def __init__(self, base, box, cutoff, capacity_multiplier=1.3):
        self.base = base
        self.box = box
        self.cutoff = cutoff
        self.capacity_multiplier = capacity_multiplier
        extent = math.ceil(cutoff / box)
        self.offsets = jnp.asarray(
            list(itertools.product(range(-extent, extent + 1), repeat=3)),
            dtype=jnp.int32,
        )

    def candidates(self, positions, base):
        n = len(positions)
        pair = jnp.concatenate(
            (
                base.idx,
                jnp.stack(
                    (jnp.arange(n, dtype=jnp.int32), jnp.arange(n, dtype=jnp.int32))
                ),
            ),
            axis=1,
        )
        valid = (pair[0] < n) & (pair[1] < n)
        safe = jnp.where(valid[None, :], pair, 0)
        d = positions[safe[1]] - positions[safe[0]]
        nearest = -jnp.round(d / self.box).astype(jnp.int32)
        unit = nearest[:, None, :] + self.offsets[None, :, :]
        vectors = d[:, None, :] + unit * self.box
        self_zero = (safe[0] == safe[1])[:, None] & jnp.all(unit == 0, axis=-1)
        keep = (
            valid[:, None]
            & ~self_zero
            & (jnp.sum(vectors * vectors, axis=-1) < self.cutoff**2)
        )
        pairs = jnp.broadcast_to(
            safe[:, :, None], (2, pair.shape[1], len(self.offsets))
        ).reshape(2, -1)
        return pairs, unit.reshape(-1, 3), keep.reshape(-1)

    def pack(self, positions, base, capacity):
        pair, unit, keep = self.candidates(positions, base)
        selected = jnp.nonzero(keep, size=capacity, fill_value=0)[0]
        count = jnp.sum(keep)
        valid = jnp.arange(capacity) < count
        edges = jnp.where(valid[None, :], pair[:, selected], len(positions)).astype(
            jnp.int32
        )
        units = jnp.where(valid[:, None], unit[selected], 0)
        return ImageGraph(base, edges, units, count > capacity)

    def allocate(self, positions):
        base = self.base.allocate(positions)
        _, _, keep = self.candidates(positions, base)
        capacity = max(
            32, math.ceil(int(np.asarray(keep).sum()) * self.capacity_multiplier) + 16
        )
        return self.pack(positions, base, capacity)

    def update(self, positions, neighbors):
        base = self.base.update(positions, neighbors.base)
        return self.pack(positions, base, neighbors.edge_index.shape[1])
