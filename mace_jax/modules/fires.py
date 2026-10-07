"""FIRES ordering restraint in distance to the nearest solute heavy atom."""

import jax.numpy as jnp


def fires_energy(qm_oxygen, mm_oxygen, heavy, box, k):
    """Return energy for fixed water identities in an orthorhombic periodic box.

    Positions and ``box`` use the same length unit; ``k`` is energy/length².
    ``qm_oxygen`` and ``mm_oxygen`` contain water oxygen positions, and ``heavy``
    contains the solute heavy atom positions. The moving boundary is the maximum
    nearest-heavy-atom distance of the QM waters. Autodiff must include that
    boundary's dependence on both water and solute coordinates.
    """
    if qm_oxygen.shape[0] == 0 or mm_oxygen.shape[0] == 0:
        return jnp.asarray(0.0, dtype=heavy.dtype)
    oxygen = jnp.concatenate((qm_oxygen, mm_oxygen), axis=0)
    delta = oxygen[:, None] - heavy[None]
    delta -= box * jnp.round(delta / box)
    # Reduce before taking square roots: one root per water, not per pair.
    distance = jnp.sqrt(jnp.min(jnp.sum(delta * delta, axis=-1), axis=-1))
    boundary = jnp.max(distance[: qm_oxygen.shape[0]])
    intrusion = jnp.maximum(boundary - distance[qm_oxygen.shape[0] :], 0.0)
    return 0.5 * k * jnp.sum(intrusion * intrusion)
