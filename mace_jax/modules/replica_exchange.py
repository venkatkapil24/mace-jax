"""Connect cached MACE-POLAR forces to JAX-MD's temperature-exchange API.

Requires venkatkapil24/jax-md working-branch. Units match our existing
MD driver: Angstrom, ps, dalton and eV. Model parameters are shared
runtime inputs, and each continuous trajectory owns its neighbor lists.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax_md import dataclasses, replica_exchange, simulate

from mace_jax.tools.physical_constants import ACCELERATION, OPENMM_BOLTZMANN


@dataclasses.dataclass
class PolarReplicaState:
    position: Any
    velocity: Any
    neighbors: Any
    potential_energy: Any
    gradient: Any
    rng: Any
    invalid: Any
    md_steps: Any


@simulate.log_probability.register(PolarReplicaState)
def _log_probability(state, energy_fn, kT, **kwargs):
    # The energy cache is refreshed at every exchange step, including FIRES.
    return -state.potential_energy / kT


@simulate.rescale_temperature.register(PolarReplicaState)
def _rescale_temperature(state, kT, new_kT, **kwargs):
    return state.set(velocity=state.velocity * jnp.sqrt(new_kT / kT))


def make_exchange(
    force_eval, neighbor_fn, shift_fn, masses, template_neighbors, temperatures,
    *, timestep_ps=.0005, thermostat_tau_ps=.1, stride=10,
):
    """Return JAX-MD exchange init/apply functions using its native CSVR."""
    # In Angstrom/ps and eV, JAX-MD's p/m convention requires this mass factor.
    md_mass = masses / ACCELERATION

    # Use JAX-MD's native CSVR thermostat with a stationary, zero-force
    # Verlet step. We then perform the physical Verlet step below, retaining
    # both energy and gradient from one model call. This avoids an additional
    # energy-only model call and a vmapped conditional at exchange steps.
    _, apply_csvr = simulate.temp_csvr(
        lambda position, **unused: jnp.zeros_like(position),
        lambda position, delta, **unused: position,
        dt=timestep_ps, kT=1., tau=thermostat_tau_ps,
        fix_center_of_mass=True,
    )
    dt = jnp.float32(timestep_ps)
    half_dt = jnp.float32(dt / 2)

    def step_md(state, *, kT, model_params):
        md = simulate.NVTLangevinState(
            state.position, state.velocity * md_mass, jnp.zeros_like(state.gradient),
            md_mass, state.rng, fix_center_of_mass=True,
        )
        md = apply_csvr(md, kT=kT)
        md = simulate.momentum_step(md.set(force=-state.gradient), half_dt)
        md = simulate.position_step(md, shift_fn, dt)
        position = md.position
        neighbors = neighbor_fn.update(position, state.neighbors)
        md_steps = state.md_steps + 1
        energy, gradient = force_eval(position, neighbors, model_params)
        md = simulate.momentum_step(md.set(force=-gradient), half_dt)
        velocity = md.velocity
        # JAX-MD's partition error bits can be uint8, rather than bool.
        invalid = (state.invalid | jnp.asarray(neighbors.did_buffer_overflow, dtype=bool)
                   | ~jnp.isfinite(energy) | ~jnp.all(jnp.isfinite(gradient))
                   | ~jnp.all(jnp.isfinite(velocity))
                   | ~jnp.all(jnp.isfinite(position)))
        return state.set(position=position, velocity=velocity, neighbors=neighbors,
                         potential_energy=energy, gradient=gradient,
                         rng=md.rng, invalid=invalid, md_steps=md_steps)

    def energy_fn(position, *, model_params):
        # Supply the shared physical energy function required by JAX-MD. Our
        # registered log_probability uses the state cache instead; JAX removes
        # this unused evaluation from the compiled exchange calculation.
        neighbors = neighbor_fn.update(position, template_neighbors)
        return force_eval(position, neighbors, model_params)[0]

    return replica_exchange.replica_exchange(
        energy_fn, step_md, jnp.asarray(temperatures) * OPENMM_BOLTZMANN,
        stride=stride,
    )
