"""Checks for the CSVR/REMD bridge against the native integrator."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytest.importorskip('jax_md.replica_exchange', reason='Requires the user JAX-MD REMD branch')
from jax_md import simulate, space
from mace_jax.modules.replica_exchange import PolarReplicaState, make_exchange
from mace_jax.tools.physical_constants import ACCELERATION, OPENMM_BOLTZMANN

jax.config.update('jax_enable_x64', True)

class Neighbors(NamedTuple):
    did_buffer_overflow: object
class NeighborFns:
    def update(self, position, neighbors):
        return neighbors

def setup(stride=10):
    r = jnp.array([[.2, .1, -.3], [-.2, -.1, .3]])
    v = jnp.array([[1., 2., -1.], [-1., -2., 1.]])
    masses = jnp.array([[2.], [2.]])
    n = Neighbors(jnp.array(False))
    force_eval = jax.jit(jax.value_and_grad(lambda r, n, k: .5*k*jnp.sum(r*r)))
    _, shift = space.free()
    s = PolarReplicaState(r, v, n, force_eval(r, n, 2.)[0], 2*r,
                          jax.random.PRNGKey(3), jnp.array(False), jnp.array(0))
    init, apply = make_exchange(force_eval, NeighborFns(), shift, masses, n,
                               [300, 400], stride=stride)
    stacked = jax.tree.map(lambda x: jnp.stack([x, x]), s)
    state = init(jax.random.PRNGKey(7), stacked)
    return s, state, apply, force_eval, shift, masses

def test_csvr_step_matches_native_integrator_and_units():
    s, state, apply, _, shift, masses = setup()
    _, native = simulate.temp_csvr(lambda r, **kw: -2*r, shift, .0005, 1., .1,
                                  fix_center_of_mass=True)
    mass = masses/ACCELERATION
    md = simulate.NVTLangevinState(s.position, s.velocity*mass, -s.gradient,
                                  mass, s.rng, fix_center_of_mass=True)
    expected = jax.jit(lambda s: native(s, kT=300*OPENMM_BOLTZMANN))(md)
    actual = jax.jit(lambda s: apply(s, model_params=2.))(state).replicas
    np.testing.assert_allclose(actual.position[0], expected.position, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.velocity[0], expected.velocity, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(actual.gradient[0], 2*actual.position[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(.5*jnp.sum(mass*expected.velocity**2),
                               .5*jnp.sum(masses*actual.velocity[0]**2)/ACCELERATION)

def test_exchange_refreshes_total_energy_and_rescales_velocity():
    _, state, apply, force_eval, _, _ = setup(stride=1)
    actual = jax.jit(lambda s: apply(s, model_params=2.))(state)
    energy = jax.vmap(lambda r: force_eval(r, Neighbors(False), 2.)[0])(actual.replicas.position)
    np.testing.assert_allclose(actual.replicas.potential_energy, energy, rtol=1e-9, atol=1e-9)
    assert np.all(np.asarray(actual.replicas.md_steps) == 1)
    assert sorted(np.asarray(actual.ensemble_indices).tolist()) == [0, 1]
    assert not np.any(actual.replicas.invalid)
    s = jax.tree.map(lambda x: x[0], actual.replicas)
    rescaled = simulate.rescale_temperature(s, 300., 400.)
    np.testing.assert_allclose(rescaled.velocity, s.velocity*jnp.sqrt(4/3))
    assert float(simulate.log_probability(s, lambda r: 999., kT=2.)) == -float(s.potential_energy)/2

def test_restart_preserves_rng_labels_neighbors_and_exact_continuation(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'examples/HydrogenMaleate-MACE-POLAR'))
    from remd import save_state, restore_state
    _, state, apply, _, _, _ = setup(stride=1)
    step = jax.jit(lambda s: apply(s, model_params=2.))
    state = step(step(state))
    checkpoint = tmp_path/'checkpoint.npz'
    save_state(checkpoint, state)
    restored = restore_state(checkpoint, state)
    uninterrupted, resumed = step(state), step(restored)
    for a, b in zip(jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(a, b)

def test_uint8_partition_error_keeps_boolean_scan_state():
    _, state, apply, _, _, _ = setup(stride=1)
    state=state.set(replicas=state.replicas.set(
        neighbors=Neighbors(jnp.ones(2,dtype=jnp.uint8))))
    actual=jax.jit(lambda s: jax.lax.fori_loop(0,2,lambda _, s:apply(s,model_params=2.),s))(state)
    assert actual.replicas.invalid.dtype == jnp.bool_
    assert np.all(actual.replicas.invalid)
