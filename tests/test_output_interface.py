"""Energy, derivative, and auxiliary outputs share one model evaluation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.utils import add_output_interface, get_outputs


@add_output_interface
class QuadraticModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, data, *, scale=1.):
        self.calls += 1
        r, h, shifts = data['positions'], data['cell'], data['shifts']
        energy = scale * (.5*jnp.sum(r*r) + .25*jnp.sum(h*h)
                          + .125*jnp.sum(shifts*shifts))
        return {'energy': energy[None], 'position_sum': jnp.sum(r, axis=0),
                'cell_sum': jnp.sum(h), 'optional': None}


def geometry():
    r = jnp.array([[.2, .3, .4], [1.1, .7, .6]], dtype=jnp.float64)
    h = jnp.array([[4., .2, 0.], [0., 5., .1], [.3, 0., 6.]], dtype=r.dtype)
    units = jnp.array([[1., 0., 0.], [-1., 0., 0.]], dtype=r.dtype)
    return dict(positions=r, cell=h[None], unit_shifts=units, shifts=units@h,
                batch=jnp.zeros(2, dtype=jnp.int32), ptr=jnp.array([0, 2]),
                edge_index=jnp.array([[0, 1], [1, 0]], dtype=jnp.int32))


def expected(data, scale):
    r, h, shifts = data['positions'], data['cell'][0], data['shifts']
    energy = scale*(.5*jnp.sum(r*r)+.25*jnp.sum(h*h)+.125*jnp.sum(shifts*shifts))
    stress = scale*(r.T@r+.5*h.T@h+.25*shifts.T@shifts)/jnp.abs(jnp.linalg.det(h))
    return energy[None], -scale*r, stress[None]


@pytest.mark.parametrize('force,stress', [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('compiled', [False,True])
def test_one_forward_and_analytic_outputs(force, stress, compiled):
    with jax.enable_x64(True):
        model, data = QuadraticModel(), geometry()
        fn = lambda d: model(d, compute_force=force, compute_stress=stress, scale=1.7)
        if compiled:
            fn = jax.jit(fn)
        out = jax.block_until_ready(fn(data))
        assert model.calls == 1
        energy, forces, pressure = expected(data, 1.7)
        np.testing.assert_allclose(out['energy'], energy, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(out['position_sum'], jnp.sum(data['positions'],axis=0))
        np.testing.assert_allclose(out['cell_sum'], jnp.sum(data['cell']))
        assert out['optional'] is None
        if force or stress:
            np.testing.assert_allclose(out['forces'], forces, atol=1e-12, rtol=1e-12)
        else:
            assert out['forces'] is None
        if stress:
            np.testing.assert_allclose(out['stress'], pressure, atol=1e-12, rtol=1e-12)
        else:
            assert out['stress'] is None
        if compiled:
            def selected(d):
                result = model(d, compute_force=force, compute_stress=stress, scale=1.7)
                return {k:result[k] for k in ['energy','forces','stress']}
            reduced = jax.jit(selected)(data)
            for key in reduced:
                if reduced[key] is not None:
                    np.testing.assert_allclose(reduced[key], out[key], atol=1e-12, rtol=1e-12)
            assert model.calls == 2  # One trace for each independently compiled path.


def test_traced_flags_keep_auxiliary_outputs_and_masks():
    with jax.enable_x64(True):
        data, model = geometry(), QuadraticModel()
        fn = jax.jit(lambda d,f,s:model(d,compute_force=f,compute_stress=s))
        energy, forces, stress = expected(data, 1.)
        for f,s in [(False,False),(True,False),(False,True),(True,True)]:
            out = fn(data,jnp.asarray(f),jnp.asarray(s))
            np.testing.assert_allclose(out['energy'], energy)
            np.testing.assert_allclose(out['forces'], forces if f or s else jnp.zeros_like(forces))
            np.testing.assert_allclose(out['stress'], stress if s else jnp.zeros_like(stress))
            np.testing.assert_allclose(out['position_sum'], jnp.sum(data['positions'],axis=0))
            assert bool(out['forces_mask']) == (f or s)
            assert bool(out['stress_mask']) == s


def test_energy_array_get_outputs_api_is_unchanged():
    with jax.enable_x64(True):
        data = geometry()
        def energy(r, shifts=None, cell=None):
            return .5*jnp.sum(r*r)[None]
        out = jax.jit(lambda d:get_outputs(energy,d,True,True))(data)
        assert len(out) == 5
        np.testing.assert_allclose(out.energy, .5*jnp.sum(data['positions']**2)[None])
        np.testing.assert_allclose(out.forces, -data['positions'])


def test_decorated_scalar_energy_preserves_output_shape():
    @add_output_interface
    class ScalarModel:
        def __call__(self, data):
            return .5*jnp.sum(data['positions']**2)

    with jax.enable_x64(True):
        data = geometry()
        model = ScalarModel()
        for force in [False, True]:
            out = jax.jit(lambda d:model(d, compute_force=force))(data)
            assert out['energy'].shape == (1,)
            if force:
                np.testing.assert_allclose(out['forces'], -data['positions'])
