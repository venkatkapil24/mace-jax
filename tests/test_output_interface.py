"""Energy, derivative, and auxiliary outputs share one model evaluation."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.utils import (
    add_output_interface, compute_forces_and_stress, get_outputs,
)


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


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('precision', ['highest', 'tensorfloat32'])
def test_vmapped_stress_uses_each_systems_triclinic_cell(dtype, precision):
    with jax.enable_x64(True), jax.default_matmul_precision(precision):
        data = geometry()
        second = dict(data, positions=data['positions'] * 1.3,
                      cell=data['cell'] * jnp.array([1.1, .9, 1.2]))
        second['shifts'] = second['unit_shifts'] @ second['cell'][0]
        systems = jax.tree.map(lambda a, b: jnp.stack((a, b)), data, second)
        systems = jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            systems,
        )
        model = QuadraticModel()
        result = jax.jit(jax.vmap(lambda d: model(
            d, compute_force=True, compute_stress=True,
        )))(systems)
        for i in range(2):
            r = np.asarray(systems['positions'][i], np.float64)
            h = np.asarray(systems['cell'][i, 0], np.float64)
            shifts = np.asarray(systems['unit_shifts'][i], np.float64) @ h
            energy = .5*np.sum(r*r)+.25*np.sum(h*h)+.125*np.sum(shifts*shifts)
            stress = (r.T@r+.5*h.T@h+.25*shifts.T@shifts)/abs(np.linalg.det(h))
            tolerance = 5e-6 if dtype == 'float32' else 1e-12
            np.testing.assert_allclose(result['energy'][i], [energy], rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(result['forces'][i], -r, rtol=tolerance, atol=tolerance)
            np.testing.assert_allclose(result['stress'][i], [stress], rtol=tolerance, atol=tolerance)
            assert result['stress'].dtype == jnp.dtype(dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('packed', [False, True])
def test_stress_helpers_match_analytic_pair_energy(dtype, packed):
    """Exercise both public stress paths and the packed-graph fallback."""
    with jax.enable_x64(True):
        data = geometry()
        if packed:
            h2 = data['cell'] * jnp.array([1.1, .9, 1.2])
            data = dict(data, positions=jnp.concatenate((data['positions'], 1.3*data['positions'])),
                        cell=jnp.concatenate((data['cell'], h2)),
                        unit_shifts=jnp.tile(data['unit_shifts'], (2, 1)),
                        edge_index=jnp.concatenate((data['edge_index'], data['edge_index']+2), axis=1),
                        batch=jnp.array([0, 0, 1, 1], dtype=jnp.int32),
                        ptr=jnp.array([0, 2, 4], dtype=jnp.int32))
        data = jax.tree.map(
            lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            data,
        )
        def energy(pos, shifts=None, cell=None):
            delta = pos[data['edge_index'][1]]-pos[data['edge_index'][0]]+shifts
            return jax.ops.segment_sum(
                .5*jnp.sum(delta**2, axis=-1), data['batch'][data['edge_index'][0]],
                num_segments=len(data['ptr'])-1,
            )
        output = jax.jit(lambda d: get_outputs(energy, d, True, True))(data)
        forces, stress = jax.jit(lambda d: compute_forces_and_stress(
            energy, d['positions'], d['cell'], d['unit_shifts'], d['edge_index'],
            d['batch'], len(d['ptr'])-1,
        ))(data)
        r = np.asarray(data['positions'], np.float64)
        h = np.asarray(data['cell'], np.float64)
        edges = np.asarray(data['edge_index'])
        group = np.asarray(data['batch'])[edges[0]]
        shifts = np.einsum('ei,eij->ej', np.asarray(data['unit_shifts'], np.float64), h[group])
        delta = r[edges[1]]-r[edges[0]]+shifts
        expected_forces = np.zeros_like(r)
        np.add.at(expected_forces, edges[0], delta)
        np.add.at(expected_forces, edges[1], -delta)
        expected_stress = np.zeros_like(h)
        np.add.at(expected_stress, group, delta[:, :, None]*delta[:, None, :])
        expected_stress /= abs(np.linalg.det(h))[:, None, None]
        expected_energy = np.zeros(len(h))
        np.add.at(expected_energy, group, .5*np.sum(delta**2, axis=-1))
        tolerance = 5e-6 if dtype == 'float32' else 1e-12
        np.testing.assert_allclose(output.energy, expected_energy, rtol=tolerance, atol=tolerance)
        for actual in [output.forces, forces]:
            np.testing.assert_allclose(actual, expected_forces, rtol=tolerance, atol=tolerance)
        for actual in [output.stress, stress]:
            np.testing.assert_allclose(actual, expected_stress, rtol=tolerance, atol=tolerance)
