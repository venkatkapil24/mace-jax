"""Independent OpenMM checks for charge PME and Langevin-middle dynamics."""
# ruff: noqa: PLC0415
# Optional OpenMM imports remain inside tests so local tests can skip them.

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mace_jax.modules.langevin import kinetic_energy, middle_step
from mace_jax.modules.point_pme import PointPME, forward_spline_weights
from mace_jax.tools.physical_constants import KJ_MOL_PER_EV


def test_forward_spline_partition_and_derivative():
    for order in (4, 5, 8):
        x = jnp.linspace(0.001, 0.999, 21)
        weights = jax.jit(lambda t: forward_spline_weights(t, order))(x)
        np.testing.assert_allclose(weights.sum(-1), 1, atol=2e-11)
        assert np.min(weights) > -2e-11
        derivative = jax.jacfwd(lambda t: forward_spline_weights(t, order).sum())(0.37)
        np.testing.assert_allclose(derivative, 0, atol=2e-11)


@pytest.mark.parametrize('charges', [[0.8, -0.2, 0.3, -0.9], [0.8, -0.2, 0.3, -0.4]])
@pytest.mark.parametrize('exception', [False, True])
def test_point_pme_against_openmm_reference(charges, exception):
    om = pytest.importorskip('openmm')
    from jax.scipy.special import erf, erfc
    from openmm import unit

    r = jnp.asarray(
        [[0.3, 0.5, 0.7], [1.1, 0.9, 0.6], [4.1, 2.7, 1.2], [7.7, 6.2, 4.3]]
    )
    q = jnp.asarray(charges)
    box = 9.0
    cell = jnp.eye(3) * box
    pme = PointPME(0.53, (24, 24, 24))
    system = om.System()
    for _ in charges:
        system.addParticle(1)
    system.setDefaultPeriodicBoxVectors(*[om.Vec3(*v) for v in np.eye(3) * 0.9])
    force = om.NonbondedForce()
    force.setNonbondedMethod(force.PME)
    force.setCutoffDistance(0.3)
    force.setPMEParameters(5.3, 24, 24, 24)
    for charge in charges:
        force.addParticle(charge, 0.1, 0)
    if exception:
        force.addException(0, 1, 0, 0.1, 0)
        force.setExceptionsUsePeriodicBoundaryConditions(True)
    system.addForce(force)
    context = om.Context(
        system, om.VerletIntegrator(0.001), om.Platform.getPlatformByName('Reference')
    )
    context.setPositions(np.asarray(r) * 0.1)
    expected = context.getState(getEnergy=True, getForces=True)
    i, j = np.triu_indices(4, 1)

    def energy(pos):
        d = pos[i] - pos[j]
        d -= box * jnp.round(d / box)
        distance = jnp.linalg.norm(d, axis=-1)
        allowed = distance < 3.0
        if exception:
            allowed = allowed & ~((i == 0) & (j == 1))
        real = pme.coulomb * jnp.sum(
            jnp.where(allowed, q[i] * q[j] * erfc(pme.alpha * distance) / distance, 0)
        )
        correction = 0.0
        if exception:
            correction = (
                -pme.coulomb * q[0] * q[1] * erf(pme.alpha * distance[0]) / distance[0]
            )
        return (
            real
            + correction
            + pme.reciprocal_energy(pos, q, cell)
            + pme.self_energy(q)
            + pme.background_energy(q, cell)
        )

    actual, grad = jax.jit(jax.value_and_grad(energy))(r)
    np.testing.assert_allclose(
        actual,
        expected.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        / KJ_MOL_PER_EV,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        -grad,
        expected.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        / KJ_MOL_PER_EV
        / 10,
        atol=2e-10,
        rtol=0,
    )
    # A complete periodic translation must preserve this discretized energy.
    np.testing.assert_allclose(energy(r + box), actual, atol=2e-10)


@pytest.mark.parametrize('friction', [0.0, 1.0])
def test_langevin_middle_against_openmm_with_supplied_noise(friction):
    om = pytest.importorskip('openmm')
    from openmm import unit

    from mace_jax.tools.physical_constants import OPENMM_BOLTZMANN

    system = om.System()
    system.addParticle(12.011)
    system.addParticle(1.008)
    spring = om.CustomExternalForce('.5*k*(x*x+y*y+z*z)')
    spring.addGlobalParameter('k', 100.0)
    spring.addParticle(0, [])
    spring.addParticle(1, [])
    system.addForce(spring)
    dt = 0.0005
    temperature = 330.0
    decay = np.exp(-friction * dt)
    if friction == 0:
        integrator = om.LangevinMiddleIntegrator(temperature, friction, dt)
    else:
        integrator = om.CustomIntegrator(dt)
        integrator.addGlobalVariable('decay', decay)
        integrator.addGlobalVariable(
            'kt', OPENMM_BOLTZMANN * KJ_MOL_PER_EV * temperature
        )
        integrator.addPerDofVariable('eta', 0)
        integrator.addComputePerDof('v', 'v+dt*f/m')
        integrator.addComputePerDof('x', 'x+0.5*dt*v')
        integrator.addComputePerDof('v', 'decay*v+sqrt((1-decay*decay)*kt/m)*eta')
        integrator.addComputePerDof('x', 'x+0.5*dt*v')
    context = om.Context(system, integrator, om.Platform.getPlatformByName('Reference'))
    r = jnp.asarray([[1.0, 0.3, 0.2], [0.5, 0.2, 0.4]])
    v = jnp.asarray([[0.7, 0.5, 0.3], [1.4, 0.2, -0.3]])
    masses = jnp.asarray([[12.011], [1.008]])
    context.setPositions(np.asarray(r) * 0.1)
    context.setVelocities(np.asarray(v) * 0.1)
    rng = np.random.default_rng(9)
    update = jax.jit(
        lambda pos, vel, noise: middle_step(
            pos,
            vel,
            -100 * pos / (KJ_MOL_PER_EV * 100),
            masses,
            noise,
            timestep_ps=dt,
            temperature=temperature,
            friction_ps=friction,
        )
    )
    for _ in range(20):
        noise = rng.normal(size=(2, 3))
        if friction:
            integrator.setPerDofVariableByName('eta', noise)
        r, v = update(r, v, jnp.asarray(noise))
        integrator.step(1)
        state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        np.testing.assert_allclose(
            r,
            state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
            atol=2e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            v,
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
            * 10,
            atol=2e-11,
            rtol=0,
        )
        np.testing.assert_allclose(
            kinetic_energy(v, masses),
            state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            / KJ_MOL_PER_EV,
            atol=2e-12,
            rtol=0,
        )


@pytest.mark.parametrize('box,cutoff', [(11.76, 6.0), (4.0, 5.0)])
def test_periodic_image_graph_matches_ase_and_jit(box, cutoff):
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from jax_md import partition, space

    from mace_jax.modules.periodic_graph import ImageGraphNeighborFns

    positions = jnp.asarray(
        [[0.1, 0.2, 0.3], [box / 2, 0.2, 0.3], [box - 0.2, box - 0.3, 0.4]]
    )
    displacement, _ = space.periodic(box)
    base = partition.neighbor_list(
        displacement, box, cutoff, dr_threshold=0.25, format=partition.Sparse
    )
    fn = ImageGraphNeighborFns(base, box, cutoff)
    neighbors = fn.allocate(positions)
    for r in (
        positions,
        positions + jnp.asarray([[box, 0.0, 0.0], [0.0, -box, 0.0], [0.0, 0.0, box]]),
    ):
        neighbors = jax.jit(fn.update)(r, neighbors)
        assert not neighbors.did_buffer_overflow
        i, j, shift = neighbor_list(
            'ijS',
            Atoms('H3', positions=np.asarray(r), cell=np.eye(3) * box, pbc=True),
            cutoff,
        )
        expected = sorted(
            (int(a), int(b), *map(int, c)) for a, b, c in zip(i, j, shift)
        )
        edge = np.asarray(neighbors.idx)
        unit = np.asarray(neighbors.unit_shifts)
        actual = sorted(
            (int(a), int(b), *map(int, c))
            for a, b, c in zip(*edge, unit)
            if a < 3 and b < 3
        )
        assert actual == expected


def test_openmm_default_force_settings(monkeypatch):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'examples/HydrogenMaleate-MACE-POLAR'
    monkeypatch.syspath_prepend(str(path))
    from simulation_settings import resolve_force_settings

    settings = resolve_force_settings('openmm', 11.76)
    assert settings.grid_shape == (30, 30, 30)
    assert settings.order == 5
    np.testing.assert_allclose(settings.alpha, 0.5232507342707401, atol=1e-15)


def test_charged_point_pme_subtraction():
    # Both regions carry net charge: the combined-minus-QM background must
    # survive. This catches cancellation errors that neutral MM water hides.
    pme = PointPME(0.53, (24, 24, 24))
    cell = jnp.eye(3) * 9.0
    r = jnp.asarray(
        [[0.3, 0.5, 0.7], [1.1, 0.9, 0.6], [4.1, 2.7, 1.2], [7.7, 6.2, 4.3]]
    )
    q = jnp.asarray([0.8, -0.2, 0.3, -0.4])

    def decomposed(pos):
        mm, cross = pme.reciprocal_parts(pos[:2], q[:2], pos[2:], q[2:], cell)
        return (
            mm
            + cross
            + pme.self_energy(q[2:])
            + pme.background_energy(q, cell)
            - pme.background_energy(q[:2], cell)
        )

    def subtractive(pos):
        return (
            pme.reciprocal_energy(pos, q, cell)
            - pme.reciprocal_energy(pos[:2], q[:2], cell)
            + pme.self_energy(q)
            - pme.self_energy(q[:2])
            + pme.background_energy(q, cell)
            - pme.background_energy(q[:2], cell)
        )

    actual, grad = jax.jit(jax.value_and_grad(decomposed))(r)
    expected, expected_grad = jax.value_and_grad(subtractive)(r)
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(grad, expected_grad, atol=2e-13)


def test_mechanical_subtraction_exceptions_beyond_cutoff():
    om = pytest.importorskip('openmm')
    from jax.scipy.special import erf, erfc
    from openmm import unit

    # QQ separation 4 A exceeds the 3 A cutoff. Regions carry nonzero charge;
    # the MM pair is excluded. This exercises three otherwise hidden terms.
    box = 9.0
    r = jnp.asarray(
        [[0.2, 0.3, 0.4], [4.2, 0.3, 0.4], [1.7, 2.8, 1.1], [2.4, 2.9, 1.2]]
    )
    q = jnp.asarray([0.8, -0.2, 0.3, -0.4])
    pme = PointPME(0.53, (24, 24, 24))
    cell = jnp.eye(3) * box
    system = om.System()
    for _ in range(4):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*[om.Vec3(*v) for v in np.eye(3) * 0.9])
    combined = om.NonbondedForce()
    subtracted = om.NonbondedForce()
    for force in (combined, subtracted):
        force.setNonbondedMethod(force.PME)
        force.setCutoffDistance(0.3)
        force.setPMEParameters(5.3, 24, 24, 24)
        force.setExceptionsUsePeriodicBoundaryConditions(True)
    for i, charge in enumerate(q):
        combined.addParticle(float(charge), 0.1, 0)
        subtracted.addParticle(float(charge) if i < 2 else 0.0, 0.1, 0)
    combined.addException(0, 1, float(q[0] * q[1]), 0.1, 0)
    combined.addException(2, 3, 0.0, 0.1, 0)
    cv = om.CustomCVForce('-qm')
    cv.addCollectiveVariable('qm', subtracted)
    system.addForce(combined)
    system.addForce(cv)
    context = om.Context(
        system, om.VerletIntegrator(0.001), om.Platform.getPlatformByName('Reference')
    )
    context.setPositions(np.asarray(r) * 0.1)
    expected = context.getState(getEnergy=True, getForces=True)
    i, j = np.triu_indices(4, 1)

    def energy(pos):
        d = pos[i] - pos[j]
        d -= box * jnp.round(d / box)
        distance = jnp.linalg.norm(d, axis=-1)
        cross = (i < 2) & (j >= 2)
        real = pme.coulomb * jnp.sum(
            jnp.where(
                cross & (distance < 3.0),
                q[i] * q[j] * erfc(pme.alpha * distance) / distance,
                0,
            )
        )
        # Restored QQ exception minus cutoff QQ PME real term.
        tail = pme.coulomb * q[0] * q[1] * erfc(pme.alpha * distance[0]) / distance[0]
        exception = (
            -pme.coulomb * q[2] * q[3] * erf(pme.alpha * distance[-1]) / distance[-1]
        )
        mm, coupling = pme.reciprocal_parts(pos[:2], q[:2], pos[2:], q[2:], cell)
        return (
            real
            + tail
            + exception
            + mm
            + coupling
            + pme.self_energy(q[2:])
            + pme.background_energy(q, cell)
            - pme.background_energy(q[:2], cell)
        )

    actual, grad = jax.jit(jax.value_and_grad(energy))(r)
    np.testing.assert_allclose(
        actual,
        expected.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        / KJ_MOL_PER_EV,
        atol=2e-10,
        rtol=0,
    )
    np.testing.assert_allclose(
        -grad,
        expected.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        / KJ_MOL_PER_EV
        / 10,
        atol=2e-10,
        rtol=0,
    )


def test_generalized_accurate_settings_and_initial_velocities(monkeypatch):
    from pathlib import Path

    from ase import Atoms

    from mace_jax.tools.physical_constants import OPENMM_MACE_SCALE

    path = Path(__file__).resolve().parents[1] / 'examples/HydrogenMaleate-MACE-POLAR'
    monkeypatch.syspath_prepend(str(path))
    from simulation_settings import generate_velocities, resolve_force_settings

    settings = resolve_force_settings('generalized-accurate', 11.76)
    assert settings.grid_shape is None  # The model creates its generalized mesh.
    assert settings.order == 8
    assert settings.cutoff == 5.8
    np.testing.assert_allclose(
        np.exp(-((settings.alpha * settings.cutoff) ** 2)) / 2, 1e-6, rtol=1e-14
    )
    assert settings.ml_energy_scale == OPENMM_MACE_SCALE
    atoms = Atoms('OH2', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    accurate = generate_velocities(atoms, 330, 123, 'generalized-accurate')
    reference = generate_velocities(atoms, 330, 123, 'openmm')
    np.testing.assert_array_equal(accurate, reference)
    np.testing.assert_array_equal(
        generate_velocities(atoms, 0, 123, 'generalized-accurate'), np.zeros((3, 3))
    )
    matched = resolve_force_settings('generalized-accurate', 11.76, tolerance=5e-5)
    np.testing.assert_allclose(
        matched.alpha, resolve_force_settings('openmm', 11.76).alpha, atol=1e-15
    )
