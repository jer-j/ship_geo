"""Native CSDL deck fitting, smoothness, and boundary compatibility checks."""

import csdl_alpha as csdl
import numpy as np
import pytest
from numpy.testing import assert_allclose

from lsdo_geo.core.ship_geometry.deck_edge import assemble_deck_edge, apply_deck_edge
from lsdo_geo.core.ship_geometry.stem_integration import integrate_stem
from lsdo_geo.core.splines.variational import VariationalSystem


@pytest.fixture
def recorder():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    yield recorder
    recorder.stop()


def test_shared_fit_and_implicit_derivative(recorder):
    u = np.linspace(0, 1, 31)
    scale = csdl.Variable(value=0.08)
    y = scale * csdl.Variable(value=u)
    z = csdl.Variable(value=0.07 - 0.03 * u)
    system = VariationalSystem("test_deck")
    assemblies = assemble_deck_edge(system, u, y, z, num_control_points=10)
    assert len(system.states) == 2
    result = system.solve()
    breadth, height = (assembly.finalize(result) for assembly in assemblies)
    assert_allclose(breadth.evaluate(u).value.ravel(), 0.08*u, atol=1.e-9)
    assert_allclose(height.evaluate(u).value.ravel(), 0.07-0.03*u, atol=1.e-9)
    assert_allclose(csdl.derivative(breadth.evaluate([0.37]), scale).value, 0.37,
                    atol=1.e-9)
    knots = np.asarray(breadth.space.knots).ravel()
    interior = np.unique(knots[(knots > 0) & (knots < 1)])
    assert all(np.count_nonzero(knots == knot) == 1 for knot in interior)
    for order in (0, 1, 2):
        left = breadth.evaluate(interior-1.e-9, order).value
        right = breadth.evaluate(interior+1.e-9, order).value
        assert_allclose(left, right, atol=1.e-8)


def test_exact_deck_waterline_two_jet_and_stem(recorder):
    p = csdl.Variable(value=np.random.default_rng(4).normal(size=(6, 6, 3)))
    deck = csdl.Variable(value=np.random.default_rng(5).normal(size=(6, 3)))
    blend = csdl.Variable(value=np.array([0., 0., 0., 1., 1., 1.]))
    updated = apply_deck_edge(p, deck, blend)
    assert_allclose(updated.value[:, -1], deck.value, atol=1.e-14)
    # First three Bezier rows fix value, first derivative, second derivative.
    assert_allclose(updated.value[:, :3], p.value[:, :3], atol=1.e-14)
    stem = updated[0, :, :].reshape((6, 3))
    displacement = np.zeros((6, 3)); displacement[3:5, 0] = -0.1
    stem = stem + csdl.Variable(value=displacement)
    longitudinal = csdl.Variable(value=np.array([1., 1., 1., 0., 0., 0.]))
    integrated = integrate_stem(updated, stem, longitudinal)
    assert_allclose(integrated.value[0], stem.value, atol=1.e-14)
    assert_allclose(integrated.value[:, -1], deck.value, atol=1.e-14)
    assert_allclose(integrated.value[:, :3], p.value[:, :3], atol=1.e-14)
    jacobian = csdl.derivative(updated[:, -1, :], deck).value
    assert_allclose(jacobian, np.eye(18), atol=1.e-14)


def test_reject_unordered_parameters(recorder):
    with pytest.raises(ValueError, match="increase strictly"):
        assemble_deck_edge(VariationalSystem(), [0, .5, .4, 1], [0]*4, [0]*4)
