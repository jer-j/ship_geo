"""Verification tests for the CSDL F-Spline kernel."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import pytest
from csdl_alpha.experimental import JaxSimulator

from lsdo_geo import FSplineCurve, FSplineProblem


def test_minimum_bending_line_and_area_moments():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    problem = FSplineProblem(num_control_points=8, degree=3)
    problem.add_point_constraint(0.0, [0.0, 0.0])
    problem.add_point_constraint(1.0, [1.0, 2.0])
    curve = problem.solve()

    parameters = np.linspace(0.0, 1.0, 9)
    expected = np.column_stack((parameters, 2.0 * parameters))
    np.testing.assert_allclose(curve.evaluate(parameters).value, expected, atol=1e-11)
    np.testing.assert_allclose(curve.signed_area().value, [1.0], atol=1e-11)
    np.testing.assert_allclose(
        curve.centroid().value, [2.0 / 3.0, 2.0 / 3.0], atol=1e-11
    )
    assert isinstance(curve.space, lfs.BSplineSpace)
    assert np.max(np.abs(curve.constraint_residual.value)) < 1e-10
    assert np.max(np.abs(curve.stationarity_residual.value)) < 1e-9
    recorder.stop()


def test_area_and_tangent_constrained_ship_section():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    problem = FSplineProblem(
        num_control_points=8,
        degree=3,
        fairness_weights={2: 1.0},
        name="test_ship_section",
    )
    problem.add_point_constraint(0.0, [0.0, 0.0])
    problem.add_point_constraint(1.0, [4.0, 5.0])
    problem.add_tangent_angle_constraint(0.0, np.deg2rad(80.0))
    problem.add_tangent_angle_constraint(1.0, np.deg2rad(15.0))
    problem.add_area_constraint(15.0, scale=0.1)
    curve = problem.solve()

    np.testing.assert_allclose(curve.signed_area().value, [15.0], atol=1e-9)
    assert np.max(np.abs(curve.constraint_residual.value)) < 1e-9
    assert np.max(np.abs(curve.stationarity_residual.value)) < 1e-8
    recorder.stop()


def test_curvature_and_moments_for_exact_parabola():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(3,),
        coefficients_shape=(4,),
    )
    coefficients = csdl.Variable(
        value=np.array(
            [
                [0.0, 0.0],
                [1.0 / 3.0, 0.0],
                [2.0 / 3.0, 1.0 / 3.0],
                [1.0, 1.0],
            ]
        )
    )
    curve = FSplineCurve(lfs.Function(space=space, coefficients=coefficients))

    np.testing.assert_allclose(curve.curvature(0.5).value, [1.0 / np.sqrt(2.0)])
    np.testing.assert_allclose(curve.signed_area().value, [1.0 / 3.0])
    np.testing.assert_allclose(curve.centroid().value, [3.0 / 4.0, 3.0 / 10.0])
    recorder.stop()


def test_implicit_area_derivative_is_preserved():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    target_area = csdl.Variable(value=15.0, name="target_area")
    problem = FSplineProblem(num_control_points=8, degree=3, name="derivative_test")
    problem.add_point_constraint(0.0, [0.0, 0.0])
    problem.add_point_constraint(1.0, [4.0, 5.0])
    problem.add_tangent_angle_constraint(0.0, np.deg2rad(80.0))
    problem.add_tangent_angle_constraint(1.0, np.deg2rad(15.0))
    problem.add_area_constraint(target_area, scale=0.1)
    curve = problem.solve()

    derivative = csdl.derivative(curve.signed_area(), target_area)
    np.testing.assert_allclose(derivative.value, [[1.0]], atol=2e-9)
    recorder.stop()


def test_backend_derivative_limit_is_reported_before_evaluation():
    problem = FSplineProblem(num_control_points=8, degree=3)
    with pytest.raises(ValueError, match="current lsdo_b_splines_cython backend"):
        problem.add_derivative_constraint(0.5, 3, [0.0, 0.0])


def test_jax_simulator_executes_implicit_fspline_solve():
    recorder = csdl.Recorder(inline=False)
    recorder.start()

    target_area = csdl.Variable(value=15.0, name="jax_target_area")
    problem = FSplineProblem(num_control_points=8, degree=3, name="jax_fspline")
    problem.add_point_constraint(0.0, [0.0, 0.0])
    problem.add_point_constraint(1.0, [4.0, 5.0])
    problem.add_tangent_angle_constraint(0.0, np.deg2rad(80.0))
    problem.add_tangent_angle_constraint(1.0, np.deg2rad(15.0))
    problem.add_area_constraint(target_area, scale=0.1)
    curve = problem.solve()
    area = curve.signed_area()

    simulator = JaxSimulator(
        recorder,
        additional_inputs=[target_area],
        additional_outputs=[area, curve.coefficients, curve.constraint_residual],
        gpu=False,
        f64=True,
    )
    simulator.run()

    np.testing.assert_allclose(area.value, [15.0], atol=2e-9)
    assert np.max(np.abs(curve.constraint_residual.value)) < 2e-9
    recorder.stop()
