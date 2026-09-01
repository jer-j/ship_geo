"""Analytic and coupled verification for variational B-spline surfaces."""

import csdl_alpha as csdl
import numpy as np

from lsdo_geo import (
    FSurfaceProblem,
    PatchConnection,
    PatchGraph,
    VariationalSystem,
)


def test_thin_plate_surface_recovers_bilinear_patch_and_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    corner_height = csdl.Variable(value=2.0, name="corner_height")
    upper_corner = csdl.concatenate(
        (
            (0.0 * corner_height + 1.0).reshape((1,)),
            (0.0 * corner_height + 1.0).reshape((1,)),
            corner_height.reshape((1,)),
        )
    )
    problem = FSurfaceProblem(
        num_control_points=(4, 4),
        degree=(3, 3),
        name="bilinear_surface",
    )
    problem.add_point_constraint((0.0, 0.0), [0.0, 0.0, 0.0])
    problem.add_point_constraint((1.0, 0.0), [1.0, 0.0, 0.0])
    problem.add_point_constraint((0.0, 1.0), [0.0, 1.0, 0.0])
    problem.add_point_constraint((1.0, 1.0), upper_corner)
    surface = problem.solve(max_iter=20)

    coordinates = np.array([[0.25, 0.30], [0.70, 0.80], [0.20, 0.90]])
    expected = np.column_stack(
        (coordinates, 2.0 * coordinates[:, 0] * coordinates[:, 1])
    )
    np.testing.assert_allclose(
        surface.evaluate(coordinates).value, expected, atol=2e-13
    )
    center_derivative = csdl.derivative(surface.evaluate([[0.5, 0.5]]), corner_height)
    np.testing.assert_allclose(
        center_derivative.value.reshape(-1), [0.0, 0.0, 0.25], atol=2e-13
    )
    assert np.max(np.abs(surface.constraint_residual.value)) < 1e-12
    assert np.max(np.abs(surface.stationarity_residual.value)) < 1e-11
    recorder.stop()


def test_two_surface_patches_share_one_global_newton_solve():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    system = VariationalSystem("coupled_surfaces")
    first_problem = FSurfaceProblem((4, 4), (3, 3), name="first_patch")
    for coordinates, target in (
        ((0.0, 0.0), [0.0, 0.0, 0.0]),
        ((1.0, 0.0), [1.0, 0.0, 0.0]),
        ((0.0, 1.0), [0.0, 1.0, 0.0]),
        ((1.0, 1.0), [1.0, 1.0, 0.0]),
    ):
        first_problem.add_point_constraint(coordinates, target)
    first_assembly = first_problem.assemble(system)

    second_problem = FSurfaceProblem((4, 4), (3, 3), name="second_patch")
    second_problem.add_point_constraint((1.0, 0.0), [2.0, 0.0, 0.0])
    second_problem.add_point_constraint((1.0, 1.0), [2.0, 1.0, 0.0])
    second_assembly = second_problem.assemble(system)

    topology = PatchGraph()
    topology.add_patch("first", first_assembly.surface)
    topology.add_patch("second", second_assembly.surface)
    topology.connect(PatchConnection("first", "u1", "second", "u0"))
    topology.add_continuity_constraints(system, samples=4)

    result = system.solve(max_iter=20)
    first = first_assembly.finalize(result)
    second = second_assembly.finalize(result)

    assert len(result.stationarity_residuals) == 2
    assert np.max(np.abs(result.constraint_residual.value)) < 1e-12
    assert np.max(np.abs(result.stationarity_residual.value)) < 2e-11
    parameters = np.linspace(0.0, 1.0, 11)
    first_edge = first.evaluate(np.column_stack((np.ones(11), parameters)))
    second_edge = second.evaluate(np.column_stack((np.zeros(11), parameters)))
    np.testing.assert_allclose(first_edge.value, second_edge.value, atol=1e-12)
    np.testing.assert_allclose(
        first.evaluate([[0.5, 0.4]]).value, [0.5, 0.4, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        second.evaluate([[0.5, 0.4]]).value, [1.5, 0.4, 0.0], atol=1e-12
    )
    recorder.stop()
