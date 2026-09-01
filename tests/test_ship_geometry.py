"""Analytic and coupled verification for first-principles ship geometry."""

import csdl_alpha as csdl
import numpy as np

from lsdo_geo import (
    ClosedSurface,
    FormCurveKind,
    FormCurveProblem,
    SectionLoftProblem,
    SectionProblem,
    SectionTemplate,
    VariationalSystem,
    bilinear_patch,
    compute_closed_surface_hydrostatics,
    compute_hydrostatics,
    curve_span_error_indicators,
    evaluate_closed_surface_closure,
    evaluate_watertight_mesh,
    export_ascii_stl,
    export_obj,
    recommended_refinement_knots,
    rectangular_box_surface,
    refine_surface,
    section_self_intersections,
    tessellate_closed_surface,
    wigley_surface,
)


def test_sectional_area_curve_integrals_and_implicit_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    target_integral = csdl.Variable(value=2.0 / 3.0, name="target_integral")
    problem = FormCurveProblem(
        FormCurveKind.SECTIONAL_AREA,
        num_control_points=6,
        name="sectional_area_curve",
    )
    problem.add_value_constraint(0.0, 0.0)
    problem.add_value_constraint(1.0, 0.0)
    problem.add_integral_constraint(target_integral)
    problem.add_integral_constraint(0.5 * target_integral, moment_order=1)
    curve = problem.solve()

    np.testing.assert_allclose(curve.integral().value, [2.0 / 3.0], atol=1e-11)
    np.testing.assert_allclose(curve.integral(1).value, [1.0 / 3.0], atol=1e-11)
    np.testing.assert_allclose(
        csdl.derivative(curve.integral(), target_integral).value,
        [[1.0]],
        atol=2e-10,
    )
    assert np.max(np.abs(curve.constraint_residual.value)) < 1e-10
    recorder.stop()


def test_exact_wigley_hydrostatics_and_design_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    length = csdl.Variable(value=100.0, name="length")
    beam = csdl.Variable(value=10.0, name="beam")
    draft = csdl.Variable(value=5.0, name="draft")
    surface = wigley_surface(length, beam, draft)
    hydrostatics = compute_hydrostatics(surface, [0.0, 0.5, 1.0])

    displacement = 4.0 / 9.0 * 100.0 * 10.0 * 5.0
    np.testing.assert_allclose(hydrostatics.displacement.value, displacement)
    np.testing.assert_allclose(
        hydrostatics.center_of_buoyancy.value,
        [0.0, 0.0, -3.0 * 5.0 / 8.0],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        hydrostatics.waterplane_area.value, 2.0 / 3.0 * 100.0 * 10.0
    )
    np.testing.assert_allclose(
        hydrostatics.longitudinal_waterplane_inertia.value,
        10.0 * 100.0**3 / 30.0,
    )
    np.testing.assert_allclose(
        hydrostatics.sectional_areas.value,
        [0.0, 2.0 / 3.0 * 10.0 * 5.0, 0.0],
        atol=2e-12,
    )
    np.testing.assert_allclose(
        csdl.derivative(hydrostatics.displacement, length).value,
        [[4.0 / 9.0 * 10.0 * 5.0]],
        atol=2e-11,
    )
    assert section_self_intersections(surface, [0.1, 0.5, 0.9]) == {
        0.1: 0,
        0.5: 0,
        0.9: 0,
    }
    recorder.stop()


def test_surface_knot_refinement_preserves_geometry_and_derivatives():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    beam = csdl.Variable(value=10.0, name="refinement_beam")
    surface = wigley_surface(100.0, beam, 5.0)
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])
    refined = refine_surface(surface, knots, knots)
    u = np.linspace(0.0, 1.0, 7)
    v = np.linspace(0.0, 1.0, 9)
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))

    np.testing.assert_allclose(
        refined.evaluate(coordinates).value,
        surface.evaluate(coordinates).value,
        atol=2e-12,
    )
    derivative = csdl.derivative(refined.evaluate([[0.5, 0.5]]), beam)
    np.testing.assert_allclose(derivative.value.reshape(-1), [0.0, 0.375, 0.0])
    recorder.stop()


def test_section_family_uses_one_global_kkt_and_compatible_loft():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    problem = SectionLoftProblem(
        length=10.0,
        station_parameters=[0.3, 0.7],
        drafts=[2.0, 2.0],
        half_breadths=[1.5, 1.5],
        half_areas=[2.0, 2.0],
        keel_tangent_angles=[0.5, 0.5],
        waterline_tangent_angles=[0.0, 0.0],
        num_section_control_points=7,
        name="coupled_section_loft",
    )
    hull = problem.solve(max_iter=30)

    assert len(hull.variational_result.stationarity_residuals) == 2
    assert np.max(np.abs(hull.variational_result.constraint_residual.value)) < 1e-9
    assert np.max(np.abs(hull.variational_result.stationarity_residual.value)) < 1e-8
    np.testing.assert_allclose(
        [section.signed_area().value[0] for section in hull.sections],
        [2.0, 2.0],
        atol=1e-9,
    )
    assert hull.surface.coefficients.shape == (7, 4, 3)
    hull.validity.assert_valid()
    recorder.stop()


def test_hard_chine_uses_repeated_knot_and_preserves_chine_point():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    system = VariationalSystem("hard_chine")
    problem = SectionProblem(
        station_parameter=0.5,
        draft=2.0,
        half_breadth=1.5,
        half_area=2.0,
        keel_tangent_angle=0.3,
        waterline_tangent_angle=0.0,
        template=SectionTemplate.HARD_CHINE,
        chine_parameter=0.5,
        chine_point=[-1.0, 1.0],
        num_control_points=8,
    )
    assembly = problem.assemble(system)
    curve = assembly.finalize(system.solve(max_iter=50))

    np.testing.assert_allclose(curve.evaluate(0.5).value, [-1.0, 1.0])
    np.testing.assert_allclose(curve.signed_area().value, [2.0])
    knots = curve.space.knots[curve.space.knot_indices[0]]
    assert np.count_nonzero(np.isclose(knots, 0.5)) == 3
    assert np.max(np.abs(curve.constraint_residual.value)) < 1e-9
    recorder.stop()


def test_closed_box_hydrostatics_mesh_and_derivative(tmp_path):
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    length = csdl.Variable(value=10.0, name="box_length")
    beam = csdl.Variable(value=4.0, name="box_beam")
    draft = csdl.Variable(value=2.0, name="box_draft")
    closed_surface = rectangular_box_surface(length, beam, draft)
    hydrostatics = compute_closed_surface_hydrostatics(closed_surface)

    np.testing.assert_allclose(hydrostatics.displacement.value, [80.0])
    np.testing.assert_allclose(hydrostatics.center_of_buoyancy.value, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(hydrostatics.waterplane_area.value, [40.0])
    np.testing.assert_allclose(
        hydrostatics.transverse_waterplane_inertia.value,
        [10.0 * 4.0**3 / 12.0],
    )
    np.testing.assert_allclose(
        hydrostatics.longitudinal_waterplane_inertia.value,
        [4.0 * 10.0**3 / 12.0],
    )
    np.testing.assert_allclose(hydrostatics.wetted_area.value, [96.0])
    np.testing.assert_allclose(
        csdl.derivative(hydrostatics.displacement, length).value,
        [[8.0]],
    )

    closure = evaluate_closed_surface_closure(closed_surface)
    closure.assert_closed()
    mesh = tessellate_closed_surface(closed_surface, (3, 3))
    mesh_report = evaluate_watertight_mesh(mesh)
    assert mesh_report.is_watertight
    stl_path = export_ascii_stl(mesh, tmp_path / "box.stl")
    obj_path = export_obj(mesh, tmp_path / "box.obj")
    assert stl_path.read_text(encoding="utf-8").startswith("solid ship_geo")
    assert "\nf " in obj_path.read_text(encoding="utf-8")
    recorder.stop()


def test_symmetric_transom_caps_close_prismatic_hull():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    length = csdl.Variable(value=10.0, name="prism_length")
    starboard = bilinear_patch(
        (
            (-0.5 * length, 0.0, -2.0),
            (0.5 * length, 0.0, -2.0),
            (-0.5 * length, 2.0, 0.0),
            (0.5 * length, 2.0, 0.0),
        ),
        name="prism_starboard",
    )
    closed_surface = ClosedSurface.from_symmetric_starboard(
        starboard,
        waterplane=True,
        transom_edges=("v0", "v1"),
        name="prism",
    )
    closure = evaluate_closed_surface_closure(closed_surface)
    closure.assert_closed()
    hydrostatics = compute_closed_surface_hydrostatics(closed_surface)

    np.testing.assert_allclose(hydrostatics.displacement.value, [40.0])
    np.testing.assert_allclose(
        csdl.derivative(hydrostatics.displacement, length).value,
        [[4.0]],
    )
    assert evaluate_watertight_mesh(
        tessellate_closed_surface(
            closed_surface,
            ((5, 7), (5, 7), (7, 7), (5, 7), (5, 7)),
        )
    ).is_watertight
    recorder.stop()


def test_residual_driven_refinement_selects_high_error_spans():
    parameters = np.linspace(0.0, 1.0, 101)
    residuals = np.where(parameters < 0.5, 0.1, 1.0)
    knots = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])

    indicators, midpoints = curve_span_error_indicators(
        parameters,
        residuals,
        knots,
        degree=2,
    )
    np.testing.assert_allclose(midpoints, [0.25, 0.75])
    np.testing.assert_allclose(indicators, [0.1, 1.0])
    np.testing.assert_allclose(
        recommended_refinement_knots(indicators, midpoints, 0.5),
        [0.75],
    )
