"""Analytic and coupled verification for first-principles ship geometry."""

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np

from lsdo_geo import (
    BasicCurveName,
    ClosedSurface,
    CompatibleLoft,
    FormCurveKind,
    FormCurveProblem,
    FormParameterHullProblem,
    FSplineCurve,
    LongitudinalFitTargets,
    LongitudinalLoftRegion,
    NavalHullParameters,
    RegionalCompatibleLoft,
    SectionBandFitTargets,
    SectionBandParameters,
    SectionLoftProblem,
    SectionProblem,
    SectionTemplate,
    SonarDomeSectionParameters,
    VariationalSystem,
    bilinear_patch,
    compute_closed_surface_hydrostatics,
    compute_hydrostatics,
    curve_span_error_indicators,
    evaluate_closed_surface_closure,
    evaluate_watertight_mesh,
    export_ascii_stl,
    export_obj,
    feature_aligned_interpolation_knots,
    recommended_refinement_knots,
    rectangular_box_surface,
    refine_surface,
    section_self_intersections,
    tessellate_closed_surface,
    wigley_surface,
)


def test_compatible_loft_interpolates_nonuniform_section_stations():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    stations = np.array([0.0, 0.05, 0.08, 0.12, 0.30, 0.70, 1.0])
    space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(3,),
        coefficients_shape=(4,),
    )
    sections = []
    for index, station in enumerate(stations):
        coefficients = np.column_stack(
            (
                np.linspace(-1.0, 0.0, 4),
                (1.0 + station**2) * np.linspace(0.0, 1.0, 4),
            )
        )
        sections.append(
            FSplineCurve(
                lfs.Function(
                    space,
                    csdl.Variable(value=coefficients),
                    name=f"nonuniform_section_{index}",
                )
            )
        )
    surface = CompatibleLoft.create(sections, stations, stations)
    regional = RegionalCompatibleLoft.create(
        sections,
        stations,
        stations,
        (
            LongitudinalLoftRegion("forward", 0.0, 0.12),
            LongitudinalLoftRegion("main", 0.12, 1.0),
        ),
    )
    transverse = np.linspace(0.0, 1.0, 9)
    for section, station in zip(sections, stations):
        coordinates = np.column_stack((transverse, np.full(transverse.size, station)))
        surface_values = np.asarray(surface.evaluate(coordinates).value)
        section_values = np.asarray(section.evaluate(transverse).value)
        np.testing.assert_allclose(surface_values[:, 1:], section_values[:, [1, 0]])
        regional_values = np.asarray(
            regional.evaluate_section(station, transverse).value
        )
        np.testing.assert_allclose(
            regional_values[:, 1:], section_values[:, [1, 0]], atol=2.0e-14
        )
    assert set(regional.patches) == {"forward", "main"}
    assert len(regional.patch_graph().connections) == 1
    for gap in regional.boundary_gaps().values():
        np.testing.assert_allclose(gap.value, 0.0, atol=2.0e-14)
    recorder.stop()


def test_feature_aligned_knots_create_one_c1_surface_and_interpolate_sections():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    stations = np.array(
        [0.0, 0.04, 0.08, 0.12, 0.15, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
    )
    features = np.array([0.12, 0.20])
    knots = feature_aligned_interpolation_knots(stations, 3, features, continuity=1)
    for feature in features:
        assert np.count_nonzero(np.isclose(knots, feature)) == 2

    space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(3,),
        coefficients_shape=(4,),
    )
    sections = []
    for index, station in enumerate(stations):
        coefficients = np.column_stack(
            (np.linspace(-1.0, 0.0, 4), station * np.linspace(0.0, 1.0, 4))
        )
        sections.append(
            FSplineCurve(
                lfs.Function(
                    space,
                    csdl.Variable(value=coefficients),
                    name=f"feature_section_{index}",
                )
            )
        )
    surface = CompatibleLoft.create(
        sections,
        stations,
        stations,
        longitudinal_knots=knots,
        name="feature_aligned_single_patch",
    )
    assert surface.function.space.coefficients_shape == (4, 12)
    for section, station in zip(sections, stations):
        coordinates = np.column_stack((np.linspace(0.0, 1.0, 5), np.full(5, station)))
        values = surface.evaluate(coordinates).value
        expected = section.evaluate(np.linspace(0.0, 1.0, 5)).value
        np.testing.assert_allclose(values[:, 1:], expected[:, [1, 0]], atol=2e-12)
    recorder.stop()


def test_form_parameter_hull_preserves_primary_naval_parameters():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    beam = csdl.Variable(value=4.0, name="form_parameter_beam")
    primary = NavalHullParameters(
        length_between_perpendiculars=10.0,
        beam=beam,
        draft=2.0,
        displacement=50.0,
        lcb=0.0,
        waterplane_coefficient=0.7,
    )
    targets = LongitudinalFitTargets(
        station_parameters=[0.25, 0.5, 1.0],
        half_breadths=[1.5, 2.0, 1.0],
        half_areas=[2.0, 3.0, 1.0],
        drafts=[2.0, 2.0, 1.5],
        deadrise_angles=[0.3, 0.2, 0.1],
        flare_angles=[0.0, 0.0, 0.1],
        maximum_beam_parameter=0.5,
        maximum_draft_parameter=0.5,
    )
    geometry = FormParameterHullProblem(
        primary,
        targets,
        num_form_control_points=6,
        num_section_control_points=7,
        name="naval_parameter_test",
    ).solve(max_iter=30)
    recovered = geometry.recovered_primary_parameters()
    controls = geometry.curve_network.evaluate_section_controls([0.5], 10.0)

    assert BasicCurveName.DESIGN_WATERLINE in geometry.curve_network.basic_curves
    np.testing.assert_allclose(controls.half_breadths.value, [2.0], atol=2e-11)
    np.testing.assert_allclose(recovered["beam"].value, [4.0], atol=2e-11)
    np.testing.assert_allclose(recovered["draft"].value, [2.0], atol=2e-11)
    np.testing.assert_allclose(recovered["displacement"].value, [50.0], atol=2e-10)
    np.testing.assert_allclose(recovered["lcb"].value, [0.0], atol=2e-11)
    np.testing.assert_allclose(
        recovered["waterplane_coefficient"].value, [0.7], atol=2e-11
    )
    np.testing.assert_allclose(
        csdl.derivative(recovered["beam"], beam).value,
        [[1.0]],
        atol=2e-10,
    )
    assert len(geometry.hull.variational_result.stationarity_residuals) == 8
    assert (
        np.max(np.abs(geometry.hull.variational_result.constraint_residual.value))
        < 1e-10
    )
    recorder.stop()


def test_section_shape_fit_preserves_exact_form_constraints():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    parameters = np.linspace(0.0, 1.0, 11)
    target = np.column_stack(
        (-1.0 + parameters, 0.5 * parameters + 0.5 * parameters**2)
    )
    system = VariationalSystem("section_shape_fit")
    problem = SectionProblem(
        station_parameter=0.5,
        draft=1.0,
        half_breadth=1.0,
        half_area=5.0 / 12.0,
        keel_tangent_angle=np.arctan(0.5),
        waterline_tangent_angle=np.arctan(1.5),
        num_control_points=6,
        fairness_weights={2: 1.0e-4},
        fit_parameters=parameters,
        fit_points=target,
        fit_weight=100.0,
    )
    assembly = problem.assemble(system)
    curve = assembly.finalize(system.solve(max_iter=20))
    residual = np.asarray(curve.evaluate(parameters).value) - target

    assert np.sqrt(np.mean(np.sum(residual**2, axis=1))) < 4.0e-6
    assert np.max(np.abs(curve.constraint_residual.value)) < 2.0e-12
    assert np.max(np.abs(curve.stationarity_residual.value)) < 2.0e-11
    recorder.stop()


def test_section_loft_accepts_station_specific_fit_parameterization():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    shared = np.linspace(0.0, 1.0, 9)
    fit_parameters = np.stack((shared**1.2, shared**0.8))
    points = np.stack(
        tuple(
            np.column_stack((-1.0 + parameter, 0.8 * parameter))
            for parameter in fit_parameters
        )
    )
    hull = SectionLoftProblem(
        length=4.0,
        station_parameters=[0.25, 0.75],
        drafts=[1.0, 1.0],
        half_breadths=[0.8, 0.8],
        half_areas=[0.4, 0.4],
        keel_tangent_angles=[np.arctan(0.8), np.arctan(0.8)],
        waterline_tangent_angles=[np.arctan(0.8), np.arctan(0.8)],
        section_fit_parameters=fit_parameters,
        section_fit_points=points,
        section_fit_weight=100.0,
        num_section_control_points=6,
        longitudinal_degree=1,
    ).solve(max_iter=30)

    for section, parameters, target in zip(hull.sections, fit_parameters, points):
        residual = np.asarray(section.evaluate(parameters).value) - target
        assert np.sqrt(np.mean(np.sum(residual**2, axis=1))) < 1.0e-5
    recorder.stop()


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
        longitudinal_degree=1,
        longitudinal_regions=(
            LongitudinalLoftRegion("bow", 0.0, 0.3),
            LongitudinalLoftRegion("midbody", 0.3, 0.7),
            LongitudinalLoftRegion("stern", 0.7, 1.0),
        ),
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
    assert hull.regional_surface is not None
    assert len(hull.regional_surface.patches) == 3
    for gap in hull.regional_surface.boundary_gaps().values():
        np.testing.assert_allclose(gap.value, 0.0, atol=2.0e-12)
    hull.validity.assert_valid()
    recorder.stop()


def test_section_family_and_free_surface_share_one_global_kkt():
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
        surface_formulation="variational",
        name="variational_section_loft",
    )
    hull = problem.solve(max_iter=30)

    assert len(hull.variational_result.stationarity_residuals) == 3
    assert hull.surface.coefficients.shape == (7, 4, 3)
    assert np.max(np.abs(hull.variational_result.constraint_residual.value)) < 1e-10
    assert np.max(np.abs(hull.variational_result.stationarity_residual.value)) < 1e-9
    transverse_parameters = np.linspace(0.0, 1.0, 11)
    for section, station in zip(hull.sections, [0.3, 0.7]):
        surface_section = hull.surface.evaluate(
            np.column_stack(
                (
                    transverse_parameters,
                    np.full(transverse_parameters.size, station),
                )
            )
        ).value
        section_points = section.evaluate(transverse_parameters).value
        np.testing.assert_allclose(
            surface_section[:, 1], section_points[:, 1], atol=2e-12
        )
        np.testing.assert_allclose(
            surface_section[:, 2], section_points[:, 0], atol=2e-12
        )
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


def test_sonar_dome_section_preserves_component_variables_and_derivative():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    dome_breadth = csdl.Variable(value=0.12, name="dome_half_breadth")
    system = VariationalSystem("sonar_dome")
    problem = SectionProblem(
        station_parameter=0.05,
        draft=0.25,
        half_breadth=0.045,
        half_area=0.018,
        keel_tangent_angle=np.pi / 2.0,
        waterline_tangent_angle=0.2,
        template=SectionTemplate.SONAR_DOME,
        sonar_dome_parameters=SonarDomeSectionParameters(
            depth=0.36,
            maximum_breadth_parameter=0.30,
            maximum_breadth_height=0.30,
            maximum_half_breadth=dome_breadth,
            attachment_parameter=0.70,
            attachment_height=0.15,
            attachment_half_breadth=0.02,
        ),
        num_control_points=10,
        fairness_weights={2: 1.0e-4},
    )
    curve = problem.assemble(system).finalize(system.solve(max_iter=30))

    np.testing.assert_allclose(curve.evaluate(0.0).value, [-0.36, 0.0], atol=2e-11)
    np.testing.assert_allclose(curve.evaluate(0.30).value, [-0.30, 0.12], atol=2e-11)
    np.testing.assert_allclose(curve.evaluate(0.70).value, [-0.15, 0.02], atol=2e-11)
    np.testing.assert_allclose(curve.evaluate(1.0).value, [0.0, 0.045], atol=2e-11)
    breadth_derivative = csdl.derivative(curve.evaluate(0.30)[1], dome_breadth)
    np.testing.assert_allclose(breadth_derivative.value, [[1.0]], atol=2e-9)
    assert np.max(np.abs(curve.constraint_residual.value)) < 2e-10
    recorder.stop()


def test_blended_sonar_dome_uses_continuous_bands_and_guide_variables():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    dome_top_breadth = csdl.Variable(value=0.4, name="dome_top_breadth")
    bands = SectionBandParameters(
        dome_top_parameter=0.4,
        dome_top_point=csdl.concatenate(
            (
                csdl.Variable(value=-0.6).reshape((1,)),
                dome_top_breadth.reshape((1,)),
            )
        ),
        hull_blend_parameter=0.75,
        hull_blend_point=np.array([-0.25, 0.75]),
        continuity=1,
    )
    system = VariationalSystem("blended_sonar_dome")
    problem = SectionProblem(
        station_parameter=0.1,
        draft=1.0,
        half_breadth=1.0,
        half_area=0.5,
        keel_tangent_angle=np.pi / 4.0,
        waterline_tangent_angle=np.pi / 4.0,
        template=SectionTemplate.BLENDED_SONAR_DOME,
        section_band_parameters=bands,
        num_control_points=10,
        fairness_weights={2: 1.0e-4},
    )
    curve = problem.assemble(system).finalize(system.solve(max_iter=30))

    np.testing.assert_allclose(curve.evaluate(0.4).value, [-0.6, 0.4], atol=2e-10)
    np.testing.assert_allclose(curve.evaluate(0.75).value, [-0.25, 0.75], atol=2e-10)
    np.testing.assert_allclose(curve.signed_area().value, [0.5], atol=2e-10)
    derivative = csdl.derivative(curve.evaluate(0.4)[1], dome_top_breadth)
    np.testing.assert_allclose(derivative.value, [[1.0]], atol=2e-8)
    knots = np.asarray(curve.space.knots[curve.space.knot_indices[0]])
    assert np.count_nonzero(np.isclose(knots, 0.4)) == 2
    assert np.count_nonzero(np.isclose(knots, 0.75)) == 2
    assert np.max(np.abs(curve.constraint_residual.value)) < 2e-9
    recorder.stop()


def test_section_band_knots_support_curvature_continuity():
    bands = SectionBandParameters(
        dome_top_parameter=0.4,
        dome_top_point=[-0.6, 0.4],
        hull_blend_parameter=0.75,
        hull_blend_point=[-0.25, 0.75],
        continuity=2,
    )
    knots = bands.knots(num_control_points=8, degree=3)

    assert np.count_nonzero(np.isclose(knots, 0.4)) == 1
    assert np.count_nonzero(np.isclose(knots, 0.75)) == 1


def test_form_parameter_hull_assembles_fair_section_guides_in_one_solve():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    stations = np.array([0.25, 0.5, 1.0])
    primary = NavalHullParameters(
        length_between_perpendiculars=10.0,
        beam=4.0,
        draft=2.0,
        displacement=50.0,
        lcb=0.0,
        waterplane_coefficient=0.7,
    )
    targets = LongitudinalFitTargets(
        station_parameters=stations,
        half_breadths=[1.5, 2.0, 1.0],
        half_areas=[2.0, 3.0, 1.0],
        drafts=[2.0, 2.0, 1.5],
        deadrise_angles=[0.3, 0.2, 0.1],
        flare_angles=[0.0, 0.0, 0.1],
        maximum_beam_parameter=0.5,
        maximum_draft_parameter=0.5,
    )
    band_targets = SectionBandFitTargets(
        station_parameters=stations,
        dome_top_points=[[-1.2, 0.5], [-1.2, 0.7], [-0.9, 0.4]],
        hull_blend_points=[[-0.5, 1.2], [-0.5, 1.6], [-0.4, 0.8]],
        dome_top_parameter=0.4,
        hull_blend_parameter=0.75,
        continuity=1,
    )
    geometry = FormParameterHullProblem(
        primary,
        targets,
        num_form_control_points=6,
        num_section_control_points=10,
        section_band_fit_targets=band_targets,
        name="guided_naval_parameter_test",
    ).solve(max_iter=40)

    assert len(geometry.section_band_curves) == 4
    assert len(geometry.hull.variational_result.stationarity_residuals) == 12
    reference_knots = np.asarray(
        geometry.hull.sections[0].space.knots[
            geometry.hull.sections[0].space.knot_indices[0]
        ]
    )
    assert np.count_nonzero(np.isclose(reference_knots, 0.4)) == 2
    assert np.count_nonzero(np.isclose(reference_knots, 0.75)) == 2
    for section in geometry.hull.sections[1:]:
        np.testing.assert_allclose(
            section.space.knots[section.space.knot_indices[0]], reference_knots
        )
    assert (
        np.max(np.abs(geometry.hull.variational_result.constraint_residual.value))
        < 2e-9
    )
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
