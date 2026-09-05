"""Regression tests for polynomial IGES and DTMB 5415 validation utilities."""

from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from lsdo_geo import SectionTemplate
from lsdo_geo.validation import (
    DTMB5415Reference,
    DTMB5415Region,
    DTMB5415SectionFitData,
    PolynomialIGESPatch,
    dtmb_5415_section_band_fit_targets,
    dtmb_5415_section_templates,
    extract_dtmb_5415_form_data,
    fit_dtmb_5415,
    read_polynomial_iges_surfaces,
)


def _ruled_section_patch(
    x_bounds: tuple[float, float], depth: float
) -> PolynomialIGESPatch:
    """Return a ruled half-hull face with a resolved waterline crossing."""
    knots = (np.array([0.0, 0.0, 1.0, 1.0]),) * 2
    coefficients = np.array([[[x, 0.0, -depth], [x, 0.4, 0.1]] for x in x_bounds])
    return PolynomialIGESPatch(
        degree=(1, 1),
        knots=knots,
        coefficients=coefficients,
        parameter_bounds=((0.0, 1.0), (0.0, 1.0)),
        directory_entry=1,
    )


def _write_bilinear_iges(path: Path) -> Path:
    parameter_values = [
        128,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        1,
        0,
        0.25,
        1,
        0,
        1,
    ]
    parameter_data = ",".join(str(value) for value in parameter_values) + ";"
    chunks = [
        parameter_data[index : index + 64]
        for index in range(0, len(parameter_data), 64)
    ]

    def directory_record(fields: list[int], sequence: int) -> str:
        payload = "".join(f"{value:>8}" for value in fields)
        return f"{payload}D{sequence:07d}"

    lines = [
        directory_record([128, 1, 0, 0, 0, 0, 0, 0, 0], 1),
        directory_record([128, 0, 0, len(chunks), 0, 0, 0, 0, 0], 2),
    ]
    for sequence, chunk in enumerate(chunks, start=1):
        lines.append(f"{chunk:<64}{1:>8}P{sequence:07d}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def test_polynomial_iges_reader_preserves_active_parameter_mapping(tmp_path):
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    patches = read_polynomial_iges_surfaces(
        _write_bilinear_iges(tmp_path / "plane.igs")
    )
    assert len(patches) == 1
    patch = patches[0]
    assert patch.degree == (1, 1)
    assert patch.coefficients.shape == (2, 2, 3)
    function = patch.build_function("test_iges_plane")

    np.testing.assert_allclose(
        patch.evaluate(function, [[0.5, 0.5]]).value,
        [0.625, 0.5, 0.0],
    )
    np.testing.assert_allclose(
        patch.evaluate(function, [[0.5, 0.5]], (1, 0)).value,
        [0.75, 0.0, 0.0],
    )
    recorder.stop()


def test_dtmb_validation_requires_and_reports_sonar_dome_region():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    knots = (np.array([0.0, 0.0, 1.0, 1.0]),) * 2

    def plane(x_offset: float) -> PolynomialIGESPatch:
        coefficients = np.array(
            [
                [[x_offset, 0.0, -1.0], [x_offset, 1.0, -1.0]],
                [[x_offset + 1.0, 0.0, 0.0], [x_offset + 1.0, 1.0, 0.0]],
            ]
        )
        return PolynomialIGESPatch(
            degree=(1, 1),
            knots=knots,
            coefficients=coefficients,
            parameter_bounds=((0.0, 1.0), (0.0, 1.0)),
            directory_entry=1,
        )

    reference = DTMB5415Reference(
        patches={
            DTMB5415Region.SONAR_DOME: plane(-2.0),
            DTMB5415Region.SONAR_DOME_TRANSITION: plane(-1.0),
            DTMB5415Region.MAIN_HULL: plane(0.0),
        },
        source_path=Path("synthetic_dtmb_5415.igs"),
    )
    approximation = fit_dtmb_5415(
        reference,
        "coarse",
        fitting_resolution=(9, 11),
        evaluation_resolution=(11, 13),
    )

    assert approximation.sonar_dome.region is DTMB5415Region.SONAR_DOME
    assert set(approximation.patches) == set(DTMB5415Region)
    assert approximation.global_rms_error < 1.0e-10
    assert approximation.sonar_dome.rms_error < 1.0e-10

    aligned = fit_dtmb_5415(
        reference,
        "fine",
        fitting_resolution=(9, 11),
        evaluation_resolution=(11, 13),
        knot_strategy="reference_aligned",
    )
    assert aligned.knot_strategy == "reference_aligned"
    assert aligned.global_rms_error < 1.0e-10
    assert all(fit.fitting_sample_count > 0 for fit in aligned.patches.values())
    recorder.stop()


def test_dtmb_form_data_separates_moulded_draft_from_sonar_dome_depth():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference = DTMB5415Reference(
        patches={
            DTMB5415Region.SONAR_DOME: _ruled_section_patch((-2.70, -2.01), 0.36),
            DTMB5415Region.SONAR_DOME_TRANSITION: _ruled_section_patch(
                (-2.01, -1.84), 0.30
            ),
            DTMB5415Region.MAIN_HULL: _ruled_section_patch((-1.84, 3.04), 0.248),
        },
        source_path=Path("synthetic_dtmb_5415.igs"),
    )
    data = extract_dtmb_5415_form_data(
        reference,
        station_parameters=np.array([0.03, 0.10, 0.30, 0.60, 1.0]),
        integration_station_count=21,
        search_resolution=21,
        transverse_resolution=41,
    )

    assert data.primary_parameters.draft == 0.248
    assert data.measured_particulars["main_hull_draft_from_sections"] < 0.249
    assert (
        data.measured_particulars["maximum_underwater_depth_including_sonar_dome"]
        > 0.35
    )
    assert data.fit_targets.maximum_draft_parameter > 0.14
    assert data.station_regions[0] is DTMB5415Region.SONAR_DOME
    assert data.station_regions[-1] is DTMB5415Region.MAIN_HULL
    recorder.stop()


def test_dtmb_section_templates_extract_dome_lobe_and_attachment():
    parameters = np.linspace(0.0, 1.0, 9)
    dome_points = np.column_stack(
        (
            np.linspace(-0.36, 0.0, 9),
            [0.0, 0.07, 0.12, 0.08, 0.03, 0.02, 0.025, 0.035, 0.045],
        )
    )
    main_points = np.column_stack(
        (np.linspace(-0.248, 0.0, 9), np.linspace(0.0, 0.30, 9))
    )
    data = DTMB5415SectionFitData(
        station_parameters=np.array([0.04, 0.50]),
        curve_parameters=parameters,
        points=np.stack((dome_points, main_points)),
        longitudinal_coordinates=np.array([-2.63, 0.0]),
        station_regions=(DTMB5415Region.SONAR_DOME, DTMB5415Region.MAIN_HULL),
    )

    templates, dome_parameters = dtmb_5415_section_templates(data)

    assert templates == (SectionTemplate.SONAR_DOME, SectionTemplate.ROUND_BILGE)
    assert dome_parameters[1] is None
    dome = dome_parameters[0]
    assert dome is not None
    assert dome.depth == 0.36
    assert dome.maximum_breadth_parameter == parameters[2]
    assert dome.maximum_half_breadth == 0.12
    assert dome.attachment_parameter == parameters[5]
    assert dome.attachment_half_breadth == 0.02


def test_dtmb_section_band_targets_span_every_section():
    parameters = np.linspace(0.0, 1.0, 9)
    points = np.stack(
        (
            np.column_stack((-0.4 + 0.4 * parameters, 0.2 * parameters)),
            np.column_stack((-0.3 + 0.3 * parameters, 0.4 * parameters)),
            np.column_stack((-0.2 + 0.2 * parameters, 0.3 * parameters)),
        )
    )
    data = DTMB5415SectionFitData(
        station_parameters=np.array([0.05, 0.5, 1.0]),
        curve_parameters=parameters,
        points=points,
        longitudinal_coordinates=np.array([-2.5, 0.0, 2.5]),
        station_regions=(
            DTMB5415Region.SONAR_DOME,
            DTMB5415Region.MAIN_HULL,
            DTMB5415Region.MAIN_HULL,
        ),
    )

    targets = dtmb_5415_section_band_fit_targets(
        data, dome_top_parameter=0.625, hull_blend_parameter=0.875
    )

    np.testing.assert_allclose(targets.dome_top_points, points[:, 5, :])
    np.testing.assert_allclose(targets.hull_blend_points, points[:, 7, :])
    assert targets.continuity == 1
