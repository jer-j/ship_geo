"""Regression tests for polynomial IGES and DTMB 5415 validation utilities."""

from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from lsdo_geo.validation import (
    DTMB5415Reference,
    DTMB5415Region,
    PolynomialIGESPatch,
    fit_dtmb_5415,
    read_polynomial_iges_surfaces,
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
    recorder.stop()
