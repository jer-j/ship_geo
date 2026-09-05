"""Calibrate and visualize the naval-variable DTMB 5415 hull model."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from lsdo_geo.validation import (
    DTMB5415FormCalibration,
    DTMB5415Reference,
    calibrate_dtmb_5415_form_hull,
    download_dtmb_5415,
    load_dtmb_5415,
)


def _derived_output(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _surface_section(
    calibration: DTMB5415FormCalibration, station: float, parameters: np.ndarray
) -> np.ndarray:
    coordinates = np.column_stack((parameters, np.full(parameters.size, station)))
    return np.asarray(calibration.geometry.hull.surface.evaluate(coordinates).value)


def _combined_section_mesh(
    calibration: DTMB5415FormCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return station-sorted canonical and generated wireframe points."""
    fitting = calibration.section_fit_data
    validation = calibration.validation_section_data
    if not np.allclose(fitting.curve_parameters, validation.curve_parameters):
        raise ValueError("fit and validation sections must share curve parameters.")
    stations = np.concatenate(
        (fitting.station_parameters, validation.station_parameters)
    )
    x_coordinates = np.concatenate(
        (fitting.longitudinal_coordinates, validation.longitudinal_coordinates)
    )
    section_points = np.concatenate((fitting.points, validation.points), axis=0)
    fitting_model_parameters = (
        np.tile(fitting.curve_parameters, (fitting.station_parameters.size, 1))
        if fitting.model_curve_parameters is None
        else fitting.model_curve_parameters
    )
    validation_model_parameters = (
        np.tile(validation.curve_parameters, (validation.station_parameters.size, 1))
        if validation.model_curve_parameters is None
        else validation.model_curve_parameters
    )
    model_parameters = np.concatenate(
        (
            fitting_model_parameters,
            validation_model_parameters,
        ),
        axis=0,
    )
    order = np.argsort(stations)
    stations = stations[order]
    x_coordinates = x_coordinates[order]
    section_points = section_points[order]
    model_parameters = model_parameters[order]
    exact = np.stack(
        (
            np.broadcast_to(x_coordinates[:, None], section_points.shape[:2]),
            section_points[:, :, 1],
            section_points[:, :, 0],
        ),
        axis=2,
    )
    generated = np.stack(
        [
            _surface_section(calibration, station, parameters)
            for station, parameters in zip(stations, model_parameters)
        ]
    )
    return exact, generated, stations, model_parameters


def _draw_wireframe(
    axis: plt.Axes,
    exact: np.ndarray,
    generated: np.ndarray,
    stations: np.ndarray,
    model_parameters: np.ndarray,
    calibration: DTMB5415FormCalibration,
) -> None:
    """Draw two coincident-comparison wireframes without opaque surfaces."""
    dome_top = calibration.section_band_fit_targets.dome_top_parameter
    hull_blend = calibration.section_band_fit_targets.hull_blend_parameter
    bands = (
        (0.0, dome_top, "#d55e00"),
        (dome_top, hull_blend, "#e69f00"),
        (hull_blend, 1.0, "#0072b2"),
    )

    for section in exact:
        axis.plot(*section.T, color="0.35", linewidth=1.5, alpha=0.72)
    for section, parameters in zip(generated, model_parameters):
        for lower, upper, color in bands:
            mask = (parameters >= lower - 1.0e-12) & (parameters <= upper + 1.0e-12)
            axis.plot(
                *section[mask].T,
                color=color,
                linewidth=1.15,
                linestyle="--",
            )
    for index in range(0, exact.shape[1], 2):
        axis.plot(*exact[:, index, :].T, color="0.35", linewidth=1.0, alpha=0.60)
        parameter = float(np.median(model_parameters[:, index]))
        color = next(
            color
            for lower, upper, color in bands
            if lower - 1.0e-12 <= parameter <= upper + 1.0e-12
        )
        axis.plot(
            *generated[:, index, :].T,
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.88,
        )


def _plot_surface_overlay(output: Path, calibration: DTMB5415FormCalibration) -> None:
    """Overlay exact and generated underwater sections in full and bow views."""
    figure = plt.figure(figsize=(13.0, 5.8))
    perspective = figure.add_subplot(1, 2, 1, projection="3d")
    bow = figure.add_subplot(1, 2, 2, projection="3d")
    exact, generated, stations, model_parameters = _combined_section_mesh(calibration)
    _draw_wireframe(
        perspective, exact, generated, stations, model_parameters, calibration
    )
    bow_mask = exact[:, 0, 0] <= -1.75
    _draw_wireframe(
        bow,
        exact[bow_mask],
        generated[bow_mask],
        stations[bow_mask],
        model_parameters[bow_mask],
        calibration,
    )
    for axis in (perspective, bow):
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.view_init(elev=19.0, azim=-64.0)
        axis.set_proj_type("ortho")
    perspective.set_box_aspect((4.5, 1.0, 0.65), zoom=1.25)
    perspective.set_title("Full starboard underwater hull")
    bow.set_box_aspect((1.4, 1.0, 1.0), zoom=1.15)
    bow.set_title("Forward bow and sonar-dome close-up")
    figure.legend(
        handles=(
            Line2D([0], [0], color="0.35", linewidth=2.2, label="Canonical IGES"),
            Line2D(
                [0],
                [0],
                color="#d55e00",
                linestyle="--",
                label="Lower hull / sonar-dome band",
            ),
            Line2D(
                [0],
                [0],
                color="#e69f00",
                linestyle="--",
                label="Hull-dome transition band",
            ),
            Line2D(
                [0],
                [0],
                color="#0072b2",
                linestyle="--",
                label="Upper hull band",
            ),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "DTMB 5415 canonical and first-principles surfaces\n"
        f"independent-section RMS = "
        f"{1e3 * calibration.validation_section_rms_error:.3f} mm, "
        f"maximum = {1e3 * calibration.validation_section_maximum_error:.3f} mm"
    )
    figure.subplots_adjust(top=0.82, bottom=0.05, left=0.01, right=0.98, wspace=0.04)
    _save(figure, output)


def _draw_reference_surface(
    axis: plt.Axes,
    reference: DTMB5415Reference,
    resolution: tuple[int, int] = (31, 31),
) -> None:
    """Draw the exact canonical patches as one neutral visual reference."""
    functions = reference.build_functions()
    for region, patch in reference.patches.items():
        _, values = patch.sample_grid(functions[region], resolution)
        points = np.asarray(values.value).reshape((*resolution, 3))
        axis.plot_surface(
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
            color="0.72",
            edgecolor="none",
            alpha=0.20,
            shade=False,
        )
        axis.plot_wireframe(
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
            rstride=5,
            cstride=5,
            color="0.48",
            linewidth=0.35,
            alpha=0.28,
        )


def _plot_curve_network(
    output: Path,
    calibration: DTMB5415FormCalibration,
    reference: DTMB5415Reference,
) -> None:
    """Show the solved curve network before constructing the loft surface."""
    figure = plt.figure(figsize=(13.0, 5.8))
    full = figure.add_subplot(1, 2, 1, projection="3d")
    bow = figure.add_subplot(1, 2, 2, projection="3d")
    geometry = calibration.geometry
    length = float(geometry.primary_parameters.length_between_perpendiculars)
    origin = calibration.form_data.coordinate_origin
    stations = np.linspace(0.0, 1.0, 241)
    x = origin + length * (stations - 0.5)

    curves = geometry.section_band_curves
    guides = (
        (
            "Central profile / keel extension",
            np.asarray(geometry.draft_curve.evaluate(stations).value).reshape(-1),
            np.zeros(stations.size),
            "#8c564b",
        ),
        (
            "Dome top / transition bottom",
            np.asarray(curves["dome_top_height"].evaluate(stations).value).reshape(-1),
            np.asarray(
                curves["dome_top_half_breadth"].evaluate(stations).value
            ).reshape(-1),
            "#d55e00",
        ),
        (
            "Transition top / upper-hull bottom",
            np.asarray(curves["hull_blend_height"].evaluate(stations).value).reshape(
                -1
            ),
            np.asarray(
                curves["hull_blend_half_breadth"].evaluate(stations).value
            ).reshape(-1),
            "#009e73",
        ),
        (
            "Design waterline",
            np.zeros(stations.size),
            np.asarray(geometry.waterline_curve.evaluate(stations).value).reshape(-1),
            "#0072b2",
        ),
    )

    for axis in (full, bow):
        _draw_reference_surface(axis, reference)
        for label, height, breadth, color in guides:
            axis.plot(x, breadth, -height, color=color, linewidth=2.4, label=label)
        for station, section in zip(
            geometry.hull.section_parameters, geometry.hull.sections
        ):
            values = np.asarray(section.evaluate(np.linspace(0.0, 1.0, 101)).value)
            x_section = np.full(values.shape[0], origin + length * (station - 0.5))
            axis.plot(
                x_section,
                values[:, 1],
                values[:, 0],
                color="#7b61a8",
                linewidth=0.7,
                alpha=0.75,
            )
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.view_init(elev=18.0, azim=-64.0)
        axis.set_proj_type("ortho")
    full.set_box_aspect((4.5, 1.0, 0.65), zoom=1.25)
    full.set_title("Complete pre-skinning curve network")
    bow.set_xlim(x.min(), origin + length * (0.28 - 0.5))
    bow.set_box_aspect((1.4, 1.0, 1.0), zoom=1.15)
    bow.set_title("Bow and sonar-dome guide curves")
    handles, labels = full.get_legend_handles_labels()
    handles.extend(
        (
            Line2D([0], [0], color="#7b61a8", label="Generating F-Spline sections"),
            Line2D(
                [0],
                [0],
                color="0.62",
                linewidth=8,
                alpha=0.40,
                label="Canonical IGES surface",
            ),
        )
    )
    labels.extend(("Generating F-Spline sections", "Canonical IGES surface"))
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "DTMB 5415 curve network before skinning\n"
        "two fair band-boundary curves partition each compatible section"
    )
    figure.subplots_adjust(top=0.78, bottom=0.05, left=0.01, right=0.98, wspace=0.04)
    _save(figure, output)


def _plot_body_plan(output: Path, calibration: DTMB5415FormCalibration) -> None:
    """Show exact/generated transverse sections at fit and holdout stations."""
    figure, axes = plt.subplots(2, 5, figsize=(13.0, 5.5), sharex=True, sharey=True)
    fitting = calibration.section_fit_data
    validation = calibration.validation_section_data
    entries = (
        (fitting, 0, True),
        (validation, 1, False),
        (fitting, 2, True),
        (validation, 3, False),
        (fitting, 4, True),
        (validation, 5, False),
        (fitting, 6, True),
        (validation, 7, False),
        (fitting, 8, True),
        (validation, 9, False),
    )
    for axis, (data, index, is_fit) in zip(axes.ravel(), entries):
        station = data.station_parameters[index]
        exact = data.points[index]
        model_parameters = (
            data.curve_parameters
            if data.model_curve_parameters is None
            else data.model_curve_parameters[index]
        )
        generated = _surface_section(calibration, station, model_parameters)
        color = "#009e73" if is_fit else "#0072b2"
        axis.plot(exact[:, 1], exact[:, 0], color="0.25", linewidth=2.3)
        axis.plot(
            generated[:, 1],
            generated[:, 2],
            color=color,
            linewidth=1.3,
            linestyle="--",
        )
        residual = generated[:, [2, 1]] - exact
        rms = 1.0e3 * np.sqrt(np.mean(np.sum(residual**2, axis=1)))
        axis.set_title(
            f"v={station:.3f} ({'fit' if is_fit else 'holdout'})\nRMS={rms:.2f} mm"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)
    for axis in axes[:, 0]:
        axis.set_ylabel("z [m]")
    for axis in axes[-1, :]:
        axis.set_xlabel("Half-breadth, y [m]")
    figure.legend(
        handles=(
            Line2D([0], [0], color="0.25", linewidth=2.3, label="Canonical IGES"),
            Line2D(
                [0],
                [0],
                color="#009e73",
                linestyle="--",
                label="Generated at a fit station",
            ),
            Line2D(
                [0],
                [0],
                color="#0072b2",
                linestyle="--",
                label="Generated at a holdout station",
            ),
        ),
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle("DTMB 5415 body-plan validation of the naval-variable hull")
    figure.subplots_adjust(bottom=0.16, top=0.84, hspace=0.40, wspace=0.25)
    _save(figure, output)


def _plot_diagnostics(output: Path, calibration: DTMB5415FormCalibration) -> None:
    """Compare fitted form functions and expose the fit/holdout error pattern."""
    figure, axes = plt.subplots(2, 3, figsize=(12.5, 7.3), sharex=True)
    stations = calibration.form_data.fit_targets.station_parameters
    geometry = calibration.geometry
    quantities = (
        (
            geometry.sectional_area_curve,
            calibration.form_data.fit_targets.half_areas,
            r"Half-sectional area [m$^2$]",
        ),
        (
            geometry.waterline_curve,
            calibration.form_data.fit_targets.half_breadths,
            "Waterline half-breadth [m]",
        ),
        (
            geometry.draft_curve,
            calibration.form_data.fit_targets.drafts,
            "Local depth [m]",
        ),
        (
            geometry.deadrise_curve,
            calibration.form_data.fit_targets.deadrise_angles,
            "Deadrise [rad]",
        ),
        (
            geometry.flare_curve,
            calibration.form_data.fit_targets.flare_angles,
            "Flare [rad]",
        ),
    )
    dense = np.linspace(0.0, 1.0, 241)
    for axis, (curve, targets, label) in zip(axes.ravel()[:5], quantities):
        axis.scatter(stations, targets, color="0.25", s=25, label="IGES extraction")
        axis.plot(
            dense,
            np.asarray(curve.evaluate(dense).value).reshape(-1),
            color="#0072b2",
            label="Fitted auxiliary function",
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    error_axis = axes.ravel()[5]
    validation = calibration.validation_section_data.station_parameters
    width = 0.025
    error_axis.bar(
        validation - 0.5 * width,
        1.0e3 * calibration.single_patch_validation_station_rms_errors,
        width=width,
        color="0.55",
        label="Averaged knots",
    )
    error_axis.bar(
        validation + 0.5 * width,
        1.0e3 * calibration.validation_station_rms_errors,
        width=width,
        color="#0072b2",
        label=r"Feature-aligned $C^1$ knots",
    )
    error_axis.set_ylabel("Section RMS error [mm]")
    error_axis.set_title("Independent holdout sections")
    error_axis.grid(alpha=0.22, axis="y")
    error_axis.legend(frameon=False)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Longitudinal parameter $v$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    max_primary = max(
        abs(value) for value in calibration.primary_parameter_errors.values()
    )
    figure.suptitle(
        "Auxiliary functions fit local shape; primary parameters remain exact\n"
        f"maximum primary-parameter residual = {max_primary:.2e}"
    )
    figure.subplots_adjust(bottom=0.11, top=0.88, hspace=0.25, wspace=0.30)
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_form_parameter_overlay.png"),
    )
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument("--backend", choices=("jax", "inline"), default="jax")
    arguments = parser.parse_args()
    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    recorder = csdl.Recorder(inline=arguments.backend == "inline")
    recorder.start()
    reference = load_dtmb_5415(source)
    calibration = calibrate_dtmb_5415_form_hull(
        reference,
        print_status=arguments.print_status,
        backend=arguments.backend,
    )
    print("primary parameter errors:", calibration.primary_parameter_errors)
    print("auxiliary RMS errors:", calibration.auxiliary_rms_errors)
    print(
        "fit-section RMS/max [mm]:",
        1.0e3 * calibration.fitting_section_rms_error,
        1.0e3 * calibration.fitting_section_maximum_error,
    )
    print(
        "validation-section RMS/max [mm]:",
        1.0e3 * calibration.validation_section_rms_error,
        1.0e3 * calibration.validation_section_maximum_error,
    )
    print(
        "averaged-knot baseline RMS/max [mm]:",
        1.0e3 * calibration.single_patch_validation_section_rms_error,
        1.0e3 * calibration.single_patch_validation_section_maximum_error,
    )
    _plot_surface_overlay(arguments.output, calibration)
    _plot_body_plan(_derived_output(arguments.output, "body_plan"), calibration)
    _plot_diagnostics(_derived_output(arguments.output, "diagnostics"), calibration)
    _plot_curve_network(
        _derived_output(arguments.output, "curve_network"), calibration, reference
    )
    recorder.stop()


if __name__ == "__main__":
    main()
