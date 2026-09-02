"""Validate and visualize regional spline reconstruction of DTMB 5415."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lsdo_geo.validation import (
    DTMB5415Approximation,
    DTMB5415Reference,
    DTMB5415Region,
    download_dtmb_5415,
    fit_dtmb_5415,
    load_dtmb_5415,
)

REGION_COLORS = {
    DTMB5415Region.SONAR_DOME: "#d55e00",
    DTMB5415Region.SONAR_DOME_TRANSITION: "#e69f00",
    DTMB5415Region.MAIN_HULL: "#0072b2",
}
REGION_LABELS = {
    DTMB5415Region.SONAR_DOME: "Forward bow and sonar dome",
    DTMB5415Region.SONAR_DOME_TRANSITION: "Dome-to-hull transition",
    DTMB5415Region.MAIN_HULL: "Main hull and transom",
}


def _derived_output(base: Path, suffix: str) -> Path:
    """Return a sibling image path derived from the requested primary output."""
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def _save(figure: plt.Figure, path: Path) -> None:
    """Save one documentation figure with consistent output settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _sample_reference(
    reference: DTMB5415Reference,
    functions: dict,
    region: DTMB5415Region,
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return structured parameters and exact reference points."""
    patch = reference.patches[region]
    coordinates, values = patch.sample_grid(functions[region], resolution)
    return coordinates, np.asarray(values.value).reshape(resolution + (3,))


def _sample_fit(
    approximation: DTMB5415Approximation,
    region: DTMB5415Region,
    coordinates: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Evaluate a fitted surface on a structured reference parameter grid."""
    values = approximation.patches[region].surface.evaluate(coordinates)
    return np.asarray(values.value).reshape(resolution + (3,))


def _set_ship_axes(axis) -> None:
    """Apply consistent labels and ship-like aspect ratio to a 3-D axis."""
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_box_aspect((6.2, 1.0, 1.0))
    axis.view_init(elev=18.0, azim=-67.0)


def _plot_patch_definition(
    output: Path,
    reference: DTMB5415Reference,
    functions: dict,
) -> None:
    """Explain which exact IGES face each regional label denotes."""
    figure = plt.figure(figsize=(12.0, 7.0))
    perspective = figure.add_subplot(121, projection="3d")
    profile = figure.add_subplot(122)
    for region in DTMB5415Region:
        _, points = _sample_reference(reference, functions, region, (45, 75))
        color = REGION_COLORS[region]
        perspective.plot_surface(
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
            color=color,
            alpha=0.83,
            linewidth=0.0,
        )
        profile.scatter(
            points[::2, ::3, 0],
            points[::2, ::3, 2],
            s=2.0,
            color=color,
            alpha=0.60,
        )
    _set_ship_axes(perspective)
    perspective.set_title("Three exact IGES faces")
    profile.set_xlabel("x [m]")
    profile.set_ylabel("z [m]")
    profile.set_aspect("equal", adjustable="box")
    profile.set_xlim(-3.15, -1.65)
    profile.set_ylim(-0.42, 0.45)
    profile.grid(alpha=0.25)
    profile.set_title("Bow close-up: the orange face includes the dome")
    handles = [
        Patch(facecolor=REGION_COLORS[region], label=REGION_LABELS[region])
        for region in DTMB5415Region
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("DTMB 5415 reference patch definition")
    figure.subplots_adjust(bottom=0.15, wspace=0.18)
    _save(figure, output)


def _plot_surface_overlay(
    output: Path,
    reference: DTMB5415Reference,
    functions: dict,
    aligned: DTMB5415Approximation,
) -> None:
    """Overlay exact and reconstructed surface evaluations on identical grids."""
    figure = plt.figure(figsize=(13.0, 8.3))
    grid = figure.add_gridspec(2, 3, height_ratios=(1.15, 1.0))
    perspective = figure.add_subplot(grid[0, :], projection="3d")
    profile = figure.add_subplot(grid[1, 0])
    plan = figure.add_subplot(grid[1, 1])
    bow = figure.add_subplot(grid[1, 2])
    for region in DTMB5415Region:
        coordinates, exact = _sample_reference(reference, functions, region, (31, 55))
        fitted = _sample_fit(aligned, region, coordinates, (31, 55))
        color = REGION_COLORS[region]
        for transverse_index in range(0, exact.shape[1], 6):
            perspective.plot(
                exact[:, transverse_index, 0],
                exact[:, transverse_index, 1],
                exact[:, transverse_index, 2],
                color="0.45",
                linewidth=1.8,
                alpha=0.65,
            )
            perspective.plot(
                fitted[:, transverse_index, 0],
                fitted[:, transverse_index, 1],
                fitted[:, transverse_index, 2],
                color=color,
                linewidth=0.9,
                linestyle="--",
            )
        for longitudinal_index in range(0, exact.shape[0], 5):
            perspective.plot(
                exact[longitudinal_index, :, 0],
                exact[longitudinal_index, :, 1],
                exact[longitudinal_index, :, 2],
                color="0.45",
                linewidth=1.8,
                alpha=0.65,
            )
            perspective.plot(
                fitted[longitudinal_index, :, 0],
                fitted[longitudinal_index, :, 1],
                fitted[longitudinal_index, :, 2],
                color=color,
                linewidth=0.9,
                linestyle="--",
            )
        exact_flat = exact.reshape((-1, 3))
        fitted_flat = fitted.reshape((-1, 3))
        for axis, x_index, y_index in (
            (profile, 0, 2),
            (plan, 0, 1),
            (bow, 0, 2),
        ):
            axis.scatter(
                exact_flat[::8, x_index],
                exact_flat[::8, y_index],
                s=7,
                facecolors="none",
                edgecolors="0.35",
                linewidths=0.55,
                alpha=0.65,
            )
            axis.scatter(
                fitted_flat[::8, x_index],
                fitted_flat[::8, y_index],
                s=2.0,
                color=color,
                alpha=0.75,
            )
    _set_ship_axes(perspective)
    perspective.set_title("Full starboard half-hull")
    profile.set(xlabel="x [m]", ylabel="z [m]", title="Profile overlay")
    plan.set(xlabel="x [m]", ylabel="y [m]", title="Plan overlay")
    bow.set(xlabel="x [m]", ylabel="z [m]", title="Bow and sonar-dome close-up")
    bow.set_xlim(-3.15, -1.65)
    bow.set_ylim(-0.42, 0.45)
    for axis in (profile, plan, bow):
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)
    figure.legend(
        handles=[
            Line2D([0], [0], color="0.45", linewidth=2.0, label="Exact IGES surface"),
            Line2D(
                [0],
                [0],
                color="#cc79a7",
                linestyle="--",
                label="Reference-aligned reconstruction",
            ),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle("DTMB 5415 exact surface and reconstructed surface")
    figure.subplots_adjust(bottom=0.10, hspace=0.30, wspace=0.25)
    _save(figure, output)


def _plot_error_comparison(
    output: Path,
    reference: DTMB5415Reference,
    functions: dict,
    uniform: DTMB5415Approximation,
    aligned: DTMB5415Approximation,
) -> None:
    """Map pointwise error for uniform and reference-aligned parameterizations."""
    figure = plt.figure(figsize=(13.0, 5.3))
    axes = [
        figure.add_subplot(121, projection="3d"),
        figure.add_subplot(122, projection="3d"),
    ]
    approximations = (uniform, aligned)
    titles = ("Uniform knots", "Feature-aligned knots")
    maxima = (
        1.0e3 * uniform.global_maximum_error,
        1.0e3 * aligned.global_maximum_error,
    )
    artists = []
    for axis, approximation, title, maximum in zip(
        axes, approximations, titles, maxima
    ):
        for region in DTMB5415Region:
            coordinates, exact = _sample_reference(
                reference, functions, region, (43, 67)
            )
            fitted = _sample_fit(approximation, region, coordinates, (43, 67))
            error = 1.0e3 * np.linalg.norm(fitted - exact, axis=2)
            flat = exact.reshape((-1, 3))
            artist = axis.scatter(
                flat[:, 0],
                flat[:, 1],
                flat[:, 2],
                c=error.ravel(),
                cmap="viridis",
                vmin=0.0,
                vmax=max(maximum, 1.0e-12),
                s=2.0,
            )
        artists.append(artist)
        _set_ship_axes(axis)
        axis.set_title(f"{title}\nmaximum error = {maximum:.4f} mm")
        axis.set_axis_off()
    for axis, artist in zip(axes, artists):
        figure.colorbar(
            artist,
            ax=axis,
            shrink=0.58,
            pad=0.08,
            label="Pointwise error [mm]",
        )
    figure.suptitle("Same fine control-net sizes, different knot distributions")
    figure.subplots_adjust(wspace=0.05)
    _save(figure, output)


def _main_hull_profiles(
    reference: DTMB5415Reference,
    functions: dict,
    approximation: DTMB5415Approximation | None,
    station_count: int = 81,
    transverse_count: int = 181,
) -> dict[str, np.ndarray]:
    """Evaluate design-like section distributions along the main hull patch."""
    u = np.linspace(0.0, 1.0, station_count)
    v = np.linspace(0.0, 1.0, transverse_count)
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
    if approximation is None:
        patch = reference.patches[DTMB5415Region.MAIN_HULL]
        values = patch.evaluate(functions[DTMB5415Region.MAIN_HULL], coordinates)
    else:
        values = approximation.patches[DTMB5415Region.MAIN_HULL].surface.evaluate(
            coordinates
        )
    points = np.asarray(values.value).reshape((station_count, transverse_count, 3))
    return {
        "x": np.mean(points[:, :, 0], axis=1),
        "half_breadth": np.max(points[:, :, 1], axis=1),
        "keel_z": np.min(points[:, :, 2], axis=1),
        "upper_z": np.max(points[:, :, 2], axis=1),
        "half_area": np.abs(np.trapz(points[:, :, 1], points[:, :, 2], axis=1)),
    }


def _plot_form_profiles(
    output: Path,
    reference: DTMB5415Reference,
    functions: dict,
    uniform: DTMB5415Approximation,
    aligned: DTMB5415Approximation,
) -> None:
    """Compare longitudinal design-variable distributions of all three surfaces."""
    exact = _main_hull_profiles(reference, functions, None)
    uniform_profiles = _main_hull_profiles(reference, functions, uniform)
    aligned_profiles = _main_hull_profiles(reference, functions, aligned)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    quantities = (
        ("half_breadth", "Half-breadth [m]"),
        ("keel_z", "Lowest section point, z [m]"),
        ("upper_z", "Highest section point, z [m]"),
        ("half_area", r"Half-sectional area [m$^2$]"),
    )
    styles = (
        (exact, "Exact IGES", "0.25", "-", 2.2),
        (uniform_profiles, "Uniform fine fit", "#cc79a7", ":", 1.8),
        (aligned_profiles, "Feature-aligned fine fit", "#0072b2", "--", 1.3),
    )
    for axis, (quantity, ylabel) in zip(axes.ravel(), quantities):
        for profiles, label, color, linestyle, width in styles:
            axis.plot(
                profiles["x"],
                profiles[quantity],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=width,
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Longitudinal location, x [m]")
    axes[1, 1].set_xlabel("Longitudinal location, x [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("Main-hull section variables recovered by inverse fitting")
    figure.subplots_adjust(bottom=0.11, hspace=0.20, wspace=0.26)
    _save(figure, output)


def _plot_body_plan(
    output: Path,
    reference: DTMB5415Reference,
    functions: dict,
    aligned: DTMB5415Approximation,
) -> None:
    """Overlay exact and reconstructed transverse sections at selected stations."""
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), sharex=True, sharey=True)
    patch = reference.patches[DTMB5415Region.MAIN_HULL]
    function = functions[DTMB5415Region.MAIN_HULL]
    surface = aligned.patches[DTMB5415Region.MAIN_HULL].surface
    v = np.linspace(0.0, 1.0, 241)
    for axis, u in zip(axes.ravel(), (0.03, 0.20, 0.40, 0.60, 0.80, 0.97)):
        coordinates = np.column_stack((np.full_like(v, u), v))
        exact = np.asarray(patch.evaluate(function, coordinates).value)
        fitted = np.asarray(surface.evaluate(coordinates).value)
        station_x = float(np.mean(exact[:, 0]))
        axis.plot(
            exact[:, 1],
            exact[:, 2],
            color="0.25",
            linewidth=2.4,
            label="Exact IGES",
        )
        axis.plot(
            fitted[:, 1],
            fitted[:, 2],
            color="#0072b2",
            linewidth=1.2,
            linestyle="--",
            label="Reconstruction",
        )
        axis.set_title(f"x = {station_x:.3f} m")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.22)
    for axis in axes[:, 0]:
        axis.set_ylabel("z [m]")
    for axis in axes[-1, :]:
        axis.set_xlabel("Half-breadth, y [m]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("DTMB 5415 main-hull body-plan overlays")
    figure.subplots_adjust(bottom=0.12, hspace=0.28, wspace=0.18)
    _save(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_validation.png"),
    )
    arguments = parser.parse_args()
    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference = load_dtmb_5415(source)
    reference_functions = reference.build_functions()
    approximations = [
        fit_dtmb_5415(reference, level) for level in ("coarse", "medium", "fine")
    ]
    uniform_fine = approximations[-1]
    aligned_fine = fit_dtmb_5415(reference, "fine", knot_strategy="reference_aligned")

    print("DTMB 5415 reference dimensions:", reference.dimensions())
    for approximation in [*approximations, aligned_fine]:
        dome = approximation.sonar_dome
        print(
            f"{approximation.level:>6} {approximation.knot_strategy:>17}: global RMS "
            f"{1e3 * approximation.global_rms_error:.6f} mm, "
            f"dome RMS {1e3 * dome.rms_error:.6f} mm, "
            f"global max {1e3 * approximation.global_maximum_error:.6f} mm"
        )

    _plot_patch_definition(arguments.output, reference, reference_functions)
    _plot_surface_overlay(
        _derived_output(arguments.output, "overlay"),
        reference,
        reference_functions,
        aligned_fine,
    )
    _plot_error_comparison(
        _derived_output(arguments.output, "error_map"),
        reference,
        reference_functions,
        uniform_fine,
        aligned_fine,
    )
    _plot_form_profiles(
        _derived_output(arguments.output, "form_profiles"),
        reference,
        reference_functions,
        uniform_fine,
        aligned_fine,
    )
    _plot_body_plan(
        _derived_output(arguments.output, "body_plan"),
        reference,
        reference_functions,
        aligned_fine,
    )
    recorder.stop()


if __name__ == "__main__":
    main()
