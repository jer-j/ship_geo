"""Reconstruct DTMB 5415 with deck-edge sections and an explicit fullness curve.

This closes two gaps between ``ship_geo``'s naval form-parameter hull and the
curve network described in Sener (2016), "Parametric Design of a Surface
Combatant for Simulation-Driven Design and Hydrodynamic Optimization":

* Every section now extends from the keel through the design waterline to the
  deck edge, using a second F-Spline segment anchored by a ``DeckEdge``
  half-breadth curve and a ``TangentCurve at Deck`` (``tanDeck``), matching
  the paper's Fig. 10 section construction.
* Each section's underwater area is driven by an explicit ``SectionFullness``
  longitudinal curve through ``Area = SectionFullness * half_breadth * draft``
  (Fig. 10's formula with the flat-of-bottom term at zero, since DTMB 5415 is
  round-bilge), rather than being read directly off the sectional-area curve.

Both curves are additional implicit states in the same
:class:`~lsdo_geo.core.splines.variational.VariationalSystem`, so the whole
network -- longitudinal curves, underwater sections, deck sections, and the
fullness curve -- is still solved with exactly one CSDL Newton call.

A coarse station count keeps the single Newton solve tractable; it is a
demonstration of the closed gap, not the fine-resolution accuracy benchmark
already reported in ``docs/src/dtmb_5415.md``.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo.validation import download_dtmb_5415, load_dtmb_5415
from lsdo_geo.validation.dtmb_5415 import (
    DTMB5415Region,
    _constant_x_section,
    extract_dtmb_5415_form_data,
)
from lsdo_geo.core.ship_geometry.form_parameter_hull import FormParameterHullProblem


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _exact_full_section(
    reference, form_data, station: float, resolution: int = 241
) -> np.ndarray:
    """Sample the exact IGES keel-to-deck section at one station."""
    length = form_data.primary_parameters.length_between_perpendiculars
    x_coordinate = (
        form_data.coordinate_origin - 0.5 * length + length * float(station)
    )
    transition = reference.patches[DTMB5415Region.SONAR_DOME_TRANSITION]
    bounds = (
        float(np.min(transition.coefficients[:, :, 0])),
        float(np.max(transition.coefficients[:, :, 0])),
    )
    if x_coordinate < bounds[0]:
        region = DTMB5415Region.SONAR_DOME
    elif x_coordinate < bounds[1]:
        region = DTMB5415Region.SONAR_DOME_TRANSITION
    else:
        region = DTMB5415Region.MAIN_HULL
    functions = reference.build_functions()
    points = _constant_x_section(
        reference.patches[region], functions[region], x_coordinate, 121, resolution
    )
    return points[np.argsort(points[:, 2])]


def _generated_full_section(
    geometry, index: int, resolution: int = 61
) -> np.ndarray:
    """Concatenate the keel-to-waterline and waterline-to-deck F-Splines."""
    parameters = np.linspace(0.0, 1.0, resolution)
    underwater = geometry.hull.sections[index].evaluate(parameters).value
    topside = geometry.hull.freeboard_sections[index].evaluate(parameters).value
    return np.concatenate((underwater, topside[1:]), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_deck_and_fullness.png"),
    )
    parser.add_argument("--num-stations", type=int, default=5)
    parser.add_argument("--num-control-points", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--print-status", action="store_true")
    arguments = parser.parse_args()

    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference = load_dtmb_5415(source)
    form_data = extract_dtmb_5415_form_data(reference)

    section_stations = np.linspace(0.08, 0.98, arguments.num_stations)

    problem = FormParameterHullProblem(
        form_data.primary_parameters,
        form_data.fit_targets,
        num_form_control_points=arguments.num_control_points,
        num_section_control_points=arguments.num_control_points,
        section_station_parameters=section_stations,
        form_fit_weight=100.0,
        use_fullness_curve=True,
        x_origin=form_data.coordinate_origin,
        name="dtmb_5415_deck_fullness_demo",
    )
    print(f"assembled a {arguments.num_stations}-station curve network; solving...")
    start = time.time()
    geometry = problem.solve(max_iter=arguments.max_iter, print_status=arguments.print_status)
    print(f"solved in {time.time() - start:.1f} s")
    result = geometry.hull.variational_result
    print(
        "max |constraint residual| =",
        float(np.max(np.abs(result.constraint_residual.value))),
    )
    print(
        "max |stationarity residual| =",
        float(np.max(np.abs(result.stationarity_residual.value))),
    )
    print("states solved simultaneously:", len(result.stationarity_residuals))

    # ---- Figure 1: body-plan comparison, keel through deck -------------
    figure, axis = plt.subplots(figsize=(7.0, 6.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(section_stations)))
    for index, (station, color) in enumerate(zip(section_stations, colors)):
        exact = _exact_full_section(reference, form_data, station)
        generated = _generated_full_section(geometry, index)
        axis.plot(exact[:, 1], exact[:, 2], color=color, linewidth=1.4, alpha=0.55)
        axis.plot(
            generated[:, 1],
            generated[:, 0],
            "--",
            color=color,
            linewidth=1.8,
            label=f"v={station:.2f}",
        )
    axis.axhline(0.0, color="0.4", linewidth=0.8, linestyle=":")
    axis.text(0.01, 0.01, "design waterline", color="0.4", fontsize=8)
    axis.set_xlabel("half-breadth, y [m]")
    axis.set_ylabel("height above baseline, z [m]")
    axis.set_title(
        "DTMB 5415 body plan: exact IGES (solid) vs. keel-to-deck\n"
        "F-Spline reconstruction (dashed) -- single global solve"
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8, loc="upper left")
    _save(figure, arguments.output)

    # ---- Figure 2: deck-edge curves (breadth, height, tangent) ---------
    fit_stations = form_data.fit_targets.station_parameters
    dense = np.linspace(fit_stations.min(), fit_stations.max(), 201)
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 3.6), constrained_layout=True)
    axes[0].plot(
        fit_stations,
        form_data.fit_targets.deck_half_breadths,
        "o",
        color="0.35",
        markersize=4,
        label="extracted from IGES",
    )
    axes[0].plot(
        dense,
        geometry.deck_edge_curve.evaluate(dense).value,
        color="#0072b2",
        label="fitted DeckEdge F-Spline",
    )
    axes[0].set_title("Deck half-breadth")
    axes[0].set_xlabel("longitudinal parameter v")
    axes[0].set_ylabel("half-breadth [m]")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    axes[1].plot(
        fit_stations, form_data.fit_targets.deck_heights, "o", color="0.35", markersize=4
    )
    axes[1].plot(
        dense, geometry.deck_height_curve.evaluate(dense).value, color="#0072b2"
    )
    axes[1].set_title("Deck height above baseline")
    axes[1].set_xlabel("longitudinal parameter v")
    axes[1].set_ylabel("height [m]")

    axes[2].plot(
        fit_stations,
        np.degrees(form_data.fit_targets.deck_tangent_angles),
        "o",
        color="0.35",
        markersize=4,
    )
    axes[2].plot(
        dense,
        np.degrees(geometry.deck_tangent_curve.evaluate(dense).value.reshape(-1)),
        color="#0072b2",
    )
    axes[2].set_title("Deck tangent angle (tanDeck)")
    axes[2].set_xlabel("longitudinal parameter v")
    axes[2].set_ylabel("angle [deg]")
    for axis in axes:
        axis.grid(alpha=0.2)
    _save(figure, arguments.output.with_name(arguments.output.stem + "_deck_curves.png"))

    # ---- Figure 3: SectionFullness curve --------------------------------
    half_areas = form_data.fit_targets.half_areas
    half_breadths = form_data.fit_targets.half_breadths
    drafts = form_data.fit_targets.drafts
    target_fullness = half_areas / (half_breadths * drafts)

    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.plot(
        fit_stations,
        target_fullness,
        "o",
        color="0.35",
        markersize=5,
        label=r"$A/(b\,T)$ from IGES sections",
    )
    axis.plot(
        dense,
        geometry.fullness_curve.evaluate(dense).value,
        color="#d55e00",
        label="fitted SectionFullness curve",
    )
    axis.plot(
        section_stations,
        geometry.fullness_curve.evaluate(section_stations).value,
        "s",
        color="#d55e00",
        markersize=6,
        markerfacecolor="none",
        label="exact at generating stations",
    )
    axis.set_xlabel("longitudinal parameter v")
    axis.set_ylabel(r"sectional fullness $A_{1/2}/(b_{wl}\,T)$")
    axis.set_title(
        "SectionFullness curve\n"
        r"Cross Section Area$_x$ = SectionFullness$_x \cdot b_{wl,x} \cdot T_x$"
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    _save(figure, arguments.output.with_name(arguments.output.stem + "_fullness.png"))

    recorder.stop()


if __name__ == "__main__":
    main()
