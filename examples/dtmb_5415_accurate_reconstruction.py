"""Accurate DTMB 5415 reconstruction with the dome-aware curve network.

This is the full-resolution counterpart to ``dtmb_5415_deck_and_fullness.py``:

* observation stations clustered through the sonar dome. The default
  ``linspace(0.03, 1.0, 13)`` lands a single sample inside a dome that ends
  near ``v = 0.12``, so every longitudinal curve describing the bulb was fit
  through one observation and the forward sections were extrapolated;
* longitudinal curves on feature-clustered knots. On a uniform vector with ten
  coefficients the first interior knot falls near ``v = 0.14``, putting the
  whole dome inside one cubic span, which cannot rise and fall. The hull has a
  second short feature aft: the skeg termination drops the keel roughly
  100 mm within ``0.04`` of length, so knots are massed at both ends and left
  sparse through the prismatic midbody;
* a generating station set clustered at both features and terminating at
  ``v = 1``, so the transom is a finite section rather than a collapsed point;
* the three-region longitudinal loft (forward dome / transition / main hull);
* deck-edge sections and the explicit ``SectionFullness`` curve; and
* sonar-dome interior waypoints driven by ``BulgeHalfBreadth`` and
  ``BulgeHeight`` longitudinal curves, so dome stations can reproduce the
  bulb's maximum half-breadth and the necking above it -- shape a monotone
  keel-to-waterline F-Spline cannot represent.

Everything is still assembled into one :class:`VariationalSystem` and solved by
a single CSDL Newton call.

Two execution backends are available:

``--backend inline``
    CSDL's eager interpreter. Every Newton iteration walks the derivative
    graph in Python, which dominates the runtime.

``--backend jax`` (default)
    Builds the same graph under a deferred recorder and JIT-compiles it
    through ``csdl_alpha``'s JAX backend. Reference geometry extraction still
    runs under its own inline recorder, because it reads sampled values back
    as arrays; only the hull solve is compiled.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from lsdo_geo.core.ship_geometry.form_curves import piecewise_open_knots
from lsdo_geo.core.ship_geometry.form_parameter_hull import FormParameterHullProblem
from lsdo_geo.validation import (
    download_dtmb_5415,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
    extract_dtmb_5415_section_fit_data,
    extract_dtmb_5415_transom_offsets,
    load_dtmb_5415,
)

DEFAULT_VALIDATION_STATIONS = (
    0.03, 0.06, 0.10, 0.15, 0.23, 0.35, 0.50, 0.67, 0.83, 0.95,
)


def _default_section_stations(regions) -> np.ndarray:
    """Cluster stations through the dome and transition, ending at the transom."""
    transition_start = regions[0].end
    transition_end = regions[1].end
    return np.asarray(
        (
            # The sonar-dome band closes near v = 0.072 (the neck between the
            # bulb and the forefoot merges into a monotone profile there), so
            # the band gets five stations of its own: a cubic dome patch needs
            # four, and a fifth keeps the closure from being carried by the
            # end span alone.
            0.012,
            0.026,
            0.042,
            0.058,
            0.070,
            0.090,
            # The transition is lofted as its own cubic patch, so it needs at
            # least four stations of its own.
            transition_start,
            transition_start + (transition_end - transition_start) / 3.0,
            transition_start + 2.0 * (transition_end - transition_start) / 3.0,
            transition_end,
            0.22,
            0.36,
            0.52,
            0.68,
            # The skeg termination drops the keel sharply between v = 0.83 and
            # v = 0.87, so the run aft needs stations as closely spaced as the
            # dome does forward. v = 0.83 is deliberately left uncovered so it
            # remains an independent holdout.
            0.79,
            0.845,
            0.90,
            0.95,
            1.0,
        )
    )


def _observation_stations(regions) -> np.ndarray:
    """Auxiliary observation stations, clustered through the sonar dome.

    The default ``linspace(0.03, 1.0, 13)`` puts a single sample inside the
    dome, so every longitudinal curve describing the bulb is fit through one
    observation and the forward sections are extrapolated. Sampling density
    here is nearly free: the observations enter as one vectorized
    least-squares term per curve, not as extra implicit states.
    """
    dome_end = regions[1].end
    # Forward of v ~ 0.009 the section is a few millimetres wide and the neck
    # detection is unresolved -- the blend height bounces between 0.016 and
    # 0.247 over four consecutive samples. Sampling starts where the feature
    # is actually resolvable and the fairness term carries the last few
    # millimetres to the stem; no generating station lives there anyway.
    band = np.linspace(0.009, 0.072, 12)
    forward = np.linspace(0.080, dome_end, 6)
    middle = np.linspace(dome_end + 0.03, 0.74, 11)
    stern = np.linspace(0.76, 1.0, 13)
    return np.concatenate((band, forward, middle, stern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--backend", choices=("jax", "inline"), default="jax")
    parser.add_argument("--num-form-control-points", type=int, default=18)
    parser.add_argument("--num-section-control-points", type=int, default=8)
    parser.add_argument(
        "--num-deck-control-points",
        type=int,
        default=5,
        help=(
            "Freeboard segments carry only two point and two tangent conditions, "
            "so they need far fewer coefficients than the underwater sections."
        ),
    )
    parser.add_argument(
        "--num-dome-control-points",
        type=int,
        default=6,
        help=(
            "The dome band carries two point conditions, two tangent conditions "
            "and an area condition, so it needs fewer coefficients than the "
            "main underwater band."
        ),
    )
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=0.75,
        help=(
            "Scales the readback mesh resolutions. These sample the solved "
            "surface and do not enter the Newton solve, but they are dense "
            "tensor-product evaluations inside the compiled function, so they "
            "drive peak compile memory. Lower this if XLA runs out of memory."
        ),
    )
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--print-status", action="store_true")
    parser.add_argument(
        "--no-dome-waypoints",
        action="store_true",
        help="Ablation: drop the sonar-dome interior waypoints.",
    )
    arguments = parser.parse_args()

    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    # ---- Phase 1: extract reference form data (needs eager evaluation) ----
    extraction_recorder = csdl.Recorder(inline=True)
    extraction_recorder.start()
    reference = load_dtmb_5415(source)
    # A first pass only to locate the dome and transition boundaries, which
    # then set where the observations need to be dense.
    form_data = extract_dtmb_5415_form_data(reference)
    regions = dtmb_5415_longitudinal_regions(reference, form_data)
    observation_stations = _observation_stations(regions)
    form_data = extract_dtmb_5415_form_data(
        reference, station_parameters=observation_stations
    )
    section_stations = _default_section_stations(regions)
    print(
        f"observations: {observation_stations.size} stations, "
        f"{int(np.sum(observation_stations <= regions[1].end))} inside the dome "
        f"and transition (v <= {regions[1].end:.4f})"
    )
    section_data = extract_dtmb_5415_section_fit_data(reference, section_stations)
    holdout_data = extract_dtmb_5415_section_fit_data(
        reference,
        DEFAULT_VALIDATION_STATIONS,
        num_curve_points=section_data.curve_parameters.size,
    )
    transom_offsets, transom_deck_offsets, transom_info = (
        extract_dtmb_5415_transom_offsets(
            reference,
            arguments.num_section_control_points,
            arguments.num_deck_control_points,
        )
    )
    print(
        "transom edge rake extent: "
        f"{1e3 * transom_info['rake_extent']:.1f} mm "
        f"(x {transom_info['minimum_x']:.4f} .. {transom_info['maximum_x']:.4f})"
    )
    extraction_recorder.stop()
    print(f"extracted {section_stations.size} generating stations "
          f"and {len(DEFAULT_VALIDATION_STATIONS)} holdout stations")

    # Report which generating stations will carry a sonar-dome band, using the
    # same rule the solver applies, so a starved dome patch is visible before
    # the expensive compile rather than after it.
    dome_depths = np.asarray(form_data.fit_targets.dome_depths, dtype=float).reshape(-1)
    blend_depths = np.asarray(
        form_data.fit_targets.blend_depths, dtype=float
    ).reshape(-1)
    draft_value = float(np.asarray(form_data.primary_parameters.draft).reshape(-1)[0])
    banded = [
        float(station)
        for station in section_stations
        if (
            np.interp(float(station), observation_stations, dome_depths)
            - np.interp(float(station), observation_stations, blend_depths)
        )
        > 0.05 * draft_value
    ]
    print(
        f"sonar-dome band on {len(banded)} generating stations: "
        + ", ".join(f"{value:.3f}" for value in banded)
    )
    if len(banded) < 4:
        raise SystemExit(
            f"the dome patch needs at least four generating stations, got "
            f"{len(banded)}; add stations forward of the dome closure."
        )

    targets = form_data.fit_targets
    if arguments.no_dome_waypoints:
        import dataclasses

        targets = dataclasses.replace(
            targets,
            bulge_half_breadths=None,
            bulge_heights=None,
            bulge_parameters=None,
        )

    # ---- Phase 2: build and solve the hull curve network -----------------
    use_jax = arguments.backend == "jax"
    recorder = csdl.Recorder(inline=not use_jax)
    recorder.start()
    build_start = time.time()
    beam_input = csdl.Variable(
        value=float(form_data.primary_parameters.beam), name="beam"
    )
    primary = type(form_data.primary_parameters)(
        length_between_perpendiculars=(
            form_data.primary_parameters.length_between_perpendiculars
        ),
        beam=beam_input,
        draft=form_data.primary_parameters.draft,
        displacement=form_data.primary_parameters.displacement,
        lcb=form_data.primary_parameters.lcb,
        waterplane_coefficient=(
            form_data.primary_parameters.waterplane_coefficient
        ),
    )
    problem = FormParameterHullProblem(
        primary,
        targets,
        num_form_control_points=arguments.num_form_control_points,
        num_section_control_points=arguments.num_section_control_points,
        num_deck_control_points=arguments.num_deck_control_points,
        num_dome_control_points=arguments.num_dome_control_points,
        section_station_parameters=section_data.station_parameters,
        section_fit_parameters=section_data.curve_parameters,
        section_fit_points=section_data.points,
        section_fit_weight=250.0,
        form_fit_weight=100.0,
        use_fullness_curve=True,
        form_knots=piecewise_open_knots(
            arguments.num_form_control_points,
            3,
            breakpoints=(float(regions[1].end), 0.72),
            weights=(0.34, 0.33, 0.33),
        ),
        transom_x_offsets=transom_offsets,
        transom_deck_x_offsets=transom_deck_offsets,
        x_origin=form_data.coordinate_origin,
        longitudinal_regions=regions,
        name="dtmb_5415_accurate",
    )
    geometry = problem.solve(
        max_iter=arguments.max_iter, print_status=arguments.print_status
    )
    print(f"[{arguments.backend}] graph built in {time.time() - build_start:.1f}s")

    # Declare everything to read back, so the compiled graph produces it.
    result = geometry.hull.variational_result
    regional = geometry.hull.regional_surface

    def resolution(rows: int, columns: int) -> tuple[int, int]:
        scale = max(arguments.mesh_scale, 0.1)
        return (max(int(round(rows * scale)), 9), max(int(round(columns * scale)), 9))

    outputs: dict[str, csdl.Variable] = {
        "constraint_residual": result.constraint_residual,
        "underwater_mesh": geometry.hull.surface.mesh(resolution(81, 161)),
    }
    if geometry.hull.freeboard_surface is not None:
        outputs["freeboard_mesh"] = geometry.hull.freeboard_surface.mesh(
            resolution(41, 161)
        )
    if geometry.hull.dome_surface is not None:
        outputs["dome_mesh"] = geometry.hull.dome_surface.mesh(resolution(41, 81))
    if regional is not None:
        for name, patch in regional.patches.items():
            outputs[f"patch_{name}"] = patch.mesh(resolution(61, 81))
    for label, data in (("fit", section_data), ("holdout", holdout_data)):
        for index, station in enumerate(data.station_parameters):
            outputs[f"{label}_section_{index}"] = regional.evaluate_section(
                float(station), data.curve_parameters
            )
    # Control nets are tiny next to sampled meshes, and the knot vectors are
    # constants, so caching the coefficients lets any later figure be rebuilt
    # at arbitrary resolution without paying for another compile.
    surface_spaces: dict[str, tuple[np.ndarray, ...]] = {}
    surfaces = {
        "underwater": geometry.hull.surface,
        "freeboard": geometry.hull.freeboard_surface,
        "dome": geometry.hull.dome_surface,
    }
    if regional is not None:
        for name, patch in regional.patches.items():
            surfaces[f"patch_{name}"] = patch
    for name, surface in surfaces.items():
        if surface is None:
            continue
        outputs[f"coefficients_{name}"] = surface.coefficients
        space = surface.space
        surface_spaces[name] = (
            np.asarray(space.degree, dtype=int),
            np.asarray(space.coefficients_shape, dtype=int),
            np.asarray(space.knots, dtype=float),
        )

    for name, curve in (
        ("sectional_area", geometry.sectional_area_curve),
        ("waterline", geometry.waterline_curve),
        ("draft", geometry.draft_curve),
        ("fullness", geometry.fullness_curve),
        ("deck_edge", geometry.deck_edge_curve),
        ("deck_height", geometry.deck_height_curve),
        ("bulge_breadth", geometry.bulge_breadth_curve),
        ("bulge_height", geometry.bulge_height_curve),
    ):
        if curve is not None:
            outputs[f"curve_{name}"] = curve.evaluate(np.linspace(0.0, 1.0, 201))
    recovered = geometry.recovered_primary_parameters()
    for name, value in recovered.items():
        outputs[f"primary_{name}"] = value
    recorder.stop()

    values: dict[str, np.ndarray] = {}
    if use_jax:
        from csdl_alpha.experimental import JaxSimulator

        compile_start = time.time()
        simulator = JaxSimulator(
            recorder=recorder,
            additional_inputs=[beam_input],
            additional_outputs=list(outputs.values()),
            gpu=False,
        )
        simulator.run()
        print(f"[jax] JIT compile + solve in {time.time() - compile_start:.1f}s")
        repeat_start = time.time()
        simulator.run()
        print(f"[jax] repeat solve in {time.time() - repeat_start:.3f}s")
        values = {
            key: np.asarray(simulator[variable]) for key, variable in outputs.items()
        }
    else:
        values = {
            key: np.asarray(variable.value) for key, variable in outputs.items()
        }

    print("states solved simultaneously:", len(result.stationarity_residuals))
    print(
        "max |constraint residual| =",
        float(np.max(np.abs(values["constraint_residual"]))),
    )
    primary_targets = {
        "beam": form_data.primary_parameters.beam,
        "draft": form_data.primary_parameters.draft,
        "displacement": form_data.primary_parameters.displacement,
        "lcb": form_data.primary_parameters.lcb,
        "waterplane_coefficient": (
            form_data.primary_parameters.waterplane_coefficient
        ),
    }
    max_primary = 0.0
    for name, target in primary_targets.items():
        error = float(values[f"primary_{name}"].reshape(-1)[0]) - float(target)
        max_primary = max(max_primary, abs(error))
    print(f"max primary-parameter residual = {max_primary:.3e}")

    payload: dict[str, np.ndarray] = {}
    section_errors: dict[str, np.ndarray] = {}
    for label, data in (("fit", section_data), ("holdout", holdout_data)):
        distances = []
        for index in range(data.station_parameters.size):
            generated = values[f"{label}_section_{index}"][:, [2, 1]]
            distances.append(np.linalg.norm(generated - data.points[index], axis=1))
        stacked = np.asarray(distances)
        section_errors[label] = stacked
        rms = float(np.sqrt(np.mean(stacked**2)))
        print(
            f"{label}-section RMS/max [mm]: {1.0e3 * rms:.3f} / "
            f"{1.0e3 * float(np.max(stacked)):.3f}"
        )
        payload[f"{label}_station_rms"] = np.sqrt(np.mean(stacked**2, axis=1))
        payload[f"{label}_stations"] = data.station_parameters

    print("per-holdout-station RMS [mm]:")
    for station, rms in zip(
        holdout_data.station_parameters, payload["holdout_station_rms"]
    ):
        print(f"    v={station:.3f}  {1.0e3 * rms:8.3f}")

    for key, value in values.items():
        if (
            key.startswith(("patch_", "curve_", "fit_section_", "holdout_section_"))
            or key.endswith("_mesh")
        ):
            payload[key] = value
    for label, data in (("fit", section_data), ("holdout", holdout_data)):
        payload[f"{label}_reference_points"] = data.points
        payload[f"{label}_curve_parameters"] = data.curve_parameters
    for name, (degrees, shape, knots) in surface_spaces.items():
        payload[f"space_{name}_degrees"] = degrees
        payload[f"space_{name}_shape"] = shape
        payload[f"space_{name}_knots"] = knots
    for key, value in values.items():
        if key.startswith("coefficients_"):
            payload[key] = value
    payload["section_stations"] = section_data.station_parameters
    payload["length"] = np.asarray(
        float(form_data.primary_parameters.length_between_perpendiculars)
    )
    payload["coordinate_origin"] = np.asarray(float(form_data.coordinate_origin))
    np.savez(arguments.cache, **payload)
    print(f"wrote {arguments.cache}")


if __name__ == "__main__":
    main()
