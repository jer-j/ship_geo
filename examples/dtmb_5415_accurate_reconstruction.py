"""Accurate DTMB 5415 reconstruction with the dome-aware curve network.

This is the full-resolution counterpart to ``dtmb_5415_deck_and_fullness.py``:

* the twelve-station clustered station set, which resolves the sonar dome, the
  dome-to-hull transition, and terminates at ``v = 1`` so the transom is a
  finite section rather than a collapsed point;
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

from lsdo_geo.core.ship_geometry.form_parameter_hull import FormParameterHullProblem
from lsdo_geo.validation import (
    download_dtmb_5415,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
    extract_dtmb_5415_section_fit_data,
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
            0.04,
            0.08,
            transition_start,
            transition_start + (transition_end - transition_start) / 3.0,
            transition_start + 2.0 * (transition_end - transition_start) / 3.0,
            transition_end,
            0.28,
            0.42,
            0.58,
            0.75,
            0.90,
            1.0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--backend", choices=("jax", "inline"), default="jax")
    parser.add_argument("--num-form-control-points", type=int, default=10)
    parser.add_argument("--num-section-control-points", type=int, default=8)
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
    form_data = extract_dtmb_5415_form_data(reference)
    regions = dtmb_5415_longitudinal_regions(reference, form_data)
    section_stations = _default_section_stations(regions)
    section_data = extract_dtmb_5415_section_fit_data(reference, section_stations)
    holdout_data = extract_dtmb_5415_section_fit_data(
        reference,
        DEFAULT_VALIDATION_STATIONS,
        num_curve_points=section_data.curve_parameters.size,
    )
    extraction_recorder.stop()
    print(f"extracted {section_stations.size} generating stations "
          f"and {len(DEFAULT_VALIDATION_STATIONS)} holdout stations")

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
        section_station_parameters=section_data.station_parameters,
        section_fit_parameters=section_data.curve_parameters,
        section_fit_points=section_data.points,
        section_fit_weight=250.0,
        form_fit_weight=100.0,
        use_fullness_curve=True,
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
    outputs: dict[str, csdl.Variable] = {
        "constraint_residual": result.constraint_residual,
        "underwater_mesh": geometry.hull.surface.mesh((81, 161)),
    }
    if geometry.hull.freeboard_surface is not None:
        outputs["freeboard_mesh"] = geometry.hull.freeboard_surface.mesh((41, 161))
    if regional is not None:
        for name, patch in regional.patches.items():
            outputs[f"patch_{name}"] = patch.mesh((61, 81))
    for label, data in (("fit", section_data), ("holdout", holdout_data)):
        for index, station in enumerate(data.station_parameters):
            outputs[f"{label}_section_{index}"] = regional.evaluate_section(
                float(station), data.curve_parameters
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
        if key.startswith(("patch_", "curve_")) or key.endswith("_mesh"):
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
