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

# Where the blend line and the waterline sit on every unified section. One
# loft needs one basis, so these are the same at every station and each
# section's own arc length is stretched to match.
BLEND_PARAMETER = 0.30
WATERLINE_PARAMETER = 0.65

DEFAULT_VALIDATION_STATIONS = (
    0.03, 0.06, 0.10, 0.15, 0.23, 0.35, 0.50, 0.67, 0.83, 0.95,
)


def _coarse_section_stations(regions) -> np.ndarray:
    """A reduced station set for machines that cannot hold the full graph.

    Carrying a lower band the whole length puts two sections at every
    station, which took the full nineteen-station graph to 236k nodes and
    past the memory of a 15 GB container. This keeps the same three-region
    structure and the same features -- four stations through the dome, the
    four the transition patch needs, and the pair bracketing the skeg -- with
    fifteen stations instead of nineteen.
    """
    transition_start = regions[0].end
    transition_end = regions[1].end
    return np.asarray(
        (
            0.012, 0.032, 0.052, 0.090,
            transition_start,
            transition_start + (transition_end - transition_start) / 3.0,
            transition_start + 2.0 * (transition_end - transition_start) / 3.0,
            transition_end,
            0.25, 0.45, 0.65, 0.79, 0.845, 0.92, 1.0,
        )
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
            # end span alone. The last sits just inside the taper, so the
            # patch runs out to nothing rather than ending at finite width.
            0.012,
            0.025,
            0.038,
            0.051,
            0.063,
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
    parser.add_argument(
        "--coarse",
        action="store_true",
        help=(
            "Use fifteen generating stations instead of nineteen. The "
            "full-length lower band doubles the sections, and the full set "
            "does not fit in 15 GB during the XLA compile."
        ),
    )
    parser.add_argument(
        "--unify-bands",
        action="store_true",
        help=(
            "Build one curve per section from keel to deck edge, so the quick "
            "work and the dead work are one surface and one patch. The "
            "waterline and the blend line become interior conditions on that "
            "curve instead of places where one surface stops and another "
            "starts."
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
    section_stations = (
        _coarse_section_stations(regions)
        if arguments.coarse
        else _default_section_stations(regions)
    )
    print(
        f"observations: {observation_stations.size} stations, "
        f"{int(np.sum(observation_stations <= regions[1].end))} inside the dome "
        f"and transition (v <= {regions[1].end:.4f})"
    )
    # The blend line each section is reparameterized around, taken from the
    # observations at the generating stations.
    blend_at_sections = np.interp(
        section_stations,
        observation_stations,
        np.asarray(form_data.fit_targets.blend_depths, dtype=float).reshape(-1),
    )
    section_data = extract_dtmb_5415_section_fit_data(
        reference,
        section_stations,
        full_section=arguments.unify_bands,
        blend_depths=blend_at_sections if arguments.unify_bands else None,
        blend_parameter=BLEND_PARAMETER,
        waterline_parameter=WATERLINE_PARAMETER,
    )
    holdout_data = extract_dtmb_5415_section_fit_data(
        reference,
        DEFAULT_VALIDATION_STATIONS,
        num_curve_points=section_data.curve_parameters.size,
        full_section=arguments.unify_bands,
        blend_depths=(
            np.interp(
                np.asarray(DEFAULT_VALIDATION_STATIONS, dtype=float),
                observation_stations,
                np.asarray(form_data.fit_targets.blend_depths, dtype=float).reshape(-1),
            )
            if arguments.unify_bands
            else None
        ),
        blend_parameter=BLEND_PARAMETER,
        waterline_parameter=WATERLINE_PARAMETER,
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
    # The rake is a profile along the girth of a keel-to-waterline section.
    # With a lower band carried below the blend line the underwater face is
    # two bands, so the profile is split at the blend and resampled onto each
    # band's own control points. Both sides take the same value at the split,
    # which keeps the shared blend point at one x.
    blend_fraction = float(
        np.asarray(form_data.fit_targets.blend_depths, dtype=float).reshape(-1)[-1]
        / np.asarray(form_data.fit_targets.dome_depths, dtype=float).reshape(-1)[-1]
    )
    girth = np.linspace(0.0, 1.0, np.asarray(transom_offsets).size)
    if arguments.unify_bands:
        # One curve spans the whole girth, so the rake profile is resampled
        # onto its control points rather than split between bands. The
        # underwater profile is carried up to the waterline and the freeboard
        # profile above it.
        waterline_share = WATERLINE_PARAMETER
        deck_girth = np.linspace(0.0, 1.0, np.asarray(transom_deck_offsets).size)
        lower = np.interp(
            np.linspace(0.0, 1.0, arguments.num_section_control_points),
            girth,
            np.asarray(transom_offsets, dtype=float),
        )
        upper = np.interp(
            np.linspace(0.0, 1.0, arguments.num_section_control_points),
            deck_girth,
            np.asarray(transom_deck_offsets, dtype=float),
        )
        blend_in = np.clip(
            (np.linspace(0.0, 1.0, arguments.num_section_control_points)
             - waterline_share) / 0.15,
            0.0,
            1.0,
        )
        transom_offsets = (1.0 - blend_in) * lower + blend_in * upper
        dome_transom_offsets = None
        transom_deck_offsets = None
        print(
            f"  rake carried on one curve, waterline at {waterline_share:.3f} "
            f"of arc: {1e3 * transom_offsets.ptp():.1f} mm"
        )
    else:
        dome_transom_offsets = np.interp(
            np.linspace(0.0, blend_fraction, arguments.num_dome_control_points),
            girth,
            np.asarray(transom_offsets, dtype=float),
        )
        transom_offsets = np.interp(
            np.linspace(blend_fraction, 1.0, arguments.num_section_control_points),
            girth,
            np.asarray(transom_offsets, dtype=float),
        )
        print(
            f"  rake split at the blend, {blend_fraction:.3f} of depth: "
            f"lower band {1e3 * dome_transom_offsets.ptp():.1f} mm, "
            f"upper band {1e3 * transom_offsets.ptp():.1f} mm"
        )
    extraction_recorder.stop()
    print(f"extracted {section_stations.size} generating stations "
          f"and {len(DEFAULT_VALIDATION_STATIONS)} holdout stations")

    # Report which generating stations will carry a sonar-dome band, using the
    # same rule the solver applies, so a starved dome patch is visible before
    # the expensive compile rather than after it.
    # The blend line runs the whole length, so every station should qualify;
    # the check below is what guarantees both bands loft on the same stations.
    dome_depths = np.asarray(form_data.fit_targets.dome_depths, dtype=float).reshape(-1)
    blend_depths = np.asarray(
        form_data.fit_targets.blend_depths, dtype=float
    ).reshape(-1)
    banded = [
        float(station)
        for station in section_stations
        if (
            np.interp(float(station), observation_stations, dome_depths)
            - np.interp(float(station), observation_stations, blend_depths)
        )
        > 0.05 * np.interp(float(station), observation_stations, dome_depths)
    ]
    print(
        f"sonar-dome band on {len(banded)} generating stations: "
        + ", ".join(f"{value:.3f}" for value in banded)
    )
    if len(banded) < 4:
        raise SystemExit(
            f"the lower band needs at least four generating stations, got "
            f"{len(banded)}."
        )
    if len(banded) != len(section_stations):
        print(
            "  note: the band is not full length, so it is lofted on its own "
            "stations and meets the hull patch only where they coincide."
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
        dome_transom_x_offsets=dome_transom_offsets,
        unify_bands=arguments.unify_bands,
        section_waterline_parameters=section_data.waterline_parameters,
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
    # With one patch there is no regional subdivision to evaluate through.
    if regional is not None:
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

    # ``evaluate_section`` samples the main hull surface. Where a sonar-dome
    # band is carried that surface begins at the blend line, so comparing it
    # against a reference running down to the keel measures the band that is
    # missing from the comparison rather than any error in the geometry -- it
    # reads 147 mm at v = 0.03 where the assembled section is within 4.2 mm.
    # Score against both surfaces, by nearest point so that no shared
    # parameterization between the reference and the two bands is assumed.
    def _assembled_section(x_target: float) -> np.ndarray | None:
        pieces = []
        for key in ("underwater_mesh", "dome_mesh"):
            if key not in values:
                continue
            # Axis 1 of these meshes runs longitudinally; axis 0 runs the girth.
            mesh = np.transpose(values[key], (1, 0, 2))
            stations = mesh[:, :, 0].mean(axis=1)
            order = np.argsort(stations)
            stations, mesh = stations[order], mesh[order]
            if x_target < stations[0] - 0.02 or x_target > stations[-1] + 0.02:
                continue
            upper = int(np.clip(np.searchsorted(stations, x_target), 1, len(stations) - 1))
            span = stations[upper] - stations[upper - 1]
            blend = 0.0 if span <= 0.0 else (x_target - stations[upper - 1]) / span
            row = (1.0 - blend) * mesh[upper - 1] + blend * mesh[upper]
            pieces.append(np.stack([row[:, 2], row[:, 1]], axis=1))
        if not pieces:
            return None
        section = np.concatenate(pieces)
        section = section[np.argsort(section[:, 0])]
        steps = np.linalg.norm(np.diff(section, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(steps)))
        if arc[-1] <= 0.0:
            return section
        dense = np.linspace(0.0, arc[-1], 2000)
        return np.stack(
            [np.interp(dense, arc, section[:, 0]), np.interp(dense, arc, section[:, 1])],
            axis=1,
        )

    origin = float(form_data.coordinate_origin)
    span_length = float(form_data.primary_parameters.length_between_perpendiculars)

    payload: dict[str, np.ndarray] = {}
    section_errors: dict[str, np.ndarray] = {}
    for label, data in (("fit", section_data), ("holdout", holdout_data)):
        distances = []
        for index in range(data.station_parameters.size):
            station = float(data.station_parameters[index])
            assembled = _assembled_section(origin - 0.5 * span_length + span_length * station)
            reference = data.points[index]
            if assembled is None:
                generated = values[f"{label}_section_{index}"][:, [2, 1]]
                distances.append(np.linalg.norm(generated - reference, axis=1))
                continue
            distances.append(
                np.linalg.norm(
                    reference[:, None, :] - assembled[None, :, :], axis=2
                ).min(axis=1)
            )
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
