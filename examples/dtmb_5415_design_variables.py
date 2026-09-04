"""Turn the reconstructed DTMB 5415 into a re-shapeable, fair, parametric hull.

Reconstruction is the means, not the end. The end is a hull that

1. reproduces DTMB 5415,
2. is driven by *known engineering quantities* that hold exactly, and
3. carries *additional shape variables* that reshape the form -- the solver
   moves the control points -- while the hull stays fair and the engineering
   quantities stay satisfied.

The two levels are enforced differently, which is the whole point of the
form-parameter approach:

``primary naval quantities``
    :math:`L_{PP}`, :math:`B`, :math:`T`, :math:`\\nabla`, LCB, :math:`C_{WP}`
    enter as **exact equality constraints** in the KKT system. Perturbing one
    of them moves the hull to a form that satisfies it to solver precision.

``auxiliary shape variables``
    Per-station deadrise, flare, deck height, and the sonar-dome bulb
    dimensions enter as **weighted least-squares objectives** competing with
    the fairness functional. Perturbing one requests a local shape change; the
    fairness term decides how that request is spread over the control net.

Every variant below is a complete single global Newton solve, JIT-compiled
once and then re-run per design in a fraction of a second.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo.core.ship_geometry.form_parameter_hull import (
    FormParameterHullProblem,
    LongitudinalFitTargets,
    NavalHullParameters,
)
from lsdo_geo.core.ship_geometry.form_curves import piecewise_open_knots
from lsdo_geo.validation import (
    download_dtmb_5415,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
    extract_dtmb_5415_section_fit_data,
    load_dtmb_5415,
)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--num-stations", type=int, default=7,
        help="Stations aft of the dome; three dome stations are added.")
    parser.add_argument("--num-form-control-points", type=int, default=14)
    parser.add_argument("--num-section-control-points", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_design_variables.png"),
    )
    arguments = parser.parse_args()

    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    # ---- Reconstruction data (eager; reads sampled geometry as arrays) ----
    extraction = csdl.Recorder(inline=True)
    extraction.start()
    reference = load_dtmb_5415(source)
    form_data = extract_dtmb_5415_form_data(reference)
    regions = dtmb_5415_longitudinal_regions(reference, form_data)
    dome_end = float(regions[1].end)
    # Observations and stations must resolve the dome, or the dome design
    # variable is inert: a bulb described by one observation cannot respond to
    # being asked to widen.
    observations = np.concatenate(
        (np.linspace(0.004, dome_end, 12), np.linspace(dome_end + 0.04, 1.0, 14))
    )
    form_data = extract_dtmb_5415_form_data(
        reference, station_parameters=observations
    )
    stations = np.concatenate(
        (
            np.array([0.02, 0.05, 0.09]),
            np.linspace(0.20, 1.0, arguments.num_stations),
        )
    )
    section_data = extract_dtmb_5415_section_fit_data(reference, stations)
    extraction.stop()
    print(
        f"observations: {observations.size} "
        f"({int(np.sum(observations <= dome_end))} in the dome), "
        f"generating stations: {stations.size}"
    )
    baseline = form_data.fit_targets
    primaries = form_data.primary_parameters
    print(f"reconstructed baseline from {stations.size} DTMB 5415 stations")

    # ---- Build the parametric hull with everything exposed as variables ---
    recorder = csdl.Recorder(inline=False)
    recorder.start()

    def variable(name: str, value) -> csdl.Variable:
        return csdl.Variable(
            value=np.atleast_1d(np.asarray(value, dtype=float)), name=name
        )

    # Level 1: known engineering quantities, held exactly.
    engineering = {
        "beam": variable("beam", primaries.beam),
        "displacement": variable("displacement", primaries.displacement),
        "lcb": variable("lcb", primaries.lcb),
        "waterplane_coefficient": variable(
            "waterplane_coefficient", primaries.waterplane_coefficient
        ),
    }
    # Level 2: additional shape variables, balanced against fairness.
    shape = {
        "deadrise_angles": variable("deadrise_angles", baseline.deadrise_angles),
        "flare_angles": variable("flare_angles", baseline.flare_angles),
        "deck_heights": variable("deck_heights", baseline.deck_heights),
        "bulge_half_breadths": variable(
            "bulge_half_breadths", baseline.bulge_half_breadths
        ),
    }

    problem = FormParameterHullProblem(
        NavalHullParameters(
            length_between_perpendiculars=(
                primaries.length_between_perpendiculars
            ),
            beam=engineering["beam"],
            draft=primaries.draft,
            displacement=engineering["displacement"],
            lcb=engineering["lcb"],
            waterplane_coefficient=engineering["waterplane_coefficient"],
        ),
        LongitudinalFitTargets(
            station_parameters=baseline.station_parameters,
            half_breadths=baseline.half_breadths,
            half_areas=baseline.half_areas,
            drafts=baseline.drafts,
            deadrise_angles=shape["deadrise_angles"],
            flare_angles=shape["flare_angles"],
            maximum_beam_parameter=baseline.maximum_beam_parameter,
            maximum_draft_parameter=baseline.maximum_draft_parameter,
            deck_half_breadths=baseline.deck_half_breadths,
            deck_heights=shape["deck_heights"],
            deck_tangent_angles=baseline.deck_tangent_angles,
            bulge_half_breadths=shape["bulge_half_breadths"],
            bulge_heights=baseline.bulge_heights,
            bulge_parameters=baseline.bulge_parameters,
        ),
        num_form_control_points=arguments.num_form_control_points,
        num_section_control_points=arguments.num_section_control_points,
        num_deck_control_points=5,
        section_station_parameters=section_data.station_parameters,
        section_fit_parameters=section_data.curve_parameters,
        section_fit_points=section_data.points,
        section_fit_weight=250.0,
        form_fit_weight=100.0,
        use_fullness_curve=True,
        form_knots=piecewise_open_knots(
            arguments.num_form_control_points, 3,
            breakpoints=(dome_end, 0.72), weights=(0.34, 0.33, 0.33),
        ),
        x_origin=form_data.coordinate_origin,
        name="dtmb_5415_design",
    )
    geometry = problem.solve(max_iter=20)

    surface = geometry.hull.surface
    recovered = geometry.recovered_primary_parameters()
    fairness = (
        surface.fairness_energy((2, 0))
        + 2.0 * surface.fairness_energy((1, 1))
        + surface.fairness_energy((0, 2))
    )
    body_parameters = np.linspace(0.0, 1.0, 61)
    body_stations = (0.12, 0.30, 0.50, 0.75, 0.95)
    outputs: dict[str, csdl.Variable] = {
        "constraint_residual": geometry.hull.variational_result.constraint_residual,
        "fairness_energy": fairness,
        "min_jacobian": csdl.minimum(geometry.hull.validity.jacobian_magnitudes),
        "min_half_breadth": csdl.minimum(geometry.hull.validity.half_breadths),
    }
    for name, value in recovered.items():
        outputs[f"primary_{name}"] = value
    for index, station in enumerate(body_stations):
        coordinates = np.column_stack(
            (body_parameters, np.full(body_parameters.size, station))
        )
        outputs[f"section_{index}"] = surface.evaluate(coordinates)
    recorder.stop()

    from csdl_alpha.experimental import JaxSimulator

    inputs = list(engineering.values()) + list(shape.values())
    simulator = JaxSimulator(
        recorder=recorder,
        additional_inputs=inputs,
        additional_outputs=list(outputs.values()),
        gpu=False,
    )

    def solve(label: str, changes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in changes.items():
            target = engineering[key] if key in engineering else shape[key]
            simulator[target] = np.atleast_1d(
                np.asarray(value, dtype=float)
            ).reshape(target.shape)
        import time

        start = time.time()
        simulator.run()
        elapsed = time.time() - start
        values = {key: np.asarray(simulator[var]) for key, var in outputs.items()}
        values["_seconds"] = np.asarray(elapsed)
        print(
            f"{label:<26} "
            f"B={float(values['primary_beam'].reshape(-1)[0]):.4f} "
            f"disp={float(values['primary_displacement'].reshape(-1)[0]):.4f} "
            f"LCB={float(values['primary_lcb'].reshape(-1)[0]):+.4f} "
            f"Cwp={float(values['primary_waterplane_coefficient'].reshape(-1)[0]):.4f} "
            f"| fair={float(values['fairness_energy'].reshape(-1)[0]):.4g} "
            f"minJ={float(values['min_jacobian'].reshape(-1)[0]):.3g} "
            f"|c|={float(np.max(np.abs(values['constraint_residual']))):.1e} "
            f"{elapsed:.3f}s"
        )
        return values

    def defaults() -> dict[str, np.ndarray]:
        values = {
            key: np.atleast_1d(np.asarray(getattr(primaries, key), dtype=float))
            for key in engineering
        }
        values.update(
            deadrise_angles=baseline.deadrise_angles,
            flare_angles=baseline.flare_angles,
            deck_heights=baseline.deck_heights,
            bulge_half_breadths=baseline.bulge_half_breadths,
        )
        return values

    print()
    print("each row is a complete single global Newton solve")
    print("-" * 118)
    variants: dict[str, dict[str, np.ndarray]] = {}
    variants["baseline (reconstruction)"] = solve(
        "baseline (reconstruction)", defaults()
    )

    # Level 1: known engineering quantities.
    changes = defaults()
    changes["displacement"] = np.asarray(primaries.displacement * 1.06)
    variants["displacement +6%"] = solve("displacement +6%", changes)

    changes = defaults()
    changes["lcb"] = np.asarray(primaries.lcb - 0.06)
    variants["LCB 60 mm aft"] = solve("LCB 60 mm aft", changes)

    changes = defaults()
    changes["beam"] = np.asarray(primaries.beam * 1.05)
    variants["beam +5%"] = solve("beam +5%", changes)

    # Level 2: additional shape variables.
    changes = defaults()
    changes["deadrise_angles"] = baseline.deadrise_angles + np.deg2rad(8.0)
    variants["deadrise +8 deg"] = solve("deadrise +8 deg", changes)

    changes = defaults()
    changes["flare_angles"] = baseline.flare_angles + np.deg2rad(10.0)
    variants["flare +10 deg"] = solve("flare +10 deg", changes)

    changes = defaults()
    changes["bulge_half_breadths"] = baseline.bulge_half_breadths * 1.25
    variants["sonar dome +25% wider"] = solve("sonar dome +25% wider", changes)
    print("-" * 118)

    # ---- Figure: what each knob does to the body plan -------------------
    figure, axes = plt.subplots(
        1, len(body_stations), figsize=(4.0 * len(body_stations), 4.4), sharey=True
    )
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(variants)))
    for index, station in enumerate(body_stations):
        axis = axes[index]
        for (label, values), color in zip(variants.items(), colors):
            points = values[f"section_{index}"]
            axis.plot(
                points[:, 1],
                points[:, 2],
                color=color,
                linewidth=2.2 if label.startswith("baseline") else 1.3,
                linestyle="-" if label.startswith("baseline") else "--",
                label=label if index == 0 else None,
            )
        axis.set_title(f"v = {station:.2f}")
        axis.set_xlabel("half-breadth y [m]")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("height z [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(variants), frameon=False)
    figure.suptitle(
        "One reconstructed DTMB 5415 hull, re-shaped by its design variables\n"
        "primary naval quantities stay exact; shape variables trade against fairness"
    )
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.93))
    _save(figure, arguments.output)


if __name__ == "__main__":
    main()
