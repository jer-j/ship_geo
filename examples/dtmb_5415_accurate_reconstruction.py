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
a single CSDL Newton call. The solve is expensive; run it in the background.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from lsdo_geo.validation import (
    calibrate_dtmb_5415_form_hull,
    download_dtmb_5415,
    load_dtmb_5415,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
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

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference = load_dtmb_5415(source)

    print("solving the accurate dome-aware curve network (single Newton call)...")
    start = time.time()
    calibration = calibrate_dtmb_5415_form_hull(
        reference,
        num_form_control_points=arguments.num_form_control_points,
        num_section_control_points=arguments.num_section_control_points,
        include_deck=True,
        use_fullness_curve=True,
        include_sonar_dome_waypoints=not arguments.no_dome_waypoints,
        max_iter=arguments.max_iter,
        print_status=arguments.print_status,
    )
    print(f"solved in {time.time() - start:.1f} s")

    geometry = calibration.geometry
    result = geometry.hull.variational_result
    print("states solved simultaneously:", len(result.stationarity_residuals))
    print(
        "max |constraint residual| =",
        float(np.max(np.abs(result.constraint_residual.value))),
    )
    max_primary = max(
        abs(value) for value in calibration.primary_parameter_errors.values()
    )
    print(f"max primary-parameter residual = {max_primary:.3e}")
    print(
        "fit-section RMS/max [mm]: "
        f"{1.0e3 * calibration.fitting_section_rms_error:.3f} / "
        f"{1.0e3 * calibration.fitting_section_maximum_error:.3f}"
    )
    print(
        "holdout-section RMS/max [mm]: "
        f"{1.0e3 * calibration.validation_section_rms_error:.3f} / "
        f"{1.0e3 * calibration.validation_section_maximum_error:.3f}"
    )
    print("per-holdout-station RMS [mm]:")
    for station, rms in zip(
        calibration.validation_section_data.station_parameters,
        calibration.validation_station_rms_errors,
    ):
        print(f"    v={station:.3f}  {1.0e3 * rms:8.3f}")
    print("auxiliary curve RMS errors:", calibration.auxiliary_rms_errors)

    payload: dict[str, np.ndarray] = {}
    regional = geometry.hull.regional_surface
    if regional is not None:
        for name, patch in regional.patches.items():
            payload[f"patch_{name}"] = patch.mesh((61, 81)).value
    payload["underwater_mesh"] = geometry.hull.surface.mesh((81, 161)).value
    if geometry.hull.freeboard_surface is not None:
        payload["freeboard_mesh"] = geometry.hull.freeboard_surface.mesh(
            (41, 161)
        ).value
    payload["section_stations"] = calibration.section_fit_data.station_parameters
    payload["holdout_stations"] = (
        calibration.validation_section_data.station_parameters
    )
    payload["holdout_station_rms"] = calibration.validation_station_rms_errors
    payload["length"] = np.asarray(
        float(calibration.form_data.primary_parameters.length_between_perpendiculars)
    )
    payload["coordinate_origin"] = np.asarray(
        float(calibration.form_data.coordinate_origin)
    )
    np.savez(arguments.cache, **payload)
    print(f"wrote {arguments.cache}")
    recorder.stop()


if __name__ == "__main__":
    main()
