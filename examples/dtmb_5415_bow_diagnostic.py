"""Diagnose the forward-region reconstruction: bow, sonar dome, and transition.

Reads the mesh cache written by ``dtmb_5415_accurate_reconstruction.py`` and
overlays generated body-plan sections on the exact IGES sections through the
bow, so the lines can be compared directly instead of through a 3D view.

It also reports whether the sonar-dome interior waypoints were actually
applied at each generating station, which is the difference between a dome
design variable that shapes the hull and one that is silently inert.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo.validation import (
    download_dtmb_5415,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
    load_dtmb_5415,
)
from lsdo_geo.validation.dtmb_5415 import DTMB5415Region, _constant_x_section


def _exact_section(reference, form_data, station: float) -> np.ndarray:
    length = form_data.primary_parameters.length_between_perpendiculars
    x = form_data.coordinate_origin - 0.5 * length + length * float(station)
    transition = reference.patches[DTMB5415Region.SONAR_DOME_TRANSITION]
    bounds = (
        float(np.min(transition.coefficients[:, :, 0])),
        float(np.max(transition.coefficients[:, :, 0])),
    )
    if x < bounds[0]:
        region = DTMB5415Region.SONAR_DOME
    elif x < bounds[1]:
        region = DTMB5415Region.SONAR_DOME_TRANSITION
    else:
        region = DTMB5415Region.MAIN_HULL
    functions = reference.build_functions()
    points = _constant_x_section(
        reference.patches[region], functions[region], x, 161, 321
    )
    return points[np.argsort(points[:, 2])]


def _generated_section(cache, regions, station: float) -> tuple[np.ndarray, np.ndarray]:
    """Pull the nearest constant-station column from the cached regional loft."""
    for region in regions:
        if region.start - 1e-12 <= station <= region.end + 1e-12:
            key = f"patch_{region.name}"
            if key not in cache.files:
                continue
            mesh = cache[key]
            local = (station - region.start) / (region.end - region.start)
            column = int(round(local * (mesh.shape[1] - 1)))
            underwater = mesh[:, column, :]
            break
    else:
        mesh = cache["underwater_mesh"]
        column = int(round(station * (mesh.shape[1] - 1)))
        underwater = mesh[:, column, :]
    freeboard = np.empty((0, 3))
    if "freeboard_mesh" in cache.files:
        deck = cache["freeboard_mesh"]
        column = int(round(station * (deck.shape[1] - 1)))
        freeboard = deck[:, column, :]
    return underwater, freeboard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_bow_diagnostic.png"),
    )
    parser.add_argument(
        "--stations",
        type=float,
        nargs="+",
        default=None,
        help="Stations to overlay (default: the forward group).",
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
    form_data = extract_dtmb_5415_form_data(reference)
    regions = dtmb_5415_longitudinal_regions(reference, form_data)

    print("longitudinal regions:")
    for region in regions:
        print(f"    {region.name:<20} v in [{region.start:.4f}, {region.end:.4f}]")

    targets = form_data.fit_targets
    stations = np.asarray(targets.station_parameters, dtype=float)
    depths = -np.asarray(targets.bulge_heights, dtype=float)
    drafts = np.asarray(targets.drafts, dtype=float)
    fractions = depths / np.maximum(drafts, 1e-12)
    parameters = np.asarray(targets.bulge_parameters, dtype=float)
    print()
    print("sonar-dome waypoint eligibility at the extracted stations")
    print(f"{'v':>7} {'bulge z':>9} {'draft':>8} {'depth frac':>11} "
          f"{'curve t':>8}  applied (>=0.15 and 0.02<t<0.98)")
    for index, station in enumerate(stations):
        applied = fractions[index] >= 0.15 and 0.02 < parameters[index] < 0.98
        print(f"{station:>7.3f} {-depths[index]:>9.4f} {drafts[index]:>8.4f} "
              f"{fractions[index]:>11.4f} {parameters[index]:>8.4f}  "
              f"{'YES' if applied else 'no'}")

    cache = np.load(arguments.cache)
    bow_stations = (
        tuple(arguments.stations)
        if arguments.stations
        else (0.02, 0.04, 0.06, 0.08, 0.12, 0.18)
    )
    figure, axes = plt.subplots(1, len(bow_stations), figsize=(3.1 * len(bow_stations), 4.6))
    for axis, station in zip(axes, bow_stations):
        exact = _exact_section(reference, form_data, station)
        underwater, freeboard = _generated_section(cache, regions, station)
        axis.plot(exact[:, 1], exact[:, 2], color="0.35", linewidth=2.0, label="exact IGES")
        axis.plot(
            underwater[:, 1], underwater[:, 2],
            "--", color="#0072b2", linewidth=1.9, label="generated underwater",
        )
        if freeboard.size:
            axis.plot(
                freeboard[:, 1], freeboard[:, 2],
                "--", color="#d55e00", linewidth=1.6, label="generated freeboard",
            )
        axis.axhline(0.0, color="0.6", linewidth=0.7, linestyle=":")
        axis.set_title(f"v = {station:.2f}")
        axis.set_xlabel("y [m]")
        axis.grid(alpha=0.2)
        axis.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("z [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle("DTMB 5415 forward sections: exact vs. reconstruction")
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 0.93))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    recorder.stop()
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
