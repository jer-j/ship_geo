"""Coincident-wireframe overlay of the canonical and reconstructed surfaces.

Follows the comparison conventions already used for this hull: the canonical
IGES is drawn in gray, the generated regional surfaces in dashed colour keyed
to their region, and neither is hidden behind an opaque patch or a control
net. Transverse sections and longitudinal lines are both drawn, so agreement
shows up as coincidence rather than as a shaded surface that could conceal a
gap.

Reads the mesh cache from ``dtmb_5415_accurate_reconstruction.py``; no solve.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lsdo_geo.validation import (
    download_dtmb_5415,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
    extract_dtmb_5415_section_fit_data,
    load_dtmb_5415,
)

REGION_COLORS = {
    "forward_sonar_dome": "#d55e00",
    "dome_transition": "#e69f00",
    "main_hull": "#0072b2",
}


def _resample_arclength(points: np.ndarray, count: int) -> np.ndarray:
    """Resample a polyline at ``count`` normalized arc-length coordinates."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 0.0:
        return np.repeat(points[:1], count, axis=0)
    cumulative /= cumulative[-1]
    targets = np.linspace(0.0, 1.0, count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, axis]) for axis in range(points.shape[1])]
    )


def _region_of(regions, station: float):
    for index, region in enumerate(regions):
        last = index == len(regions) - 1
        if region.start <= station < region.end or (
            last and station <= region.end + 1e-12
        ):
            return region
    return regions[-1]


def _generated_section(cache, regions, station: float, count: int) -> np.ndarray:
    """Interpolate a constant-station underwater section from the cached loft."""
    region = _region_of(regions, station)
    key = f"patch_{region.name}"
    if key in cache.files:
        mesh = cache[key]
        local = (station - region.start) / (region.end - region.start)
    else:
        mesh = cache["underwater_mesh"]
        local = station
    columns = mesh.shape[1]
    position = np.clip(local, 0.0, 1.0) * (columns - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, columns - 1)
    fraction = position - lower
    section = (1.0 - fraction) * mesh[:, lower, :] + fraction * mesh[:, upper, :]
    return _resample_arclength(section, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--num-stations", type=int, default=24)
    parser.add_argument("--num-curve-points", type=int, default=17)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_surface_overlay.png"),
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

    dome_end = float(regions[1].end)
    stations = np.unique(
        np.concatenate(
            (
                np.linspace(0.012, dome_end, max(6, arguments.num_stations // 3)),
                np.linspace(dome_end + 0.02, 1.0, arguments.num_stations),
            )
        )
    )
    exact_data = extract_dtmb_5415_section_fit_data(
        reference, stations, num_curve_points=arguments.num_curve_points
    )
    recorder.stop()

    length = float(form_data.primary_parameters.length_between_perpendiculars)
    origin = float(form_data.coordinate_origin)
    x_of = origin - 0.5 * length + length * stations

    # (station, curve point, xyz); exact_data.points is (z, y) ordered.
    exact = np.stack(
        (
            np.broadcast_to(x_of[:, None], exact_data.points.shape[:2]),
            exact_data.points[:, :, 1],
            exact_data.points[:, :, 0],
        ),
        axis=2,
    )
    cache = np.load(arguments.cache)
    generated = np.stack(
        [
            _generated_section(cache, regions, float(s), arguments.num_curve_points)
            for s in stations
        ]
    )

    deviation = np.linalg.norm(generated - exact, axis=2)
    print(
        f"overlay sampled at {stations.size} stations: "
        f"RMS {1e3 * float(np.sqrt(np.mean(deviation ** 2))):.3f} mm, "
        f"max {1e3 * float(np.max(deviation)):.3f} mm"
    )

    def draw(axis, exact_set, generated_set, station_set, step: int) -> None:
        for section in exact_set:
            axis.plot(*section.T, color="0.35", linewidth=1.5, alpha=0.72)
        for section, station in zip(generated_set, station_set):
            axis.plot(
                *section.T,
                color=REGION_COLORS[_region_of(regions, float(station)).name],
                linewidth=0.95,
                linestyle="--",
            )
        for index in range(0, exact_set.shape[1], step):
            axis.plot(
                *exact_set[:, index, :].T, color="0.35", linewidth=1.0, alpha=0.60
            )
            for region in regions:
                mask = (station_set >= region.start - 1e-12) & (
                    station_set <= region.end + 1e-12
                )
                if np.count_nonzero(mask) >= 2:
                    axis.plot(
                        *generated_set[mask, index, :].T,
                        color=REGION_COLORS[region.name],
                        linewidth=0.8,
                        linestyle="--",
                        alpha=0.88,
                    )

    figure = plt.figure(figsize=(13.0, 5.8))
    perspective = figure.add_subplot(1, 2, 1, projection="3d")
    bow = figure.add_subplot(1, 2, 2, projection="3d")
    draw(perspective, exact, generated, stations, 2)
    bow_mask = stations <= dome_end + 0.06
    draw(bow, exact[bow_mask], generated[bow_mask], stations[bow_mask], 1)

    for axis in (perspective, bow):
        axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.view_init(elev=19.0, azim=-64.0)
        axis.set_proj_type("ortho")
        axis.tick_params(labelsize=8)
    # The compressed transverse axis of the full-hull view collides its own
    # tick labels at the default count.
    perspective.set_yticks(np.linspace(0.0, 0.4, 3))
    perspective.set_zticks(np.linspace(-0.35, 0.0, 4))
    perspective.set_box_aspect((4.5, 1.0, 0.65), zoom=1.25)
    perspective.set_title("Full starboard underwater hull")
    bow.set_box_aspect((1.4, 1.0, 1.0), zoom=1.15)
    bow.set_title("Forward bow and sonar-dome close-up")

    figure.legend(
        handles=(
            Line2D([0], [0], color="0.35", linewidth=2.2, label="Canonical IGES"),
            Line2D([0], [0], color=REGION_COLORS["forward_sonar_dome"],
                   linestyle="--", label="Forward bow / sonar dome"),
            Line2D([0], [0], color=REGION_COLORS["dome_transition"],
                   linestyle="--", label="Dome transition"),
            Line2D([0], [0], color=REGION_COLORS["main_hull"],
                   linestyle="--", label="Main hull"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "DTMB 5415 canonical and first-principles surfaces\n"
        f"overlay RMS = {1e3 * float(np.sqrt(np.mean(deviation ** 2))):.3f} mm, "
        f"maximum = {1e3 * float(np.max(deviation)):.3f} mm"
    )
    figure.subplots_adjust(top=0.82, bottom=0.05, left=0.01, right=0.98, wspace=0.04)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
