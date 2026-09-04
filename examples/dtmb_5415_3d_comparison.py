"""3D comparisons of the reconstructed DTMB 5415 hull against the exact IGES.

Loads the mesh cache written by ``dtmb_5415_deck_and_fullness.py --cache``
(no Newton solve needed here) and overlays it on the exact reference surface:
one full-hull view plus several zoomed close-ups (sonar dome, midship, and
stern/transom) so local shape agreement is visible, not just the aggregate.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from lsdo_geo.validation import download_dtmb_5415, load_dtmb_5415
from lsdo_geo.validation.dtmb_5415 import (
    DTMB5415Region,
    dtmb_5415_longitudinal_regions,
    extract_dtmb_5415_form_data,
)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _exact_region_mesh(patch, function, resolution=(71, 71)):
    _, values = patch.sample_grid(function, resolution)
    return np.asarray(values.value, dtype=float).reshape(resolution + (3,))


def _draw_exact(axis, mesh, color="0.55", alpha=0.55, stride=1):
    axis.plot_surface(
        mesh[:, :, 0],
        mesh[:, :, 1],
        mesh[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        rstride=stride,
        cstride=stride,
    )
    # Port side too, since the reference is a starboard-only half-hull.
    axis.plot_surface(
        mesh[:, :, 0],
        -mesh[:, :, 1],
        mesh[:, :, 2],
        color=color,
        alpha=alpha,
        linewidth=0.0,
        antialiased=True,
        shade=True,
        rstride=stride,
        cstride=stride,
    )


def _draw_generated(axis, mesh, color, stride_u=4, stride_v=8, label=None):
    axis.plot_wireframe(
        mesh[:, :, 0],
        mesh[:, :, 1],
        mesh[:, :, 2],
        color=color,
        linewidth=0.9,
        rstride=stride_u,
        cstride=stride_v,
        alpha=0.95,
    )
    axis.plot_wireframe(
        mesh[:, :, 0],
        -mesh[:, :, 1],
        mesh[:, :, 2],
        color=color,
        linewidth=0.9,
        rstride=stride_u,
        cstride=stride_v,
        alpha=0.95,
    )
    if label is not None:
        axis.plot([], [], color=color, label=label)


def _set_equal_aspect(axis, x, y, z):
    ranges = [np.ptp(x), np.ptp(y), np.ptp(z)]
    max_range = max(ranges) / 2.0
    mid_x, mid_y, mid_z = np.mean(x), np.mean(y), np.mean(z)
    axis.set_xlim(mid_x - max_range, mid_x + max_range)
    axis.set_ylim(mid_y - max_range, mid_y + max_range)
    axis.set_zlim(mid_z - max_range, mid_z + max_range)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_3d_comparison.png"),
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
    length = form_data.primary_parameters.length_between_perpendiculars
    origin = form_data.coordinate_origin

    def x_of(v: float) -> float:
        return origin - 0.5 * length + length * float(v)

    cache = np.load(arguments.cache)
    underwater = cache["underwater_mesh"]
    freeboard = cache["freeboard_mesh"]

    functions = reference.build_functions()
    exact_meshes = {
        region: _exact_region_mesh(reference.patches[region], functions[region])
        for region in DTMB5415Region
    }

    exact_color = "#8c8c8c"
    generated_color_under = "#0072b2"
    generated_color_deck = "#d55e00"

    # ---- Figure 1: full-hull overview -----------------------------------
    figure = plt.figure(figsize=(11.0, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    for region, mesh in exact_meshes.items():
        _draw_exact(axis, mesh, stride=2)
    _draw_generated(
        axis, underwater, generated_color_under, stride_u=4, stride_v=8,
        label="generated: underwater hull",
    )
    _draw_generated(
        axis, freeboard, generated_color_deck, stride_u=4, stride_v=8,
        label="generated: freeboard (new)",
    )
    all_x = np.concatenate([m[:, :, 0].ravel() for m in exact_meshes.values()])
    all_y = np.concatenate(
        [m[:, :, 1].ravel() for m in exact_meshes.values()]
        + [-m[:, :, 1].ravel() for m in exact_meshes.values()]
    )
    all_z = np.concatenate([m[:, :, 2].ravel() for m in exact_meshes.values()])
    _set_equal_aspect(axis, all_x, all_y, all_z)
    axis.view_init(elev=18, azim=-60)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(
        "DTMB 5415: exact IGES (gray) vs. F-Spline reconstruction\n"
        "underwater hull (blue) + new freeboard surface (orange) -- single global solve"
    )
    axis.legend(loc="upper left", frameon=False)
    _save(figure, arguments.output)

    # ---- Figure 2: close-ups ---------------------------------------------
    closeups = [
        ("sonar dome & bow", x_of(0.0) - 0.05, x_of(regions[1].end) + 0.05, -55, 12),
        ("midship", x_of(0.42), x_of(0.58), -55, 12),
        ("stern & transom", x_of(0.90), x_of(1.0) + 0.05, -55, 12),
    ]
    figure = plt.figure(figsize=(16.0, 5.4))
    for panel_index, (title, x_lo, x_hi, azim, elev) in enumerate(closeups):
        axis = figure.add_subplot(1, 3, panel_index + 1, projection="3d")
        for region, mesh in exact_meshes.items():
            mask_rows = np.any(
                (mesh[:, :, 0] >= x_lo) & (mesh[:, :, 0] <= x_hi), axis=1
            )
            if not np.any(mask_rows):
                continue
            _draw_exact(axis, mesh[mask_rows], stride=1)
        for mesh, color in (
            (underwater, generated_color_under),
            (freeboard, generated_color_deck),
        ):
            mask_rows = np.any((mesh[:, :, 0] >= x_lo) & (mesh[:, :, 0] <= x_hi), axis=1)
            if not np.any(mask_rows):
                continue
            _draw_generated(
                axis, mesh[mask_rows], color, stride_u=1, stride_v=3
            )
        window_x = np.concatenate(
            [
                mesh[np.any((mesh[:, :, 0] >= x_lo) & (mesh[:, :, 0] <= x_hi), axis=1)][:, :, 0].ravel()
                for mesh in exact_meshes.values()
            ]
        )
        window_y = np.concatenate(
            [
                mesh[np.any((mesh[:, :, 0] >= x_lo) & (mesh[:, :, 0] <= x_hi), axis=1)][:, :, 1].ravel()
                for mesh in exact_meshes.values()
            ]
        )
        window_z = np.concatenate(
            [
                mesh[np.any((mesh[:, :, 0] >= x_lo) & (mesh[:, :, 0] <= x_hi), axis=1)][:, :, 2].ravel()
                for mesh in exact_meshes.values()
            ]
        )
        window_y = np.concatenate((window_y, -window_y))
        _set_equal_aspect(axis, window_x, window_y, window_z)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("x [m]", fontsize=8)
        axis.set_ylabel("y [m]", fontsize=8)
        axis.set_zlabel("z [m]", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(
        "Close-ups: exact IGES (gray, shaded) vs. generated wireframe "
        "(blue = underwater, orange = freeboard)"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save(figure, arguments.output.with_name(arguments.output.stem + "_closeups.png"))
    recorder.stop()


if __name__ == "__main__":
    main()
