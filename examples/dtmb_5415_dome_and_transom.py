"""Bow, sonar-dome, and transom close-ups of the DTMB 5415 reconstruction.

Reads the cache written by ``dtmb_5415_accurate_reconstruction.py --cache``
and draws the two places the surface topology changed:

* the **sonar-dome band**, now a separate F-Spline carried below a blend
  line and lofted as its own partial-length patch. The main band starts on
  the blend point with the blend tangent, so the two meet with position and
  slope agreeing by construction rather than by a fitted compromise. The
  panels draw the bands in different colours so the junction is visible.
* the **raked transom**. The exact transom edge is not a plane of constant
  ``x``: it rakes about 29 mm between the keel and the deck. The generating
  station at ``v = 1`` now carries a per-control-point ``x`` offset, so the
  aft edge follows the rake instead of standing square.

No Newton solve happens here; everything is read back from the cache.
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
    extract_dtmb_5415_form_data,
    extract_dtmb_5415_section_fit_data,
    load_dtmb_5415,
)

DOME_STATIONS = (0.012, 0.026, 0.042, 0.058, 0.070, 0.090)


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _section_at(mesh: np.ndarray, x_target: float) -> np.ndarray | None:
    """Linearly interpolate a transverse section out of a lofted mesh.

    Axis 0 of the cached meshes runs longitudinally and axis 1 runs around
    the girth, so a section is a blend of two neighbouring rows.
    """
    if mesh is None:
        return None
    x_rows = mesh[:, :, 0].mean(axis=1)
    order = np.argsort(x_rows)
    x_rows, mesh = x_rows[order], mesh[order]
    if x_target < x_rows[0] - 1.0e-9 or x_target > x_rows[-1] + 1.0e-9:
        return None
    upper = int(np.searchsorted(x_rows, x_target).clip(1, len(x_rows) - 1))
    lower = upper - 1
    span = x_rows[upper] - x_rows[lower]
    weight = 0.0 if span <= 0.0 else (x_target - x_rows[lower]) / span
    return (1.0 - weight) * mesh[lower] + weight * mesh[upper]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output-dome",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_dome_band.png"),
    )
    parser.add_argument(
        "--output-transom",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_transom_rake.png"),
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
    length = float(form_data.primary_parameters.length_between_perpendiculars)
    origin = float(form_data.coordinate_origin)
    exact = extract_dtmb_5415_section_fit_data(reference, DOME_STATIONS)
    recorder.stop()

    def x_of(v: float) -> float:
        return origin - 0.5 * length + length * float(v)

    cache = np.load(arguments.cache)
    underwater = cache["underwater_mesh"]
    freeboard = cache["freeboard_mesh"] if "freeboard_mesh" in cache.files else None
    dome = cache["dome_mesh"] if "dome_mesh" in cache.files else None
    print("dome patch in cache:", dome is not None)

    exact_color, main_color, dome_color = "0.35", "#0072b2", "#009e73"

    # ---- Figure 1: dome band sections ------------------------------------
    figure, axes = plt.subplots(
        1, len(DOME_STATIONS), figsize=(3.0 * len(DOME_STATIONS), 3.9), sharey=True
    )
    for axis, station in zip(np.atleast_1d(axes), DOME_STATIONS):
        index = list(DOME_STATIONS).index(station)
        points = exact.points[index]  # columns are (z, y)
        axis.plot(
            points[:, 1], points[:, 0], "-", color=exact_color, linewidth=3.2,
            alpha=0.55, label="exact IGES", solid_capstyle="round",
        )
        x_target = x_of(station)
        main_section = _section_at(underwater, x_target)
        if main_section is not None:
            axis.plot(
                main_section[:, 1], main_section[:, 2], "-", color=main_color,
                linewidth=1.5, label="main band",
            )
        dome_section = _section_at(dome, x_target)
        if dome_section is not None:
            axis.plot(
                dome_section[:, 1], dome_section[:, 2], "-", color=dome_color,
                linewidth=1.5, label="sonar-dome band",
            )
            # The blend point is where the two bands meet.
            axis.plot(
                dome_section[-1, 1], dome_section[-1, 2], "o", color="#d55e00",
                markersize=4.5, zorder=5, label="blend point",
            )
        axis.set_title(f"v = {station:.3f}", fontsize=10)
        axis.set_xlabel("y [m]")
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.set_aspect("equal", adjustable="datalim")
    np.atleast_1d(axes)[0].set_ylabel("z [m]")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if len(handles) < 4:
        for candidate in np.atleast_1d(axes):
            handles, labels = candidate.get_legend_handles_labels()
            if len(handles) >= 4:
                break
    figure.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.06),
    )
    figure.suptitle(
        "DTMB 5415 sonar dome: the bulb is a separate F-Spline band below a "
        "blend line,\njoined to the main band with matching position and tangent",
        fontsize=11,
    )
    figure.tight_layout()
    _save(figure, arguments.output_dome)

    # ---- Figure 2: transom rake ------------------------------------------
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))

    aft_edge = underwater[np.argmax(underwater[:, :, 0].mean(axis=1))]
    axis = axes[0]
    axis.plot(
        aft_edge[:, 2], aft_edge[:, 0], "-o", color=main_color, markersize=2.5,
        linewidth=1.4, label="generated: underwater transom edge",
    )
    if freeboard is not None:
        deck_edge = freeboard[np.argmax(freeboard[:, :, 0].mean(axis=1))]
        axis.plot(
            deck_edge[:, 2], deck_edge[:, 0], "-o", color="#d55e00",
            markersize=2.5, linewidth=1.4,
            label="generated: freeboard transom edge",
        )
    # The exact aft edge, taken as the largest x on the reference surface at
    # each height.
    functions = reference.build_functions()
    exact_points = []
    for region, patch in reference.patches.items():
        _, values = patch.sample_grid(functions[region], (81, 81))
        exact_points.append(np.asarray(values.value, dtype=float).reshape(-1, 3))
    exact_points = np.concatenate(exact_points)
    aft = exact_points[exact_points[:, 0] > exact_points[:, 0].max() - 0.06]
    edges = []
    for lo, hi in zip(
        np.linspace(aft[:, 2].min(), aft[:, 2].max(), 41)[:-1],
        np.linspace(aft[:, 2].min(), aft[:, 2].max(), 41)[1:],
    ):
        band = aft[(aft[:, 2] >= lo) & (aft[:, 2] < hi)]
        if band.size:
            edges.append((0.5 * (lo + hi), band[:, 0].max()))
    edges = np.asarray(edges)
    axis.plot(
        edges[:, 0], edges[:, 1], "-", color=exact_color, linewidth=3.0,
        alpha=0.55, label="exact IGES aft edge",
    )
    axis.set_xlabel("z [m]")
    axis.set_ylabel("x [m]")
    axis.set_title(
        "Transom rake: aft edge x against height\n"
        "(a square transom would be a vertical line)"
    )
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1]
    axis.plot(
        aft_edge[:, 1], aft_edge[:, 2], "-o", color=main_color, markersize=2.5,
        linewidth=1.4, label="generated: underwater",
    )
    if freeboard is not None:
        axis.plot(
            deck_edge[:, 1], deck_edge[:, 2], "-o", color="#d55e00",
            markersize=2.5, linewidth=1.4, label="generated: freeboard",
        )
    axis.plot(
        aft[:, 1], aft[:, 2], ".", color=exact_color, markersize=1.6, alpha=0.35,
        label="exact IGES (aft 60 mm)",
    )
    axis.set_xlabel("y [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title("Transom section shape")
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    _save(figure, arguments.output_transom)

    # ---- Figure 3: forward waterlines -----------------------------------
    # Section RMS can look acceptable while the longitudinal lines still read
    # wrong, which is what a plan view of the waterlines shows directly.
    levels = (-0.02, -0.06, -0.11, -0.17, -0.23)
    forward_stations = np.linspace(0.008, 0.34, 60)
    recorder2 = csdl.Recorder(inline=True)
    recorder2.start()
    exact_forward = extract_dtmb_5415_section_fit_data(reference, forward_stations)
    recorder2.stop()

    def waterline_from_sections(points_by_station, x_values, level):
        """Half-breadth at a given height, per station."""
        xs, ys = [], []
        for x_value, points in zip(x_values, points_by_station):
            z, y = points[:, 0], points[:, 1]
            order = np.argsort(z)
            z, y = z[order], y[order]
            if level < z[0] or level > z[-1]:
                continue
            xs.append(x_value)
            ys.append(float(np.interp(level, z, y)))
        return np.asarray(xs), np.asarray(ys)

    def waterline_from_mesh(meshes, level, x_lo, x_hi):
        xs, ys = [], []
        rows = []
        for mesh in meshes:
            if mesh is None:
                continue
            rows.extend(mesh[i] for i in range(mesh.shape[0]))
        rows.sort(key=lambda row: row[:, 0].mean())
        for row in rows:
            x_value = float(row[:, 0].mean())
            if not x_lo <= x_value <= x_hi:
                continue
            z, y = row[:, 2], row[:, 1]
            order = np.argsort(z)
            z, y = z[order], y[order]
            if level < z[0] or level > z[-1]:
                continue
            xs.append(x_value)
            ys.append(float(np.interp(level, z, y)))
        return np.asarray(xs), np.asarray(ys)

    figure, axis = plt.subplots(figsize=(12.0, 5.2))
    x_lo, x_hi = x_of(0.008), x_of(0.34)
    for index, level in enumerate(levels):
        ex, ey = waterline_from_sections(
            exact_forward.points, [x_of(v) for v in forward_stations], level
        )
        gx, gy = waterline_from_mesh([underwater, dome], level, x_lo, x_hi)
        axis.plot(
            ex, ey, "-", color=exact_color, linewidth=3.0, alpha=0.5,
            label="exact IGES" if index == 0 else None,
        )
        axis.plot(
            gx, gy, "-", color=main_color, linewidth=1.4,
            label="reconstruction" if index == 0 else None,
        )
        if ex.size:
            axis.annotate(
                f"z = {level:.2f} m", (ex[-1], ey[-1]), fontsize=8,
                xytext=(4, 0), textcoords="offset points", color="0.3",
            )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]  (half-breadth)")
    axis.set_title(
        "Forward waterlines in plan view: exact IGES (grey) vs. reconstruction"
        " (blue)\nthe two lowest levels pass through the sonar dome"
    )
    axis.grid(alpha=0.25, linewidth=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    _save(
        figure,
        arguments.output_transom.with_name("dtmb_5415_forward_waterlines.png"),
    )

    span = float(aft_edge[:, 0].max() - aft_edge[:, 0].min())
    exact_span = float(edges[:, 1].max() - edges[:, 1].min())
    print(f"generated underwater rake extent: {1.0e3 * span:.1f} mm")
    print(f"exact aft-edge rake extent:       {1.0e3 * exact_span:.1f} mm")


if __name__ == "__main__":
    main()
