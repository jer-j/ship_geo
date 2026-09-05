"""3D view of the reconstructed sonar dome, and the seam at its patch boundary.

The dome is a partial-length patch carried below a blend line. Within a
section it meets the main band by construction: both bands are built from the
same blend-point and blend-tangent expressions, so position and slope agree
exactly rather than by a fitted compromise.

Along the loft they are two separate surfaces. The main hull interpolates
every generating section; the dome interpolates only those that carry a band,
with its own knot vector and its own local parameterization. Nothing ties
them between sections, so the panels here draw the seam and report how far
apart the two edges actually run.

Surfaces are evaluated from the cached control nets rather than from the
sampled meshes, so the pictures show the geometry itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lsdo_function_spaces as lfs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _surface(cache, name):
    degree = tuple(int(v) for v in cache[f"space_{name}_degrees"])
    shape = tuple(int(v) for v in cache[f"space_{name}_shape"])
    knots = np.asarray(cache[f"space_{name}_knots"], dtype=float)
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2, degree=degree,
        coefficients_shape=shape, knots=knots,
    )
    return space, np.asarray(cache[f"coefficients_{name}"], dtype=float).reshape(-1, 3)


def _grid(cache, name, resolution=(60, 120)):
    space, coefficients = _surface(cache, name)
    u = np.linspace(0.0, 1.0, resolution[0])
    t = np.linspace(0.0, 1.0, resolution[1])
    uu, tt = np.meshgrid(u, t, indexing="ij")
    uv = np.stack([uu.ravel(), tt.ravel()], axis=1)
    points = np.asarray(space.compute_basis_matrix(uv) @ coefficients)
    return points.reshape(resolution + (3,))


def _edge(cache, name, girth, count=3000):
    space, coefficients = _surface(cache, name)
    t = np.linspace(0.0, 1.0, count)
    uv = np.stack([np.full(count, float(girth)), t], axis=1)
    return np.asarray(space.compute_basis_matrix(uv) @ coefficients)


def _equal_aspect(axis, points):
    spans = points.max(axis=0) - points.min(axis=0)
    centre = 0.5 * (points.max(axis=0) + points.min(axis=0))
    radius = 0.5 * float(spans.max())
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("docs/src/images/dtmb_5415_dome_3d.png"),
    )
    arguments = parser.parse_args()
    cache = np.load(arguments.cache)

    dome = _grid(cache, "dome", (50, 110))
    main = _grid(cache, "underwater", (60, 260))
    dome_top = _edge(cache, "dome", 1.0)
    main_low = _edge(cache, "underwater", 0.0)

    x_lo, x_hi = dome[:, :, 0].min(), dome[:, :, 0].max()
    # The stretch of main hull alongside the dome, plus a little aft of it.
    keep = (main[0, :, 0] >= x_lo - 0.02) & (main[0, :, 0] <= x_hi + 0.35)
    main_near = main[:, keep]

    dome_colour, main_colour, seam_colour = "#009e73", "#0072b2", "#d55e00"
    figure = plt.figure(figsize=(16.5, 5.4))

    views = [
        ("sonar dome and the hull above it", 22, -62),
        ("looking up from below and forward", -18, -128),
        ("the blend seam along the patch boundary", 8, -95),
    ]
    for index, (title, elev, azim) in enumerate(views):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        if index < 2:
            axis.plot_surface(
                main_near[:, :, 0], main_near[:, :, 1], main_near[:, :, 2],
                color=main_colour, alpha=0.35, linewidth=0.0, shade=True,
                rstride=2, cstride=4,
            )
        axis.plot_surface(
            dome[:, :, 0], dome[:, :, 1], dome[:, :, 2],
            color=dome_colour, alpha=0.95, linewidth=0.0, shade=True,
            rstride=1, cstride=2,
        )
        band = main_low[
            (main_low[:, 0] >= x_lo - 1.0e-9) & (main_low[:, 0] <= x_hi + 1.0e-9)
        ]
        axis.plot(dome_top[:, 0], dome_top[:, 1], dome_top[:, 2],
                  color=seam_colour, linewidth=2.2, label="dome band, upper edge")
        axis.plot(band[:, 0], band[:, 1], band[:, 2],
                  color="black", linewidth=1.2, linestyle="--",
                  label="main band, lower edge")
        if index == 2:
            for station in (0.012, 0.025, 0.038, 0.051, 0.063):
                length = float(cache["length"])
                origin = float(cache["coordinate_origin"])
                x = origin - 0.5 * length + length * station
                j = int(np.argmin(np.abs(dome_top[:, 0] - x)))
                axis.scatter(*dome_top[j], color="black", s=16, zorder=6)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]"); axis.set_zlabel("z [m]")
        pts = dome.reshape(-1, 3) if index == 2 else np.concatenate(
            [dome.reshape(-1, 3), main_near.reshape(-1, 3)]
        )
        _equal_aspect(axis, pts)
        if index == 0:
            axis.legend(loc="upper left", fontsize=8, frameon=False)

    figure.suptitle(
        "DTMB 5415 sonar dome as its own F-Spline patch. Black dots mark the five "
        "generating sections,\nthe only places the dome patch and the hull patch are "
        "tied together.",
        fontsize=11,
    )
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170, bbox_inches="tight")
    print(f"wrote {arguments.output}")

    band = main_low[
        (main_low[:, 0] >= x_lo - 1.0e-9) & (main_low[:, 0] <= x_hi + 1.0e-9)
    ]
    gap = np.linalg.norm(
        dome_top[:, None, :] - band[None, :, :], axis=2
    ).min(axis=1)
    print(f"seam gap: max {1.0e3 * gap.max():.3f} mm, mean {1.0e3 * gap.mean():.3f} mm")


if __name__ == "__main__":
    main()
