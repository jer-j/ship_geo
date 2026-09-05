"""3D view of the hull's lower band and the blend line it meets the hull on.

The blend line runs the whole length of the hull: the neck above the sonar
dome where there is one, and a fixed fraction of the local depth aft of it.
Everything below it is a band of its own.

The two bands meet twice over. Within a section they are built from the same
blend-point and blend-tangent expressions, so position and slope agree
exactly rather than by a fitted compromise. Along the loft, because the band
now exists at every station, it is lofted on the hull's own stations,
parameters and x coordinates -- so the shared edge is the same B-spline on
both surfaces, not two curves that happen to agree where sections fall. The
script reports the distance between them as a check.

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

    # The bulb itself, for the close views.
    bulb_x = dome[:, :, 0].min() + 0.45
    x_lo, x_hi = dome[:, :, 0].min(), bulb_x
    keep = (main[0, :, 0] >= x_lo - 0.02) & (main[0, :, 0] <= x_hi + 0.10)
    main_near = main[:, keep]
    dome_keep = (dome[0, :, 0] >= x_lo - 1.0e-9) & (dome[0, :, 0] <= x_hi)
    dome_near = dome[:, dome_keep]
    stations = np.asarray(cache["section_stations"], dtype=float)

    dome_colour, main_colour, seam_colour = "#009e73", "#0072b2", "#d55e00"
    figure = plt.figure(figsize=(16.5, 5.4))

    views = [
        ("sonar dome and the hull above it", 22, -62),
        ("looking up from below and forward", -18, -128),
        ("the band runs the whole length", 16, -68),
    ]
    for index, (title, elev, azim) in enumerate(views):
        axis = figure.add_subplot(1, 3, index + 1, projection="3d")
        upper = main if index == 2 else main_near
        lower = dome if index == 2 else dome_near
        axis.plot_surface(
            upper[:, :, 0], upper[:, :, 1], upper[:, :, 2],
            color=main_colour, alpha=0.35, linewidth=0.0, shade=True,
            rstride=2, cstride=4,
        )
        axis.plot_surface(
            lower[:, :, 0], lower[:, :, 1], lower[:, :, 2],
            color=dome_colour, alpha=0.95, linewidth=0.0, shade=True,
            rstride=1, cstride=2,
        )
        span = (dome_top[:, 0] >= lower[:, :, 0].min() - 1.0e-9) & (
            dome_top[:, 0] <= lower[:, :, 0].max() + 1.0e-9
        )
        axis.plot(dome_top[span, 0], dome_top[span, 1], dome_top[span, 2],
                  color=seam_colour, linewidth=2.4, label="lower band, upper edge")
        band = main_low[span]
        axis.plot(band[:, 0], band[:, 1], band[:, 2],
                  color="black", linewidth=1.2, linestyle="--",
                  label="upper band, lower edge")
        if index == 2:
            length = float(cache["length"])
            origin = float(cache["coordinate_origin"])
            for station in stations:
                x = origin - 0.5 * length + length * float(station)
                j = int(np.argmin(np.abs(dome_top[:, 0] - x)))
                axis.scatter(*dome_top[j], color="black", s=14, zorder=6)
        axis.view_init(elev=elev, azim=azim)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("x [m]"); axis.set_ylabel("y [m]"); axis.set_zlabel("z [m]")
        pts = np.concatenate([lower.reshape(-1, 3), upper.reshape(-1, 3)])
        if index == 2:
            # Equal aspect over the whole 5.7 m hull flattens it to a line, so
            # the full-length view is stretched across the beam and depth.
            axis.set_box_aspect((3.4, 1.0, 0.9))
            for setter, column in (
                (axis.set_xlim, 0), (axis.set_ylim, 1), (axis.set_zlim, 2)
            ):
                setter(pts[:, column].min(), pts[:, column].max())
        else:
            _equal_aspect(axis, pts)
        if index == 0:
            axis.legend(loc="upper left", fontsize=8, frameon=False)

    figure.suptitle(
        "DTMB 5415: everything below the blend line is a band of its own, lofted on "
        "the hull's own stations.\nBlack dots are the generating sections; the two "
        "edges are now one B-spline, not two that meet at them.",
        fontsize=11,
    )
    figure.tight_layout()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=170, bbox_inches="tight")
    print(f"wrote {arguments.output}")

    gap = np.linalg.norm(
        dome_top[:, None, :] - main_low[None, :, :], axis=2
    ).min(axis=1)
    print(f"seam gap: max {1.0e3 * gap.max():.3f} mm, mean {1.0e3 * gap.mean():.3f} mm")


if __name__ == "__main__":
    main()
