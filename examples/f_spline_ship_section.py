"""Generate a family of fairness-optimized transverse ship sections."""

from __future__ import annotations

import argparse
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo import FSplineProblem


def solve_section(target_area: float, station_index: int):
    """Solve one half-section with area and endpoint tangent constraints."""
    problem = FSplineProblem(
        num_control_points=8,
        degree=3,
        fairness_weights={2: 1.0},
        name=f"ship_section_{station_index}",
    )
    # Coordinates are (z, y): height above keel and half-breadth.
    problem.add_point_constraint(0.0, [0.0, 0.0])
    problem.add_point_constraint(1.0, [4.0, 5.0])
    problem.add_tangent_angle_constraint(0.0, np.deg2rad(80.0))
    problem.add_tangent_angle_constraint(1.0, np.deg2rad(15.0))
    problem.add_area_constraint(target_area, scale=0.1)
    return problem.solve()


def run_demo(output: Path) -> None:
    """Solve, report, and plot the demonstration sections."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    parameters = np.linspace(0.0, 1.0, 201)
    target_areas = (10.0, 13.0, 16.0)
    curves = [
        solve_section(target_area, index)
        for index, target_area in enumerate(target_areas)
    ]

    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    colors = ("#2f6f9f", "#d07a2d", "#4c956c")
    for target_area, curve, color in zip(target_areas, curves, colors):
        points = curve.evaluate(parameters).value
        control_points = curve.coefficients.value
        area = float(curve.signed_area().value[0])
        constraint_norm = float(np.max(np.abs(curve.constraint_residual.value)))
        stationarity_norm = float(np.max(np.abs(curve.stationarity_residual.value)))
        print(
            f"target={target_area:5.1f} m^2  area={area:12.9f} m^2  "
            f"|g|inf={constraint_norm:.3e}  "
            f"|dL/dP|inf={stationarity_norm:.3e}"
        )

        axis.fill_betweenx(points[:, 0], 0.0, points[:, 1], color=color, alpha=0.10)
        axis.plot(
            points[:, 1],
            points[:, 0],
            color=color,
            linewidth=2.4,
            label=rf"$A={target_area:.0f}\,\mathrm{{m}}^2$",
        )
        axis.plot(
            control_points[:, 1],
            control_points[:, 0],
            color=color,
            linewidth=0.9,
            marker="o",
            markersize=3.2,
            alpha=0.58,
        )

    axis.axvline(0.0, color="#30343b", linewidth=1.1)
    axis.axhline(4.0, color="#5b8db8", linewidth=1.0, linestyle="--")
    axis.text(0.08, 4.08, "design waterline", color="#376d99")
    axis.set_xlabel("half-breadth, $y$ [m]")
    axis.set_ylabel("height above keel, $z$ [m]")
    axis.set_title("CSDL F-Spline ship sections")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    recorder.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("f_spline_ship_section.png"),
        help="path for the generated figure",
    )
    run_demo(parser.parse_args().output)
