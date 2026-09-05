"""Validate and visualize a fully spatial F-Spline curve with JAX."""

from __future__ import annotations

import argparse
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np
from csdl_alpha.experimental import JaxSimulator

from lsdo_geo import FSplineProblem


def run_demo(output: Path) -> None:
    """Solve the spatial curve and plot it with all coordinate projections."""
    recorder = csdl.Recorder(inline=False)
    recorder.start()

    problem = FSplineProblem(
        num_control_points=4,
        degree=3,
        physical_dimension=3,
        fairness_weights={2: 1.0},
        name="spatial_cubic_demo",
    )
    end_point = csdl.Variable(value=np.array([1.0, 1.0, 1.0]), name="end_point")
    problem.add_point_constraint(0.0, [0.0, 0.0, 0.0])
    problem.add_point_constraint(1.0, end_point)
    problem.add_derivative_constraint(0.0, 1, [1.0, 0.0, 0.0])
    problem.add_derivative_constraint(1.0, 1, [1.0, 2.0, 3.0])
    curve = problem.solve()

    parameters = np.linspace(0.0, 1.0, 201)
    sampled_curve = curve.evaluate(parameters)
    start_tangent = curve.evaluate(0.0, derivative_order=1)
    end_tangent = curve.evaluate(1.0, derivative_order=1)
    midpoint_curvature = curve.curvature_magnitude(0.5)

    simulator = JaxSimulator(
        recorder,
        additional_inputs=[end_point],
        additional_outputs=[
            sampled_curve,
            curve.coefficients,
            curve.constraint_residual,
            start_tangent,
            end_tangent,
            midpoint_curvature,
        ],
        gpu=False,
        f64=True,
    )
    simulator.run()

    points = np.asarray(sampled_curve.value)
    control_points = np.asarray(curve.coefficients.value)
    exact = np.column_stack((parameters, parameters**2, parameters**3))
    max_error = float(np.max(np.linalg.norm(points - exact, axis=1)))
    residual = float(np.max(np.abs(curve.constraint_residual.value)))

    velocity = np.array([1.0, 1.0, 0.75])
    acceleration = np.array([0.0, 2.0, 3.0])
    exact_curvature = np.linalg.norm(np.cross(velocity, acceleration)) / (
        np.linalg.norm(velocity) ** 3
    )
    curvature_error = abs(float(midpoint_curvature.value[0]) - exact_curvature)

    print(f"maximum curve error: {max_error:.3e}")
    print(f"maximum constraint residual: {residual:.3e}")
    print(f"midpoint curvature error: {curvature_error:.3e}")

    figure = plt.figure(figsize=(11.0, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    spatial_axis = figure.add_subplot(grid[0, 0], projection="3d")
    xy_axis = figure.add_subplot(grid[0, 1])
    xz_axis = figure.add_subplot(grid[1, 0])
    yz_axis = figure.add_subplot(grid[1, 1])

    spatial_axis.plot(
        points[:, 0], points[:, 1], points[:, 2], color="#176b87", linewidth=2.8
    )
    spatial_axis.plot(
        control_points[:, 0],
        control_points[:, 1],
        control_points[:, 2],
        color="#a84a32",
        marker="o",
        markersize=5,
        linewidth=1.2,
        label="control polygon",
    )
    for point, tangent, color in (
        (points[0], np.asarray(start_tangent.value).reshape(3), "#5c8d3e"),
        (points[-1], np.asarray(end_tangent.value).reshape(3), "#5c8d3e"),
    ):
        direction = tangent / np.linalg.norm(tangent)
        spatial_axis.quiver(
            point[0],
            point[1],
            point[2],
            direction[0],
            direction[1],
            direction[2],
            length=0.28,
            color=color,
            linewidth=1.8,
        )
    spatial_axis.set_xlabel("$x$")
    spatial_axis.set_ylabel("$y$")
    spatial_axis.set_zlabel("$z$")
    spatial_axis.set_title("Spatial curve and control polygon")
    spatial_axis.view_init(elev=25, azim=-58)
    spatial_axis.set_box_aspect((1.0, 1.0, 1.0))

    projections = (
        (xy_axis, 0, 1, "$x$", "$y$", "$x$-$y$ projection"),
        (xz_axis, 0, 2, "$x$", "$z$", "$x$-$z$ projection"),
        (yz_axis, 1, 2, "$y$", "$z$", "$y$-$z$ projection"),
    )
    for axis, first, second, xlabel, ylabel, title in projections:
        axis.plot(
            exact[:, first],
            exact[:, second],
            color="#747b82",
            linewidth=5.0,
            alpha=0.30,
            label="analytic reference",
        )
        axis.plot(
            points[:, first],
            points[:, second],
            color="#176b87",
            linewidth=2.2,
            label="F-Spline",
        )
        axis.plot(
            control_points[:, first],
            control_points[:, second],
            color="#a84a32",
            marker="o",
            markersize=3.5,
            linewidth=0.9,
            label="control polygon",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)

    xy_axis.legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Arbitrary 3D F-Spline: $\\mathbf{r}(t)=(t,t^2,t^3)$\n"
        rf"JAX solve, max point error ${max_error:.1e}$, "
        rf"max constraint residual ${residual:.1e}$"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    recorder.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("f_spline_spatial_curve.png"),
        help="path for the generated figure",
    )
    run_demo(parser.parse_args().output)
