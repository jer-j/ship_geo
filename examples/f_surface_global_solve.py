"""Generate two continuity-coupled F-Surface patches with one Newton solve."""

from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo import (
    FSurfaceProblem,
    PatchConnection,
    PatchGraph,
    VariationalSystem,
)


def build_coupled_surfaces():
    """Return two fairness-optimized patches and their global KKT result."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    system = VariationalSystem("f_surface_demo")

    first_problem = FSurfaceProblem((4, 4), (3, 3), name="fore_patch")
    for coordinates, target in (
        ((0.0, 0.0), [0.0, 0.0, 0.0]),
        ((1.0, 0.0), [1.0, 0.0, 0.2]),
        ((0.0, 1.0), [0.0, 1.0, 0.5]),
        ((1.0, 1.0), [1.0, 1.0, 0.7]),
    ):
        first_problem.add_point_constraint(coordinates, target)
    first_assembly = first_problem.assemble(system)

    second_problem = FSurfaceProblem((4, 4), (3, 3), name="aft_patch")
    second_problem.add_point_constraint((1.0, 0.0), [2.0, 0.0, -0.2])
    second_problem.add_point_constraint((1.0, 1.0), [2.0, 1.0, 0.1])
    second_assembly = second_problem.assemble(system)

    topology = PatchGraph()
    topology.add_patch("fore", first_assembly.surface)
    topology.add_patch("aft", second_assembly.surface)
    topology.connect(PatchConnection("fore", "u1", "aft", "u0"))
    topology.add_continuity_constraints(system, samples=4)

    result = system.solve(max_iter=20)
    surfaces = (
        first_assembly.finalize(result),
        second_assembly.finalize(result),
    )
    return recorder, surfaces, result


def save_figure(path: Path) -> None:
    """Render the two solved patches and their shared edge."""
    recorder, surfaces, result = build_coupled_surfaces()
    figure = plt.figure(figsize=(8.0, 4.8))
    axes = figure.add_subplot(111, projection="3d")
    colors = ("#247ba0", "#f25f5c")
    for surface, color in zip(surfaces, colors):
        mesh = surface.mesh((21, 21)).value
        axes.plot_surface(
            mesh[:, :, 0],
            mesh[:, :, 1],
            mesh[:, :, 2],
            color=color,
            alpha=0.78,
            linewidth=0.25,
            edgecolor="white",
        )
    edge_parameters = np.linspace(0.0, 1.0, 41)
    edge = (
        surfaces[0]
        .evaluate(np.column_stack((np.ones_like(edge_parameters), edge_parameters)))
        .value
    )
    axes.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="#202020", linewidth=2.5)
    axes.set_xlabel("$x$")
    axes.set_ylabel("$y$")
    axes.set_zlabel("$z$")
    axes.set_title("Two F-Surface states, one global CSDL Newton solve")
    axes.view_init(elev=24.0, azim=-62.0)
    axes.set_box_aspect((2.0, 1.0, 0.8))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(
        "states:",
        len(result.stationarity_residuals),
        "max constraint residual:",
        float(np.max(np.abs(result.constraint_residual.value))),
        "max stationarity residual:",
        float(np.max(np.abs(result.stationarity_residual.value))),
    )
    recorder.stop()


if __name__ == "__main__":
    save_figure(Path("docs/src/images/f_surface_global_solve.png"))
