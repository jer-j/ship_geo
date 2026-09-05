"""Generate a first-principles F-Spline section family and Wigley-like hull."""

from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo import SectionLoftProblem


def main(show: bool = True) -> None:
    """Solve one coupled geometry system and render the resulting hull."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    length = csdl.Variable(value=100.0, name="length")
    beam = csdl.Variable(value=10.0, name="beam")
    draft = csdl.Variable(value=5.0, name="draft")
    stations = np.array([0.2, 0.5, 0.8])
    longitudinal_coordinate = 2.0 * stations - 1.0
    breadth_shape = 1.0 - longitudinal_coordinate**2
    half_breadths = 0.5 * breadth_shape * beam
    half_areas = (2.0 / 3.0) * half_breadths * draft
    baseline_keel_angles = np.arctan2(
        10.0 * breadth_shape,
        5.0,
    )

    problem = SectionLoftProblem(
        length=length,
        station_parameters=stations,
        drafts=[draft, draft, draft],
        half_breadths=half_breadths,
        half_areas=half_areas,
        keel_tangent_angles=baseline_keel_angles,
        waterline_tangent_angles=np.zeros(3),
        longitudinal_fairness_weight=1.0e-6,
        name="wigley_like_fspline_hull",
    )
    hull = problem.solve()
    displacement_sensitivity = csdl.derivative(hull.hydrostatics.displacement, beam)

    mesh = hull.surface.mesh((41, 81)).value
    figure = plt.figure(figsize=(13.0, 8.5), constrained_layout=True)
    surface_axis = figure.add_subplot(2, 2, 1, projection="3d")
    surface_axis.plot_surface(
        mesh[:, :, 0],
        mesh[:, :, 1],
        mesh[:, :, 2],
        color="#4c78a8",
        alpha=0.75,
        linewidth=0.2,
        edgecolor="#24415c",
    )
    surface_axis.plot_surface(
        mesh[:, :, 0],
        -mesh[:, :, 1],
        mesh[:, :, 2],
        color="#72a8d4",
        alpha=0.55,
        linewidth=0.2,
        edgecolor="#24415c",
    )
    section_parameter = np.linspace(0.0, 1.0, 161)
    plotted_sections: list[np.ndarray] = []
    for station in stations:
        coordinates = np.column_stack(
            (section_parameter, np.full(section_parameter.shape, station))
        )
        section = hull.surface.evaluate(coordinates).value
        plotted_sections.append(section)
        surface_axis.plot(section[:, 0], section[:, 1], section[:, 2], color="#f58518")
        surface_axis.plot(section[:, 0], -section[:, 1], section[:, 2], color="#f58518")

    displacement = float(hull.hydrostatics.displacement.value[0])
    expected = 4.0 / 9.0 * 100.0 * 10.0 * 5.0
    residual = np.max(np.abs(hull.variational_result.constraint_residual.value))
    title = (
        "One-solve F-Spline hull\n"
        f"$\\nabla={displacement:.2f}\\,\\mathrm{{m}}^3$ "
        f"(Wigley reference {expected:.2f}, error "
        f"{100.0 * (displacement / expected - 1.0):+.3f}%), "
        f"$\\max|R_c|={residual:.1e}$"
    )
    figure.suptitle(title, fontsize=14)
    surface_axis.set_title("Hull surface and generating sections")
    surface_axis.set_xlabel("$x$ [m]")
    surface_axis.set_ylabel("$y$ [m]")
    surface_axis.set_zlabel("$z$ [m]")
    surface_axis.set_box_aspect((5.0, 1.2, 1.0))
    surface_axis.view_init(elev=24.0, azim=-58.0)

    plan_axis = figure.add_subplot(2, 2, 2)
    waterline = mesh[-1]
    plan_axis.fill_between(
        waterline[:, 0],
        -waterline[:, 1],
        waterline[:, 1],
        color="#72a8d4",
        alpha=0.65,
    )
    plan_axis.plot(waterline[:, 0], waterline[:, 1], color="#24415c")
    plan_axis.plot(waterline[:, 0], -waterline[:, 1], color="#24415c")
    plan_axis.set_title("Design waterline")
    plan_axis.set_xlabel("$x$ [m]")
    plan_axis.set_ylabel("$y$ [m]")
    plan_axis.set_aspect("equal", adjustable="box")
    plan_axis.grid(alpha=0.25)

    body_axis = figure.add_subplot(2, 2, 3)
    for section, station in zip(plotted_sections, stations):
        color = plt.cm.viridis(float(station))
        body_axis.plot(section[:, 1], section[:, 2], color=color)
        body_axis.plot(-section[:, 1], section[:, 2], color=color)
    body_axis.set_title("Body plan at F-Spline stations")
    body_axis.set_xlabel("$y$ [m]")
    body_axis.set_ylabel("$z$ [m]")
    body_axis.set_aspect("equal", adjustable="box")
    body_axis.grid(alpha=0.25)

    area_axis = figure.add_subplot(2, 2, 4)
    area_parameters = hull.hydrostatics.section_parameters
    area_x = 100.0 * (area_parameters - 0.5)
    area_axis.plot(
        area_x,
        hull.hydrostatics.sectional_areas.value,
        color="#e45756",
        linewidth=2.0,
        label="F-Spline loft",
    )
    area_axis.plot(
        area_x,
        2.0 / 3.0 * 10.0 * 5.0 * (1.0 - (2.0 * area_parameters - 1.0) ** 2),
        "--",
        color="#444444",
        label="Wigley reference",
    )
    area_axis.set_title("Sectional-area curve")
    area_axis.set_xlabel("$x$ [m]")
    area_axis.set_ylabel("$A(x)$ [$\\mathrm{m}^2$]")
    area_axis.grid(alpha=0.25)
    area_axis.legend()

    output = Path(__file__).parents[1] / "docs" / "src" / "images"
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "first_principles_fspline_hull.png", dpi=180)
    print(f"displacement: {displacement:.12g} m^3")
    print(f"d(displacement)/d(beam): {displacement_sensitivity.value[0, 0]:.12g} m^2")
    print(f"maximum constraint residual: {residual:.3e}")
    print(hull.validity.report())
    if show:
        plt.show()
    recorder.stop()


if __name__ == "__main__":
    main()
