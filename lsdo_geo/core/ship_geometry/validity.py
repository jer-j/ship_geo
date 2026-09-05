"""Sampled differentiable and diagnostic hull-validity checks."""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np

from .surfaces import TensorProductSurface


@dataclass
class SurfaceValidity:
    """Raw CSDL arrays suitable for optimizer constraints and diagnostics."""

    half_breadths: csdl.Variable
    jacobian_magnitudes: csdl.Variable
    outward_normal_y: csdl.Variable
    longitudinal_x_derivatives: csdl.Variable
    section_z_derivatives: csdl.Variable
    centerplane_offsets: csdl.Variable
    waterline_heights: csdl.Variable
    resolution: tuple[int, int]

    def report(self) -> dict[str, float]:
        """Return current numerical extrema without altering the CSDL graph."""
        return {
            "minimum_half_breadth": float(np.min(self.half_breadths.value)),
            "minimum_surface_jacobian": float(np.min(self.jacobian_magnitudes.value)),
            "minimum_outward_normal_y": float(np.min(self.outward_normal_y.value)),
            "minimum_dx_dv": float(np.min(self.longitudinal_x_derivatives.value)),
            "minimum_dz_du": float(np.min(self.section_z_derivatives.value)),
            "maximum_centerplane_gap": float(
                np.max(np.abs(self.centerplane_offsets.value))
            ),
            "maximum_waterline_error": float(
                np.max(np.abs(self.waterline_heights.value))
            ),
        }

    def assert_valid(
        self,
        breadth_tolerance: float = 1.0e-10,
        jacobian_tolerance: float = 1.0e-10,
        monotonicity_tolerance: float = 1.0e-10,
        boundary_tolerance: float = 1.0e-8,
    ) -> None:
        """Raise ``ValueError`` if sampled geometric invariants are violated."""
        values = self.report()
        failures: list[str] = []
        if values["minimum_half_breadth"] < -breadth_tolerance:
            failures.append("negative half-breadth")
        if values["minimum_surface_jacobian"] < jacobian_tolerance:
            failures.append("collapsed surface parameterization")
        if values["minimum_outward_normal_y"] < -jacobian_tolerance:
            failures.append("reversed starboard surface orientation")
        if values["minimum_dx_dv"] < -monotonicity_tolerance:
            failures.append("non-monotone longitudinal coordinate")
        if values["minimum_dz_du"] < -monotonicity_tolerance:
            failures.append("non-monotone section height")
        if values["maximum_centerplane_gap"] > boundary_tolerance:
            failures.append("centerplane edge gap")
        if values["maximum_waterline_error"] > boundary_tolerance:
            failures.append("waterline edge is not at z=0")
        if failures:
            raise ValueError("invalid sampled hull geometry: " + ", ".join(failures))


def evaluate_surface_validity(
    surface: TensorProductSurface,
    resolution: tuple[int, int] = (17, 33),
) -> SurfaceValidity:
    """Evaluate sampled validity fields while preserving design derivatives."""
    if resolution[0] < 3 or resolution[1] < 3:
        raise ValueError("validity resolution must be at least (3, 3).")
    u = np.linspace(0.0, 1.0, resolution[0])
    v = np.linspace(0.0, 1.0, resolution[1])
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
    points = surface.evaluate(coordinates)
    tangent_u = surface.evaluate(coordinates, (1, 0))
    tangent_v = surface.evaluate(coordinates, (0, 1))
    area_vectors = csdl.cross(tangent_u, tangent_v, axis=1)

    centerplane_coordinates = np.column_stack((np.zeros_like(v), v))
    waterline_coordinates = np.column_stack((np.ones_like(v), v))
    centerplane = surface.evaluate(centerplane_coordinates)
    waterline = surface.evaluate(waterline_coordinates)
    return SurfaceValidity(
        half_breadths=points[:, 1],
        jacobian_magnitudes=csdl.norm(area_vectors, axes=(1,)),
        outward_normal_y=area_vectors[:, 1],
        longitudinal_x_derivatives=tangent_v[:, 0],
        section_z_derivatives=tangent_u[:, 2],
        centerplane_offsets=centerplane[:, 1],
        waterline_heights=waterline[:, 2],
        resolution=resolution,
    )
