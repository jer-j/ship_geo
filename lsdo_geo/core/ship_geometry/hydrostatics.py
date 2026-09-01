"""Differentiable hydrostatics for symmetric untrimmed hull surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt

from .closed_surface import ClosedSurface
from .surfaces import TensorProductSurface, _composite_rule


@dataclass
class Hydrostatics:
    """Hydrostatic quantities evaluated in the CSDL graph."""

    displacement: csdl.Variable
    center_of_buoyancy: csdl.Variable
    waterplane_area: csdl.Variable
    center_of_flotation: csdl.Variable
    transverse_waterplane_inertia: csdl.Variable
    longitudinal_waterplane_inertia: csdl.Variable
    wetted_area: csdl.Variable
    sectional_areas: csdl.Variable | None = None
    section_parameters: np.ndarray | None = None
    waterplane_centroid: csdl.Variable | None = None


def compute_hydrostatics(
    surface: TensorProductSurface,
    section_parameters: npt.ArrayLike | None = None,
    normal_sign: float = 1.0,
) -> Hydrostatics:
    """Compute hydrostatics from a starboard half-hull side surface.

    The surface convention is ``u=0`` at the keel/centerplane, ``u=1`` at the
    waterline, ``v=0`` at the bow, and ``v=1`` at the stern.  With increasing
    ``x`` in the ``v`` direction, :math:`S_u \times S_v` points outward on the
    starboard side and ``normal_sign=1``.

    The waterplane is assumed to lie at ``z=0`` and the hull is symmetric about
    ``y=0``.  Pointed bow and stern boundaries may collapse to the centerplane;
    transom hulls require a separately connected transom patch for fully closed
    volume integrals.
    """
    sign = float(normal_sign)
    if sign not in (-1.0, 1.0):
        raise ValueError("normal_sign must be either +1 or -1.")
    coordinates, weights = surface.quadrature()
    points = surface.evaluate(coordinates)
    area_vectors = sign * surface.area_vectors(coordinates)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    nx = area_vectors[:, 0]
    ny = area_vectors[:, 1]
    nz = area_vectors[:, 2]

    displacement = 2.0 * csdl.sum(weights * y * ny)
    first_moment_x = csdl.sum(weights * x**2 * nx)
    first_moment_z = csdl.sum(weights * z**2 * nz)
    zero_y = 0.0 * displacement
    center_of_buoyancy = csdl.concatenate(
        (
            (first_moment_x / displacement).reshape((1,)),
            zero_y.reshape((1,)),
            (first_moment_z / displacement).reshape((1,)),
        )
    )
    wetted_area = 2.0 * csdl.sum(weights * csdl.norm(area_vectors, axes=(1,)))

    waterline_points, waterline_weights = _waterline_quadrature(surface)
    waterline = surface.evaluate(waterline_points)
    waterline_tangent = surface.evaluate(waterline_points, (0, 1))
    x_waterline = waterline[:, 0]
    breadth = waterline[:, 1]
    dx_dv = waterline_tangent[:, 0]
    waterplane_area = 2.0 * csdl.sum(waterline_weights * breadth * dx_dv)
    first_waterplane_moment = 2.0 * csdl.sum(
        waterline_weights * x_waterline * breadth * dx_dv
    )
    center_of_flotation = first_waterplane_moment / waterplane_area
    transverse_inertia = (2.0 / 3.0) * csdl.sum(waterline_weights * breadth**3 * dx_dv)
    longitudinal_inertia_origin = 2.0 * csdl.sum(
        waterline_weights * x_waterline**2 * breadth * dx_dv
    )
    longitudinal_inertia = (
        longitudinal_inertia_origin - waterplane_area * center_of_flotation**2
    )

    sectional_areas = None
    parameters_array = None
    if section_parameters is not None:
        parameters_array = np.asarray(section_parameters, dtype=float).reshape(-1)
        if np.any(parameters_array < 0.0) or np.any(parameters_array > 1.0):
            raise ValueError("section_parameters must lie in [0, 1].")
        sectional_areas = compute_sectional_areas(surface, parameters_array)

    return Hydrostatics(
        displacement=displacement,
        center_of_buoyancy=center_of_buoyancy,
        waterplane_area=waterplane_area,
        center_of_flotation=center_of_flotation,
        transverse_waterplane_inertia=transverse_inertia,
        longitudinal_waterplane_inertia=longitudinal_inertia,
        wetted_area=wetted_area,
        sectional_areas=sectional_areas,
        section_parameters=parameters_array,
        waterplane_centroid=csdl.concatenate(
            (
                center_of_flotation.reshape((1,)),
                zero_y.reshape((1,)),
                zero_y.reshape((1,)),
            )
        ),
    )


def compute_closed_surface_hydrostatics(
    closed_surface: ClosedSurface,
) -> Hydrostatics:
    r"""Integrate hydrostatics over an explicitly closed patch collection.

    Volume and first moments follow directly from the divergence theorem:

    .. math::

       \nabla=\frac{1}{3}\int_{\partial V}\mathbf r\cdot\mathbf n\,dA,
       \qquad
       \int_V x_i\,dV=\frac{1}{2}\int_{\partial V}x_i^2n_i\,dA.

    Every patch supplies an outward orientation sign. Patches marked as the
    waterplane are used for flotation properties and excluded from wetted area.
    """
    volume_terms: list[csdl.Variable] = []
    first_x_terms: list[csdl.Variable] = []
    first_y_terms: list[csdl.Variable] = []
    first_z_terms: list[csdl.Variable] = []
    wetted_terms: list[csdl.Variable] = []
    waterplane_area_terms: list[csdl.Variable] = []
    waterplane_x_terms: list[csdl.Variable] = []
    waterplane_y_terms: list[csdl.Variable] = []
    waterplane_x2_terms: list[csdl.Variable] = []
    waterplane_y2_terms: list[csdl.Variable] = []

    for patch in closed_surface.patches:
        coordinates, weights = patch.surface.quadrature()
        points = patch.surface.evaluate(coordinates)
        area_vectors = patch.normal_sign * patch.surface.area_vectors(coordinates)
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        nx = area_vectors[:, 0]
        ny = area_vectors[:, 1]
        nz = area_vectors[:, 2]
        volume_terms.append(csdl.sum(weights * (x * nx + y * ny + z * nz)) / 3.0)
        first_x_terms.append(0.5 * csdl.sum(weights * x**2 * nx))
        first_y_terms.append(0.5 * csdl.sum(weights * y**2 * ny))
        first_z_terms.append(0.5 * csdl.sum(weights * z**2 * nz))
        if patch.wetted:
            wetted_terms.append(csdl.sum(weights * csdl.norm(area_vectors, axes=(1,))))
        if patch.waterplane:
            projected_area = nz
            waterplane_area_terms.append(csdl.sum(weights * projected_area))
            waterplane_x_terms.append(csdl.sum(weights * x * projected_area))
            waterplane_y_terms.append(csdl.sum(weights * y * projected_area))
            waterplane_x2_terms.append(csdl.sum(weights * x**2 * projected_area))
            waterplane_y2_terms.append(csdl.sum(weights * y**2 * projected_area))

    if not waterplane_area_terms:
        raise ValueError("closed surface has no patch marked as waterplane.")

    displacement = _sum_variables(volume_terms)
    first_x = _sum_variables(first_x_terms)
    first_y = _sum_variables(first_y_terms)
    first_z = _sum_variables(first_z_terms)
    center = csdl.concatenate(
        (
            (first_x / displacement).reshape((1,)),
            (first_y / displacement).reshape((1,)),
            (first_z / displacement).reshape((1,)),
        )
    )
    waterplane_area = _sum_variables(waterplane_area_terms)
    waterplane_x = _sum_variables(waterplane_x_terms) / waterplane_area
    waterplane_y = _sum_variables(waterplane_y_terms) / waterplane_area
    zero_z = 0.0 * waterplane_x
    waterplane_centroid = csdl.concatenate(
        (
            waterplane_x.reshape((1,)),
            waterplane_y.reshape((1,)),
            zero_z.reshape((1,)),
        )
    )
    transverse_inertia = (
        _sum_variables(waterplane_y2_terms) - waterplane_area * waterplane_y**2
    )
    longitudinal_inertia = (
        _sum_variables(waterplane_x2_terms) - waterplane_area * waterplane_x**2
    )
    wetted_area = _sum_variables(wetted_terms)
    return Hydrostatics(
        displacement=displacement,
        center_of_buoyancy=center,
        waterplane_area=waterplane_area,
        center_of_flotation=waterplane_x,
        transverse_waterplane_inertia=transverse_inertia,
        longitudinal_waterplane_inertia=longitudinal_inertia,
        wetted_area=wetted_area,
        waterplane_centroid=waterplane_centroid,
    )


def _sum_variables(values: list[csdl.Variable]) -> csdl.Variable:
    if not values:
        raise ValueError("cannot sum an empty variable sequence.")
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def compute_sectional_areas(
    surface: TensorProductSurface,
    section_parameters: npt.ArrayLike,
) -> csdl.Variable:
    """Compute full immersed section areas at fixed longitudinal parameters."""
    parameters = np.asarray(section_parameters, dtype=float).reshape(-1)
    indices = surface.space.knot_indices[0]
    knots = np.asarray(surface.space.knots[indices], dtype=float)
    u_points, u_weights = _composite_rule(
        knots,
        int(surface.space.degree[0]),
        int(surface.quadrature_order[0]),
    )
    areas: list[csdl.Variable] = []
    for parameter in parameters:
        coordinates = np.column_stack((u_points, np.full(u_points.shape, parameter)))
        section = surface.evaluate(coordinates)
        tangent = surface.evaluate(coordinates, (1, 0))
        half_area = csdl.sum(u_weights * section[:, 1] * tangent[:, 2])
        areas.append((2.0 * half_area).reshape((1,)))
    if len(areas) == 1:
        return areas[0]
    return csdl.concatenate(tuple(areas))


def _waterline_quadrature(
    surface: TensorProductSurface,
) -> tuple[np.ndarray, np.ndarray]:
    indices = surface.space.knot_indices[1]
    knots = np.asarray(surface.space.knots[indices], dtype=float)
    v_points, weights = _composite_rule(
        knots,
        int(surface.space.degree[1]),
        int(surface.quadrature_order[1]),
    )
    return np.column_stack((np.ones_like(v_points), v_points)), weights
