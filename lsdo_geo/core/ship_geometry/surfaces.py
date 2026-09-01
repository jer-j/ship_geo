"""Compatible B-spline lofts and surface-patch relationships."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.f_spline import FSplineCurve
from lsdo_geo.core.splines.variational import ConstraintHandle, VariationalSystem


def _composite_rule(
    knots: np.ndarray, degree: int, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed quadrature data over every nonzero knot span."""
    local_points, local_weights = np.polynomial.legendre.leggauss(order)
    active = np.unique(knots[degree:-degree])
    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for lower, upper in pairwise(active):
        if upper <= lower:
            continue
        half = 0.5 * (upper - lower)
        points.append(0.5 * (upper + lower) + half * local_points)
        weights.append(half * local_weights)
    return np.concatenate(points), np.concatenate(weights)


@dataclass
class TensorProductSurface:
    """A three-dimensional tensor-product B-spline surface."""

    function: lfs.Function
    quadrature_order: tuple[int, int] = (6, 6)
    lagrange_multipliers: csdl.Variable | None = None
    constraint_residual: csdl.Variable | None = None
    stationarity_residual: csdl.Variable | None = None
    fairness_objective: csdl.Variable | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.function.space, lfs.BSplineSpace):
            raise TypeError("surface function must use BSplineSpace.")
        if self.function.space.num_parametric_dimensions != 2:
            raise ValueError("surface function must have two parametric dimensions.")
        if self.function.num_physical_dimensions != 3:
            raise ValueError("surface function must have three physical dimensions.")

    @property
    def coefficients(self) -> csdl.Variable:
        """Surface control net."""
        return self.function.coefficients

    @property
    def space(self) -> lfs.BSplineSpace:
        """Underlying tensor-product B-spline space."""
        return self.function.space

    def evaluate(
        self,
        parametric_coordinates: npt.ArrayLike,
        derivative_orders: tuple[int, int] = (0, 0),
    ) -> csdl.Variable:
        """Evaluate the surface or one of its parametric derivatives."""
        coordinates = np.asarray(parametric_coordinates, dtype=float).reshape((-1, 2))
        if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
            raise ValueError("parametric coordinates must lie in [0, 1]^2.")
        if len(derivative_orders) != 2 or any(
            order < 0 or order > 2 for order in derivative_orders
        ):
            raise ValueError("surface derivative orders must lie in [0, 2].")
        return self.function.evaluate(
            coordinates,
            parametric_derivative_orders=derivative_orders,
        )

    def quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """Return tensor-product quadrature coordinates and weights."""
        axis_data: list[tuple[np.ndarray, np.ndarray]] = []
        for axis in range(2):
            indices = self.space.knot_indices[axis]
            knots = np.asarray(self.space.knots[indices], dtype=float)
            axis_data.append(
                _composite_rule(
                    knots,
                    int(self.space.degree[axis]),
                    int(self.quadrature_order[axis]),
                )
            )
        u_grid, v_grid = np.meshgrid(axis_data[0][0], axis_data[1][0], indexing="ij")
        wu_grid, wv_grid = np.meshgrid(axis_data[0][1], axis_data[1][1], indexing="ij")
        coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
        return coordinates, (wu_grid * wv_grid).ravel()

    def area_vectors(self, coordinates: npt.ArrayLike) -> csdl.Variable:
        """Return oriented :math:`S_u \times S_v` vectors."""
        coordinates = np.asarray(coordinates, dtype=float).reshape((-1, 2))
        tangent_u = self.evaluate(coordinates, (1, 0))
        tangent_v = self.evaluate(coordinates, (0, 1))
        return csdl.cross(tangent_u, tangent_v, axis=1)

    def area(self) -> csdl.Variable:
        """Integrate the physical surface area."""
        coordinates, weights = self.quadrature()
        vectors = self.area_vectors(coordinates)
        magnitudes = csdl.norm(vectors, axes=(1,))
        return csdl.sum(weights * magnitudes)

    def fairness_energy(self, derivative_orders: tuple[int, int]) -> csdl.Variable:
        """Integrate the squared norm of a parametric surface derivative."""
        if derivative_orders == (0, 0):
            raise ValueError("fairness requires a nonzero derivative order.")
        coordinates, weights = self.quadrature()
        derivative = self.evaluate(coordinates, derivative_orders)
        return csdl.sum(weights * csdl.sum(derivative**2, axes=(1,)))

    def mesh(self, resolution: tuple[int, int] = (31, 61)) -> csdl.Variable:
        """Evaluate a structured surface mesh in CSDL."""
        u = np.linspace(0.0, 1.0, resolution[0])
        v = np.linspace(0.0, 1.0, resolution[1])
        u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
        coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
        return self.evaluate(coordinates).reshape(resolution + (3,))


class CompatibleLoft:
    """Skin compatible transverse control polygons longitudinally."""

    @staticmethod
    def create(
        sections: Sequence[FSplineCurve],
        station_parameters: npt.ArrayLike,
        x_coordinates: Sequence[Any] | csdl.Variable | npt.ArrayLike,
        longitudinal_degree: int = 3,
        longitudinal_num_control_points: int | None = None,
        longitudinal_knots: npt.ArrayLike | None = None,
        quadrature_order: tuple[int, int] = (6, 6),
        name: str = "hull_surface",
    ) -> TensorProductSurface:
        """Create a differentiable tensor-product surface through sections.

        The longitudinal fit is a fixed linear map applied with CSDL
        operations.  It therefore adds no nested nonlinear solve.
        """
        if len(sections) < longitudinal_degree + 1:
            raise ValueError("insufficient sections for the longitudinal degree.")
        parameters = np.asarray(station_parameters, dtype=float).reshape(-1)
        if parameters.size != len(sections):
            raise ValueError("station_parameters must match the section count.")
        if np.any(np.diff(parameters) <= 0.0):
            raise ValueError("station_parameters must be strictly increasing.")
        if parameters[0] < 0.0 or parameters[-1] > 1.0:
            raise ValueError("station_parameters must lie in [0, 1].")
        CompatibleLoft._check_compatibility(sections)

        if isinstance(x_coordinates, csdl.Variable):
            if x_coordinates.size != len(sections):
                raise ValueError("x_coordinates must match the section count.")
            x_values: list[Any] = [
                x_coordinates[index] for index in range(len(sections))
            ]
        else:
            x_values = list(x_coordinates)
            if len(x_values) != len(sections):
                raise ValueError("x_coordinates must match the section count.")

        num_longitudinal = longitudinal_num_control_points or len(sections)
        if num_longitudinal <= longitudinal_degree:
            raise ValueError(
                "longitudinal_num_control_points must exceed longitudinal_degree."
            )
        longitudinal_space = lfs.BSplineSpace(
            num_parametric_dimensions=1,
            degree=(longitudinal_degree,),
            coefficients_shape=(num_longitudinal,),
            knots=(
                None
                if longitudinal_knots is None
                else np.asarray(longitudinal_knots, dtype=float)
            ),
        )
        basis = longitudinal_space.compute_basis_matrix(parameters[:, None]).toarray()
        if basis.shape[0] == basis.shape[1]:
            fitting_map = np.linalg.solve(basis, np.eye(basis.shape[0]))
        else:
            normal_matrix = basis.T @ basis
            if np.linalg.matrix_rank(normal_matrix) < normal_matrix.shape[0]:
                raise ValueError("longitudinal section fit is rank deficient.")
            fitting_map = np.linalg.solve(normal_matrix, basis.T)

        section_points: list[csdl.Variable] = []
        for section, x_coordinate in zip(sections, x_values):
            coefficients = section.coefficients
            z = coefficients[:, 0].reshape((coefficients.shape[0], 1))
            y = coefficients[:, 1].reshape((coefficients.shape[0], 1))
            x = 0.0 * z + x_coordinate
            section_points.append(csdl.concatenate((x, y, z), axis=1))

        transverse_count = sections[0].coefficients.shape[0]
        longitudinal_rows: list[csdl.Variable] = []
        for transverse_index in range(transverse_count):
            values = csdl.concatenate(
                tuple(
                    points[transverse_index, :].reshape((1, 3))
                    for points in section_points
                ),
                axis=0,
            )
            row = fitting_map @ values
            longitudinal_rows.append(row.reshape((1, num_longitudinal, 3)))
        surface_coefficients = csdl.concatenate(tuple(longitudinal_rows), axis=0)

        transverse_space = sections[0].space
        transverse_knots = np.asarray(
            transverse_space.knots[transverse_space.knot_indices[0]], dtype=float
        )
        longitudinal_knot_values = np.asarray(
            longitudinal_space.knots[longitudinal_space.knot_indices[0]], dtype=float
        )
        surface_space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=(transverse_space.degree[0], longitudinal_degree),
            coefficients_shape=(transverse_count, num_longitudinal),
            knots=np.concatenate((transverse_knots, longitudinal_knot_values)),
        )
        return TensorProductSurface(
            function=lfs.Function(surface_space, surface_coefficients, name=name),
            quadrature_order=quadrature_order,
        )

    @staticmethod
    def _check_compatibility(sections: Sequence[FSplineCurve]) -> None:
        reference = sections[0].space
        reference_knots = np.asarray(
            reference.knots[reference.knot_indices[0]], dtype=float
        )
        reference_shape = sections[0].coefficients.shape
        for section in sections[1:]:
            knots = np.asarray(
                section.space.knots[section.space.knot_indices[0]], dtype=float
            )
            if section.coefficients.shape != reference_shape:
                raise ValueError("all loft sections must have compatible coefficients.")
            if section.space.degree != reference.degree or not np.array_equal(
                knots, reference_knots
            ):
                raise ValueError("all loft sections must share degree and knots.")


Edge = Literal["u0", "u1", "v0", "v1"]


@dataclass(frozen=True)
class PatchConnection:
    """A sampled relationship between two surface-patch edges."""

    first: str
    first_edge: Edge
    second: str
    second_edge: Edge
    reverse: bool = False
    tangent_continuity: bool = False
    scale: float = 1.0


class PatchGraph:
    """Explicit representation topology for connected surface patches."""

    def __init__(self) -> None:
        self.patches: dict[str, TensorProductSurface] = {}
        self.connections: list[PatchConnection] = []

    def add_patch(self, name: str, surface: TensorProductSurface) -> None:
        """Register a uniquely named surface patch."""
        if name in self.patches:
            raise ValueError(f"patch {name!r} already exists.")
        self.patches[name] = surface

    def connect(self, connection: PatchConnection) -> None:
        """Register a positional and optional tangent edge relationship."""
        if (
            connection.first not in self.patches
            or connection.second not in self.patches
        ):
            raise KeyError("both patches must be registered before connecting them.")
        self.connections.append(connection)

    def add_continuity_constraints(
        self, system: VariationalSystem, samples: int = 7
    ) -> list[ConstraintHandle]:
        """Add sampled ``C0`` and optional tangent constraints to one KKT solve."""
        if samples < 2:
            raise ValueError("at least two edge samples are required.")
        parameters = np.linspace(0.0, 1.0, samples)
        handles: list[ConstraintHandle] = []
        for connection in self.connections:
            first_coords, first_normal_order = self._edge_coordinates(
                connection.first_edge, parameters
            )
            second_parameters = parameters[::-1] if connection.reverse else parameters
            second_coords, second_normal_order = self._edge_coordinates(
                connection.second_edge, second_parameters
            )
            first_surface = self.patches[connection.first]
            second_surface = self.patches[connection.second]
            gap = connection.scale * (
                first_surface.evaluate(first_coords)
                - second_surface.evaluate(second_coords)
            )
            handles.append(system.add_constraint(gap.flatten()))
            if connection.tangent_continuity:
                first_tangent = first_surface.evaluate(first_coords, first_normal_order)
                second_tangent = second_surface.evaluate(
                    second_coords, second_normal_order
                )
                cross_residual = csdl.cross(first_tangent, second_tangent, axis=1)
                handles.append(
                    system.add_constraint(connection.scale * cross_residual.flatten())
                )
        return handles

    @staticmethod
    def _edge_coordinates(
        edge: Edge, parameters: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if edge == "u0":
            return np.column_stack((np.zeros_like(parameters), parameters)), (1, 0)
        if edge == "u1":
            return np.column_stack((np.ones_like(parameters), parameters)), (1, 0)
        if edge == "v0":
            return np.column_stack((parameters, np.zeros_like(parameters))), (0, 1)
        return np.column_stack((parameters, np.ones_like(parameters))), (0, 1)


def wigley_surface(
    length: Any,
    beam: Any,
    draft: Any,
    name: str = "wigley_surface",
) -> TensorProductSurface:
    r"""Return an exact quadratic B-spline representation of a Wigley hull.

    The starboard half surface is

    .. math::

       x=L(v-1/2),\quad z=-T(1-u),\quad
       y=\frac{B}{2}(2u-u^2)4v(1-v).
    """
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(2, 2),
        coefficients_shape=(3, 3),
    )
    u_shape = np.array([0.0, 1.0, 1.0])
    v_shape = np.array([0.0, 2.0, 0.0])
    x_basis = np.array([-0.5, 0.0, 0.5])
    z_basis = np.array([-1.0, -0.5, 0.0])
    rows: list[csdl.Variable] = []
    anchor = length if isinstance(length, csdl.Variable) else beam
    if not isinstance(anchor, csdl.Variable):
        anchor = draft
    if not isinstance(anchor, csdl.Variable):
        anchor = csdl.Variable(value=float(length))
    length_variable = (
        length if isinstance(length, csdl.Variable) else 0.0 * anchor + length
    )
    beam_variable = beam if isinstance(beam, csdl.Variable) else 0.0 * anchor + beam
    draft_variable = draft if isinstance(draft, csdl.Variable) else 0.0 * anchor + draft
    for u_index in range(3):
        points: list[csdl.Variable] = []
        for v_index in range(3):
            x = length_variable * x_basis[v_index]
            y = 0.5 * beam_variable * u_shape[u_index] * v_shape[v_index]
            z = draft_variable * z_basis[u_index]
            points.append(
                csdl.concatenate(
                    (x.reshape((1,)), y.reshape((1,)), z.reshape((1,)))
                ).reshape((1, 3))
            )
        rows.append(csdl.concatenate(tuple(points), axis=0).reshape((1, 3, 3)))
    coefficients = csdl.concatenate(tuple(rows), axis=0)
    return TensorProductSurface(lfs.Function(space, coefficients, name=name))
