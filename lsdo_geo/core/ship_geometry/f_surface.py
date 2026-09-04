"""Variational tensor-product B-spline surfaces.

An :class:`FSurfaceProblem` contributes a free surface control net, fairness
terms, and geometric equality constraints to a shared
:class:`~lsdo_geo.core.splines.variational.VariationalSystem`.  The primary
``assemble`` interface lets curves, sections, and several surface patches share
one KKT system and one CSDL Newton solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.variational import (
    ConstraintHandle,
    VariationalResult,
    VariationalSystem,
)

from .surfaces import Edge, TensorProductSurface


def _coerce_target(value: Any) -> Any:
    """Convert constant targets while preserving differentiable CSDL inputs."""
    if isinstance(value, csdl.Variable):
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


def _value_array(value: Any) -> np.ndarray:
    """Return the current value of a CSDL or ordinary array-like object."""
    if isinstance(value, csdl.Variable):
        return np.asarray(value.value, dtype=float)
    return np.asarray(value, dtype=float)


def _target_size(value: Any) -> int:
    """Return a target's scalar count from static shape information."""
    if isinstance(value, csdl.Variable):
        return int(value.size)
    return int(np.asarray(value, dtype=float).size)


def _maybe_value_array(value: Any) -> np.ndarray | None:
    """Return current numeric values, or ``None`` under a deferred recorder."""
    if isinstance(value, csdl.Variable):
        if value.value is None:
            return None
        return np.asarray(value.value, dtype=float)
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class SurfacePointConstraint:
    """A complete surface-position constraint at one parameter pair."""

    coordinates: tuple[float, float]
    target: Any
    scale: float


@dataclass(frozen=True)
class SurfaceDerivativeConstraint:
    """A complete parametric-derivative constraint at one parameter pair."""

    coordinates: tuple[float, float]
    derivative_orders: tuple[int, int]
    target: Any
    scale: float


@dataclass(frozen=True)
class SurfaceSamplesConstraint:
    """A vectorized set of complete surface-position constraints."""

    coordinates: np.ndarray
    target: Any
    scale: float


@dataclass(frozen=True)
class SurfaceControlPointConstraint:
    """A direct constraint on one physical control point."""

    indices: tuple[int, int]
    target: Any
    scale: float


SurfaceConstraint = (
    SurfacePointConstraint
    | SurfaceDerivativeConstraint
    | SurfaceSamplesConstraint
    | SurfaceControlPointConstraint
)


@dataclass
class FSurfaceAssembly:
    """A free surface-control-net contribution to a global KKT system."""

    surface: TensorProductSurface
    state_index: int
    constraint_handle: ConstraintHandle
    fairness_objective: csdl.Variable
    lagrange_multipliers: csdl.Variable | None = None
    constraint_residual: csdl.Variable | None = None
    stationarity_residual: csdl.Variable | None = None

    def finalize(self, result: VariationalResult) -> TensorProductSurface:
        """Attach KKT diagnostics to the surface and return it."""
        self.stationarity_residual = result.stationarity_residuals[self.state_index]
        if result.constraint_residual is not None:
            self.constraint_residual = result.constraint_residual[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        if result.lagrange_multipliers is not None:
            self.lagrange_multipliers = result.lagrange_multipliers[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        self.surface.fairness_objective = self.fairness_objective
        self.surface.constraint_residual = self.constraint_residual
        self.surface.stationarity_residual = self.stationarity_residual
        self.surface.lagrange_multipliers = self.lagrange_multipliers
        return self.surface


class FSurfaceProblem:
    r"""Define a fairness-optimized tensor-product B-spline surface.

    The default thin-plate objective is

    .. math::

       J = \int_0^1\!\int_0^1
       \left(\lVert S_{uu}\rVert^2
       +2\lVert S_{uv}\rVert^2
       +\lVert S_{vv}\rVert^2\right)\,du\,dv.

    ``assemble`` is the primary interface.  It registers the surface control
    net in an existing :class:`VariationalSystem` without running a solver.
    ``solve`` is a convenience for a standalone surface and still uses the
    same assembly path.

    Parameters
    ----------
    num_control_points
        Control-point counts in the two parametric directions.
    degree
        Polynomial degrees in the two parametric directions.
    knots
        Optional pair of complete knot vectors.
    fairness_weights
        Nonnegative weights keyed by derivative orders ``(du, dv)``.
    quadrature_order
        Fixed Gauss-Legendre order in each parametric direction.
    regularization
        Optional squared-control-net regularization.
    name
        Prefix for CSDL variables and the standalone Newton solver.
    """

    def __init__(
        self,
        num_control_points: tuple[int, int] = (6, 8),
        degree: tuple[int, int] = (3, 3),
        knots: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
        fairness_weights: Mapping[tuple[int, int], float] | None = None,
        quadrature_order: tuple[int, int] = (6, 6),
        regularization: float = 0.0,
        name: str = "f_surface",
    ) -> None:
        if len(num_control_points) != 2 or len(degree) != 2:
            raise ValueError("surface control-point counts and degrees need two axes.")
        counts = tuple(int(value) for value in num_control_points)
        degrees = tuple(int(value) for value in degree)
        if any(value < 1 for value in degrees):
            raise ValueError("surface degrees must be positive.")
        if any(count <= order for count, order in zip(counts, degrees)):
            raise ValueError("each control-point count must exceed its degree.")
        if len(quadrature_order) != 2 or any(
            int(value) < 1 for value in quadrature_order
        ):
            raise ValueError("surface quadrature orders must be positive.")
        if not np.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative.")

        knot_values: np.ndarray | None = None
        if knots is not None:
            if len(knots) != 2:
                raise ValueError("knots must contain one knot vector per axis.")
            knot_values = np.concatenate(
                tuple(np.asarray(vector, dtype=float).reshape(-1) for vector in knots)
            )
        self.space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=degrees,
            coefficients_shape=counts,
            knots=knot_values,
        )
        self.num_control_points = counts
        self.degree = degrees
        self.quadrature_order = tuple(int(value) for value in quadrature_order)
        self.regularization = float(regularization)
        self.name = name
        self.constraints: list[SurfaceConstraint] = []

        if fairness_weights is None:
            if min(degrees) < 2:
                raise ValueError(
                    "the default thin-plate objective requires degree >= 2 on both axes."
                )
            fairness_weights = {(2, 0): 1.0, (1, 1): 2.0, (0, 2): 1.0}
        self.fairness_weights = {
            tuple(int(order) for order in orders): float(weight)
            for orders, weight in fairness_weights.items()
            if float(weight) != 0.0
        }
        for orders, weight in self.fairness_weights.items():
            if len(orders) != 2 or orders == (0, 0):
                raise ValueError("fairness derivative orders must be a nonzero pair.")
            if any(order < 0 or order > 2 for order in orders):
                raise ValueError(
                    "surface derivative orders must lie in [0, 2] for the "
                    "current lsdo_b_splines_cython backend."
                )
            if any(order > axis_degree for order, axis_degree in zip(orders, degrees)):
                raise ValueError("fairness derivative order exceeds the spline degree.")
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("fairness weights must be finite and nonnegative.")
        if not self.fairness_weights and self.regularization == 0.0:
            raise ValueError(
                "at least one fairness or regularization term is required."
            )

    def add_point_constraint(
        self,
        coordinates: tuple[float, float],
        target: Any,
        scale: float = 1.0,
    ) -> None:
        """Constrain the full physical position at one parameter pair."""
        parameter_pair = self._validate_coordinates(coordinates)
        self._validate_target(target)
        self.constraints.append(
            SurfacePointConstraint(
                parameter_pair,
                _coerce_target(target),
                self._validate_scale(scale),
            )
        )

    def add_derivative_constraint(
        self,
        coordinates: tuple[float, float],
        derivative_orders: tuple[int, int],
        target: Any,
        scale: float = 1.0,
    ) -> None:
        """Constrain one complete parametric surface derivative."""
        parameter_pair = self._validate_coordinates(coordinates)
        orders = tuple(int(value) for value in derivative_orders)
        if len(orders) != 2 or orders == (0, 0):
            raise ValueError("derivative_orders must be a nonzero pair.")
        if any(order < 0 or order > 2 for order in orders):
            raise ValueError("surface derivative orders must lie in [0, 2].")
        if any(order > axis_degree for order, axis_degree in zip(orders, self.degree)):
            raise ValueError("derivative order exceeds the spline degree.")
        self._validate_target(target)
        self.constraints.append(
            SurfaceDerivativeConstraint(
                parameter_pair,
                orders,
                _coerce_target(target),
                self._validate_scale(scale),
            )
        )

    def add_points_constraint(
        self,
        coordinates: npt.ArrayLike,
        targets: Any,
        scale: float = 1.0,
    ) -> None:
        """Constrain many physical positions through one vectorized residual."""
        coordinate_array = np.asarray(coordinates, dtype=float).reshape((-1, 2))
        if coordinate_array.shape[0] < 1:
            raise ValueError("at least one surface sample is required.")
        if np.any(coordinate_array < 0.0) or np.any(coordinate_array > 1.0):
            raise ValueError("surface coordinates must lie in [0, 1]^2.")
        expected_shape = (coordinate_array.shape[0], 3)
        target_shape = (
            tuple(targets.shape)
            if isinstance(targets, csdl.Variable)
            else np.asarray(targets, dtype=float).shape
        )
        if target_shape != expected_shape:
            raise ValueError(
                f"surface sample targets must have shape {expected_shape}."
            )
        self.constraints.append(
            SurfaceSamplesConstraint(
                coordinate_array.copy(),
                _coerce_target(targets),
                self._validate_scale(scale),
            )
        )

    def add_control_point_constraint(
        self,
        indices: tuple[int, int],
        target: Any,
        scale: float = 1.0,
    ) -> None:
        """Constrain one control point, useful for exact compatible boundaries."""
        if len(indices) != 2:
            raise ValueError("control-point indices must contain two values.")
        index_pair = tuple(int(value) for value in indices)
        if any(
            index < 0 or index >= count
            for index, count in zip(index_pair, self.num_control_points)
        ):
            raise IndexError("surface control-point index is outside the control net.")
        self._validate_target(target)
        self.constraints.append(
            SurfaceControlPointConstraint(
                index_pair,
                _coerce_target(target),
                self._validate_scale(scale),
            )
        )

    def add_edge_constraints(
        self,
        edge: Edge,
        targets: Sequence[Any] | csdl.Variable | npt.ArrayLike,
        parameters: npt.ArrayLike | None = None,
        scale: float = 1.0,
    ) -> None:
        """Constrain sampled physical positions along one patch edge."""
        if parameters is None:
            target_count = (
                int(targets.shape[0])
                if isinstance(targets, csdl.Variable)
                else len(targets)
            )
            parameters = np.linspace(0.0, 1.0, target_count)
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if isinstance(targets, csdl.Variable):
            if targets.shape != (values.size, 3):
                raise ValueError("edge targets must have shape (num_samples, 3).")
            target_values: Sequence[Any] = [
                targets[index] for index in range(values.size)
            ]
        else:
            target_values = list(targets)
            if len(target_values) != values.size:
                raise ValueError("edge targets must match the parameter count.")
        for parameter, target in zip(values, target_values):
            self.add_point_constraint(
                self._edge_coordinates(edge, float(parameter)), target, scale
            )

    def assemble(
        self,
        system: VariationalSystem,
        initial_control_points: npt.ArrayLike | None = None,
    ) -> FSurfaceAssembly:
        """Register this free control net with a shared variational system."""
        try:
            csdl.get_current_recorder()
        except ValueError as error:
            raise RuntimeError(
                "FSurfaceProblem.assemble requires an active csdl.Recorder."
            ) from error
        if not self.constraints:
            raise ValueError("at least one surface form constraint is required.")

        initial = self._initial_control_points(initial_control_points)
        coefficients = csdl.ImplicitVariable(
            value=initial,
            name=f"{self.name}_coefficients",
        )
        surface = TensorProductSurface(
            function=lfs.Function(self.space, coefficients, name=self.name),
            quadrature_order=self.quadrature_order,
        )

        objective: Any = 0.0
        for orders, weight in self.fairness_weights.items():
            objective = objective + weight * surface.fairness_energy(orders)
        if self.regularization:
            objective = objective + self.regularization * csdl.sum(coefficients**2)

        residuals = tuple(
            self._constraint_residual(surface, constraint)
            for constraint in self.constraints
        )
        constraint_residual = (
            residuals[0] if len(residuals) == 1 else csdl.concatenate(residuals)
        )
        coefficient_count = int(np.prod(self.num_control_points) * 3)
        if constraint_residual.size > coefficient_count:
            raise ValueError(
                f"received {constraint_residual.size} scalar constraints for "
                f"{coefficient_count} coefficient values."
            )
        state_index = system.add_state(coefficients, name=self.name)
        system.add_objective(objective)
        constraint_handle = system.add_constraint(constraint_residual)
        return FSurfaceAssembly(
            surface=surface,
            state_index=state_index,
            constraint_handle=constraint_handle,
            fairness_objective=objective,
        )

    def solve(
        self,
        initial_control_points: npt.ArrayLike | None = None,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
    ) -> TensorProductSurface:
        """Solve a standalone surface through the shared assembly API."""
        system = VariationalSystem(name=self.name)
        assembly = self.assemble(system, initial_control_points)
        result = system.solve(tolerance, max_iter, print_status)
        return assembly.finalize(result)

    def _constraint_residual(
        self, surface: TensorProductSurface, constraint: SurfaceConstraint
    ) -> csdl.Variable:
        if isinstance(constraint, SurfacePointConstraint):
            value = surface.evaluate([constraint.coordinates]).reshape((3,))
        elif isinstance(constraint, SurfaceDerivativeConstraint):
            value = surface.evaluate(
                [constraint.coordinates], constraint.derivative_orders
            ).reshape((3,))
        elif isinstance(constraint, SurfaceSamplesConstraint):
            value = surface.evaluate(constraint.coordinates)
        elif isinstance(constraint, SurfaceControlPointConstraint):
            value = surface.coefficients[
                constraint.indices[0], constraint.indices[1], :
            ]
        else:
            raise TypeError(f"unsupported surface constraint {type(constraint)!r}.")
        return (constraint.scale * (value - constraint.target)).flatten()

    def _initial_control_points(
        self, initial_control_points: npt.ArrayLike | None
    ) -> np.ndarray:
        expected_shape = self.num_control_points + (3,)
        if initial_control_points is not None:
            initial = np.asarray(initial_control_points, dtype=float)
            if initial.shape != expected_shape:
                raise ValueError(
                    f"initial_control_points must have shape {expected_shape}, "
                    f"received {initial.shape}."
                )
            return initial.copy()

        corners = {
            (0, 0): np.array([0.0, 0.0, 0.0]),
            (self.num_control_points[0] - 1, 0): np.array([1.0, 0.0, 0.0]),
            (0, self.num_control_points[1] - 1): np.array([0.0, 1.0, 0.0]),
            (
                self.num_control_points[0] - 1,
                self.num_control_points[1] - 1,
            ): np.array([1.0, 1.0, 0.0]),
        }
        for constraint in self.constraints:
            if isinstance(constraint, SurfaceControlPointConstraint):
                target = _maybe_value_array(constraint.target)
                if target is not None and constraint.indices in corners:
                    corners[constraint.indices] = target.reshape(3)
            elif isinstance(constraint, SurfacePointConstraint):
                u, v = constraint.coordinates
                index = (
                    0 if np.isclose(u, 0.0) else self.num_control_points[0] - 1,
                    0 if np.isclose(v, 0.0) else self.num_control_points[1] - 1,
                )
                target = _maybe_value_array(constraint.target)
                if (
                    target is not None
                    and (np.isclose(u, 0.0) or np.isclose(u, 1.0))
                    and (np.isclose(v, 0.0) or np.isclose(v, 1.0))
                ):
                    corners[index] = target.reshape(3)
            elif isinstance(constraint, SurfaceSamplesConstraint):
                sample_values = _maybe_value_array(constraint.target)
                if sample_values is None:
                    continue
                targets = sample_values.reshape((-1, 3))
                for coordinates, target in zip(constraint.coordinates, targets):
                    u, v = coordinates
                    if not (
                        (np.isclose(u, 0.0) or np.isclose(u, 1.0))
                        and (np.isclose(v, 0.0) or np.isclose(v, 1.0))
                    ):
                        continue
                    index = (
                        0 if np.isclose(u, 0.0) else self.num_control_points[0] - 1,
                        0 if np.isclose(v, 0.0) else self.num_control_points[1] - 1,
                    )
                    corners[index] = target

        u_values = np.linspace(0.0, 1.0, self.num_control_points[0])
        v_values = np.linspace(0.0, 1.0, self.num_control_points[1])
        initial = np.empty(expected_shape)
        p00 = corners[(0, 0)]
        p10 = corners[(self.num_control_points[0] - 1, 0)]
        p01 = corners[(0, self.num_control_points[1] - 1)]
        p11 = corners[(self.num_control_points[0] - 1, self.num_control_points[1] - 1)]
        for index_u, u in enumerate(u_values):
            for index_v, v in enumerate(v_values):
                initial[index_u, index_v] = (
                    (1.0 - u) * (1.0 - v) * p00
                    + u * (1.0 - v) * p10
                    + (1.0 - u) * v * p01
                    + u * v * p11
                )
        return initial

    @staticmethod
    def _validate_coordinates(
        coordinates: tuple[float, float],
    ) -> tuple[float, float]:
        if len(coordinates) != 2:
            raise ValueError("surface coordinates must contain two values.")
        result = tuple(float(value) for value in coordinates)
        if any(value < 0.0 or value > 1.0 for value in result):
            raise ValueError("surface coordinates must lie in [0, 1]^2.")
        return result

    @staticmethod
    def _edge_coordinates(edge: Edge, parameter: float) -> tuple[float, float]:
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("edge parameters must lie in [0, 1].")
        if edge == "u0":
            return 0.0, parameter
        if edge == "u1":
            return 1.0, parameter
        if edge == "v0":
            return parameter, 0.0
        if edge == "v1":
            return parameter, 1.0
        raise ValueError(f"unknown surface edge {edge!r}.")

    @staticmethod
    def _validate_target(target: Any) -> None:
        if _target_size(target) != 3:
            raise ValueError("surface constraint targets must contain three values.")

    @staticmethod
    def _validate_scale(scale: float) -> float:
        value = float(scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("constraint scale must be finite and positive.")
        return value
