"""CSDL-native fairness-optimized B-spline curves.

The B-spline representation and sparse basis evaluation are supplied by
``lsdo_function_spaces.BSplineSpace`` and ``lsdo_b_splines_cython``. Curve
evaluation, fairness measures, form constraints, the Lagrangian, the KKT
residual, the Newton solution, and implicit derivatives are CSDL operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt

from .constraints import (
    AreaConstraint,
    CentroidConstraint,
    CurvatureConstraint,
    DerivativeConstraint,
    FSplineConstraint,
    PointConstraint,
    TangentAngleConstraint,
)
from .variational import ConstraintHandle, VariationalResult, VariationalSystem

_MAX_BACKEND_DERIVATIVE_ORDER = 2


def _value_array(value: Any) -> np.ndarray:
    """Return the current numerical value of a CSDL or array-like object."""
    if isinstance(value, csdl.Variable):
        return np.asarray(value.value, dtype=float)
    return np.asarray(value, dtype=float)


def _coerce_target(value: Any) -> Any:
    """Convert ordinary array-like targets while preserving CSDL variables."""
    if isinstance(value, csdl.Variable):
        return value
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    return array


def _sin(value: Any) -> Any:
    return csdl.sin(value) if isinstance(value, csdl.Variable) else np.sin(value)


def _cos(value: Any) -> Any:
    return csdl.cos(value) if isinstance(value, csdl.Variable) else np.cos(value)


@dataclass
class FSplineCurve:
    """A solved F-Spline represented by an LSDO function-space function."""

    function: lfs.Function
    quadrature_order: int = 8
    lagrange_multipliers: csdl.Variable | None = None
    constraint_residual: csdl.Variable | None = None
    stationarity_residual: csdl.Variable | None = None
    fairness_objective: csdl.Variable | None = None

    @property
    def coefficients(self) -> csdl.Variable:
        """CSDL B-spline coefficient state."""
        return self.function.coefficients

    @property
    def space(self) -> lfs.BSplineSpace:
        """Underlying one-dimensional B-spline space."""
        return self.function.space

    @property
    def physical_dimension(self) -> int:
        """Number of physical coordinate dimensions."""
        return int(self.function.num_physical_dimensions)

    def evaluate(
        self,
        parametric_coordinates: float | npt.ArrayLike,
        derivative_order: int = 0,
    ) -> csdl.Variable:
        """Evaluate the curve or one of its parametric derivatives."""
        derivative_order = int(derivative_order)
        max_order = min(int(self.space.degree[0]), _MAX_BACKEND_DERIVATIVE_ORDER)
        if derivative_order < 0 or derivative_order > max_order:
            raise ValueError(
                f"derivative_order must lie in [0, {max_order}] for the "
                "current lsdo_b_splines_cython backend."
            )
        coordinates = np.asarray(parametric_coordinates, dtype=float).reshape((-1, 1))
        if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
            raise ValueError("parametric coordinates must lie in [0, 1].")
        return self.function.evaluate(
            coordinates,
            parametric_derivative_orders=(derivative_order,),
        )

    def quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """Return composite Gauss-Legendre data over nonzero knot spans."""
        local_points, local_weights = np.polynomial.legendre.leggauss(
            self.quadrature_order
        )
        knot_indices = self.space.knot_indices[0]
        knots = np.asarray(self.space.knots[knot_indices], dtype=float)
        degree = int(self.space.degree[0])
        active_knots = np.unique(knots[degree:-degree])
        points: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for lower, upper in pairwise(active_knots):
            if upper <= lower:
                continue
            half_span = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            points.append(midpoint + half_span * local_points)
            weights.append(half_span * local_weights)
        return np.concatenate(points), np.concatenate(weights)

    def fairness_energy(self, derivative_order: int) -> csdl.Variable:
        """Evaluate a squared-derivative fairness pseudo-norm."""
        points, weights = self.quadrature()
        derivatives = self.evaluate(points, derivative_order=derivative_order)
        squared_norm = csdl.sum(derivatives**2, axes=(1,))
        return csdl.sum(weights * squared_norm)

    def arc_length(self) -> csdl.Variable:
        """Evaluate curve arc length by composite Gaussian quadrature."""
        points, weights = self.quadrature()
        derivatives = self.evaluate(points, derivative_order=1)
        squared_speed = csdl.sum(derivatives**2, axes=(1,))
        return csdl.sum(weights * csdl.sqrt(squared_speed))

    def curvature(self, parameter: float, epsilon: float = 1.0e-14) -> csdl.Variable:
        """Evaluate signed planar curvature."""
        if self.physical_dimension != 2:
            raise ValueError("curvature currently requires a planar curve.")
        first = self.evaluate(parameter, derivative_order=1).reshape((2,))
        second = self.evaluate(parameter, derivative_order=2).reshape((2,))
        numerator = first[0] * second[1] - first[1] * second[0]
        speed_squared = csdl.sum(first**2)
        return numerator / (speed_squared + epsilon) ** 1.5

    def area_moments(
        self,
    ) -> tuple[csdl.Variable, csdl.Variable, csdl.Variable]:
        """Return signed area and its first moments for a planar curve.

        The curve is closed to the first coordinate axis by vertical segments.
        The result is ``(area, first_moment_about_y,
        first_moment_about_x)``.
        """
        if self.physical_dimension != 2:
            raise ValueError("area and centroid require a planar curve.")

        points, weights = self.quadrature()
        coordinates = self.evaluate(points)
        derivatives = self.evaluate(points, derivative_order=1)
        start = self.evaluate(0.0).reshape((2,))
        end = self.evaluate(1.0).reshape((2,))

        line_integrand = (
            coordinates[:, 1] * derivatives[:, 0]
            - coordinates[:, 0] * derivatives[:, 1]
        )
        line_integral = csdl.sum(weights * line_integrand)
        area = 0.5 * (line_integral + end[0] * end[1] - start[0] * start[1])

        moment_y = (
            csdl.sum(weights * coordinates[:, 0] * line_integrand)
            + end[0] ** 2 * end[1]
            - start[0] ** 2 * start[1]
        ) / 3.0
        moment_x = (
            2.0 * csdl.sum(weights * coordinates[:, 1] * line_integrand)
            + end[0] * end[1] ** 2
            - start[0] * start[1] ** 2
        ) / 6.0
        return area, moment_y, moment_x

    def signed_area(self) -> csdl.Variable:
        """Return signed area between the curve and the first axis."""
        return self.area_moments()[0]

    def centroid(self) -> csdl.Variable:
        """Return the centroid of signed area under a planar curve."""
        area, moment_y, moment_x = self.area_moments()
        return csdl.concatenate(
            (
                (moment_y / area).reshape((1,)),
                (moment_x / area).reshape((1,)),
            )
        )


@dataclass
class FSplineAssembly:
    """An F-Spline contribution registered with a global variational system."""

    curve: FSplineCurve
    state_index: int
    constraint_handle: ConstraintHandle

    def finalize(self, result: VariationalResult) -> FSplineCurve:
        """Attach global KKT diagnostics to the assembled curve."""
        self.curve.stationarity_residual = result.stationarity_residuals[
            self.state_index
        ]
        if result.constraint_residual is not None:
            self.curve.constraint_residual = result.constraint_residual[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        if result.lagrange_multipliers is not None:
            self.curve.lagrange_multipliers = result.lagrange_multipliers[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        return self.curve


class FSplineProblem:
    """Define and solve a constrained F-Spline problem.

    Parameters
    ----------
    num_control_points
        Number of B-spline control points.
    degree
        Polynomial degree.
    physical_dimension
        Number of physical coordinates.
    knots
        Optional knot vector on :math:`[0,1]`.
    fairness_weights
        Mapping from derivative order to nonnegative objective weight.
    quadrature_order
        Gauss-Legendre order used on every nonzero knot span.
    regularization
        Optional coefficient norm added to the fairness objective.
    name
        Prefix used for CSDL variables and the Newton solver.
    """

    def __init__(
        self,
        num_control_points: int = 8,
        degree: int = 3,
        physical_dimension: int = 2,
        knots: npt.ArrayLike | None = None,
        fairness_weights: Mapping[int, float] | None = None,
        quadrature_order: int = 8,
        regularization: float = 0.0,
        name: str = "f_spline",
    ) -> None:
        if physical_dimension < 1:
            raise ValueError("physical_dimension must be positive.")
        if degree < 1 or num_control_points <= degree:
            raise ValueError("num_control_points must be greater than degree >= 1.")
        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive.")
        if regularization < 0.0:
            raise ValueError("regularization must be nonnegative.")

        self.space = lfs.BSplineSpace(
            num_parametric_dimensions=1,
            degree=(degree,),
            coefficients_shape=(num_control_points,),
            knots=None if knots is None else np.asarray(knots, dtype=float),
        )
        self.degree = int(degree)
        self.num_control_points = int(num_control_points)
        self.physical_dimension = int(physical_dimension)
        self.quadrature_order = int(quadrature_order)
        self.regularization = float(regularization)
        self.name = name
        self.constraints: list[FSplineConstraint] = []

        max_derivative_order = min(degree, _MAX_BACKEND_DERIVATIVE_ORDER)
        if fairness_weights is None:
            fairness_weights = {max_derivative_order: 1.0}
        self.fairness_weights = {
            int(order): float(weight)
            for order, weight in fairness_weights.items()
            if float(weight) != 0.0
        }
        for order, weight in self.fairness_weights.items():
            if order < 1 or order > max_derivative_order:
                raise ValueError(
                    f"fairness derivative order {order} is outside "
                    f"[1, {max_derivative_order}] for the current "
                    "lsdo_b_splines_cython backend."
                )
            if weight < 0.0:
                raise ValueError("fairness weights must be nonnegative.")
        if not self.fairness_weights and regularization == 0.0:
            raise ValueError(
                "at least one fairness or regularization term is required."
            )

    def add_point_constraint(
        self, parameter: float, target: Any, scale: float = 1.0
    ) -> None:
        """Add a complete point-position constraint."""
        self._validate_parameter(parameter)
        self._validate_vector_target(target)
        self.constraints.append(
            PointConstraint(parameter, _coerce_target(target), scale)
        )

    def add_derivative_constraint(
        self,
        parameter: float,
        derivative_order: int,
        target: Any,
        scale: float = 1.0,
    ) -> None:
        """Add a complete parametric-derivative constraint."""
        self._validate_parameter(parameter)
        max_order = min(self.degree, _MAX_BACKEND_DERIVATIVE_ORDER)
        if derivative_order < 1 or derivative_order > max_order:
            raise ValueError(
                f"derivative_order must lie in [1, {max_order}] for the "
                "current lsdo_b_splines_cython backend."
            )
        self._validate_vector_target(target)
        self.constraints.append(
            DerivativeConstraint(
                parameter,
                derivative_order,
                _coerce_target(target),
                scale,
            )
        )

    def add_tangent_angle_constraint(
        self, parameter: float, angle: Any, scale: float = 1.0
    ) -> None:
        """Add a planar tangent-angle constraint in radians."""
        self._require_planar("tangent-angle")
        self._validate_parameter(parameter)
        self.constraints.append(
            TangentAngleConstraint(parameter, _coerce_target(angle), scale)
        )

    def add_curvature_constraint(
        self, parameter: float, target: Any, scale: float = 1.0
    ) -> None:
        """Add a signed planar-curvature constraint."""
        self._require_planar("curvature")
        if self.degree < 2:
            raise ValueError("curvature constraints require degree >= 2.")
        self._validate_parameter(parameter)
        self.constraints.append(
            CurvatureConstraint(parameter, _coerce_target(target), scale)
        )

    def add_area_constraint(self, target: Any, scale: float = 1.0) -> None:
        """Add a signed planar-area constraint."""
        self._require_planar("area")
        self.constraints.append(AreaConstraint(_coerce_target(target), scale))

    def add_centroid_constraint(self, target: Any, scale: float = 1.0) -> None:
        """Add a planar area-centroid constraint."""
        self._require_planar("centroid")
        if _value_array(target).size != 2:
            raise ValueError("centroid target must contain two values.")
        self.constraints.append(CentroidConstraint(_coerce_target(target), scale))

    def solve(
        self,
        initial_control_points: npt.ArrayLike | None = None,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
    ) -> FSplineCurve:
        """Solve one curve through the shared variational assembly API."""
        system = VariationalSystem(name=self.name)
        assembly = self.assemble(
            system=system,
            initial_control_points=initial_control_points,
        )
        result = system.solve(
            tolerance=tolerance,
            max_iter=max_iter,
            print_status=print_status,
        )
        return assembly.finalize(result)

    def assemble(
        self,
        system: VariationalSystem,
        initial_control_points: npt.ArrayLike | None = None,
    ) -> FSplineAssembly:
        """Add this curve to a shared KKT system without running Newton."""
        try:
            csdl.get_current_recorder()
        except ValueError as error:
            raise RuntimeError(
                "FSplineProblem.assemble requires an active csdl.Recorder."
            ) from error
        if not self.constraints:
            raise ValueError("at least one form constraint is required.")

        coefficients = csdl.ImplicitVariable(
            value=self._initial_control_points(initial_control_points),
            name=f"{self.name}_coefficients",
        )
        function = lfs.Function(
            space=self.space,
            coefficients=coefficients,
            name=self.name,
        )
        curve = FSplineCurve(function, quadrature_order=self.quadrature_order)

        fairness_objective: Any = 0.0
        for derivative_order, weight in self.fairness_weights.items():
            fairness_objective = fairness_objective + weight * curve.fairness_energy(
                derivative_order
            )
        if self.regularization:
            fairness_objective = fairness_objective + self.regularization * csdl.sum(
                coefficients**2
            )

        residuals = [
            self._constraint_residual(curve, item) for item in self.constraints
        ]
        constraint_residual = csdl.concatenate(tuple(residuals))
        num_constraints = int(constraint_residual.size)
        num_coefficient_values = self.num_control_points * self.physical_dimension
        if num_constraints > num_coefficient_values:
            raise ValueError(
                f"received {num_constraints} scalar constraints for "
                f"{num_coefficient_values} coefficient values."
            )
        curve.constraint_residual = constraint_residual
        curve.fairness_objective = fairness_objective
        state_index = system.add_state(coefficients, name=self.name)
        system.add_objective(fairness_objective)
        constraint_handle = system.add_constraint(constraint_residual)
        return FSplineAssembly(
            curve=curve,
            state_index=state_index,
            constraint_handle=constraint_handle,
        )

    def _constraint_residual(
        self, curve: FSplineCurve, constraint: FSplineConstraint
    ) -> csdl.Variable:
        scale = self._validate_scale(constraint.scale)
        if isinstance(constraint, PointConstraint):
            residual = (
                curve.evaluate(constraint.parameter).reshape((self.physical_dimension,))
                - constraint.target
            )
        elif isinstance(constraint, DerivativeConstraint):
            residual = (
                curve.evaluate(
                    constraint.parameter,
                    derivative_order=constraint.derivative_order,
                ).reshape((self.physical_dimension,))
                - constraint.target
            )
        elif isinstance(constraint, TangentAngleConstraint):
            tangent = curve.evaluate(constraint.parameter, derivative_order=1).reshape(
                (2,)
            )
            residual = (
                tangent[1] * _cos(constraint.angle)
                - tangent[0] * _sin(constraint.angle)
            ).reshape((1,))
        elif isinstance(constraint, CurvatureConstraint):
            residual = (
                curve.curvature(constraint.parameter) - constraint.target
            ).reshape((1,))
        elif isinstance(constraint, AreaConstraint):
            residual = (curve.signed_area() - constraint.target).reshape((1,))
        elif isinstance(constraint, CentroidConstraint):
            residual = curve.centroid() - constraint.target
        else:
            raise TypeError(f"unsupported F-Spline constraint {type(constraint)!r}.")
        return (scale * residual).flatten()

    def _initial_control_points(
        self, initial_control_points: npt.ArrayLike | None
    ) -> np.ndarray:
        expected_shape = (self.num_control_points, self.physical_dimension)
        if initial_control_points is not None:
            initial = np.asarray(initial_control_points, dtype=float)
            if initial.shape != expected_shape:
                raise ValueError(
                    f"initial_control_points must have shape {expected_shape}, "
                    f"received {initial.shape}."
                )
            return initial.copy()

        start = np.zeros(self.physical_dimension)
        end = np.ones(self.physical_dimension)
        for constraint in self.constraints:
            if isinstance(constraint, PointConstraint):
                if np.isclose(constraint.parameter, 0.0):
                    start = _value_array(constraint.target).reshape(-1)
                elif np.isclose(constraint.parameter, 1.0):
                    end = _value_array(constraint.target).reshape(-1)

        knots = np.asarray(self.space.knots[self.space.knot_indices[0]], dtype=float)
        greville = np.asarray(
            [
                np.mean(knots[index + 1 : index + self.degree + 1])
                for index in range(self.num_control_points)
            ]
        )
        initial = (1.0 - greville[:, None]) * start[None, :] + greville[:, None] * end[
            None, :
        ]

        if self.physical_dimension == 2:
            chord_length = np.linalg.norm(end - start)
            tangent_distance = chord_length / max(self.num_control_points - 1, 1)
            for constraint in self.constraints:
                if not isinstance(constraint, TangentAngleConstraint):
                    continue
                angle = float(_value_array(constraint.angle).reshape(-1)[0])
                direction = np.array([np.cos(angle), np.sin(angle)])
                if np.isclose(constraint.parameter, 0.0):
                    initial[1] = start + tangent_distance * direction
                elif np.isclose(constraint.parameter, 1.0):
                    initial[-2] = end - tangent_distance * direction
        return initial

    @staticmethod
    def _validate_parameter(parameter: float) -> None:
        if not 0.0 <= float(parameter) <= 1.0:
            raise ValueError("constraint parameter must lie in [0, 1].")

    def _validate_vector_target(self, target: Any) -> None:
        size = int(_value_array(target).size)
        if size != self.physical_dimension:
            raise ValueError(
                f"constraint target must contain {self.physical_dimension} values, "
                f"received {size}."
            )

    def _require_planar(self, constraint_name: str) -> None:
        if self.physical_dimension != 2:
            raise ValueError(
                f"{constraint_name} constraints currently require physical_dimension=2."
            )

    @staticmethod
    def _validate_scale(scale: float) -> float:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("constraint scale must be finite and positive.")
        return scale
