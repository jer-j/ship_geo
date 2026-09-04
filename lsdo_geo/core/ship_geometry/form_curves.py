"""Longitudinal curves of form for first-principles ship geometry.

The independent coordinate is the fixed nondimensional longitudinal parameter
``v`` on :math:`[0, 1]`.  The ordinate is a dimensional or nondimensional ship
form quantity such as sectional area, half-breadth, keel height, deadrise, or
flare.  Only the ordinate coefficients are implicit states, which avoids
introducing redundant longitudinal-coordinate degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
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


class FormCurveKind(str, Enum):
    """Naval-architecture meaning assigned to a longitudinal distribution."""

    SECTIONAL_AREA = "sectional_area"
    WATERLINE_HALF_BREADTH = "waterline_half_breadth"
    KEEL_PROFILE = "keel_profile"
    DECK_EDGE = "deck_edge"
    DECK_HEIGHT = "deck_height"
    DECK_TANGENT = "deck_tangent"
    BULGE_HALF_BREADTH = "bulge_half_breadth"
    BULGE_HEIGHT = "bulge_height"
    CHINE = "chine"
    DEADRISE = "deadrise"
    FLARE = "flare"
    FULLNESS = "fullness"


def _coerce_target(value: Any) -> Any:
    if isinstance(value, csdl.Variable):
        return value
    array = np.asarray(value, dtype=float)
    return float(array) if array.ndim == 0 else array


def _current_scalar(value: Any) -> float | None:
    """Return a scalar's current value, or ``None`` under a deferred recorder."""
    if isinstance(value, csdl.Variable):
        if value.value is None:
            return None
        return float(np.asarray(value.value).reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def clustered_open_knots(
    num_control_points: int,
    degree: int,
    breakpoint: float,
    forward_fraction: float = 0.5,
) -> np.ndarray:
    """Return an open knot vector with interior knots massed forward.

    A uniform knot vector distributes polynomial freedom evenly along the
    ship, which starves any short feature. With ten control points at degree
    three the first interior knot falls near ``v = 0.14``, so a sonar dome
    ending at ``v = 0.12`` lies entirely inside one cubic span and cannot
    show a bulb that rises and falls. Concentrating interior knots ahead of
    ``breakpoint`` adds spans exactly where the shape changes fastest,
    without adding coefficients elsewhere.

    Parameters
    ----------
    num_control_points
        Coefficient count of the target space.
    degree
        Polynomial degree.
    breakpoint
        Parameter separating the clustered forward region from the rest. A
        knot is placed exactly here, so the two regions meet at a knot.
    forward_fraction
        Share of the interior knots placed ahead of ``breakpoint``.
    """
    if not 0.0 < breakpoint < 1.0:
        raise ValueError("breakpoint must lie strictly inside (0, 1).")
    if not 0.0 < forward_fraction < 1.0:
        raise ValueError("forward_fraction must lie strictly inside (0, 1).")
    interior_count = num_control_points - degree - 1
    if interior_count < 1:
        raise ValueError("clustered knots require num_control_points > degree + 1.")
    forward_count = int(round(forward_fraction * (interior_count - 1)))
    forward_count = max(0, min(forward_count, interior_count - 1))
    aft_count = interior_count - 1 - forward_count
    forward = (
        np.linspace(0.0, breakpoint, forward_count + 2)[1:-1]
        if forward_count
        else np.empty(0)
    )
    aft = (
        np.linspace(breakpoint, 1.0, aft_count + 2)[1:-1]
        if aft_count
        else np.empty(0)
    )
    interior = np.concatenate((forward, [float(breakpoint)], aft))
    return np.concatenate(
        (np.zeros(degree + 1), interior, np.ones(degree + 1))
    )


@dataclass(frozen=True)
class _DistributionConstraint:
    kind: str
    target: Any
    parameter: float | None = None
    derivative_order: int = 0
    moment_order: int = 0
    scale: float = 1.0


@dataclass
class FormCurve:
    """Solved scalar B-spline distribution with naval-architecture semantics."""

    function: lfs.Function
    kind: FormCurveKind
    quadrature_order: int = 8
    lagrange_multipliers: csdl.Variable | None = None
    constraint_residual: csdl.Variable | None = None
    stationarity_residual: csdl.Variable | None = None
    fairness_objective: csdl.Variable | None = None

    @property
    def coefficients(self) -> csdl.Variable:
        """CSDL coefficient state."""
        return self.function.coefficients

    @property
    def space(self) -> lfs.BSplineSpace:
        """Underlying B-spline space."""
        return self.function.space

    def evaluate(
        self,
        parameters: float | npt.ArrayLike,
        derivative_order: int = 0,
    ) -> csdl.Variable:
        """Evaluate the distribution or a parametric derivative."""
        derivative_order = int(derivative_order)
        if derivative_order < 0 or derivative_order > min(2, self.space.degree[0]):
            raise ValueError("derivative_order must lie between zero and two.")
        coordinates = np.asarray(parameters, dtype=float).reshape((-1, 1))
        if np.any(coordinates < 0.0) or np.any(coordinates > 1.0):
            raise ValueError("parameters must lie in [0, 1].")
        return self.function.evaluate(
            coordinates,
            parametric_derivative_orders=(derivative_order,),
        )

    def quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed composite Gauss-Legendre nodes and weights."""
        points, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        indices = self.space.knot_indices[0]
        knots = np.asarray(self.space.knots[indices], dtype=float)
        degree = int(self.space.degree[0])
        active = np.unique(knots[degree:-degree])
        global_points: list[np.ndarray] = []
        global_weights: list[np.ndarray] = []
        for lower, upper in pairwise(active):
            if upper <= lower:
                continue
            half = 0.5 * (upper - lower)
            global_points.append(0.5 * (upper + lower) + half * points)
            global_weights.append(half * weights)
        return np.concatenate(global_points), np.concatenate(global_weights)

    def integral(self, moment_order: int = 0) -> csdl.Variable:
        """Integrate :math:`v^m f(v)` over the unit interval."""
        if moment_order < 0:
            raise ValueError("moment_order must be nonnegative.")
        points, weights = self.quadrature()
        values = self.evaluate(points).reshape((points.size,))
        return csdl.sum(weights * points**moment_order * values)

    def fairness_energy(self, derivative_order: int = 2) -> csdl.Variable:
        """Return an integrated squared-derivative fairness energy."""
        points, weights = self.quadrature()
        values = self.evaluate(points, derivative_order).reshape((points.size,))
        return csdl.sum(weights * values**2)


@dataclass
class FormCurveAssembly:
    """A form-curve contribution awaiting the shared global solve."""

    curve: FormCurve
    state_index: int
    constraint_handle: ConstraintHandle | None

    def finalize(self, result: VariationalResult) -> FormCurve:
        """Attach diagnostics from the global KKT solution."""
        self.curve.stationarity_residual = result.stationarity_residuals[
            self.state_index
        ]
        if (
            result.constraint_residual is not None
            and self.constraint_handle is not None
        ):
            self.curve.constraint_residual = result.constraint_residual[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        if (
            result.lagrange_multipliers is not None
            and self.constraint_handle is not None
        ):
            self.curve.lagrange_multipliers = result.lagrange_multipliers[
                self.constraint_handle.start : self.constraint_handle.stop
            ]
        return self.curve


class FormCurveProblem:
    """Constrained variational problem for one longitudinal curve of form."""

    def __init__(
        self,
        kind: FormCurveKind,
        num_control_points: int = 8,
        degree: int = 3,
        knots: npt.ArrayLike | None = None,
        fairness_weights: dict[int, float] | None = None,
        quadrature_order: int = 8,
        regularization: float = 0.0,
        name: str | None = None,
    ) -> None:
        if degree < 1 or num_control_points <= degree:
            raise ValueError("num_control_points must exceed degree >= 1.")
        if quadrature_order < 1:
            raise ValueError("quadrature_order must be positive.")
        if regularization < 0.0:
            raise ValueError("regularization must be nonnegative.")
        self.kind = FormCurveKind(kind)
        self.name = name or self.kind.value
        self.num_control_points = int(num_control_points)
        self.degree = int(degree)
        self.quadrature_order = int(quadrature_order)
        self.regularization = float(regularization)
        self.space = lfs.BSplineSpace(
            num_parametric_dimensions=1,
            degree=(degree,),
            coefficients_shape=(num_control_points,),
            knots=None if knots is None else np.asarray(knots, dtype=float),
        )
        if fairness_weights is None:
            fairness_weights = {min(2, degree): 1.0}
        self.fairness_weights = {
            int(order): float(weight)
            for order, weight in fairness_weights.items()
            if float(weight) != 0.0
        }
        for order, weight in self.fairness_weights.items():
            if order < 1 or order > min(2, degree):
                raise ValueError("fairness derivative orders must lie in [1, 2].")
            if weight < 0.0:
                raise ValueError("fairness weights must be nonnegative.")
        if not self.fairness_weights and regularization == 0.0:
            raise ValueError("at least one objective term is required.")
        self.constraints: list[_DistributionConstraint] = []

    def add_value_constraint(
        self, parameter: float, target: Any, scale: float = 1.0
    ) -> None:
        """Constrain the distribution value at a fixed parameter."""
        self._validate_parameter(parameter)
        self.constraints.append(
            _DistributionConstraint(
                "value", _coerce_target(target), parameter=parameter, scale=scale
            )
        )

    def add_derivative_constraint(
        self,
        parameter: float,
        target: Any,
        derivative_order: int = 1,
        scale: float = 1.0,
    ) -> None:
        """Constrain a first or second parametric derivative."""
        self._validate_parameter(parameter)
        if derivative_order < 1 or derivative_order > min(2, self.degree):
            raise ValueError("derivative_order must lie in [1, 2].")
        self.constraints.append(
            _DistributionConstraint(
                "derivative",
                _coerce_target(target),
                parameter=parameter,
                derivative_order=derivative_order,
                scale=scale,
            )
        )

    def add_integral_constraint(
        self, target: Any, moment_order: int = 0, scale: float = 1.0
    ) -> None:
        """Constrain an integral or longitudinal moment."""
        if moment_order < 0:
            raise ValueError("moment_order must be nonnegative.")
        self.constraints.append(
            _DistributionConstraint(
                "integral",
                _coerce_target(target),
                moment_order=moment_order,
                scale=scale,
            )
        )

    def assemble(
        self,
        system: VariationalSystem,
        initial_coefficients: npt.ArrayLike | None = None,
    ) -> FormCurveAssembly:
        """Add this distribution to a shared KKT system."""
        coefficients = csdl.ImplicitVariable(
            value=self._initial_coefficients(initial_coefficients),
            name=f"{self.name}_coefficients",
        )
        curve = FormCurve(
            function=lfs.Function(self.space, coefficients, name=self.name),
            kind=self.kind,
            quadrature_order=self.quadrature_order,
        )
        objective: Any = 0.0
        for order, weight in self.fairness_weights.items():
            objective = objective + weight * curve.fairness_energy(order)
        if self.regularization:
            objective = objective + self.regularization * csdl.sum(coefficients**2)
        residuals: list[csdl.Variable] = []
        for constraint in self.constraints:
            scale = self._validate_scale(constraint.scale)
            if constraint.kind == "value":
                value = curve.evaluate(constraint.parameter).reshape((1,))
            elif constraint.kind == "derivative":
                value = curve.evaluate(
                    constraint.parameter, constraint.derivative_order
                ).reshape((1,))
            else:
                value = curve.integral(constraint.moment_order).reshape((1,))
            residuals.append(scale * (value - constraint.target))
        curve.fairness_objective = objective
        state_index = system.add_state(coefficients, name=self.name)
        system.add_objective(objective)
        handle = None
        if residuals:
            residual = (
                residuals[0]
                if len(residuals) == 1
                else csdl.concatenate(tuple(residuals))
            )
            curve.constraint_residual = residual
            handle = system.add_constraint(residual)
        return FormCurveAssembly(curve, state_index, handle)

    def solve(
        self,
        initial_coefficients: npt.ArrayLike | None = None,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
    ) -> FormCurve:
        """Solve this curve in a one-state variational system."""
        system = VariationalSystem(name=self.name)
        assembly = self.assemble(system, initial_coefficients)
        return assembly.finalize(system.solve(tolerance, max_iter, print_status))

    def _initial_coefficients(
        self, initial_coefficients: npt.ArrayLike | None
    ) -> np.ndarray:
        if initial_coefficients is not None:
            initial = np.asarray(initial_coefficients, dtype=float)
            if initial.shape != (self.num_control_points,):
                raise ValueError(
                    "initial_coefficients must have shape "
                    f"({self.num_control_points},)."
                )
            return initial.copy()
        start = 0.0
        end = 0.0
        for constraint in self.constraints:
            if constraint.kind != "value":
                continue
            target = _current_scalar(constraint.target)
            if target is None:
                continue
            if np.isclose(constraint.parameter, 0.0):
                start = target
            elif np.isclose(constraint.parameter, 1.0):
                end = target
        knots = np.asarray(self.space.knots[self.space.knot_indices[0]])
        greville = np.array(
            [
                np.mean(knots[index + 1 : index + self.degree + 1])
                for index in range(self.num_control_points)
            ]
        )
        return (1.0 - greville) * start + greville * end

    @staticmethod
    def _validate_parameter(parameter: float) -> None:
        if not 0.0 <= float(parameter) <= 1.0:
            raise ValueError("parameter must lie in [0, 1].")

    @staticmethod
    def _validate_scale(scale: float) -> float:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("constraint scale must be finite and positive.")
        return scale
