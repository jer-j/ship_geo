"""Compatible transverse section problems for ship hull generation.

Section coordinates use ``(z, y)`` ordering.  The parameter runs from the
keel or centerplane endpoint to the waterline endpoint.  Consequently,
``FSplineCurve.signed_area()`` is the physical immersed half-sectional area
when the section is monotone in ``z``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np

from lsdo_geo.core.splines.f_spline import (
    FSplineAssembly,
    FSplineCurve,
    FSplineProblem,
)
from lsdo_geo.core.splines.variational import VariationalResult, VariationalSystem


class SectionTemplate(str, Enum):
    """Supported representation topologies for a half section."""

    ROUND_BILGE = "round_bilge"
    HARD_CHINE = "hard_chine"


def _vector2(first: Any, second: Any) -> Any:
    """Build a two-vector without severing CSDL derivative paths."""
    if not isinstance(first, csdl.Variable) and not isinstance(second, csdl.Variable):
        return np.array([first, second], dtype=float)
    anchor = first if isinstance(first, csdl.Variable) else second
    first_variable = first if isinstance(first, csdl.Variable) else 0.0 * anchor + first
    second_variable = (
        second if isinstance(second, csdl.Variable) else 0.0 * anchor + second
    )
    return csdl.concatenate(
        (first_variable.reshape((1,)), second_variable.reshape((1,)))
    )


def _repeated_chine_knots(
    num_control_points: int, degree: int, chine_parameter: float
) -> np.ndarray:
    """Construct an open knot vector with a ``C0`` interior hard chine."""
    if not 0.0 < chine_parameter < 1.0:
        raise ValueError("chine_parameter must lie strictly inside (0, 1).")
    interior_count = num_control_points - degree - 1
    if interior_count < degree:
        raise ValueError(
            "a C0 hard chine requires at least 2 * degree + 1 control points."
        )
    remaining = interior_count - degree
    before_count = remaining // 2
    after_count = remaining - before_count
    before = (
        np.linspace(0.0, chine_parameter, before_count + 2)[1:-1]
        if before_count
        else np.array([])
    )
    after = (
        np.linspace(chine_parameter, 1.0, after_count + 2)[1:-1]
        if after_count
        else np.array([])
    )
    return np.concatenate(
        (
            np.zeros(degree + 1),
            before,
            np.full(degree, chine_parameter),
            after,
            np.ones(degree + 1),
        )
    )


@dataclass
class SectionAssembly:
    """A named transverse section in a shared variational system."""

    station_parameter: float
    fspline: FSplineAssembly
    template: SectionTemplate

    @property
    def curve(self) -> FSplineCurve:
        """Unsolved curve graph, usable for coupled surface assembly."""
        return self.fspline.curve

    def finalize(self, result: VariationalResult) -> FSplineCurve:
        """Attach global KKT diagnostics and return the solved curve."""
        return self.fspline.finalize(result)


class SectionProblem:
    """Fairness-optimized round-bilge or hard-chine half section.

    Parameters may be ordinary constants or CSDL expressions.  ``draft``,
    ``half_breadth``, and ``half_area`` are therefore suitable direct outputs
    of longitudinal curves of form in the same global KKT system.
    """

    def __init__(
        self,
        station_parameter: float,
        draft: Any,
        half_breadth: Any,
        half_area: Any,
        keel_tangent_angle: Any = 0.0,
        waterline_tangent_angle: Any = 0.0,
        template: SectionTemplate = SectionTemplate.ROUND_BILGE,
        chine_parameter: float = 0.5,
        chine_point: Any | None = None,
        num_control_points: int = 8,
        degree: int = 3,
        fairness_weights: dict[int, float] | None = None,
        quadrature_order: int = 8,
        fit_parameters: Any | None = None,
        fit_points: Any | None = None,
        fit_weight: float = 0.0,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= float(station_parameter) <= 1.0:
            raise ValueError("station_parameter must lie in [0, 1].")
        self.station_parameter = float(station_parameter)
        self.template = SectionTemplate(template)
        self.name = name or f"section_{station_parameter:.4f}"
        self.draft = draft
        self.half_breadth = half_breadth
        if (fit_parameters is None) != (fit_points is None):
            raise ValueError("fit_parameters and fit_points must be supplied together.")
        if fit_weight < 0.0:
            raise ValueError("fit_weight must be nonnegative.")
        self.fit_parameters = None
        self.fit_points = fit_points
        self.fit_weight = float(fit_weight)
        if fit_parameters is not None:
            parameters = np.asarray(fit_parameters, dtype=float).reshape(-1)
            if (
                parameters.size < 2
                or np.any(np.diff(parameters) <= 0.0)
                or parameters[0] < 0.0
                or parameters[-1] > 1.0
            ):
                raise ValueError("fit_parameters must increase inside [0, 1].")
            shape = (
                fit_points.shape
                if isinstance(fit_points, csdl.Variable)
                else np.shape(fit_points)
            )
            if tuple(shape) != (parameters.size, 2):
                raise ValueError("fit_points must have shape (num_fit_parameters, 2).")
            self.fit_parameters = parameters
        knots = None
        if self.template is SectionTemplate.HARD_CHINE:
            if chine_point is None:
                raise ValueError("hard-chine sections require chine_point=(z, y).")
            knots = _repeated_chine_knots(num_control_points, degree, chine_parameter)
        self.problem = FSplineProblem(
            num_control_points=num_control_points,
            degree=degree,
            physical_dimension=2,
            knots=knots,
            fairness_weights=fairness_weights,
            quadrature_order=quadrature_order,
            name=self.name,
        )
        self.problem.add_point_constraint(0.0, _vector2(-draft, 0.0))
        self.problem.add_point_constraint(1.0, _vector2(0.0, half_breadth))
        self.problem.add_tangent_angle_constraint(0.0, keel_tangent_angle)
        self.problem.add_tangent_angle_constraint(1.0, waterline_tangent_angle)
        if self.template is SectionTemplate.HARD_CHINE:
            if isinstance(chine_point, csdl.Variable):
                target = chine_point
            else:
                target = np.asarray(chine_point, dtype=float)
            self.problem.add_point_constraint(chine_parameter, target)
        self.problem.add_area_constraint(half_area)

    def assemble(
        self,
        system: VariationalSystem,
        initial_control_points: np.ndarray | None = None,
    ) -> SectionAssembly:
        """Add the section to a shared KKT system without solving it."""
        if initial_control_points is None and self.fit_parameters is not None:
            target_values = (
                np.asarray(self.fit_points.value, dtype=float)
                if isinstance(self.fit_points, csdl.Variable)
                else np.asarray(self.fit_points, dtype=float)
            )
            basis = self.problem.space.compute_basis_matrix(
                self.fit_parameters[:, None]
            ).toarray()
            initial_control_points = np.linalg.lstsq(basis, target_values, rcond=None)[
                0
            ]
        assembly = SectionAssembly(
            station_parameter=self.station_parameter,
            fspline=self.problem.assemble(system, initial_control_points),
            template=self.template,
        )
        if self.fit_parameters is not None and self.fit_weight:
            values = assembly.curve.evaluate(self.fit_parameters)
            target = self.fit_points
            z_residual = (values[:, 0] - target[:, 0]) / self.draft
            y_residual = (values[:, 1] - target[:, 1]) / self.half_breadth
            system.add_objective(
                self.fit_weight * (csdl.sum(z_residual**2) + csdl.sum(y_residual**2))
            )
        return assembly


def collapsed_section(
    reference: FSplineCurve, name: str = "collapsed_section"
) -> FSplineCurve:
    """Create a centerplane section with the reference vertical distribution.

    This is useful at a pointed bow or stern.  The construction is explicit and
    remains differentiated with respect to the reference section coefficients.
    """
    coefficients = reference.coefficients
    z_coordinates = coefficients[:, 0].reshape((coefficients.shape[0], 1))
    zero_breadth = 0.0 * coefficients[:, 1].reshape((coefficients.shape[0], 1))
    collapsed_coefficients = csdl.concatenate((z_coordinates, zero_breadth), axis=1)
    return FSplineCurve(
        function=lfs.Function(reference.space, collapsed_coefficients, name=name),
        quadrature_order=reference.quadrature_order,
    )
