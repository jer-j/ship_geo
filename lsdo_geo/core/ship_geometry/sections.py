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


def _linear_polygon(
    problem: FSplineProblem,
    start: tuple[float, float],
    end: tuple[float, float],
) -> np.ndarray:
    """Return a Greville-spaced straight control polygon between two points.

    Used as a numeric starting guess when the endpoint targets are CSDL
    expressions whose values are not available during graph construction.
    """
    knots = np.asarray(
        problem.space.knots[problem.space.knot_indices[0]], dtype=float
    )
    greville = np.asarray(
        [
            np.mean(knots[index + 1 : index + problem.degree + 1])
            for index in range(problem.num_control_points)
        ]
    )
    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    return (1.0 - greville[:, None]) * start_array[None, :] + greville[
        :, None
    ] * end_array[None, :]


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
    topside: FSplineAssembly | None = None
    dome: FSplineAssembly | None = None

    @property
    def curve(self) -> FSplineCurve:
        """Unsolved underwater curve graph, usable for coupled surface assembly."""
        return self.fspline.curve

    @property
    def topside_curve(self) -> FSplineCurve | None:
        """Unsolved freeboard curve graph, when deck data was supplied."""
        return None if self.topside is None else self.topside.curve

    @property
    def dome_curve(self) -> FSplineCurve | None:
        """Unsolved sonar-dome curve graph, when dome data was supplied."""
        return None if self.dome is None else self.dome.curve

    def finalize(self, result: VariationalResult) -> FSplineCurve:
        """Attach global KKT diagnostics and return the solved underwater curve."""
        curve = self.fspline.finalize(result)
        if self.topside is not None:
            self.topside.finalize(result)
        if self.dome is not None:
            self.dome.finalize(result)
        return curve


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
        keel_half_breadth: Any = 0.0,
        dome_depth: Any | None = None,
        dome_half_area: Any | None = None,
        dome_fit_points: Any | None = None,
        dome_bottom_tangent_angle: Any | None = None,
        num_dome_control_points: int | None = None,
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
        interior_points: Any | None = None,
        deck_height: Any | None = None,
        deck_half_breadth: Any | None = None,
        deck_tangent_angle: Any | None = None,
        num_deck_control_points: int | None = None,
        deck_degree: int | None = None,
        deck_fairness_weights: dict[int, float] | None = None,
        initial_geometry_hint: dict[str, float] | None = None,
        name: str | None = None,
    ) -> None:
        if not 0.0 <= float(station_parameter) <= 1.0:
            raise ValueError("station_parameter must lie in [0, 1].")
        self.station_parameter = float(station_parameter)
        self.template = SectionTemplate(template)
        self.name = name or f"section_{station_parameter:.4f}"
        self.draft = draft
        self.half_breadth = half_breadth
        # The dome band is fit against the part of the reference section below
        # the blend line, on the same parameters as the main band. Its depth
        # scale is the section's deepest point, not the blend line the main
        # band starts from.
        self.dome_fit_points = dome_fit_points
        self.dome_depth = dome_depth
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
        # The lower endpoint is the band's own lower boundary. Where a sonar
        # dome is carried below, that boundary is the blend line rather than
        # the centreplane keel, so it has a half-breadth of its own.
        self.problem.add_point_constraint(0.0, _vector2(-draft, keel_half_breadth))
        self.problem.add_point_constraint(1.0, _vector2(0.0, half_breadth))
        self.problem.add_tangent_angle_constraint(0.0, keel_tangent_angle)
        self.problem.add_tangent_angle_constraint(1.0, waterline_tangent_angle)
        if self.template is SectionTemplate.HARD_CHINE:
            if isinstance(chine_point, csdl.Variable):
                target = chine_point
            else:
                target = np.asarray(chine_point, dtype=float)
            self.problem.add_point_constraint(chine_parameter, target)
        if interior_points is not None:
            for parameter, point in interior_points:
                value = float(parameter)
                if not 0.0 < value < 1.0:
                    raise ValueError(
                        "interior waypoint parameters must lie strictly inside (0, 1)."
                    )
                self.problem.add_point_constraint(value, point)
        self.problem.add_area_constraint(half_area)

        deck_given = (
            deck_height is not None
            or deck_half_breadth is not None
            or deck_tangent_angle is not None
        )
        if deck_given and (
            deck_height is None or deck_half_breadth is None or deck_tangent_angle is None
        ):
            raise ValueError(
                "deck_height, deck_half_breadth, and deck_tangent_angle must be "
                "supplied together."
            )
        # Sonar-dome band, carried below the blend line. It is a separate
        # F-Spline that ends on the same blend point and the same blend
        # tangent expressions the main band starts from, so position and
        # tangent agree by construction rather than by a fitted compromise.
        self.dome_problem: FSplineProblem | None = None
        if dome_depth is not None:
            self.dome_problem = FSplineProblem(
                num_control_points=num_dome_control_points or num_control_points,
                degree=degree,
                physical_dimension=2,
                fairness_weights=fairness_weights,
                quadrature_order=quadrature_order,
                name=f"{self.name}_dome",
            )
            self.dome_problem.add_point_constraint(0.0, _vector2(-dome_depth, 0.0))
            self.dome_problem.add_tangent_angle_constraint(
                0.0,
                0.5 * np.pi
                if dome_bottom_tangent_angle is None
                else dome_bottom_tangent_angle,
            )
            self.dome_problem.add_point_constraint(
                1.0, _vector2(-draft, keel_half_breadth)
            )
            self.dome_problem.add_tangent_angle_constraint(1.0, keel_tangent_angle)
            if dome_half_area is not None:
                self.dome_problem.add_area_constraint(dome_half_area)

        self.initial_geometry_hint = initial_geometry_hint or {}
        self.deck_problem: FSplineProblem | None = None
        if deck_given:
            self.deck_problem = FSplineProblem(
                num_control_points=num_deck_control_points or num_control_points,
                degree=deck_degree or degree,
                physical_dimension=2,
                fairness_weights=deck_fairness_weights or fairness_weights,
                quadrature_order=quadrature_order,
                name=f"{self.name}_topside",
            )
            self.deck_problem.add_point_constraint(0.0, _vector2(0.0, half_breadth))
            self.deck_problem.add_tangent_angle_constraint(
                0.0, waterline_tangent_angle
            )
            self.deck_problem.add_point_constraint(
                1.0, _vector2(deck_height, deck_half_breadth)
            )
            self.deck_problem.add_tangent_angle_constraint(1.0, deck_tangent_angle)

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
        hint = self.initial_geometry_hint
        if initial_control_points is None and {"draft", "half_breadth"} <= hint.keys():
            initial_control_points = _linear_polygon(
                self.problem,
                (-float(hint["draft"]), 0.0),
                (0.0, float(hint["half_breadth"])),
            )
        topside_initial = None
        if self.deck_problem is not None and {
            "half_breadth",
            "deck_height",
            "deck_half_breadth",
        } <= hint.keys():
            topside_initial = _linear_polygon(
                self.deck_problem,
                (0.0, float(hint["half_breadth"])),
                (float(hint["deck_height"]), float(hint["deck_half_breadth"])),
            )
        topside_assembly = (
            None
            if self.deck_problem is None
            else self.deck_problem.assemble(system, topside_initial)
        )
        dome_initial = None
        if self.dome_problem is not None and {"draft", "dome_depth"} <= hint.keys():
            dome_initial = _linear_polygon(
                self.dome_problem,
                (-float(hint["dome_depth"]), 0.0),
                (-float(hint["draft"]), float(hint.get("keel_half_breadth", 0.0))),
            )
        dome_assembly = (
            None
            if self.dome_problem is None
            else self.dome_problem.assemble(system, dome_initial)
        )
        assembly = SectionAssembly(
            station_parameter=self.station_parameter,
            fspline=self.problem.assemble(system, initial_control_points),
            template=self.template,
            topside=topside_assembly,
            dome=dome_assembly,
        )
        if self.fit_parameters is not None and self.fit_weight:
            values = assembly.curve.evaluate(self.fit_parameters)
            target = self.fit_points
            z_residual = (values[:, 0] - target[:, 0]) / self.draft
            y_residual = (values[:, 1] - target[:, 1]) / self.half_breadth
            system.add_objective(
                self.fit_weight * (csdl.sum(z_residual**2) + csdl.sum(y_residual**2))
            )
        # Without this the bulb is described only by two endpoints, two
        # tangents and an area, which pins how much section it encloses but
        # not its profile.
        if (
            dome_assembly is not None
            and self.dome_fit_points is not None
            and self.fit_parameters is not None
            and self.fit_weight
        ):
            values = dome_assembly.curve.evaluate(self.fit_parameters)
            target = self.dome_fit_points
            scale = self.draft if self.dome_depth is None else self.dome_depth
            z_residual = (values[:, 0] - target[:, 0]) / scale
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
