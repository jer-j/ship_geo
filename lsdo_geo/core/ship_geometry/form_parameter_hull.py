"""MIT-style naval form-parameter hull assembly.

The primary variables are naval-architecture particulars. Auxiliary sampled
distributions provide additional local shape freedom, but they enter as
least-squares objectives rather than replacing the exact displacement,
longitudinal center of buoyancy, beam, draft, and waterplane constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.variational import VariationalResult, VariationalSystem

from .form_curves import (
    FormCurve,
    FormCurveAssembly,
    FormCurveKind,
    FormCurveProblem,
)
from .hull import HullGeometry, SectionLoftAssembly, SectionLoftProblem


def _scalar_value(value: Any) -> float:
    """Return the current scalar value without changing its derivative path."""
    if isinstance(value, csdl.Variable):
        return float(np.asarray(value.value).reshape(-1)[0])
    return float(value)


def _sequence_values(values: Any, count: int, name: str) -> Any:
    """Validate a one-dimensional auxiliary target sequence."""
    if isinstance(values, csdl.Variable):
        if values.size != count:
            raise ValueError(f"{name} must contain {count} values.")
        return values.reshape((count,))
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count:
        raise ValueError(f"{name} must contain {count} values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _current_array(values: Any) -> np.ndarray:
    """Return current numeric values for initialization only."""
    if isinstance(values, csdl.Variable):
        return np.asarray(values.value, dtype=float).reshape(-1)
    return np.asarray(values, dtype=float).reshape(-1)


@dataclass(frozen=True)
class NavalHullParameters:
    """Primary hull particulars enforced by the implicit geometry system.

    ``lcb`` is measured from midships, positive toward increasing longitudinal
    parameter. ``displacement`` is geometric volume rather than mass.
    """

    length_between_perpendiculars: Any
    beam: Any
    draft: Any
    displacement: Any
    lcb: Any
    waterplane_coefficient: Any

    def validate_current_values(self) -> None:
        """Validate the current numeric values of design-dependent particulars."""
        for name in (
            "length_between_perpendiculars",
            "beam",
            "draft",
            "displacement",
        ):
            if _scalar_value(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        coefficient = _scalar_value(self.waterplane_coefficient)
        if not 0.0 < coefficient <= 1.0:
            raise ValueError("waterplane_coefficient must lie in (0, 1].")


@dataclass(frozen=True)
class LongitudinalFitTargets:
    """Auxiliary local shape observations used to select fair form curves."""

    station_parameters: npt.ArrayLike
    half_breadths: Any
    half_areas: Any
    drafts: Any
    deadrise_angles: Any
    flare_angles: Any
    maximum_beam_parameter: float
    maximum_draft_parameter: float

    def validated(self) -> LongitudinalFitTargets:
        """Return a normalized target object after deterministic checks."""
        stations = np.asarray(self.station_parameters, dtype=float).reshape(-1)
        if stations.size < 3:
            raise ValueError("at least three auxiliary stations are required.")
        if np.any(np.diff(stations) <= 0.0) or np.any(
            (stations <= 0.0) | (stations > 1.0)
        ):
            raise ValueError("stations must increase strictly inside (0, 1].")
        for value, name in (
            (self.maximum_beam_parameter, "maximum_beam_parameter"),
            (self.maximum_draft_parameter, "maximum_draft_parameter"),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        count = stations.size
        return LongitudinalFitTargets(
            station_parameters=stations,
            half_breadths=_sequence_values(self.half_breadths, count, "half_breadths"),
            half_areas=_sequence_values(self.half_areas, count, "half_areas"),
            drafts=_sequence_values(self.drafts, count, "drafts"),
            deadrise_angles=_sequence_values(
                self.deadrise_angles, count, "deadrise_angles"
            ),
            flare_angles=_sequence_values(self.flare_angles, count, "flare_angles"),
            maximum_beam_parameter=float(self.maximum_beam_parameter),
            maximum_draft_parameter=float(self.maximum_draft_parameter),
        )


@dataclass
class FormParameterHullGeometry:
    """Solved hull plus its naval-architecture longitudinal form curves."""

    hull: HullGeometry
    sectional_area_curve: FormCurve
    waterline_curve: FormCurve
    draft_curve: FormCurve
    deadrise_curve: FormCurve
    flare_curve: FormCurve
    primary_parameters: NavalHullParameters
    maximum_beam_parameter: float
    maximum_draft_parameter: float

    def recovered_primary_parameters(self) -> dict[str, csdl.Variable]:
        """Recover the exact naval particulars represented by the form curves."""
        length = self.primary_parameters.length_between_perpendiculars
        beam = self.primary_parameters.beam
        area_integral = self.sectional_area_curve.integral()
        return {
            "beam": 2.0 * self.waterline_curve.evaluate(self.maximum_beam_parameter),
            "draft": self.draft_curve.evaluate(self.maximum_draft_parameter),
            "displacement": 2.0 * length * area_integral,
            "lcb": length
            * (
                self.sectional_area_curve.integral(moment_order=1) / area_integral - 0.5
            ),
            "waterplane_coefficient": 2.0 * self.waterline_curve.integral() / beam,
        }


@dataclass
class FormParameterHullAssembly:
    """All curve, section, and surface states awaiting one global solve."""

    curve_assemblies: dict[FormCurveKind, FormCurveAssembly]
    section_loft: SectionLoftAssembly
    primary_parameters: NavalHullParameters
    maximum_beam_parameter: float
    maximum_draft_parameter: float

    def finalize(self, result: VariationalResult) -> FormParameterHullGeometry:
        """Attach shared KKT diagnostics to every solved geometry component."""
        curves = {
            kind: assembly.finalize(result)
            for kind, assembly in self.curve_assemblies.items()
        }
        return FormParameterHullGeometry(
            hull=self.section_loft.finalize(result),
            sectional_area_curve=curves[FormCurveKind.SECTIONAL_AREA],
            waterline_curve=curves[FormCurveKind.WATERLINE_HALF_BREADTH],
            draft_curve=curves[FormCurveKind.KEEL_PROFILE],
            deadrise_curve=curves[FormCurveKind.DEADRISE],
            flare_curve=curves[FormCurveKind.FLARE],
            primary_parameters=self.primary_parameters,
            maximum_beam_parameter=self.maximum_beam_parameter,
            maximum_draft_parameter=self.maximum_draft_parameter,
        )


class FormParameterHullProblem:
    """Construct a hull from exact particulars and fitted auxiliary functions.

    This follows the form-parameter hierarchy used by Nestoras' MIT hull
    modeler: principal particulars constrain longitudinal curves of form, those
    curves drive compatible transverse sections, and the sections define the
    surface. All implicit curve and section coefficients participate in one
    :class:`VariationalSystem` and one Newton solve.
    """

    def __init__(
        self,
        primary_parameters: NavalHullParameters,
        fit_targets: LongitudinalFitTargets,
        num_form_control_points: int = 10,
        num_section_control_points: int = 8,
        form_fit_weight: float = 1.0,
        form_fairness_weight: float = 1.0e-4,
        name: str = "form_parameter_hull",
    ) -> None:
        primary_parameters.validate_current_values()
        targets = fit_targets.validated()
        if num_form_control_points < targets.station_parameters.size + 1:
            raise ValueError(
                "num_form_control_points must exceed the auxiliary station count."
            )
        if form_fit_weight <= 0.0 or form_fairness_weight <= 0.0:
            raise ValueError("form fitting and fairness weights must be positive.")
        self.primary = primary_parameters
        self.targets = targets
        self.num_form_control_points = int(num_form_control_points)
        self.num_section_control_points = int(num_section_control_points)
        self.form_fit_weight = float(form_fit_weight)
        self.form_fairness_weight = float(form_fairness_weight)
        self.name = name

    def solve(
        self,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
    ) -> FormParameterHullGeometry:
        """Assemble and solve all form curves, sections, and the surface once."""
        system = VariationalSystem(self.name)
        assembly = self.assemble(system)
        result = system.solve(tolerance, max_iter, print_status)
        return assembly.finalize(result)

    def assemble(self, system: VariationalSystem) -> FormParameterHullAssembly:
        """Register the complete form-parameter hull without invoking Newton."""
        length = self.primary.length_between_perpendiculars
        beam = self.primary.beam
        draft = self.primary.draft
        displacement = self.primary.displacement
        lcb = self.primary.lcb
        cwp = self.primary.waterplane_coefficient
        stations = np.asarray(self.targets.station_parameters)
        mean_volume_parameter = 0.5 + lcb / length
        problems: dict[FormCurveKind, FormCurveProblem] = {}
        area_problem = self._problem(FormCurveKind.SECTIONAL_AREA)
        area_problem.add_value_constraint(0.0, 0.0)
        area_problem.add_value_constraint(1.0, self.targets.half_areas[-1])
        area_problem.add_integral_constraint(displacement / (2.0 * length))
        area_problem.add_integral_constraint(
            displacement / (2.0 * length) * mean_volume_parameter,
            moment_order=1,
        )
        problems[FormCurveKind.SECTIONAL_AREA] = area_problem

        waterline_problem = self._problem(FormCurveKind.WATERLINE_HALF_BREADTH)
        waterline_problem.add_value_constraint(0.0, 0.0)
        waterline_problem.add_value_constraint(
            self.targets.maximum_beam_parameter, 0.5 * beam
        )
        waterline_problem.add_derivative_constraint(
            self.targets.maximum_beam_parameter, 0.0
        )
        waterline_problem.add_value_constraint(1.0, self.targets.half_breadths[-1])
        waterline_problem.add_integral_constraint(0.5 * cwp * beam)
        problems[FormCurveKind.WATERLINE_HALF_BREADTH] = waterline_problem

        draft_problem = self._problem(FormCurveKind.KEEL_PROFILE)
        draft_problem.add_value_constraint(self.targets.maximum_draft_parameter, draft)
        draft_problem.add_derivative_constraint(
            self.targets.maximum_draft_parameter, 0.0
        )
        draft_problem.add_value_constraint(1.0, self.targets.drafts[-1])
        problems[FormCurveKind.KEEL_PROFILE] = draft_problem

        for kind, values in (
            (FormCurveKind.DEADRISE, self.targets.deadrise_angles),
            (FormCurveKind.FLARE, self.targets.flare_angles),
        ):
            problem = self._problem(kind)
            for station_index, station in enumerate(stations):
                problem.add_value_constraint(station, values[station_index])
            problems[kind] = problem

        target_map = {
            FormCurveKind.SECTIONAL_AREA: self.targets.half_areas,
            FormCurveKind.WATERLINE_HALF_BREADTH: self.targets.half_breadths,
            FormCurveKind.KEEL_PROFILE: self.targets.drafts,
            FormCurveKind.DEADRISE: self.targets.deadrise_angles,
            FormCurveKind.FLARE: self.targets.flare_angles,
        }
        assemblies = {
            kind: problem.assemble(
                system,
                self._initial_form_coefficients(
                    problem,
                    stations,
                    target_map[kind],
                    zero_at_bow=kind
                    in (
                        FormCurveKind.SECTIONAL_AREA,
                        FormCurveKind.WATERLINE_HALF_BREADTH,
                    ),
                ),
            )
            for kind, problem in problems.items()
        }
        scales = {
            FormCurveKind.SECTIONAL_AREA: 1.0
            / (_scalar_value(beam) * _scalar_value(draft)),
            FormCurveKind.WATERLINE_HALF_BREADTH: 1.0 / _scalar_value(beam),
            FormCurveKind.KEEL_PROFILE: 1.0 / _scalar_value(draft),
            FormCurveKind.DEADRISE: 1.0,
            FormCurveKind.FLARE: 1.0,
        }
        for kind, assembly in assemblies.items():
            residual = scales[kind] * (
                assembly.curve.evaluate(stations).reshape((stations.size,))
                - target_map[kind]
            )
            system.add_objective(self.form_fit_weight * csdl.sum(residual**2))

        curves = {kind: assembly.curve for kind, assembly in assemblies.items()}
        section_problem = SectionLoftProblem(
            length=length,
            station_parameters=stations,
            drafts=curves[FormCurveKind.KEEL_PROFILE]
            .evaluate(stations)
            .reshape((stations.size,)),
            half_breadths=curves[FormCurveKind.WATERLINE_HALF_BREADTH]
            .evaluate(stations)
            .reshape((stations.size,)),
            half_areas=curves[FormCurveKind.SECTIONAL_AREA]
            .evaluate(stations)
            .reshape((stations.size,)),
            keel_tangent_angles=curves[FormCurveKind.DEADRISE]
            .evaluate(stations)
            .reshape((stations.size,)),
            waterline_tangent_angles=curves[FormCurveKind.FLARE]
            .evaluate(stations)
            .reshape((stations.size,)),
            num_section_control_points=self.num_section_control_points,
            pointed_ends=(True, stations[-1] < 1.0),
            name=self.name,
        )
        section_assembly = section_problem.assemble(system)
        return FormParameterHullAssembly(
            curve_assemblies=assemblies,
            section_loft=section_assembly,
            primary_parameters=self.primary,
            maximum_beam_parameter=self.targets.maximum_beam_parameter,
            maximum_draft_parameter=self.targets.maximum_draft_parameter,
        )

    def _problem(self, kind: FormCurveKind) -> FormCurveProblem:
        """Create one consistently scaled longitudinal form-curve problem."""
        return FormCurveProblem(
            kind,
            num_control_points=self.num_form_control_points,
            fairness_weights={2: self.form_fairness_weight},
            regularization=1.0e-12,
            name=f"{self.name}_{kind.value}",
        )

    @staticmethod
    def _initial_form_coefficients(
        problem: FormCurveProblem,
        stations: np.ndarray,
        targets: Any,
        zero_at_bow: bool,
    ) -> np.ndarray:
        """Interpolate auxiliary observations at the control-point abscissae."""
        knots = np.asarray(problem.space.knots[problem.space.knot_indices[0]])
        greville = np.asarray(
            [
                np.mean(knots[index + 1 : index + problem.degree + 1])
                for index in range(problem.num_control_points)
            ]
        )
        target_values = _current_array(targets)
        parameters = stations
        values = target_values
        if zero_at_bow:
            parameters = np.concatenate(([0.0], parameters))
            values = np.concatenate(([0.0], target_values))
        return np.interp(greville, parameters, values)
