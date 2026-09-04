"""MIT-style naval form-parameter hull assembly.

The primary variables are naval-architecture particulars. Auxiliary sampled
distributions provide additional local shape freedom, but they enter as
least-squares objectives rather than replacing the exact displacement,
longitudinal center of buoyancy, beam, draft, and waterplane constraints.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from .surfaces import LongitudinalLoftRegion


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
        # Reshaping is an operation, and an operation's result carries no value
        # under a deferred recorder. Keeping an already-correct shape preserves
        # the numeric values that initial guesses interpolate from.
        if tuple(values.shape) == (count,):
            return values
        return values.reshape((count,))
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != count:
        raise ValueError(f"{name} must contain {count} values.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values.")
    return array


def _interpolation_weights(stations: np.ndarray, station: float) -> np.ndarray:
    """Return linear-interpolation weights over ``stations`` at ``station``."""
    weights = np.zeros(stations.size)
    if station <= stations[0]:
        weights[0] = 1.0
    elif station >= stations[-1]:
        weights[-1] = 1.0
    else:
        upper = int(np.searchsorted(stations, station))
        lower = upper - 1
        span = stations[upper] - stations[lower]
        fraction = (station - stations[lower]) / span
        weights[lower] = 1.0 - fraction
        weights[upper] = fraction
    return weights


def _weighted_target(weights: np.ndarray, values: Any) -> Any:
    """Interpolate an auxiliary target, keeping design variables differentiable."""
    if isinstance(values, csdl.Variable):
        return csdl.vdot(
            csdl.Variable(value=weights, name="interpolation_weights"), values
        ).reshape((1,))
    return np.asarray(
        [float(np.dot(weights, np.asarray(values, dtype=float).reshape(-1)))]
    )


def _stack_pair(first: Any, second: Any) -> Any:
    """Build a two-vector from scalars that may be CSDL expressions."""
    if isinstance(first, csdl.Variable) or isinstance(second, csdl.Variable):
        anchor = first if isinstance(first, csdl.Variable) else second
        first_variable = (
            first if isinstance(first, csdl.Variable) else 0.0 * anchor + float(first)
        )
        second_variable = (
            second
            if isinstance(second, csdl.Variable)
            else 0.0 * anchor + float(second)
        )
        return csdl.concatenate(
            (first_variable.reshape((1,)), second_variable.reshape((1,)))
        )
    return np.asarray([float(first), float(second)])


def _current_array(values: Any) -> np.ndarray:
    """Return current numeric values for initialization only."""
    if isinstance(values, csdl.Variable):
        if values.value is None:
            raise ValueError(
                "an auxiliary target has no value available for initialization. "
                "Supply design variables as csdl.Variable inputs carrying an "
                "initial value rather than as computed expressions."
            )
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
    deck_half_breadths: Any | None = None
    deck_heights: Any | None = None
    deck_tangent_angles: Any | None = None
    bulge_half_breadths: Any | None = None
    bulge_heights: Any | None = None
    bulge_parameters: Any | None = None
    blend_depths: Any | None = None
    blend_half_breadths: Any | None = None
    blend_tangent_angles: Any | None = None
    dome_depths: Any | None = None
    dome_half_areas: Any | None = None

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
        deck_fields = (
            self.deck_half_breadths,
            self.deck_heights,
            self.deck_tangent_angles,
        )
        if any(field is not None for field in deck_fields) and any(
            field is None for field in deck_fields
        ):
            raise ValueError(
                "deck_half_breadths, deck_heights, and deck_tangent_angles must be "
                "supplied together."
            )
        bulge_fields = (
            self.bulge_half_breadths,
            self.bulge_heights,
            self.bulge_parameters,
        )
        if any(field is not None for field in bulge_fields) and any(
            field is None for field in bulge_fields
        ):
            raise ValueError(
                "bulge_half_breadths, bulge_heights, and bulge_parameters must be "
                "supplied together."
            )
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
            deck_half_breadths=(
                None
                if self.deck_half_breadths is None
                else _sequence_values(
                    self.deck_half_breadths, count, "deck_half_breadths"
                )
            ),
            deck_heights=(
                None
                if self.deck_heights is None
                else _sequence_values(self.deck_heights, count, "deck_heights")
            ),
            deck_tangent_angles=(
                None
                if self.deck_tangent_angles is None
                else _sequence_values(
                    self.deck_tangent_angles, count, "deck_tangent_angles"
                )
            ),
            bulge_half_breadths=(
                None
                if self.bulge_half_breadths is None
                else _sequence_values(
                    self.bulge_half_breadths, count, "bulge_half_breadths"
                )
            ),
            bulge_heights=(
                None
                if self.bulge_heights is None
                else _sequence_values(self.bulge_heights, count, "bulge_heights")
            ),
            bulge_parameters=(
                None
                if self.bulge_parameters is None
                else _sequence_values(self.bulge_parameters, count, "bulge_parameters")
            ),
            blend_depths=(
                None
                if self.blend_depths is None
                else _sequence_values(self.blend_depths, count, "blend_depths")
            ),
            blend_tangent_angles=(
                None
                if self.blend_tangent_angles is None
                else _sequence_values(
                    self.blend_tangent_angles, count, "blend_tangent_angles"
                )
            ),
            blend_half_breadths=(
                None
                if self.blend_half_breadths is None
                else _sequence_values(
                    self.blend_half_breadths, count, "blend_half_breadths"
                )
            ),
            dome_depths=(
                None
                if self.dome_depths is None
                else _sequence_values(self.dome_depths, count, "dome_depths")
            ),
            dome_half_areas=(
                None
                if self.dome_half_areas is None
                else _sequence_values(self.dome_half_areas, count, "dome_half_areas")
            ),
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
    fullness_curve: FormCurve | None = None
    deck_edge_curve: FormCurve | None = None
    deck_height_curve: FormCurve | None = None
    deck_tangent_curve: FormCurve | None = None
    bulge_breadth_curve: FormCurve | None = None
    bulge_height_curve: FormCurve | None = None
    blend_breadth_curve: FormCurve | None = None
    dome_depth_curve: FormCurve | None = None
    dome_area_curve: FormCurve | None = None

    def recovered_primary_parameters(self) -> dict[str, csdl.Variable]:
        """Recover the exact naval particulars represented by the form curves."""
        length = self.primary_parameters.length_between_perpendiculars
        beam = self.primary_parameters.beam
        area_integral = self.sectional_area_curve.integral()
        return {
            "length_between_perpendiculars": 0.0 * area_integral + length,
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
            fullness_curve=curves.get(FormCurveKind.FULLNESS),
            deck_edge_curve=curves.get(FormCurveKind.DECK_EDGE),
            deck_height_curve=curves.get(FormCurveKind.DECK_HEIGHT),
            deck_tangent_curve=curves.get(FormCurveKind.DECK_TANGENT),
            bulge_breadth_curve=curves.get(FormCurveKind.BULGE_HALF_BREADTH),
            bulge_height_curve=curves.get(FormCurveKind.BULGE_HEIGHT),
            blend_breadth_curve=curves.get(FormCurveKind.BLEND_HALF_BREADTH),
            dome_depth_curve=curves.get(FormCurveKind.DOME_DEPTH),
            dome_area_curve=curves.get(FormCurveKind.DOME_AREA),
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
        num_deck_control_points: int | None = None,
        num_dome_control_points: int | None = None,
        section_station_parameters: npt.ArrayLike | None = None,
        section_fit_parameters: npt.ArrayLike | None = None,
        section_fit_points: Any | None = None,
        section_fit_weight: float = 0.0,
        form_fit_weight: float = 1.0,
        form_fairness_weight: float = 1.0e-4,
        x_origin: Any = 0.0,
        longitudinal_regions: Sequence[LongitudinalLoftRegion] | None = None,
        use_fullness_curve: bool = False,
        form_knots: npt.ArrayLike | None = None,
        transom_x_offsets: Any | None = None,
        transom_deck_x_offsets: Any | None = None,
        bulge_depth_threshold: float = 0.15,
        dome_depth_threshold: float = 0.05,
        name: str = "form_parameter_hull",
    ) -> None:
        primary_parameters.validate_current_values()
        targets = fit_targets.validated()
        if num_form_control_points < 6:
            raise ValueError(
                "num_form_control_points must be at least six to support the "
                "waterline form constraints."
            )
        if form_fit_weight <= 0.0 or form_fairness_weight <= 0.0:
            raise ValueError("form fitting and fairness weights must be positive.")
        self.primary = primary_parameters
        self.targets = targets
        self.num_form_control_points = int(num_form_control_points)
        self.num_section_control_points = int(num_section_control_points)
        self.num_deck_control_points = (
            None if num_deck_control_points is None else int(num_deck_control_points)
        )
        self.num_dome_control_points = (
            None if num_dome_control_points is None else int(num_dome_control_points)
        )
        section_stations = (
            targets.station_parameters
            if section_station_parameters is None
            else np.asarray(section_station_parameters, dtype=float).reshape(-1)
        )
        if (
            section_stations.size < 2
            or np.any(np.diff(section_stations) <= 0.0)
            or np.any((section_stations <= 0.0) | (section_stations > 1.0))
        ):
            raise ValueError("section_station_parameters must increase inside (0, 1].")
        self.section_station_parameters = section_stations
        self.section_fit_parameters = section_fit_parameters
        self.section_fit_points = section_fit_points
        self.section_fit_weight = float(section_fit_weight)
        self.form_fit_weight = float(form_fit_weight)
        self.form_fairness_weight = float(form_fairness_weight)
        self.x_origin = x_origin
        self.longitudinal_regions = (
            None if longitudinal_regions is None else tuple(longitudinal_regions)
        )
        self.use_fullness_curve = bool(use_fullness_curve)
        self.model_deck = targets.deck_half_breadths is not None
        self.model_bulge = targets.bulge_half_breadths is not None
        self.model_dome_band = targets.dome_depths is not None
        self.form_knots = (
            None if form_knots is None else np.asarray(form_knots, dtype=float)
        )
        self.transom_x_offsets = transom_x_offsets
        self.transom_deck_x_offsets = transom_deck_x_offsets
        self.bulge_depth_threshold = float(bulge_depth_threshold)
        self.dome_depth_threshold = float(dome_depth_threshold)
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
        fit_stations = np.asarray(self.targets.station_parameters)
        section_stations = self.section_station_parameters
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

        for kind in (FormCurveKind.DEADRISE, FormCurveKind.FLARE):
            problems[kind] = self._problem(kind)

        target_map = {
            FormCurveKind.SECTIONAL_AREA: self.targets.half_areas,
            FormCurveKind.WATERLINE_HALF_BREADTH: self.targets.half_breadths,
            FormCurveKind.KEEL_PROFILE: self.targets.drafts,
            FormCurveKind.DEADRISE: self.targets.deadrise_angles,
            FormCurveKind.FLARE: self.targets.flare_angles,
        }
        zero_at_bow_kinds = [
            FormCurveKind.SECTIONAL_AREA,
            FormCurveKind.WATERLINE_HALF_BREADTH,
        ]
        if self.model_deck:
            deck_edge_problem = self._problem(FormCurveKind.DECK_EDGE)
            deck_edge_problem.add_value_constraint(0.0, 0.0)
            problems[FormCurveKind.DECK_EDGE] = deck_edge_problem
            problems[FormCurveKind.DECK_HEIGHT] = self._problem(
                FormCurveKind.DECK_HEIGHT
            )
            problems[FormCurveKind.DECK_TANGENT] = self._problem(
                FormCurveKind.DECK_TANGENT
            )
            target_map[FormCurveKind.DECK_EDGE] = self.targets.deck_half_breadths
            target_map[FormCurveKind.DECK_HEIGHT] = self.targets.deck_heights
            target_map[FormCurveKind.DECK_TANGENT] = self.targets.deck_tangent_angles
            zero_at_bow_kinds.append(FormCurveKind.DECK_EDGE)

        if self.model_bulge:
            problems[FormCurveKind.BULGE_HALF_BREADTH] = self._problem(
                FormCurveKind.BULGE_HALF_BREADTH
            )
            problems[FormCurveKind.BULGE_HEIGHT] = self._problem(
                FormCurveKind.BULGE_HEIGHT
            )
            target_map[FormCurveKind.BULGE_HALF_BREADTH] = (
                self.targets.bulge_half_breadths
            )
            target_map[FormCurveKind.BULGE_HEIGHT] = self.targets.bulge_heights

        if self.model_dome_band:
            target_map[FormCurveKind.KEEL_PROFILE] = self.targets.blend_depths
            target_map[FormCurveKind.DEADRISE] = (
                0.5 * np.pi - _current_array(self.targets.blend_tangent_angles)
            )
            problems[FormCurveKind.BLEND_HALF_BREADTH] = self._problem(
                FormCurveKind.BLEND_HALF_BREADTH
            )
            problems[FormCurveKind.DOME_DEPTH] = self._problem(
                FormCurveKind.DOME_DEPTH
            )
            problems[FormCurveKind.DOME_AREA] = self._problem(
                FormCurveKind.DOME_AREA
            )
            target_map[FormCurveKind.BLEND_HALF_BREADTH] = (
                self.targets.blend_half_breadths
            )
            target_map[FormCurveKind.DOME_DEPTH] = self.targets.dome_depths
            target_map[FormCurveKind.DOME_AREA] = self.targets.dome_half_areas

        assemblies = {
            kind: problem.assemble(
                system,
                self._initial_form_coefficients(
                    problem,
                    fit_stations,
                    target_map[kind],
                    zero_at_bow=kind in zero_at_bow_kinds,
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
            FormCurveKind.DECK_EDGE: 1.0 / _scalar_value(beam),
            FormCurveKind.DECK_HEIGHT: 1.0 / _scalar_value(draft),
            FormCurveKind.DECK_TANGENT: 1.0,
            FormCurveKind.BULGE_HALF_BREADTH: 1.0 / _scalar_value(beam),
            FormCurveKind.BULGE_HEIGHT: 1.0 / _scalar_value(draft),
            FormCurveKind.BLEND_HALF_BREADTH: 1.0 / _scalar_value(beam),
            FormCurveKind.DOME_DEPTH: 1.0 / _scalar_value(draft),
            FormCurveKind.DOME_AREA: 1.0
            / (_scalar_value(beam) * _scalar_value(draft)),
        }
        for kind, assembly in assemblies.items():
            residual = scales[kind] * (
                assembly.curve.evaluate(fit_stations).reshape((fit_stations.size,))
                - target_map[kind]
            )
            system.add_objective(self.form_fit_weight * csdl.sum(residual**2))

        curves = {kind: assembly.curve for kind, assembly in assemblies.items()}

        num_stations = section_stations.size
        # Fullness is defined against the section's full depth. Where a sonar
        # dome is carried, KEEL_PROFILE is the blend line rather than the
        # deepest point, and measuring fullness against it inflates the ratio
        # by the depth ratio -- about 2.4x through the DTMB 5415 dome, and
        # without bound as the waterline breadth vanishes at the stem. The
        # dome-depth curve is the true keel profile over the whole length
        # (aft of the dome its observations coincide with the keel), so it is
        # the right denominator whenever it exists.
        fullness_depth_kind = (
            FormCurveKind.DOME_DEPTH
            if self.model_dome_band
            else FormCurveKind.KEEL_PROFILE
        )
        if self.use_fullness_curve:
            fullness_problem = FormCurveProblem(
                FormCurveKind.FULLNESS,
                num_control_points=max(
                    self.num_form_control_points, num_stations + 5
                ),
                fairness_weights={2: self.form_fairness_weight},
                regularization=1.0e-12,
                name=f"{self.name}_{FormCurveKind.FULLNESS.value}",
            )
            for station in section_stations:
                station_value = float(station)
                sac_value = curves[FormCurveKind.SECTIONAL_AREA].evaluate(
                    station_value
                ).reshape((1,))
                breadth_value = curves[FormCurveKind.WATERLINE_HALF_BREADTH].evaluate(
                    station_value
                ).reshape((1,))
                draft_value = curves[fullness_depth_kind].evaluate(
                    station_value
                ).reshape((1,))
                fullness_problem.add_value_constraint(
                    station_value, sac_value / (breadth_value * draft_value)
                )
            fullness_assembly = fullness_problem.assemble(
                system, np.full(fullness_problem.num_control_points, 0.7)
            )
            assemblies[FormCurveKind.FULLNESS] = fullness_assembly
            curves[FormCurveKind.FULLNESS] = fullness_assembly.curve
            half_areas_for_sections = (
                fullness_assembly.curve.evaluate(section_stations).reshape(
                    (num_stations,)
                )
                * curves[FormCurveKind.WATERLINE_HALF_BREADTH]
                .evaluate(section_stations)
                .reshape((num_stations,))
                * curves[fullness_depth_kind]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
        else:
            half_areas_for_sections = (
                curves[FormCurveKind.SECTIONAL_AREA]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )

        deck_heights_for_sections = None
        deck_half_breadths_for_sections = None
        deck_tangent_angles_for_sections = None
        if self.model_deck:
            deck_heights_for_sections = (
                curves[FormCurveKind.DECK_HEIGHT]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
            deck_half_breadths_for_sections = (
                curves[FormCurveKind.DECK_EDGE]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
            deck_tangent_angles_for_sections = (
                curves[FormCurveKind.DECK_TANGENT]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )

        # Which generating stations carry a sonar-dome band. Topology is
        # decided from the observations, not from the solve, and it is needed
        # before the bulge waypoints because the band supersedes them.
        dome_mask = None
        if self.model_dome_band:
            observed_depths = _current_array(self.targets.dome_depths)
            observed_blend = _current_array(self.targets.blend_depths)
            station_values = np.asarray(fit_stations, dtype=float)
            threshold = self.dome_depth_threshold * _scalar_value(draft)
            dome_mask = [
                (
                    float(np.interp(float(station), station_values, observed_depths))
                    - float(np.interp(float(station), station_values, observed_blend))
                )
                > threshold
                for station in section_stations
            ]

        section_interior_points = None
        if self.model_bulge:
            fit_station_values = np.asarray(fit_stations, dtype=float)
            bulge_depths = -_current_array(self.targets.bulge_heights)
            local_drafts = _current_array(self.targets.drafts)
            depth_fractions = bulge_depths / np.maximum(local_drafts, 1.0e-12)
            bulge_curve_parameters = _current_array(self.targets.bulge_parameters)
            section_interior_points = []
            for index, station in enumerate(section_stations):
                station_value = float(station)
                # Where a dome band is carried, the bulb is described by the
                # band's own endpoints, tangents and area. The waypoint was a
                # way to fake a neck inside a single monotone keel-to-waterline
                # curve; with the band present it would also ask the main band
                # to pass through a point below its own lower boundary.
                if dome_mask is not None and dome_mask[index]:
                    section_interior_points.append(())
                    continue
                depth_fraction = float(
                    np.interp(station_value, fit_station_values, depth_fractions)
                )
                parameter = float(
                    np.interp(
                        station_value, fit_station_values, bulge_curve_parameters
                    )
                )
                if (
                    depth_fraction < self.bulge_depth_threshold
                    or not 0.02 < parameter < 0.98
                ):
                    section_interior_points.append(())
                    continue
                # Drive the waypoint from the observations directly rather
                # than from the fitted bulge curves. Those curves are tied to
                # their targets only by least squares, while the section is
                # pinned by its endpoints, tangents, area, and section-fit
                # objective. Routing the constraint through them lets the
                # section drag the curve instead of the request moving the
                # section, which leaves the dome variable with no authority.
                weights = _interpolation_weights(fit_station_values, station_value)
                waypoint = _stack_pair(
                    _weighted_target(weights, self.targets.bulge_heights),
                    _weighted_target(weights, self.targets.bulge_half_breadths),
                )
                section_interior_points.append(((parameter, waypoint),))

        # Numeric starting guesses for the section control polygons. The
        # endpoint targets themselves are CSDL expressions of the longitudinal
        # curves, so under a deferred recorder they cannot be evaluated during
        # graph construction. The auxiliary observations are ordinary arrays,
        # which makes them a sound and always-available starting point.
        hint_stations = np.asarray(fit_stations, dtype=float)
        section_geometry_hints: list[dict[str, float]] = []
        hint_sources = {
            "draft": _current_array(
                self.targets.blend_depths
                if self.model_dome_band
                else self.targets.drafts
            ),
            "half_breadth": _current_array(self.targets.half_breadths),
        }
        if self.model_deck:
            hint_sources["deck_height"] = _current_array(self.targets.deck_heights)
            hint_sources["deck_half_breadth"] = _current_array(
                self.targets.deck_half_breadths
            )
        if self.model_dome_band:
            hint_sources["dome_depth"] = _current_array(self.targets.dome_depths)
            hint_sources["keel_half_breadth"] = _current_array(
                self.targets.blend_half_breadths
            )
        for station in section_stations:
            station_value = float(station)
            section_geometry_hints.append(
                {
                    key: float(np.interp(station_value, hint_stations, values))
                    for key, values in hint_sources.items()
                }
            )

        # Sonar-dome band. The main band's lower boundary becomes the blend
        # line, and the dome carries what lies below it. The sectional-area
        # curve still describes the *total* immersed area, so the main band
        # receives that total minus the dome's share and the displacement and
        # LCB constraints are untouched.
        keel_half_breadths_for_sections = None
        dome_depths_for_sections = None
        dome_areas_for_sections = None
        if self.model_dome_band:
            keel_half_breadths_for_sections = (
                curves[FormCurveKind.BLEND_HALF_BREADTH]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
            dome_depths_for_sections = (
                curves[FormCurveKind.DOME_DEPTH]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
            dome_areas_for_sections = (
                curves[FormCurveKind.DOME_AREA]
                .evaluate(section_stations)
                .reshape((num_stations,))
            )
            band = np.asarray(dome_mask, dtype=float)
            dome_areas_for_sections = dome_areas_for_sections * band
            keel_half_breadths_for_sections = keel_half_breadths_for_sections * band
            half_areas_for_sections = (
                half_areas_for_sections - dome_areas_for_sections
            )
        # The section-fit objective compares the main band against the
        # reference section. Where a dome band is carried the main band spans
        # only blend line to waterline, so fitting it against points that run
        # all the way to the keel asks it to reproduce the bulb it no longer
        # covers, and fights its own lower endpoint. The residual is scaled by
        # the band's depth, which is the blend depth there, so the conflict is
        # amplified exactly where it is worst. Resample the part of the
        # reference that the main band actually spans.
        section_fit_points = self.section_fit_points
        dome_points: np.ndarray | None = None
        dome_fit_available: list[bool] = []
        if (
            self.model_dome_band
            and dome_mask is not None
            and section_fit_points is not None
            and self.section_fit_parameters is not None
            and any(dome_mask)
        ):
            # _current_array flattens; the fit points are (station, sample, 2).
            points = np.array(
                section_fit_points.value
                if isinstance(section_fit_points, csdl.Variable)
                else section_fit_points,
                dtype=float,
            )
            parameters = np.asarray(self.section_fit_parameters, dtype=float).reshape(-1)
            observed_blend = _current_array(self.targets.blend_depths)
            station_values = np.asarray(fit_stations, dtype=float)
            dome_points = np.zeros_like(points)
            dome_fit_available = [False] * int(points.shape[0])
            for index, banded in enumerate(dome_mask):
                if not banded:
                    continue
                blend_z = -float(
                    np.interp(
                        float(section_stations[index]), station_values, observed_blend
                    )
                )
                section = points[index]
                keep = section[:, 0] >= blend_z
                if int(np.count_nonzero(keep)) < 4:
                    continue
                upper = section[keep]
                # Reparameterize the retained arc onto [0, 1] by arc length,
                # which is the same convention the band's own parameter uses.
                steps = np.linalg.norm(np.diff(upper, axis=0), axis=1)
                arc = np.concatenate(([0.0], np.cumsum(steps)))
                if arc[-1] <= 0.0:
                    continue
                arc /= arc[-1]
                points[index] = np.stack(
                    [np.interp(parameters, arc, upper[:, 0]),
                     np.interp(parameters, arc, upper[:, 1])],
                    axis=1,
                )
                # The other half of the split. Without it the bulb keeps only
                # its area and endpoints, and its profile is free.
                lower = section[section[:, 0] <= blend_z]
                if lower.shape[0] >= 4:
                    steps = np.linalg.norm(np.diff(lower, axis=0), axis=1)
                    arc = np.concatenate(([0.0], np.cumsum(steps)))
                    if arc[-1] > 0.0:
                        arc /= arc[-1]
                        # The dome band runs bottom-to-blend, the same
                        # direction the reference samples run.
                        dome_points[index] = np.stack(
                            [np.interp(parameters, arc, lower[:, 0]),
                             np.interp(parameters, arc, lower[:, 1])],
                            axis=1,
                        )
                        dome_fit_available[index] = True
            section_fit_points = points

        dome_fit_points = (
            dome_points if dome_points is not None and any(dome_fit_available) else None
        )

        section_problem = SectionLoftProblem(
            length=length,
            station_parameters=section_stations,
            drafts=curves[FormCurveKind.KEEL_PROFILE]
            .evaluate(section_stations)
            .reshape((section_stations.size,)),
            half_breadths=curves[FormCurveKind.WATERLINE_HALF_BREADTH]
            .evaluate(section_stations)
            .reshape((section_stations.size,)),
            half_areas=half_areas_for_sections,
            keel_tangent_angles=(
                0.5 * np.pi
                - curves[FormCurveKind.DEADRISE]
                .evaluate(section_stations)
                .reshape((section_stations.size,))
            ),
            waterline_tangent_angles=curves[FormCurveKind.FLARE]
            .evaluate(section_stations)
            .reshape((section_stations.size,)),
            deck_heights=deck_heights_for_sections,
            deck_half_breadths=deck_half_breadths_for_sections,
            deck_tangent_angles=deck_tangent_angles_for_sections,
            keel_half_breadths=keel_half_breadths_for_sections,
            dome_depths=dome_depths_for_sections,
            dome_half_areas=dome_areas_for_sections,
            dome_fit_points=dome_fit_points,
            dome_mask=dome_mask,
            num_dome_control_points=self.num_dome_control_points,
            section_interior_points=section_interior_points,
            section_geometry_hints=section_geometry_hints,
            transom_x_offsets=self.transom_x_offsets,
            transom_deck_x_offsets=self.transom_deck_x_offsets,
            num_section_control_points=self.num_section_control_points,
            num_deck_control_points=self.num_deck_control_points,
            pointed_ends=(True, section_stations[-1] < 1.0),
            x_origin=self.x_origin,
            section_fit_parameters=self.section_fit_parameters,
            section_fit_points=section_fit_points,
            section_fit_weight=self.section_fit_weight,
            longitudinal_regions=self.longitudinal_regions,
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
            knots=self.form_knots,
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
