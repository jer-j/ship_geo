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

from .curve_network import (
    BasicCurveName,
    ControlCurveName,
    HullCurveNetwork,
)
from .form_curves import (
    FormCurve,
    FormCurveAssembly,
    FormCurveKind,
    FormCurveProblem,
)
from .hull import HullGeometry, SectionLoftAssembly, SectionLoftProblem
from .sections import (
    SectionBandParameters,
    SectionTemplate,
    SonarDomeSectionParameters,
)
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


@dataclass(frozen=True)
class SectionBandFitTargets:
    """Auxiliary observations for fair hull/dome interface guide curves.

    Points use the transverse section convention ``(z, y)``. The two fixed
    transverse parameters define the representation topology. Their physical
    coordinates are fitted longitudinally, so the resulting interface curves
    are smooth CSDL expressions rather than independent section observations.
    """

    station_parameters: npt.ArrayLike
    dome_top_points: Any
    hull_blend_points: Any
    dome_top_parameter: float
    hull_blend_parameter: float
    continuity: int = 1

    def validated(self) -> SectionBandFitTargets:
        """Return normalized targets after deterministic topology checks."""
        stations = np.asarray(self.station_parameters, dtype=float).reshape(-1)
        if (
            stations.size < 3
            or np.any(np.diff(stations) <= 0.0)
            or np.any((stations <= 0.0) | (stations > 1.0))
        ):
            raise ValueError(
                "section-band stations must increase strictly inside (0, 1]."
            )
        dome_top = float(self.dome_top_parameter)
        hull_blend = float(self.hull_blend_parameter)
        if not 0.0 < dome_top < hull_blend < 1.0:
            raise ValueError("section bands require 0 < dome top < hull blend < 1.")
        continuity = int(self.continuity)
        if continuity not in (1, 2):
            raise ValueError("section-band continuity must be one or two.")
        points = []
        for values, name in (
            (self.dome_top_points, "dome_top_points"),
            (self.hull_blend_points, "hull_blend_points"),
        ):
            shape = (
                values.shape if isinstance(values, csdl.Variable) else np.shape(values)
            )
            if tuple(shape) != (stations.size, 2):
                raise ValueError(f"{name} must have shape ({stations.size}, 2).")
            if not isinstance(values, csdl.Variable) and not np.all(
                np.isfinite(np.asarray(values, dtype=float))
            ):
                raise ValueError(f"{name} must contain finite values.")
            points.append(
                values
                if isinstance(values, csdl.Variable)
                else np.asarray(values, dtype=float)
            )
        return SectionBandFitTargets(
            station_parameters=stations,
            dome_top_points=points[0],
            hull_blend_points=points[1],
            dome_top_parameter=dome_top,
            hull_blend_parameter=hull_blend,
            continuity=continuity,
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
    section_band_curves: dict[str, FormCurve]
    curve_network: HullCurveNetwork
    primary_parameters: NavalHullParameters
    maximum_beam_parameter: float
    maximum_draft_parameter: float

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
    curve_network: HullCurveNetwork

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
            section_band_curves={
                kind.value: curves[kind]
                for kind in (
                    FormCurveKind.DOME_TOP_HEIGHT,
                    FormCurveKind.DOME_TOP_HALF_BREADTH,
                    FormCurveKind.HULL_BLEND_HEIGHT,
                    FormCurveKind.HULL_BLEND_HALF_BREADTH,
                )
                if kind in curves
            },
            curve_network=self.curve_network,
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
        section_station_parameters: npt.ArrayLike | None = None,
        section_fit_parameters: npt.ArrayLike | None = None,
        section_fit_points: Any | None = None,
        section_fit_weight: float = 0.0,
        section_templates: Sequence[SectionTemplate] | None = None,
        sonar_dome_parameters: Sequence[SonarDomeSectionParameters | None]
        | None = None,
        section_band_fit_targets: SectionBandFitTargets | None = None,
        form_fit_weight: float = 1.0,
        form_fairness_weight: float = 1.0e-4,
        x_origin: Any = 0.0,
        longitudinal_regions: Sequence[LongitudinalLoftRegion] | None = None,
        longitudinal_feature_parameters: npt.ArrayLike | None = None,
        longitudinal_feature_continuity: int = 1,
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
        self.section_templates = section_templates
        self.sonar_dome_parameters = sonar_dome_parameters
        self.section_band_fit_targets = (
            None
            if section_band_fit_targets is None
            else section_band_fit_targets.validated()
        )
        if self.section_band_fit_targets is not None:
            band_templates = (SectionTemplate.BLENDED_SONAR_DOME,) * len(
                section_stations
            )
            if (
                section_templates is not None
                and tuple(SectionTemplate(item) for item in section_templates)
                != band_templates
            ):
                raise ValueError(
                    "section_band_fit_targets require blended sonar-dome "
                    "templates at every loft station."
                )
            if sonar_dome_parameters is not None and any(
                item is not None for item in sonar_dome_parameters
            ):
                raise ValueError(
                    "section_band_fit_targets replace isolated sonar-dome "
                    "section parameters."
                )
            self.section_templates = band_templates
            self.sonar_dome_parameters = (None,) * len(section_stations)
        self.form_fit_weight = float(form_fit_weight)
        self.form_fairness_weight = float(form_fairness_weight)
        self.x_origin = x_origin
        self.longitudinal_regions = (
            None if longitudinal_regions is None else tuple(longitudinal_regions)
        )
        self.longitudinal_feature_parameters = longitudinal_feature_parameters
        self.longitudinal_feature_continuity = int(longitudinal_feature_continuity)
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
        band_targets = self.section_band_fit_targets
        if band_targets is not None:
            for kind in (
                FormCurveKind.DOME_TOP_HEIGHT,
                FormCurveKind.DOME_TOP_HALF_BREADTH,
                FormCurveKind.HULL_BLEND_HEIGHT,
                FormCurveKind.HULL_BLEND_HALF_BREADTH,
            ):
                problems[kind] = self._problem(kind)
            target_map.update(
                {
                    FormCurveKind.DOME_TOP_HEIGHT: -band_targets.dome_top_points[:, 0],
                    FormCurveKind.DOME_TOP_HALF_BREADTH: (
                        band_targets.dome_top_points[:, 1]
                    ),
                    FormCurveKind.HULL_BLEND_HEIGHT: -band_targets.hull_blend_points[
                        :, 0
                    ],
                    FormCurveKind.HULL_BLEND_HALF_BREADTH: (
                        band_targets.hull_blend_points[:, 1]
                    ),
                }
            )
        assemblies = {
            kind: problem.assemble(
                system,
                self._initial_form_coefficients(
                    problem,
                    (
                        fit_stations
                        if kind
                        not in (
                            FormCurveKind.DOME_TOP_HEIGHT,
                            FormCurveKind.DOME_TOP_HALF_BREADTH,
                            FormCurveKind.HULL_BLEND_HEIGHT,
                            FormCurveKind.HULL_BLEND_HALF_BREADTH,
                        )
                        else np.asarray(band_targets.station_parameters)
                    ),
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
        scales.update(
            {
                FormCurveKind.DOME_TOP_HEIGHT: 1.0 / _scalar_value(draft),
                FormCurveKind.DOME_TOP_HALF_BREADTH: 2.0 / _scalar_value(beam),
                FormCurveKind.HULL_BLEND_HEIGHT: 1.0 / _scalar_value(draft),
                FormCurveKind.HULL_BLEND_HALF_BREADTH: 2.0 / _scalar_value(beam),
            }
        )
        for kind, assembly in assemblies.items():
            target_stations = (
                fit_stations
                if kind
                not in (
                    FormCurveKind.DOME_TOP_HEIGHT,
                    FormCurveKind.DOME_TOP_HALF_BREADTH,
                    FormCurveKind.HULL_BLEND_HEIGHT,
                    FormCurveKind.HULL_BLEND_HALF_BREADTH,
                )
                else np.asarray(band_targets.station_parameters)
            )
            residual = scales[kind] * (
                assembly.curve.evaluate(target_stations).reshape(
                    (target_stations.size,)
                )
                - target_map[kind]
            )
            system.add_objective(self.form_fit_weight * csdl.sum(residual**2))

        curves = {kind: assembly.curve for kind, assembly in assemblies.items()}
        curve_network = HullCurveNetwork(
            basic_curves={
                BasicCurveName.CENTRAL_PROFILE: curves[FormCurveKind.KEEL_PROFILE],
                BasicCurveName.DESIGN_WATERLINE: curves[
                    FormCurveKind.WATERLINE_HALF_BREADTH
                ],
                BasicCurveName.SECTIONAL_AREA: curves[FormCurveKind.SECTIONAL_AREA],
            },
            control_curves={
                ControlCurveName.TANGENT_AT_DESIGN_WATERLINE: curves[
                    FormCurveKind.FLARE
                ],
                ControlCurveName.TANGENT_AT_KEEL: curves[FormCurveKind.DEADRISE],
            },
            local_feature_curves={
                kind.value: curves[kind]
                for kind in (
                    FormCurveKind.DOME_TOP_HEIGHT,
                    FormCurveKind.DOME_TOP_HALF_BREADTH,
                    FormCurveKind.HULL_BLEND_HEIGHT,
                    FormCurveKind.HULL_BLEND_HALF_BREADTH,
                )
                if kind in curves
            },
        )
        section_controls = curve_network.evaluate_section_controls(
            section_stations, length
        )
        section_band_parameters = None
        if band_targets is not None:
            dome_heights = curves[FormCurveKind.DOME_TOP_HEIGHT].evaluate(
                section_stations
            )
            dome_breadths = curves[FormCurveKind.DOME_TOP_HALF_BREADTH].evaluate(
                section_stations
            )
            blend_heights = curves[FormCurveKind.HULL_BLEND_HEIGHT].evaluate(
                section_stations
            )
            blend_breadths = curves[FormCurveKind.HULL_BLEND_HALF_BREADTH].evaluate(
                section_stations
            )
            section_band_parameters = tuple(
                SectionBandParameters(
                    dome_top_parameter=band_targets.dome_top_parameter,
                    dome_top_point=csdl.concatenate(
                        (
                            -dome_heights[index].reshape((1,)),
                            dome_breadths[index].reshape((1,)),
                        )
                    ),
                    hull_blend_parameter=band_targets.hull_blend_parameter,
                    hull_blend_point=csdl.concatenate(
                        (
                            -blend_heights[index].reshape((1,)),
                            blend_breadths[index].reshape((1,)),
                        )
                    ),
                    continuity=band_targets.continuity,
                )
                for index in range(section_stations.size)
            )
        section_problem = SectionLoftProblem(
            length=length,
            station_parameters=section_stations,
            drafts=section_controls.depths,
            half_breadths=section_controls.half_breadths,
            half_areas=section_controls.half_areas,
            keel_tangent_angles=section_controls.keel_tangent_angles,
            waterline_tangent_angles=section_controls.waterline_tangent_angles,
            num_section_control_points=self.num_section_control_points,
            pointed_ends=(True, section_stations[-1] < 1.0),
            x_origin=self.x_origin,
            section_fit_parameters=self.section_fit_parameters,
            section_fit_points=self.section_fit_points,
            section_fit_weight=self.section_fit_weight,
            section_templates=self.section_templates,
            sonar_dome_parameters=self.sonar_dome_parameters,
            section_band_parameters=section_band_parameters,
            longitudinal_regions=self.longitudinal_regions,
            longitudinal_feature_parameters=self.longitudinal_feature_parameters,
            longitudinal_feature_continuity=self.longitudinal_feature_continuity,
            name=self.name,
        )
        section_assembly = section_problem.assemble(system)
        return FormParameterHullAssembly(
            curve_assemblies=assemblies,
            section_loft=section_assembly,
            primary_parameters=self.primary,
            maximum_beam_parameter=self.targets.maximum_beam_parameter,
            maximum_draft_parameter=self.targets.maximum_draft_parameter,
            curve_network=curve_network,
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
