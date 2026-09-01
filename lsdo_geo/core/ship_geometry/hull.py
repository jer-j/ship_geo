"""One-solve assembly of transverse sections and a compatible hull surface."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.f_spline import FSplineCurve
from lsdo_geo.core.splines.variational import VariationalResult, VariationalSystem

from .hydrostatics import Hydrostatics, compute_hydrostatics
from .sections import (
    SectionAssembly,
    SectionProblem,
    SectionTemplate,
    collapsed_section,
)
from .surfaces import CompatibleLoft, TensorProductSurface
from .validity import SurfaceValidity, evaluate_surface_validity


def _sequence_item(values: Any, index: int) -> Any:
    if isinstance(values, csdl.Variable):
        return values[index]
    return values[index]


def _sequence_length(values: Any) -> int:
    if isinstance(values, csdl.Variable):
        return int(values.size)
    return len(values)


@dataclass
class HullGeometry:
    """Solved differentiable hull geometry and its principal diagnostics."""

    surface: TensorProductSurface
    sections: tuple[FSplineCurve, ...]
    section_parameters: np.ndarray
    hydrostatics: Hydrostatics
    validity: SurfaceValidity
    variational_result: VariationalResult


class SectionLoftProblem:
    """Assemble all F-Spline sections and surface fairness into one Newton solve.

    The section parameters identify interior longitudinal stations.  When
    ``pointed_ends`` is true, explicit collapsed bow and stern sections are
    derived from the nearest interior section and included in the compatible
    loft without introducing additional implicit states.
    """

    def __init__(
        self,
        length: Any,
        station_parameters: npt.ArrayLike,
        drafts: Sequence[Any] | csdl.Variable,
        half_breadths: Sequence[Any] | csdl.Variable,
        half_areas: Sequence[Any] | csdl.Variable,
        keel_tangent_angles: Sequence[Any] | csdl.Variable | None = None,
        waterline_tangent_angles: Sequence[Any] | csdl.Variable | None = None,
        num_section_control_points: int = 8,
        section_degree: int = 3,
        longitudinal_degree: int = 3,
        section_fairness_weights: dict[int, float] | None = None,
        longitudinal_fairness_weight: float = 0.0,
        pointed_ends: bool = True,
        x_origin: Any = 0.0,
        name: str = "hull_geometry",
    ) -> None:
        parameters = np.asarray(station_parameters, dtype=float).reshape(-1)
        if parameters.size < 2:
            raise ValueError("at least two interior sections are required.")
        if np.any(np.diff(parameters) <= 0.0):
            raise ValueError("station_parameters must be strictly increasing.")
        if pointed_ends and not (parameters[0] > 0.0 and parameters[-1] < 1.0):
            raise ValueError("pointed-end interior stations must lie inside (0, 1).")
        for values, label in (
            (drafts, "drafts"),
            (half_breadths, "half_breadths"),
            (half_areas, "half_areas"),
        ):
            if _sequence_length(values) != parameters.size:
                raise ValueError(f"{label} must match station_parameters.")
        if keel_tangent_angles is None:
            keel_tangent_angles = np.zeros(parameters.size)
        if waterline_tangent_angles is None:
            waterline_tangent_angles = np.zeros(parameters.size)
        for values, label in (
            (keel_tangent_angles, "keel_tangent_angles"),
            (waterline_tangent_angles, "waterline_tangent_angles"),
        ):
            if _sequence_length(values) != parameters.size:
                raise ValueError(f"{label} must match station_parameters.")
        if longitudinal_fairness_weight < 0.0:
            raise ValueError("longitudinal_fairness_weight must be nonnegative.")
        self.length = length
        self.parameters = parameters
        self.drafts = drafts
        self.half_breadths = half_breadths
        self.half_areas = half_areas
        self.keel_tangent_angles = keel_tangent_angles
        self.waterline_tangent_angles = waterline_tangent_angles
        self.num_section_control_points = int(num_section_control_points)
        self.section_degree = int(section_degree)
        self.longitudinal_degree = int(longitudinal_degree)
        self.section_fairness_weights = section_fairness_weights
        self.longitudinal_fairness_weight = float(longitudinal_fairness_weight)
        self.pointed_ends = bool(pointed_ends)
        self.x_origin = x_origin
        self.name = name

    def solve(
        self,
        tolerance: float = 1.0e-10,
        max_iter: int = 100,
        print_status: bool = False,
        hydrostatic_section_parameters: npt.ArrayLike | None = None,
    ) -> HullGeometry:
        """Build and solve the entire section/surface system once."""
        system = VariationalSystem(name=self.name)
        assemblies: list[SectionAssembly] = []
        for index, parameter in enumerate(self.parameters):
            problem = SectionProblem(
                station_parameter=float(parameter),
                draft=_sequence_item(self.drafts, index),
                half_breadth=_sequence_item(self.half_breadths, index),
                half_area=_sequence_item(self.half_areas, index),
                keel_tangent_angle=_sequence_item(self.keel_tangent_angles, index),
                waterline_tangent_angle=_sequence_item(
                    self.waterline_tangent_angles, index
                ),
                template=SectionTemplate.ROUND_BILGE,
                num_control_points=self.num_section_control_points,
                degree=self.section_degree,
                fairness_weights=self.section_fairness_weights,
                name=f"{self.name}_section_{index}",
            )
            assemblies.append(problem.assemble(system))

        loft_sections: list[FSplineCurve] = [assembly.curve for assembly in assemblies]
        loft_parameters = self.parameters.copy()
        if self.pointed_ends:
            loft_sections = [
                collapsed_section(loft_sections[0], f"{self.name}_bow"),
                *loft_sections,
                collapsed_section(loft_sections[-1], f"{self.name}_stern"),
            ]
            loft_parameters = np.concatenate(([0.0], loft_parameters, [1.0]))

        x_coordinates = [
            self.x_origin + self.length * (parameter - 0.5)
            for parameter in loft_parameters
        ]
        surface = CompatibleLoft.create(
            sections=loft_sections,
            station_parameters=loft_parameters,
            x_coordinates=x_coordinates,
            longitudinal_degree=self.longitudinal_degree,
            name=f"{self.name}_surface",
        )
        if self.longitudinal_fairness_weight:
            system.add_objective(
                self.longitudinal_fairness_weight * surface.fairness_energy((0, 2))
            )

        result = system.solve(
            tolerance=tolerance,
            max_iter=max_iter,
            print_status=print_status,
        )
        solved_sections = tuple(assembly.finalize(result) for assembly in assemblies)
        section_parameters = (
            np.linspace(0.0, 1.0, 21)
            if hydrostatic_section_parameters is None
            else np.asarray(hydrostatic_section_parameters, dtype=float)
        )
        hydrostatics = compute_hydrostatics(surface, section_parameters)
        validity = evaluate_surface_validity(surface)
        return HullGeometry(
            surface=surface,
            sections=solved_sections,
            section_parameters=self.parameters.copy(),
            hydrostatics=hydrostatics,
            validity=validity,
            variational_result=result,
        )
