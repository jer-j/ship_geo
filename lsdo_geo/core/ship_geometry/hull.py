"""One-solve assembly of transverse sections and hull surfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.f_spline import FSplineCurve
from lsdo_geo.core.splines.variational import VariationalResult, VariationalSystem

from .f_surface import FSurfaceAssembly, FSurfaceProblem
from .hydrostatics import Hydrostatics, compute_hydrostatics
from .sections import (
    SectionAssembly,
    SectionProblem,
    SectionTemplate,
    collapsed_section,
)
from .surfaces import (
    CompatibleLoft,
    LongitudinalLoftRegion,
    RegionalCompatibleLoft,
    TensorProductSurface,
)
from .validity import SurfaceValidity, evaluate_surface_validity


def _sequence_item(values: Any, index: int) -> Any:
    if isinstance(values, csdl.Variable):
        return values[index]
    return values[index]


def _sequence_length(values: Any) -> int:
    if isinstance(values, csdl.Variable):
        return int(values.size)
    return len(values)


def _current_scalar(value: Any) -> float:
    """Return a scalar's current value without changing its derivative path.

    Falls back to zero when the value is not yet available, which happens under
    a deferred recorder. Only initial guesses use this path.
    """
    if isinstance(value, csdl.Variable):
        if value.value is None:
            return 0.0
        return float(np.asarray(value.value).reshape(-1)[0])
    return float(value)


@dataclass
class HullGeometry:
    """Solved differentiable hull geometry and its principal diagnostics."""

    surface: TensorProductSurface
    sections: tuple[FSplineCurve, ...]
    section_parameters: np.ndarray
    hydrostatics: Hydrostatics
    validity: SurfaceValidity
    variational_result: VariationalResult
    regional_surface: RegionalCompatibleLoft | None = None
    freeboard_surface: TensorProductSurface | None = None
    freeboard_sections: tuple[FSplineCurve, ...] | None = None


@dataclass
class SectionLoftAssembly:
    """Section and surface states registered in one variational system."""

    surface: TensorProductSurface
    section_assemblies: tuple[SectionAssembly, ...]
    section_parameters: np.ndarray
    surface_assembly: FSurfaceAssembly | None = None
    regional_surface: RegionalCompatibleLoft | None = None
    freeboard_surface: TensorProductSurface | None = None

    def finalize(
        self,
        result: VariationalResult,
        hydrostatic_section_parameters: npt.ArrayLike | None = None,
    ) -> HullGeometry:
        """Attach KKT diagnostics and construct differentiable analyses."""
        solved_sections = tuple(
            assembly.finalize(result) for assembly in self.section_assemblies
        )
        freeboard_sections = None
        if all(assembly.topside is not None for assembly in self.section_assemblies):
            freeboard_sections = tuple(
                assembly.topside_curve for assembly in self.section_assemblies
            )
        surface = (
            self.surface
            if self.surface_assembly is None
            else self.surface_assembly.finalize(result)
        )
        analysis_parameters = (
            np.linspace(0.0, 1.0, 21)
            if hydrostatic_section_parameters is None
            else np.asarray(hydrostatic_section_parameters, dtype=float)
        )
        return HullGeometry(
            surface=surface,
            sections=solved_sections,
            section_parameters=self.section_parameters.copy(),
            hydrostatics=compute_hydrostatics(surface, analysis_parameters),
            validity=evaluate_surface_validity(surface),
            variational_result=result,
            regional_surface=self.regional_surface,
            freeboard_surface=self.freeboard_surface,
            freeboard_sections=freeboard_sections,
        )


class SectionLoftProblem:
    """Assemble all F-Spline sections and surface fairness into one Newton solve.

    The section parameters identify interior longitudinal stations.  When
    ``pointed_ends`` is true, explicit collapsed bow and stern sections are
    derived from the nearest interior section and included in the surface
    construction without introducing additional implicit states.

    ``surface_formulation="compatible_loft"`` retains the fixed linear loft.
    ``surface_formulation="variational"`` adds a free surface control net and
    exact section-incidence constraints to the same global KKT system.  It does
    not introduce a second Newton solve.
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
        deck_heights: Sequence[Any] | csdl.Variable | None = None,
        deck_half_breadths: Sequence[Any] | csdl.Variable | None = None,
        deck_tangent_angles: Sequence[Any] | csdl.Variable | None = None,
        section_interior_points: Sequence[Any] | None = None,
        section_geometry_hints: Sequence[dict[str, float]] | None = None,
        num_section_control_points: int = 8,
        num_deck_control_points: int | None = None,
        section_degree: int = 3,
        longitudinal_degree: int = 3,
        section_fairness_weights: dict[int, float] | None = None,
        longitudinal_fairness_weight: float = 0.0,
        surface_formulation: Literal["compatible_loft", "variational"] = (
            "compatible_loft"
        ),
        surface_num_longitudinal_control_points: int | None = None,
        surface_fairness_weights: dict[tuple[int, int], float] | None = None,
        surface_constraint_scale: float = 1.0,
        pointed_ends: bool | tuple[bool, bool] = True,
        x_origin: Any = 0.0,
        section_fit_parameters: npt.ArrayLike | None = None,
        section_fit_points: Any | None = None,
        section_fit_weight: float = 0.0,
        longitudinal_regions: Sequence[LongitudinalLoftRegion] | None = None,
        name: str = "hull_geometry",
    ) -> None:
        parameters = np.asarray(station_parameters, dtype=float).reshape(-1)
        if parameters.size < 2:
            raise ValueError("at least two interior sections are required.")
        if np.any(np.diff(parameters) <= 0.0):
            raise ValueError("station_parameters must be strictly increasing.")
        if isinstance(pointed_ends, tuple):
            if len(pointed_ends) != 2:
                raise ValueError("pointed_ends must be a bool or a two-bool tuple.")
            pointed_bow, pointed_stern = map(bool, pointed_ends)
        else:
            pointed_bow = pointed_stern = bool(pointed_ends)
        if pointed_bow and parameters[0] <= 0.0:
            raise ValueError("a pointed bow requires the first station inside (0, 1].")
        if not pointed_bow and parameters[0] < 0.0:
            raise ValueError("the first station must lie in [0, 1].")
        if pointed_stern and parameters[-1] >= 1.0:
            raise ValueError("a pointed stern requires the last station inside [0, 1).")
        if not pointed_stern and parameters[-1] > 1.0:
            raise ValueError("the last station must lie in [0, 1].")
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
        deck_values = (deck_heights, deck_half_breadths, deck_tangent_angles)
        if any(value is not None for value in deck_values) and any(
            value is None for value in deck_values
        ):
            raise ValueError(
                "deck_heights, deck_half_breadths, and deck_tangent_angles must be "
                "supplied together."
            )
        for values, label in (
            (deck_heights, "deck_heights"),
            (deck_half_breadths, "deck_half_breadths"),
            (deck_tangent_angles, "deck_tangent_angles"),
        ):
            if values is not None and _sequence_length(values) != parameters.size:
                raise ValueError(f"{label} must match station_parameters.")
        if longitudinal_fairness_weight < 0.0:
            raise ValueError("longitudinal_fairness_weight must be nonnegative.")
        if surface_formulation not in ("compatible_loft", "variational"):
            raise ValueError(
                "surface_formulation must be 'compatible_loft' or 'variational'."
            )
        if not np.isfinite(surface_constraint_scale) or surface_constraint_scale <= 0:
            raise ValueError("surface_constraint_scale must be finite and positive.")
        self.length = length
        self.parameters = parameters
        self.drafts = drafts
        self.half_breadths = half_breadths
        self.half_areas = half_areas
        self.keel_tangent_angles = keel_tangent_angles
        self.waterline_tangent_angles = waterline_tangent_angles
        self.deck_heights = deck_heights
        self.deck_half_breadths = deck_half_breadths
        self.deck_tangent_angles = deck_tangent_angles
        if section_interior_points is not None and len(section_interior_points) != (
            parameters.size
        ):
            raise ValueError("section_interior_points must match station_parameters.")
        self.section_interior_points = section_interior_points
        if section_geometry_hints is not None and len(section_geometry_hints) != (
            parameters.size
        ):
            raise ValueError("section_geometry_hints must match station_parameters.")
        self.section_geometry_hints = section_geometry_hints
        self.num_section_control_points = int(num_section_control_points)
        self.num_deck_control_points = (
            None if num_deck_control_points is None else int(num_deck_control_points)
        )
        self.section_degree = int(section_degree)
        self.longitudinal_degree = int(longitudinal_degree)
        self.section_fairness_weights = section_fairness_weights
        self.longitudinal_fairness_weight = float(longitudinal_fairness_weight)
        self.surface_formulation = surface_formulation
        self.surface_num_longitudinal_control_points = (
            None
            if surface_num_longitudinal_control_points is None
            else int(surface_num_longitudinal_control_points)
        )
        self.surface_fairness_weights = surface_fairness_weights
        self.surface_constraint_scale = float(surface_constraint_scale)
        self.pointed_ends = (pointed_bow, pointed_stern)
        self.x_origin = x_origin
        if (section_fit_parameters is None) != (section_fit_points is None):
            raise ValueError(
                "section_fit_parameters and section_fit_points must be supplied "
                "together."
            )
        self.section_fit_parameters = (
            None
            if section_fit_parameters is None
            else np.asarray(section_fit_parameters, dtype=float).reshape(-1)
        )
        self.section_fit_points = section_fit_points
        self.section_fit_weight = float(section_fit_weight)
        if self.section_fit_weight < 0.0:
            raise ValueError("section_fit_weight must be nonnegative.")
        if self.section_fit_parameters is not None:
            shape = (
                section_fit_points.shape
                if isinstance(section_fit_points, csdl.Variable)
                else np.shape(section_fit_points)
            )
            expected = (
                parameters.size,
                self.section_fit_parameters.size,
                2,
            )
            if tuple(shape) != expected:
                raise ValueError(
                    "section_fit_points must have shape "
                    "(num_sections, num_fit_parameters, 2)."
                )
        self.longitudinal_regions = (
            None if longitudinal_regions is None else tuple(longitudinal_regions)
        )
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
        assembly = self.assemble(system)
        result = system.solve(
            tolerance=tolerance,
            max_iter=max_iter,
            print_status=print_status,
        )
        return assembly.finalize(result, hydrostatic_section_parameters)

    def assemble(self, system: VariationalSystem) -> SectionLoftAssembly:
        """Register every section and the hull surface without running Newton."""
        model_deck = self.deck_heights is not None
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
                num_deck_control_points=self.num_deck_control_points,
                degree=self.section_degree,
                fairness_weights=self.section_fairness_weights,
                fit_parameters=self.section_fit_parameters,
                fit_points=(
                    None
                    if self.section_fit_points is None
                    else self.section_fit_points[index]
                ),
                fit_weight=self.section_fit_weight,
                interior_points=(
                    None
                    if self.section_interior_points is None
                    else self.section_interior_points[index]
                ),
                initial_geometry_hint=(
                    None
                    if self.section_geometry_hints is None
                    else self.section_geometry_hints[index]
                ),
                deck_height=(
                    _sequence_item(self.deck_heights, index) if model_deck else None
                ),
                deck_half_breadth=(
                    _sequence_item(self.deck_half_breadths, index)
                    if model_deck
                    else None
                ),
                deck_tangent_angle=(
                    _sequence_item(self.deck_tangent_angles, index)
                    if model_deck
                    else None
                ),
                name=f"{self.name}_section_{index}",
            )
            assemblies.append(problem.assemble(system))

        loft_sections: list[FSplineCurve] = [assembly.curve for assembly in assemblies]
        loft_parameters = self.parameters.copy()
        pointed_bow, pointed_stern = self.pointed_ends
        if pointed_bow:
            loft_sections.insert(
                0, collapsed_section(loft_sections[0], f"{self.name}_bow")
            )
            loft_parameters = np.concatenate(([0.0], loft_parameters))
        if pointed_stern:
            loft_sections.append(
                collapsed_section(loft_sections[-1], f"{self.name}_stern")
            )
            loft_parameters = np.concatenate((loft_parameters, [1.0]))

        x_coordinates = [
            self.x_origin + self.length * (parameter - 0.5)
            for parameter in loft_parameters
        ]
        surface_assembly: FSurfaceAssembly | None = None
        if self.surface_formulation == "compatible_loft":
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
        else:
            surface_assembly = self._assemble_variational_surface(
                system,
                loft_sections,
                loft_parameters,
                x_coordinates,
            )
            surface = surface_assembly.surface

        regional_surface = None
        if self.longitudinal_regions is not None:
            regional_surface = RegionalCompatibleLoft.create(
                sections=loft_sections,
                station_parameters=loft_parameters,
                x_coordinates=x_coordinates,
                regions=self.longitudinal_regions,
                longitudinal_degree=self.longitudinal_degree,
                name=f"{self.name}_regional_surface",
            )

        freeboard_surface: TensorProductSurface | None = None
        if model_deck:
            freeboard_sections: list[FSplineCurve] = [
                assembly.topside_curve for assembly in assemblies
            ]
            if pointed_bow:
                freeboard_sections.insert(
                    0,
                    collapsed_section(freeboard_sections[0], f"{self.name}_bow_deck"),
                )
            if pointed_stern:
                freeboard_sections.append(
                    collapsed_section(
                        freeboard_sections[-1], f"{self.name}_stern_deck"
                    )
                )
            freeboard_surface = CompatibleLoft.create(
                sections=freeboard_sections,
                station_parameters=loft_parameters,
                x_coordinates=x_coordinates,
                longitudinal_degree=self.longitudinal_degree,
                name=f"{self.name}_freeboard_surface",
            )

        return SectionLoftAssembly(
            surface=surface,
            section_assemblies=tuple(assemblies),
            section_parameters=self.parameters.copy(),
            surface_assembly=surface_assembly,
            regional_surface=regional_surface,
            freeboard_surface=freeboard_surface,
        )

    def _assemble_variational_surface(
        self,
        system: VariationalSystem,
        sections: Sequence[FSplineCurve],
        station_parameters: np.ndarray,
        x_coordinates: Sequence[Any],
    ) -> FSurfaceAssembly:
        """Couple a free surface state exactly to all generating sections."""
        transverse_space = sections[0].space
        transverse_knots = np.asarray(
            transverse_space.knots[transverse_space.knot_indices[0]], dtype=float
        )
        longitudinal_count = self.surface_num_longitudinal_control_points
        if longitudinal_count is None:
            longitudinal_count = max(len(sections), self.longitudinal_degree + 1)
        if longitudinal_count <= self.longitudinal_degree:
            raise ValueError(
                "surface_num_longitudinal_control_points must exceed "
                "longitudinal_degree."
            )
        if longitudinal_count < len(sections):
            raise ValueError(
                "surface_num_longitudinal_control_points must be at least the "
                "number of generating sections."
            )
        longitudinal_space = lfs.BSplineSpace(
            num_parametric_dimensions=1,
            degree=(self.longitudinal_degree,),
            coefficients_shape=(longitudinal_count,),
        )
        longitudinal_knots = np.asarray(
            longitudinal_space.knots[longitudinal_space.knot_indices[0]], dtype=float
        )
        fairness_weights = self.surface_fairness_weights
        if fairness_weights is None and self.longitudinal_fairness_weight:
            fairness_weights = {
                (2, 0): 1.0,
                (1, 1): 2.0,
                (0, 2): self.longitudinal_fairness_weight,
            }
        problem = FSurfaceProblem(
            num_control_points=(
                sections[0].coefficients.shape[0],
                longitudinal_count,
            ),
            degree=(self.section_degree, self.longitudinal_degree),
            knots=(transverse_knots, longitudinal_knots),
            fairness_weights=fairness_weights,
            name=f"{self.name}_surface",
        )

        degree = int(transverse_space.degree[0])
        transverse_count = sections[0].coefficients.shape[0]
        greville = np.asarray(
            [
                np.mean(transverse_knots[index + 1 : index + degree + 1])
                for index in range(transverse_count)
            ]
        )
        surface_coordinates: list[np.ndarray] = []
        surface_targets: list[csdl.Variable] = []
        for section, station, x_coordinate in zip(
            sections, station_parameters, x_coordinates
        ):
            section_points = section.evaluate(greville)
            z = section_points[:, 0].reshape((transverse_count, 1))
            y = section_points[:, 1].reshape((transverse_count, 1))
            x = 0.0 * z + x_coordinate
            surface_targets.append(csdl.concatenate((x, y, z), axis=1))
            surface_coordinates.append(
                np.column_stack((greville, np.full(transverse_count, float(station))))
            )
        problem.add_points_constraint(
            np.concatenate(tuple(surface_coordinates), axis=0),
            csdl.concatenate(tuple(surface_targets), axis=0),
            scale=self.surface_constraint_scale,
        )
        initial_control_points = None
        if longitudinal_count == len(sections):
            longitudinal_basis = longitudinal_space.compute_basis_matrix(
                station_parameters[:, None]
            ).toarray()
            longitudinal_map = np.linalg.solve(
                longitudinal_basis, np.eye(longitudinal_count)
            )
            physical_sections: list[np.ndarray] = []
            for section, x_coordinate in zip(sections, x_coordinates):
                coefficients = np.asarray(section.coefficients.value, dtype=float)
                physical_sections.append(
                    np.column_stack(
                        (
                            np.full(transverse_count, _current_scalar(x_coordinate)),
                            coefficients[:, 1],
                            coefficients[:, 0],
                        )
                    )
                )
            initial_control_points = np.empty((transverse_count, longitudinal_count, 3))
            for transverse_index in range(transverse_count):
                station_values = np.asarray(
                    [points[transverse_index] for points in physical_sections]
                )
                initial_control_points[transverse_index] = (
                    longitudinal_map @ station_values
                )
        return problem.assemble(system, initial_control_points)
