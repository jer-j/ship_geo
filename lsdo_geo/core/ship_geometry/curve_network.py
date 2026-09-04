"""Named naval curve network for one-patch meta-surface generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import csdl_alpha as csdl
import numpy as np
import numpy.typing as npt

from .form_curves import FormCurve


class BasicCurveName(str, Enum):
    """Longitudinal basic curves used by the surface-combatant model."""

    CENTRAL_PROFILE = "central_profile"
    DECK = "deck"
    DESIGN_WATERLINE = "design_waterline"
    FLAT_OF_BOTTOM = "flat_of_bottom"
    ADAPTIVE_STEM = "adaptive_stem"
    SECTIONAL_AREA = "sectional_area"


class ControlCurveName(str, Enum):
    """Longitudinal section controls used by the meta-surface."""

    ENTRANCE_ANGLE = "entrance_angle"
    TANGENT_AT_DESIGN_WATERLINE = "tangent_at_design_waterline"
    TANGENT_AT_DECK = "tangent_at_deck"
    TANGENT_AT_KEEL = "tangent_at_keel"
    SECTIONAL_FULLNESS = "sectional_fullness"


@dataclass(frozen=True)
class SectionControlValues:
    """CSDL expressions imported by transverse sections at given stations."""

    half_areas: csdl.Variable
    half_breadths: csdl.Variable
    depths: csdl.Variable
    keel_tangent_angles: csdl.Variable
    waterline_tangent_angles: csdl.Variable
    entrance_slopes: csdl.Variable
    sectional_fullness: csdl.Variable


@dataclass
class HullCurveNetwork:
    """Coordinate named basic and control curves for a single hull surface.

    The network makes design intent explicit without creating another solve.
    Every member is an already assembled CSDL-backed :class:`FormCurve`, and
    section inputs are evaluated as expressions in the same global graph.
    Optional curves allow deck, flat-of-bottom, adaptive-stem, and sonar-dome
    extensions to be added without changing the surface assembly interface.
    """

    basic_curves: dict[BasicCurveName, FormCurve]
    control_curves: dict[ControlCurveName, FormCurve]
    local_feature_curves: dict[str, FormCurve] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_basic = {
            BasicCurveName.CENTRAL_PROFILE,
            BasicCurveName.DESIGN_WATERLINE,
            BasicCurveName.SECTIONAL_AREA,
        }
        required_control = {
            ControlCurveName.TANGENT_AT_DESIGN_WATERLINE,
            ControlCurveName.TANGENT_AT_KEEL,
        }
        missing_basic = required_basic.difference(self.basic_curves)
        missing_control = required_control.difference(self.control_curves)
        if missing_basic or missing_control:
            raise ValueError(
                "curve network is missing required curves: "
                f"basic={sorted(item.value for item in missing_basic)}, "
                f"control={sorted(item.value for item in missing_control)}."
            )

    def evaluate_section_controls(
        self,
        parameters: npt.ArrayLike,
        length: Any,
    ) -> SectionControlValues:
        """Evaluate the curve network inputs imported by design sections."""
        stations = np.asarray(parameters, dtype=float).reshape(-1)
        count = stations.size
        areas = self.basic_curves[BasicCurveName.SECTIONAL_AREA].evaluate(stations)
        breadths = self.basic_curves[BasicCurveName.DESIGN_WATERLINE].evaluate(stations)
        depths = self.basic_curves[BasicCurveName.CENTRAL_PROFILE].evaluate(stations)
        deadrise = self.control_curves[ControlCurveName.TANGENT_AT_KEEL].evaluate(
            stations
        )
        flare = self.control_curves[
            ControlCurveName.TANGENT_AT_DESIGN_WATERLINE
        ].evaluate(stations)
        entrance_slopes = (
            self.basic_curves[BasicCurveName.DESIGN_WATERLINE].evaluate(stations, 1)
            / length
        )
        fullness = areas / (breadths * depths)
        return SectionControlValues(
            half_areas=areas.reshape((count,)),
            half_breadths=breadths.reshape((count,)),
            depths=depths.reshape((count,)),
            keel_tangent_angles=(0.5 * np.pi - deadrise).reshape((count,)),
            waterline_tangent_angles=flare.reshape((count,)),
            entrance_slopes=entrance_slopes.reshape((count,)),
            sectional_fullness=fullness.reshape((count,)),
        )
