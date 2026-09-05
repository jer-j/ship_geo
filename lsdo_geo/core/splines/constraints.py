"""Form-parameter constraint specifications for F-Spline curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PointConstraint:
    """Constrain a curve point at a parametric coordinate."""

    parameter: float
    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class DerivativeConstraint:
    """Constrain a complete parametric derivative vector."""

    parameter: float
    derivative_order: int
    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class TangentAngleConstraint:
    """Constrain a planar tangent direction measured from the first axis."""

    parameter: float
    angle: Any
    scale: float = 1.0


@dataclass(frozen=True)
class TangentDirectionConstraint:
    """Constrain tangent direction in an arbitrary physical dimension."""

    parameter: float
    direction: Any
    pivot_index: int
    scale: float = 1.0


@dataclass(frozen=True)
class CurvatureConstraint:
    """Constrain signed planar curvature."""

    parameter: float
    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class CurvatureMagnitudeConstraint:
    """Constrain unsigned curvature in any physical dimension."""

    parameter: float
    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class AreaConstraint:
    """Constrain signed area between a planar curve and the first axis."""

    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class CentroidConstraint:
    """Constrain the centroid of signed area under a planar curve."""

    target: Any
    scale: float = 1.0


FSplineConstraint = (
    PointConstraint
    | DerivativeConstraint
    | TangentAngleConstraint
    | TangentDirectionConstraint
    | CurvatureConstraint
    | CurvatureMagnitudeConstraint
    | AreaConstraint
    | CentroidConstraint
)
