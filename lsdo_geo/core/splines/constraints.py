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
class CurvatureConstraint:
    """Constrain signed planar curvature."""

    parameter: float
    target: Any
    scale: float = 1.0


@dataclass(frozen=True)
class AreaConstraint:
    """Constrain signed area between a planar curve and the first axis.

    ``parameter_range`` restricts the constraint to a sub-arc, which is how a
    section spanning keel to deck edge constrains the area of its immersed
    part alone.
    """

    target: Any
    scale: float = 1.0
    parameter_range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class CentroidConstraint:
    """Constrain the centroid of signed area under a planar curve."""

    target: Any
    scale: float = 1.0


FSplineConstraint = (
    PointConstraint
    | DerivativeConstraint
    | TangentAngleConstraint
    | CurvatureConstraint
    | AreaConstraint
    | CentroidConstraint
)
