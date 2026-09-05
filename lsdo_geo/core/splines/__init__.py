"""Spline primitives for first-principles ship geometry."""

from .f_spline import FSplineAssembly, FSplineCurve, FSplineProblem
from .variational import VariationalResult, VariationalSystem

__all__ = [
    "FSplineAssembly",
    "FSplineCurve",
    "FSplineProblem",
    "VariationalResult",
    "VariationalSystem",
]
