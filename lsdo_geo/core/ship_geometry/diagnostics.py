"""Numerical topology and self-intersection diagnostics for solved geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt

from .closed_surface import ClosedSurface
from .surfaces import TensorProductSurface


@dataclass(frozen=True)
class ClosureReport:
    """Current numerical closure state of a multi-patch boundary."""

    maximum_matched_gap: float
    unmatched_edges: tuple[str, ...]
    num_edges: int

    @property
    def is_closed(self) -> bool:
        """Return whether every sampled edge has a matching partner."""
        return not self.unmatched_edges

    def assert_closed(self) -> None:
        """Raise when one or more sampled patch edges are unmatched."""
        if self.unmatched_edges:
            raise ValueError(
                "closed-surface topology has unmatched edges: "
                + ", ".join(self.unmatched_edges)
            )


def evaluate_closed_surface_closure(
    closed_surface: ClosedSurface,
    samples: int = 17,
    tolerance: float = 1.0e-8,
) -> ClosureReport:
    """Match sampled patch edges independent of orientation."""
    if samples < 2:
        raise ValueError("at least two edge samples are required.")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    parameter = np.linspace(0.0, 1.0, samples)
    edge_coordinates = {
        "u0": np.column_stack((np.zeros_like(parameter), parameter)),
        "u1": np.column_stack((np.ones_like(parameter), parameter)),
        "v0": np.column_stack((parameter, np.zeros_like(parameter))),
        "v1": np.column_stack((parameter, np.ones_like(parameter))),
    }
    edges: list[tuple[str, np.ndarray]] = []
    for patch in closed_surface.patches:
        for edge_name, coordinates in edge_coordinates.items():
            values = np.asarray(patch.surface.evaluate(coordinates).value, dtype=float)
            edges.append((f"{patch.name}:{edge_name}", values))

    unmatched: list[str] = []
    matched_gaps: list[float] = []
    for index, (name, edge) in enumerate(edges):
        if float(np.max(np.linalg.norm(edge - edge[0], axis=1))) <= tolerance:
            # A collapsed parametric edge is a point, not an open physical edge.
            matched_gaps.append(0.0)
            continue
        best_gap = np.inf
        for other_index, (_, other) in enumerate(edges):
            if index == other_index:
                continue
            direct = float(np.max(np.linalg.norm(edge - other, axis=1)))
            reversed_gap = float(np.max(np.linalg.norm(edge - other[::-1], axis=1)))
            best_gap = min(best_gap, direct, reversed_gap)
        if best_gap > tolerance:
            unmatched.append(name)
        else:
            matched_gaps.append(best_gap)
    return ClosureReport(
        maximum_matched_gap=max(matched_gaps, default=0.0),
        unmatched_edges=tuple(unmatched),
        num_edges=len(edges),
    )


def section_self_intersections(
    surface: TensorProductSurface,
    section_parameters: Sequence[float],
    samples: int = 201,
    tolerance: float = 1.0e-10,
) -> dict[float, int]:
    """Count nonadjacent intersections in sampled ``(y, z)`` sections."""
    if samples < 4:
        raise ValueError("at least four section samples are required.")
    u = np.linspace(0.0, 1.0, samples)
    result: dict[float, int] = {}
    for parameter in section_parameters:
        parameter = float(parameter)
        if not 0.0 <= parameter <= 1.0:
            raise ValueError("section parameters must lie in [0, 1].")
        coordinates = np.column_stack((u, np.full(u.shape, parameter)))
        points = np.asarray(surface.evaluate(coordinates).value[:, 1:3], dtype=float)
        result[parameter] = _polyline_intersections(points, tolerance)
    return result


def curve_span_error_indicators(
    parameters: npt.ArrayLike,
    residuals: npt.ArrayLike,
    knots: npt.ArrayLike,
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return RMS residuals and midpoints for every nonzero knot span."""
    parameters = np.asarray(parameters, dtype=float).reshape(-1)
    errors = np.asarray(residuals, dtype=float)
    if errors.shape[0] != parameters.size:
        raise ValueError("residuals must have one row per parameter.")
    if errors.ndim == 1:
        errors = errors[:, None]
    knot_values = np.asarray(knots, dtype=float).reshape(-1)
    active = np.unique(knot_values[degree:-degree])
    indicators: list[float] = []
    midpoints: list[float] = []
    for span_index, (lower, upper) in enumerate(pairwise(active)):
        if span_index == len(active) - 2:
            mask = (parameters >= lower) & (parameters <= upper)
        else:
            mask = (parameters >= lower) & (parameters < upper)
        if not np.any(mask):
            indicator = 0.0
        else:
            indicator = float(np.sqrt(np.mean(errors[mask] ** 2)))
        indicators.append(indicator)
        midpoints.append(0.5 * (lower + upper))
    return np.asarray(indicators), np.asarray(midpoints)


def recommended_refinement_knots(
    indicators: npt.ArrayLike,
    midpoints: npt.ArrayLike,
    relative_threshold: float = 0.5,
) -> np.ndarray:
    """Select span midpoints whose indicator exceeds a relative threshold."""
    indicators = np.asarray(indicators, dtype=float).reshape(-1)
    midpoints = np.asarray(midpoints, dtype=float).reshape(-1)
    if indicators.size != midpoints.size:
        raise ValueError("indicators and midpoints must have equal length.")
    if not 0.0 <= relative_threshold <= 1.0:
        raise ValueError("relative_threshold must lie in [0, 1].")
    maximum = float(np.max(indicators, initial=0.0))
    if maximum == 0.0:
        return np.array([], dtype=float)
    return midpoints[indicators >= relative_threshold * maximum]


def _polyline_intersections(points: np.ndarray, tolerance: float) -> int:
    count = 0
    num_segments = points.shape[0] - 1
    for first in range(num_segments):
        for second in range(first + 2, num_segments):
            if second == first + 1:
                continue
            if _segments_intersect(
                points[first],
                points[first + 1],
                points[second],
                points[second + 1],
                tolerance,
            ):
                count += 1
    return count


def _segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        first = b - a
        second = c - a
        return float(first[0] * second[1] - first[1] * second[0])

    o1 = orientation(first_start, first_end, second_start)
    o2 = orientation(first_start, first_end, second_end)
    o3 = orientation(second_start, second_end, first_start)
    o4 = orientation(second_start, second_end, first_end)
    return o1 * o2 < -(tolerance**2) and o3 * o4 < -(tolerance**2)
