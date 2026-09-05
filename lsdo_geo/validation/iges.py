"""Minimal polynomial IGES type-128 reader for validation geometry.

The production CAD import path remains outside this module. This reader has a
deliberately narrow contract: it extracts untrimmed polynomial B-spline
surfaces and rectangular parameter bounds from IGES entity 128. Rational
weights other than one are rejected explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PolynomialIGESPatch:
    """A polynomial IGES type-128 surface and its active parameter rectangle."""

    degree: tuple[int, int]
    knots: tuple[np.ndarray, np.ndarray]
    coefficients: np.ndarray
    parameter_bounds: tuple[tuple[float, float], tuple[float, float]]
    directory_entry: int

    def __post_init__(self) -> None:
        expected = self.coefficients.shape[:2]
        if self.coefficients.ndim != 3 or self.coefficients.shape[2] != 3:
            raise ValueError("IGES surface coefficients must have shape (nu, nv, 3).")
        for axis in range(2):
            count = len(self.knots[axis]) - self.degree[axis] - 1
            if count != expected[axis]:
                raise ValueError("IGES knot and coefficient dimensions disagree.")

    def build_function(self, name: str = "iges_patch") -> lfs.Function:
        """Build the exact polynomial B-spline in the active CSDL recorder."""
        space = lfs.BSplineSpace(
            num_parametric_dimensions=2,
            degree=self.degree,
            coefficients_shape=self.coefficients.shape[:2],
            knots=np.concatenate(self.knots),
        )
        return lfs.Function(
            space,
            csdl.Variable(value=self.coefficients.copy()),
            name=name,
        )

    def map_coordinates(self, coordinates: npt.ArrayLike) -> np.ndarray:
        """Map a local unit square into the IGES active parameter rectangle."""
        local = np.asarray(coordinates, dtype=float).reshape((-1, 2))
        if np.any(local < 0.0) or np.any(local > 1.0):
            raise ValueError("local surface coordinates must lie in [0, 1]^2.")
        mapped = np.empty_like(local)
        for axis, (lower, upper) in enumerate(self.parameter_bounds):
            mapped[:, axis] = lower + local[:, axis] * (upper - lower)
        return mapped

    def evaluate(
        self,
        function: lfs.Function,
        coordinates: npt.ArrayLike,
        derivative_orders: tuple[int, int] = (0, 0),
    ) -> csdl.Variable:
        """Evaluate in local coordinates, including derivative scale factors."""
        mapped = self.map_coordinates(coordinates)
        values = function.evaluate(
            mapped,
            parametric_derivative_orders=derivative_orders,
        )
        scale = 1.0
        for order, (lower, upper) in zip(derivative_orders, self.parameter_bounds):
            scale *= (upper - lower) ** order
        return scale * values

    def sample_grid(
        self,
        function: lfs.Function,
        resolution: tuple[int, int],
    ) -> tuple[np.ndarray, csdl.Variable]:
        """Evaluate a structured grid over the active rectangular face."""
        if resolution[0] < 2 or resolution[1] < 2:
            raise ValueError("surface sampling resolution must be at least (2, 2).")
        first = np.linspace(0.0, 1.0, resolution[0])
        second = np.linspace(0.0, 1.0, resolution[1])
        first_grid, second_grid = np.meshgrid(first, second, indexing="ij")
        coordinates = np.column_stack((first_grid.ravel(), second_grid.ravel()))
        return coordinates, self.evaluate(function, coordinates)


def read_polynomial_iges_surfaces(path: str | Path) -> list[PolynomialIGESPatch]:
    """Read all polynomial type-128 surfaces from an ASCII IGES file."""
    lines = Path(path).read_text(encoding="ascii", errors="strict").splitlines()
    directory_lines = [line for line in lines if _section(line) == "D"]
    if len(directory_lines) % 2:
        raise ValueError("IGES directory section contains an odd number of records.")
    parameter_lines = {
        int(line[73:80]): line[:64] for line in lines if _section(line) == "P"
    }
    patches: list[PolynomialIGESPatch] = []
    for offset in range(0, len(directory_lines), 2):
        first = _fields(directory_lines[offset])
        second = _fields(directory_lines[offset + 1])
        if int(first[0]) != 128:
            continue
        start = int(first[1])
        count = int(second[3])
        try:
            data = "".join(
                parameter_lines[index] for index in range(start, start + count)
            )
        except KeyError as error:
            raise ValueError("IGES parameter records are incomplete.") from error
        terminator = data.find(";")
        if terminator < 0:
            raise ValueError("IGES type-128 parameter data has no terminator.")
        patches.append(_parse_type_128(data[:terminator], offset + 1))
    if not patches:
        raise ValueError("IGES file contains no polynomial type-128 surfaces.")
    return patches


def _section(line: str) -> str:
    return line[72] if len(line) > 72 else ""


def _fields(line: str) -> list[str]:
    return [line[index : index + 8].strip() for index in range(0, 72, 8)]


def _parse_type_128(data: str, directory_entry: int) -> PolynomialIGESPatch:
    tokens = [token.strip() for token in data.split(",")]
    try:
        values = [float(token.replace("D", "E").replace("d", "e")) for token in tokens]
    except ValueError as error:
        raise ValueError(
            "unsupported IGES parameter delimiter or number format."
        ) from error
    if len(values) < 10 or int(values[0]) != 128:
        raise ValueError("invalid IGES type-128 parameter record.")
    upper_u, upper_v, degree_u, degree_v = (int(value) for value in values[1:5])
    count_u = upper_u + 1
    count_v = upper_v + 1
    knot_count_u = upper_u + degree_u + 2
    knot_count_v = upper_v + degree_v + 2
    coefficient_count = count_u * count_v
    cursor = 10
    required = (
        cursor
        + knot_count_u
        + knot_count_v
        + coefficient_count
        + 3 * coefficient_count
        + 4
    )
    if len(values) != required:
        raise ValueError("IGES type-128 parameter count is inconsistent.")
    knots_u = np.asarray(values[cursor : cursor + knot_count_u], dtype=float)
    cursor += knot_count_u
    knots_v = np.asarray(values[cursor : cursor + knot_count_v], dtype=float)
    cursor += knot_count_v
    weights = np.asarray(values[cursor : cursor + coefficient_count], dtype=float)
    cursor += coefficient_count
    if not np.allclose(weights, 1.0, rtol=0.0, atol=1.0e-14):
        raise NotImplementedError("rational IGES surfaces are outside ship_geo scope.")
    coefficients = np.asarray(
        values[cursor : cursor + 3 * coefficient_count], dtype=float
    )
    cursor += 3 * coefficient_count
    # IGES varies the first surface index fastest. NumPy/CSDL store the second
    # index fastest, so transpose the first two axes after reshaping.
    coefficients = coefficients.reshape((count_v, count_u, 3)).transpose((1, 0, 2))
    bounds = np.asarray(values[cursor : cursor + 4], dtype=float)

    scales = np.array([knots_u[-1], knots_v[-1]], dtype=float)
    if np.any(scales <= 0.0):
        raise ValueError("IGES surface knot domains must have positive extent.")
    knots_u = knots_u / scales[0]
    knots_v = knots_v / scales[1]
    normalized_bounds = (
        (float(bounds[0] / scales[0]), float(bounds[1] / scales[0])),
        (float(bounds[2] / scales[1]), float(bounds[3] / scales[1])),
    )
    for lower, upper in normalized_bounds:
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("IGES active parameter bounds must lie inside its knots.")
    return PolynomialIGESPatch(
        degree=(degree_u, degree_v),
        knots=(knots_u, knots_v),
        coefficients=coefficients,
        parameter_bounds=normalized_bounds,
        directory_entry=directory_entry,
    )
