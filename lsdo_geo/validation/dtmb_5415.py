"""DTMB 5415 multi-patch approximation validation, including the sonar dome."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import lsdo_function_spaces as lfs
import numpy as np

from ..core.ship_geometry.refinement import fit_offset_surface
from ..core.ship_geometry.surfaces import TensorProductSurface
from .iges import PolynomialIGESPatch, read_polynomial_iges_surfaces

DTMB5415_SOURCE_COMMIT = "afad0f04a2ce5ee03f37dcefbe697a5281bc0168"
DTMB5415_SOURCE_URL = (
    "https://raw.githubusercontent.com/mathLab/WaveBEMapp-DTMB-5415/"
    f"{DTMB5415_SOURCE_COMMIT}/goteborg_original.iges"
)
DTMB5415_SOURCE_SHA256 = (
    "a3d1e9d82ff5aec16525f39e1708af4bd71960c56e9584d494554633e60dc8ad"
)


class DTMB5415Region(str, Enum):
    """Representation regions present in the reference half-hull."""

    SONAR_DOME = "sonar_dome"
    SONAR_DOME_TRANSITION = "sonar_dome_transition"
    MAIN_HULL = "main_hull"


@dataclass
class DTMB5415Reference:
    """Exact polynomial B-spline faces from the Gothenburg DTMB 5415 IGES."""

    patches: dict[DTMB5415Region, PolynomialIGESPatch]
    source_path: Path

    def build_functions(self) -> dict[DTMB5415Region, lfs.Function]:
        """Build all exact reference functions in the active CSDL recorder."""
        return {
            region: patch.build_function(f"dtmb_5415_{region.value}_reference")
            for region, patch in self.patches.items()
        }

    def dimensions(self, resolution: tuple[int, int] = (81, 81)) -> dict[str, float]:
        """Return sampled model dimensions in the IGES metre coordinate system."""
        functions = self.build_functions()
        points = []
        for region, patch in self.patches.items():
            _, values = patch.sample_grid(functions[region], resolution)
            points.append(np.asarray(values.value, dtype=float))
        coordinates = np.vstack(points)
        return {
            "overall_length": float(np.ptp(coordinates[:, 0])),
            "beam": float(2.0 * np.max(np.abs(coordinates[:, 1]))),
            "draft_below_z0": float(max(0.0, -np.min(coordinates[:, 2]))),
            "maximum_z": float(np.max(coordinates[:, 2])),
        }


@dataclass
class DTMB5415PatchFit:
    """One approximating patch and its fixed-parameter pointwise errors."""

    region: DTMB5415Region
    surface: TensorProductSurface
    coefficient_shape: tuple[int, int]
    rms_error: float
    maximum_error: float
    sample_count: int
    fitting_sample_count: int


@dataclass
class DTMB5415Approximation:
    """Global and region-resolved DTMB 5415 approximation result."""

    level: str
    patches: dict[DTMB5415Region, DTMB5415PatchFit]
    global_rms_error: float
    global_maximum_error: float
    length_normalized_rms_error: float
    knot_strategy: str = "uniform"

    @property
    def sonar_dome(self) -> DTMB5415PatchFit:
        """Return the mandatory sonar-dome fit."""
        return self.patches[DTMB5415Region.SONAR_DOME]


_FIT_LEVELS: dict[str, dict[DTMB5415Region, tuple[int, int]]] = {
    "coarse": {
        DTMB5415Region.SONAR_DOME: (6, 6),
        DTMB5415Region.SONAR_DOME_TRANSITION: (4, 10),
        DTMB5415Region.MAIN_HULL: (8, 6),
    },
    "medium": {
        DTMB5415Region.SONAR_DOME: (10, 8),
        DTMB5415Region.SONAR_DOME_TRANSITION: (4, 20),
        DTMB5415Region.MAIN_HULL: (12, 8),
    },
    "fine": {
        DTMB5415Region.SONAR_DOME: (18, 14),
        DTMB5415Region.SONAR_DOME_TRANSITION: (4, 28),
        DTMB5415Region.MAIN_HULL: (18, 12),
    },
}


def _reference_aligned_axis_knots(
    patch: PolynomialIGESPatch,
    axis: int,
    target_degree: int,
    target_count: int,
) -> np.ndarray:
    """Construct local knots that retain reference feature locations.

    Interior reference knots are mapped from the active IGES parameter range
    to ``[0, 1]``. Their continuity is retained when the target degree permits
    it. If the requested space has more or fewer interior knots, simple knots
    are inserted in the widest spans or removed to minimize the largest
    remaining span.
    """
    lower, upper = patch.parameter_bounds[axis]
    local = np.clip((patch.knots[axis] - lower) / (upper - lower), 0.0, 1.0)
    values, multiplicities = np.unique(np.round(local, decimals=14), return_counts=True)
    interior: list[float] = []
    for value, source_multiplicity in zip(values, multiplicities):
        if value <= 1.0e-12 or value >= 1.0 - 1.0e-12:
            continue
        source_continuity = patch.degree[axis] - int(source_multiplicity)
        target_multiplicity = max(1, target_degree - source_continuity)
        interior.extend([float(value)] * min(target_multiplicity, target_degree))

    required = target_count - target_degree - 1
    while len(interior) > required:
        candidates: list[tuple[float, int]] = []
        for index, value in enumerate(interior):
            if interior.count(value) > 1:
                continue
            trial = interior[:index] + interior[index + 1 :]
            unique = np.asarray([0.0, *sorted(set(trial)), 1.0])
            candidates.append((float(np.max(np.diff(unique))), index))
        removal_index = min(candidates)[1] if candidates else len(interior) // 2
        interior.pop(removal_index)

    while len(interior) < required:
        unique = np.asarray([0.0, *sorted(set(interior)), 1.0])
        gaps = np.diff(unique)
        span_index = int(np.argmax(gaps))
        interior.append(float(0.5 * (unique[span_index] + unique[span_index + 1])))

    return np.concatenate(
        (
            np.zeros(target_degree + 1),
            np.sort(np.asarray(interior)),
            np.ones(target_degree + 1),
        )
    )


def _reference_aligned_knots(
    patch: PolynomialIGESPatch,
    degree: tuple[int, int],
    coefficient_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference-feature-aligned knots for both surface axes."""
    return tuple(
        _reference_aligned_axis_knots(
            patch,
            axis,
            degree[axis],
            coefficient_shape[axis],
        )
        for axis in range(2)
    )


def _feature_aware_grid(
    knots: tuple[np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    coefficient_shape: tuple[int, int],
) -> np.ndarray:
    """Sample every knot span while retaining a uniform validation baseline."""
    axes: list[np.ndarray] = []
    for axis in range(2):
        unique = np.unique(knots[axis])
        midpoints = 0.5 * (unique[:-1] + unique[1:])
        uniform_count = max(int(resolution[axis]), 2 * coefficient_shape[axis] + 1)
        axes.append(
            np.unique(
                np.concatenate(
                    (np.linspace(0.0, 1.0, uniform_count), unique, midpoints)
                )
            )
        )
    first, second = np.meshgrid(axes[0], axes[1], indexing="ij")
    return np.column_stack((first.ravel(), second.ravel()))


def download_dtmb_5415(destination: str | Path) -> Path:
    """Download the pinned MIT-licensed reference IGES and verify its digest."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DTMB5415_SOURCE_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DTMB5415_SOURCE_SHA256:
        raise ValueError("downloaded DTMB 5415 IGES failed its SHA-256 check.")
    output.write_bytes(payload)
    return output


def load_dtmb_5415(path: str | Path) -> DTMB5415Reference:
    """Load and classify the three DTMB 5415 half-hull B-spline faces."""
    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != DTMB5415_SOURCE_SHA256:
        raise ValueError("DTMB 5415 reference does not match the pinned source digest.")
    surfaces = read_polynomial_iges_surfaces(source)
    if len(surfaces) != 3:
        raise ValueError(
            "DTMB 5415 reference must contain exactly three type-128 faces."
        )
    main_hull = max(surfaces, key=lambda patch: np.max(patch.coefficients[:, :, 0]))
    remaining = [patch for patch in surfaces if patch is not main_hull]
    sonar_dome = min(remaining, key=lambda patch: np.min(patch.coefficients[:, :, 0]))
    transition = next(patch for patch in remaining if patch is not sonar_dome)
    return DTMB5415Reference(
        patches={
            DTMB5415Region.SONAR_DOME: sonar_dome,
            DTMB5415Region.SONAR_DOME_TRANSITION: transition,
            DTMB5415Region.MAIN_HULL: main_hull,
        },
        source_path=source,
    )


def fit_dtmb_5415(
    reference: DTMB5415Reference,
    level: str,
    fitting_resolution: tuple[int, int] = (41, 61),
    evaluation_resolution: tuple[int, int] = (57, 83),
    knot_strategy: Literal["uniform", "reference_aligned"] = "uniform",
) -> DTMB5415Approximation:
    """Fit all regions and report global and dome-resolved errors.

    ``reference_aligned`` treats the longitudinal and transverse knot
    locations as representation variables. It maps feature locations and
    continuity from the source patch into the fitted space, then samples every
    resulting knot span. This separates control-net resolution error from the
    much larger error that can be caused by an unsuitable uniform
    parameterization.
    """
    if level not in _FIT_LEVELS:
        raise ValueError(f"unknown DTMB 5415 fit level {level!r}.")
    if knot_strategy not in ("uniform", "reference_aligned"):
        raise ValueError(f"unknown knot strategy {knot_strategy!r}.")
    functions = reference.build_functions()
    fits: dict[DTMB5415Region, DTMB5415PatchFit] = {}
    all_squared_errors: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    all_reference_points: list[np.ndarray] = []
    for region, patch in reference.patches.items():
        coefficient_shape = _FIT_LEVELS[level][region]
        degree = tuple(min(3, count - 1) for count in coefficient_shape)
        knots = None
        if knot_strategy == "reference_aligned":
            axis_knots = _reference_aligned_knots(patch, degree, coefficient_shape)
            fit_coordinates = _feature_aware_grid(
                axis_knots, fitting_resolution, coefficient_shape
            )
            fit_values = patch.evaluate(functions[region], fit_coordinates)
            knots = np.concatenate(axis_knots)
        else:
            fit_coordinates, fit_values = patch.sample_grid(
                functions[region], fitting_resolution
            )
        surface = fit_offset_surface(
            fit_values,
            fit_coordinates,
            degree=degree,
            coefficients_shape=coefficient_shape,
            knots=knots,
            regularization=(
                1.0e-14 if knot_strategy == "reference_aligned" else 1.0e-12
            ),
            name=f"dtmb_5415_{region.value}_{level}",
        )
        evaluation_coordinates, reference_values = patch.sample_grid(
            functions[region], evaluation_resolution
        )
        fitted_values = surface.evaluate(evaluation_coordinates)
        residual = np.asarray(fitted_values.value - reference_values.value, dtype=float)
        errors = np.linalg.norm(residual, axis=1)
        fits[region] = DTMB5415PatchFit(
            region=region,
            surface=surface,
            coefficient_shape=coefficient_shape,
            rms_error=float(np.sqrt(np.mean(errors**2))),
            maximum_error=float(np.max(errors)),
            sample_count=errors.size,
            fitting_sample_count=fit_coordinates.shape[0],
        )
        all_squared_errors.append(errors**2)
        all_errors.append(errors)
        all_reference_points.append(np.asarray(reference_values.value, dtype=float))
    squared = np.concatenate(all_squared_errors)
    errors = np.concatenate(all_errors)
    reference_points = np.vstack(all_reference_points)
    length = float(np.ptp(reference_points[:, 0]))
    global_rms = float(np.sqrt(np.mean(squared)))
    return DTMB5415Approximation(
        level=level,
        patches=fits,
        global_rms_error=global_rms,
        global_maximum_error=float(np.max(errors)),
        length_normalized_rms_error=global_rms / length,
        knot_strategy=knot_strategy,
    )
