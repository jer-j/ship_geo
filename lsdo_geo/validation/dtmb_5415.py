"""DTMB 5415 multi-patch approximation validation, including the sonar dome."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

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


@dataclass
class DTMB5415Approximation:
    """Global and region-resolved DTMB 5415 approximation result."""

    level: str
    patches: dict[DTMB5415Region, DTMB5415PatchFit]
    global_rms_error: float
    global_maximum_error: float
    length_normalized_rms_error: float

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
) -> DTMB5415Approximation:
    """Fit all three regions and report global and dome-resolved errors."""
    if level not in _FIT_LEVELS:
        raise ValueError(f"unknown DTMB 5415 fit level {level!r}.")
    functions = reference.build_functions()
    fits: dict[DTMB5415Region, DTMB5415PatchFit] = {}
    all_squared_errors: list[np.ndarray] = []
    all_errors: list[np.ndarray] = []
    all_reference_points: list[np.ndarray] = []
    for region, patch in reference.patches.items():
        fit_coordinates, fit_values = patch.sample_grid(
            functions[region], fitting_resolution
        )
        coefficient_shape = _FIT_LEVELS[level][region]
        degree = tuple(min(3, count - 1) for count in coefficient_shape)
        surface = fit_offset_surface(
            fit_values,
            fit_coordinates,
            degree=degree,
            coefficients_shape=coefficient_shape,
            regularization=1.0e-12,
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
    )
