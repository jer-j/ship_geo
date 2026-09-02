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
import numpy.typing as npt

from ..core.ship_geometry.form_parameter_hull import (
    LongitudinalFitTargets,
    NavalHullParameters,
)
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


@dataclass(frozen=True)
class DTMB5415FormData:
    """Canonical particulars and extracted underwater auxiliary distributions.

    ``fit_targets`` follows the complete wetted envelope, so its forward draft
    observations include the sonar-dome protrusion. ``station_regions`` makes
    that component identity explicit. The primary ``draft`` and its maximum
    location are measured only on the main-hull face; the dome depth is retained
    separately in ``measured_particulars``.
    """

    primary_parameters: NavalHullParameters
    fit_targets: LongitudinalFitTargets
    measured_particulars: dict[str, float]
    longitudinal_coordinates: np.ndarray
    station_regions: tuple[DTMB5415Region, ...]


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


def _constant_x_section(
    patch: PolynomialIGESPatch,
    function: lfs.Function,
    x_coordinate: float,
    search_resolution: int,
    transverse_resolution: int,
) -> np.ndarray:
    """Intersect one regular patch with a constant-x transverse plane."""
    u = np.linspace(0.0, 1.0, search_resolution)
    v = np.linspace(0.0, 1.0, transverse_resolution)
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
    values = np.asarray(patch.evaluate(function, coordinates).value).reshape(
        (search_resolution, transverse_resolution, 3)
    )
    intersections: list[tuple[float, float]] = []
    for transverse_index, transverse_parameter in enumerate(v):
        distance = values[:, transverse_index, 0] - x_coordinate
        closest = int(np.argmin(np.abs(distance)))
        tolerance = 1.0e-12 * max(1.0, float(np.ptp(values[:, transverse_index, 0])))
        if abs(distance[closest]) <= tolerance:
            intersections.append((float(u[closest]), float(transverse_parameter)))
            continue
        crossing = np.flatnonzero(distance[:-1] * distance[1:] <= 0.0)
        if crossing.size == 0:
            continue
        index = int(crossing[0])
        denominator = distance[index] - distance[index + 1]
        fraction = 0.0 if abs(denominator) < 1.0e-14 else distance[index] / denominator
        intersections.append(
            (
                float(u[index] + fraction * (u[index + 1] - u[index])),
                float(transverse_parameter),
            )
        )
    if not intersections:
        raise ValueError(f"no patch intersection found at x={x_coordinate:.8g}.")
    return np.asarray(patch.evaluate(function, intersections).value, dtype=float)


def _underwater_section_properties(points: np.ndarray) -> tuple[float, ...]:
    """Measure waterline breadth, draft, area, deadrise, and flare."""
    ordered = points[np.argsort(points[:, 2])]
    submerged = ordered[ordered[:, 2] <= 0.0]
    emerged = ordered[ordered[:, 2] > 0.0]
    if submerged.shape[0] < 3 or emerged.shape[0] == 0:
        raise ValueError("section does not contain a resolved z=0 waterline crossing.")
    lower = submerged[-1]
    upper = emerged[0]
    fraction = -lower[2] / (upper[2] - lower[2])
    waterline = lower + fraction * (upper - lower)
    underwater = np.vstack((submerged, waterline))
    z = underwater[:, 2]
    y = underwater[:, 1]
    half_area = abs(float(np.trapz(y, z)))
    deadrise = float(np.arctan2(z[1] - z[0], y[1] - y[0]))
    flare = float(np.arctan2(y[-1] - y[-2], z[-1] - z[-2]))
    return float(y[-1]), float(-np.min(z)), half_area, deadrise, flare


def extract_dtmb_5415_form_data(
    reference: DTMB5415Reference,
    station_parameters: npt.ArrayLike | None = None,
    integration_station_count: int = 61,
    search_resolution: int = 121,
    transverse_resolution: int = 241,
) -> DTMB5415FormData:
    """Extract MIT-style underwater form data from the canonical geometry.

    The canonical model particulars follow the SIMMAN DTMB 5415 definition:
    ``Lpp=5.719 m``, ``Bwl=0.768 m``, ``T=0.248 m``, and
    ``displacement=0.554 m^3``. LCB and waterplane coefficient are measured
    from constant-x intersections of the pinned IGES surface at ``z=0``.
    The forward-bow face, including the sonar dome, remains a distinct source
    component during extraction.
    """
    if integration_station_count < 11:
        raise ValueError("integration_station_count must be at least 11.")
    if search_resolution < 11 or transverse_resolution < 21:
        raise ValueError("section intersection resolutions are too small.")
    requested = (
        np.linspace(0.03, 1.0, 13)
        if station_parameters is None
        else np.asarray(station_parameters, dtype=float).reshape(-1)
    )
    if (
        requested.size < 3
        or np.any(np.diff(requested) <= 0.0)
        or np.any((requested <= 0.0) | (requested > 1.0))
    ):
        raise ValueError("station_parameters must increase inside (0, 1].")

    functions = reference.build_functions()
    length = 5.719
    beam = 0.768
    draft = 0.248
    displacement = 0.554
    main_patch = reference.patches[DTMB5415Region.MAIN_HULL]
    main_function = functions[DTMB5415Region.MAIN_HULL]
    aft_edge = main_patch.evaluate(
        main_function,
        np.column_stack((np.ones(101), np.linspace(0.0, 1.0, 101))),
    )
    aft_perpendicular = float(np.mean(np.asarray(aft_edge.value)[:, 0]))
    forward_perpendicular = aft_perpendicular - length
    transition_patch = reference.patches[DTMB5415Region.SONAR_DOME_TRANSITION]
    transition_bounds = (
        float(np.min(transition_patch.coefficients[:, :, 0])),
        float(np.max(transition_patch.coefficients[:, :, 0])),
    )

    def region_at(x_coordinate: float) -> DTMB5415Region:
        if x_coordinate < transition_bounds[0]:
            return DTMB5415Region.SONAR_DOME
        if x_coordinate < transition_bounds[1]:
            return DTMB5415Region.SONAR_DOME_TRANSITION
        return DTMB5415Region.MAIN_HULL

    def properties(parameters: np.ndarray) -> np.ndarray:
        output = np.zeros((parameters.size, 5))
        for index, parameter in enumerate(parameters):
            x_coordinate = forward_perpendicular + length * float(parameter)
            region = region_at(x_coordinate)
            points = _constant_x_section(
                reference.patches[region],
                functions[region],
                x_coordinate,
                search_resolution,
                transverse_resolution,
            )
            output[index] = _underwater_section_properties(points)
        return output

    integration_parameters = np.linspace(0.005, 1.0, integration_station_count)
    integrated = properties(integration_parameters)
    full_parameters = np.concatenate(([0.0], integration_parameters))
    half_breadths = np.concatenate(([0.0], integrated[:, 0]))
    half_areas = np.concatenate(([0.0], integrated[:, 2]))
    x_relative = length * (full_parameters - 0.5)
    measured_volume = 2.0 * float(np.trapz(half_areas, x_relative))
    measured_lcb = float(
        np.trapz(x_relative * half_areas, x_relative) / np.trapz(half_areas, x_relative)
    )
    waterplane_area = 2.0 * float(np.trapz(half_breadths, x_relative))
    waterplane_coefficient = waterplane_area / (length * beam)

    sampled = properties(requested)
    maximum_beam_index = int(np.argmax(integrated[:, 0]))
    integration_coordinates = forward_perpendicular + length * integration_parameters
    main_hull_indices = np.flatnonzero(integration_coordinates >= transition_bounds[1])
    maximum_draft_index = int(
        main_hull_indices[np.argmax(integrated[main_hull_indices, 1])]
    )
    maximum_underwater_depth = float(np.max(integrated[:, 1]))
    requested_coordinates = forward_perpendicular + length * requested
    targets = LongitudinalFitTargets(
        station_parameters=requested,
        half_breadths=sampled[:, 0],
        half_areas=sampled[:, 2],
        drafts=sampled[:, 1],
        deadrise_angles=sampled[:, 3],
        flare_angles=sampled[:, 4],
        maximum_beam_parameter=float(integration_parameters[maximum_beam_index]),
        maximum_draft_parameter=float(integration_parameters[maximum_draft_index]),
    )
    return DTMB5415FormData(
        primary_parameters=NavalHullParameters(
            length_between_perpendiculars=length,
            beam=beam,
            draft=draft,
            displacement=displacement,
            lcb=measured_lcb,
            waterplane_coefficient=waterplane_coefficient,
        ),
        fit_targets=targets,
        measured_particulars={
            "length_between_perpendiculars": length,
            "beam_at_waterline": 2.0 * float(np.max(integrated[:, 0])),
            "canonical_moulded_draft": draft,
            "main_hull_draft_from_sections": float(integrated[maximum_draft_index, 1]),
            "maximum_underwater_depth_including_sonar_dome": (maximum_underwater_depth),
            "sonar_dome_protrusion_below_moulded_draft": (
                maximum_underwater_depth - draft
            ),
            "displacement_from_sections": measured_volume,
            "canonical_displacement": displacement,
            "lcb_from_midships": measured_lcb,
            "waterplane_area": waterplane_area,
            "waterplane_coefficient": waterplane_coefficient,
        },
        longitudinal_coordinates=requested_coordinates,
        station_regions=tuple(region_at(x) for x in requested_coordinates),
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
