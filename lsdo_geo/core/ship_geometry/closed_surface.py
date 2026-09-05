"""Closed multi-patch hull topology and exact cap construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np

from .surfaces import TensorProductSurface


def _as_variable(value: Any, anchor: csdl.Variable) -> csdl.Variable:
    """Preserve a CSDL value or lift a scalar using an existing graph anchor."""
    return value if isinstance(value, csdl.Variable) else 0.0 * anchor + value


def _vector3(x: Any, y: Any, z: Any, anchor: csdl.Variable) -> csdl.Variable:
    values = tuple(_as_variable(value, anchor).reshape((1,)) for value in (x, y, z))
    return csdl.concatenate(values)


@dataclass(frozen=True)
class OrientedSurfacePatch:
    """A surface patch with outward-orientation and hydrostatic roles."""

    surface: TensorProductSurface
    normal_sign: float = 1.0
    wetted: bool = True
    waterplane: bool = False
    name: str = "patch"

    def __post_init__(self) -> None:
        if float(self.normal_sign) not in (-1.0, 1.0):
            raise ValueError("normal_sign must be either +1 or -1.")
        if self.waterplane and self.wetted:
            raise ValueError("a waterplane closure cannot be marked as wetted.")


@dataclass
class ClosedSurface:
    """An explicitly oriented collection of closed boundary patches."""

    patches: tuple[OrientedSurfacePatch, ...]

    def __init__(self, patches: Sequence[OrientedSurfacePatch]) -> None:
        if len(patches) < 1:
            raise ValueError("a closed surface requires at least one patch.")
        self.patches = tuple(patches)

    @classmethod
    def from_symmetric_starboard(
        cls,
        starboard: TensorProductSurface,
        waterplane: bool = True,
        transom_edges: Sequence[str] = (),
        name: str = "hull",
    ) -> ClosedSurface:
        """Close a symmetric half-hull with mirrored side and exact caps.

        ``transom_edges`` may contain ``"v0"`` and/or ``"v1"``. Edges not
        listed are assumed to collapse to the centerplane.
        """
        invalid_edges = set(transom_edges) - {"v0", "v1"}
        if invalid_edges:
            raise ValueError(f"unsupported transom edges: {sorted(invalid_edges)}")
        port = mirror_surface_y(starboard, name=f"{name}_port")
        patches = [
            OrientedSurfacePatch(starboard, 1.0, True, False, f"{name}_starboard"),
            OrientedSurfacePatch(port, -1.0, True, False, f"{name}_port"),
        ]
        if waterplane:
            cap = create_waterplane_cap(starboard, name=f"{name}_waterplane")
            patches.append(
                OrientedSurfacePatch(cap, -1.0, False, True, f"{name}_waterplane")
            )
        for edge in transom_edges:
            cap = create_transom_cap(starboard, edge=edge, name=f"{name}_{edge}_cap")
            normal_sign = 1.0 if edge == "v0" else -1.0
            patches.append(
                OrientedSurfacePatch(
                    cap,
                    normal_sign,
                    True,
                    False,
                    f"{name}_{edge}_cap",
                )
            )
        return cls(patches)


def mirror_surface_y(
    surface: TensorProductSurface,
    name: str = "mirrored_surface",
) -> TensorProductSurface:
    """Mirror a surface across ``y=0`` without breaking derivatives."""
    coefficients = surface.coefficients
    mirrored = csdl.concatenate(
        (
            coefficients[:, :, 0].reshape(coefficients.shape[:2] + (1,)),
            (-coefficients[:, :, 1]).reshape(coefficients.shape[:2] + (1,)),
            coefficients[:, :, 2].reshape(coefficients.shape[:2] + (1,)),
        ),
        axis=2,
    )
    return TensorProductSurface(
        lfs.Function(surface.space, mirrored, name=name),
        quadrature_order=surface.quadrature_order,
    )


def create_waterplane_cap(
    starboard: TensorProductSurface,
    name: str = "waterplane_cap",
) -> TensorProductSurface:
    """Create the full symmetric waterplane from the ``u=1`` control row."""
    edge = starboard.coefficients[-1, :, :]
    port = csdl.concatenate(
        (
            edge[:, 0].reshape((edge.shape[0], 1)),
            (-edge[:, 1]).reshape((edge.shape[0], 1)),
            edge[:, 2].reshape((edge.shape[0], 1)),
        ),
        axis=1,
    )
    starboard_edge = edge
    coefficients = csdl.concatenate(
        (port.reshape((1,) + port.shape), starboard_edge.reshape((1,) + edge.shape)),
        axis=0,
    )
    transverse_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(1,),
        coefficients_shape=(2,),
    )
    transverse_knots = transverse_space.knots[transverse_space.knot_indices[0]]
    longitudinal_knots = starboard.space.knots[starboard.space.knot_indices[1]]
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, starboard.space.degree[1]),
        coefficients_shape=(2, edge.shape[0]),
        knots=np.concatenate((transverse_knots, longitudinal_knots)),
    )
    return TensorProductSurface(
        lfs.Function(space, coefficients, name=name),
        quadrature_order=(2, starboard.quadrature_order[1]),
    )


def create_transom_cap(
    starboard: TensorProductSurface,
    edge: str = "v1",
    name: str = "transom_cap",
) -> TensorProductSurface:
    """Create a full symmetric transom cap from a longitudinal boundary row."""
    if edge not in {"v0", "v1"}:
        raise ValueError("transom edge must be 'v0' or 'v1'.")
    boundary = (
        starboard.coefficients[:, 0, :]
        if edge == "v0"
        else starboard.coefficients[:, -1, :]
    )
    port = csdl.concatenate(
        (
            boundary[:, 0].reshape((boundary.shape[0], 1)),
            (-boundary[:, 1]).reshape((boundary.shape[0], 1)),
            boundary[:, 2].reshape((boundary.shape[0], 1)),
        ),
        axis=1,
    )
    coefficients = csdl.concatenate(
        (
            port.reshape((boundary.shape[0], 1, 3)),
            boundary.reshape((boundary.shape[0], 1, 3)),
        ),
        axis=1,
    )
    transverse_knots = starboard.space.knots[starboard.space.knot_indices[0]]
    symmetry_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(1,),
        coefficients_shape=(2,),
    )
    symmetry_knots = symmetry_space.knots[symmetry_space.knot_indices[0]]
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(starboard.space.degree[0], 1),
        coefficients_shape=(boundary.shape[0], 2),
        knots=np.concatenate((transverse_knots, symmetry_knots)),
    )
    return TensorProductSurface(
        lfs.Function(space, coefficients, name=name),
        quadrature_order=(starboard.quadrature_order[0], 2),
    )


def bilinear_patch(
    corners: Sequence[Sequence[Any]],
    name: str = "bilinear_patch",
) -> TensorProductSurface:
    """Create a degree-one patch from corners ordered ``(u, v)``."""
    if len(corners) != 4 or any(len(point) != 3 for point in corners):
        raise ValueError("corners must contain four three-dimensional points.")
    anchor: csdl.Variable | None = None
    for point in corners:
        for value in point:
            if isinstance(value, csdl.Variable):
                anchor = value
                break
        if anchor is not None:
            break
    if anchor is None:
        anchor = csdl.Variable(value=float(corners[0][0]))
    vectors = [_vector3(*point, anchor=anchor).reshape((1, 3)) for point in corners]
    coefficients = csdl.concatenate(
        (
            csdl.concatenate((vectors[0], vectors[1]), axis=0).reshape((1, 2, 3)),
            csdl.concatenate((vectors[2], vectors[3]), axis=0).reshape((1, 2, 3)),
        ),
        axis=0,
    )
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=(1, 1),
        coefficients_shape=(2, 2),
    )
    return TensorProductSurface(
        lfs.Function(space, coefficients, name=name),
        quadrature_order=(2, 2),
    )


def rectangular_box_surface(
    length: Any,
    beam: Any,
    draft: Any,
    name: str = "box",
) -> ClosedSurface:
    """Create an exactly closed box spanning ``z in [-draft, 0]``."""
    anchor = length if isinstance(length, csdl.Variable) else beam
    if not isinstance(anchor, csdl.Variable):
        anchor = draft
    if not isinstance(anchor, csdl.Variable):
        anchor = csdl.Variable(value=float(length))
    length = _as_variable(length, anchor)
    beam = _as_variable(beam, anchor)
    draft = _as_variable(draft, anchor)
    x0, x1 = -0.5 * length, 0.5 * length
    y0, y1 = -0.5 * beam, 0.5 * beam
    z0, z1 = -draft, 0.0 * draft

    definitions = (
        (
            "bottom",
            ((x0, y0, z0), (x1, y0, z0), (x0, y1, z0), (x1, y1, z0)),
            1.0,
            True,
            False,
        ),
        (
            "waterplane",
            ((x0, y0, z1), (x1, y0, z1), (x0, y1, z1), (x1, y1, z1)),
            -1.0,
            False,
            True,
        ),
        (
            "port",
            ((x0, y0, z0), (x1, y0, z0), (x0, y0, z1), (x1, y0, z1)),
            -1.0,
            True,
            False,
        ),
        (
            "starboard",
            ((x0, y1, z0), (x1, y1, z0), (x0, y1, z1), (x1, y1, z1)),
            1.0,
            True,
            False,
        ),
        (
            "bow",
            ((x0, y0, z0), (x0, y1, z0), (x0, y0, z1), (x0, y1, z1)),
            1.0,
            True,
            False,
        ),
        (
            "stern",
            ((x1, y0, z0), (x1, y1, z0), (x1, y0, z1), (x1, y1, z1)),
            -1.0,
            True,
            False,
        ),
    )
    patches = []
    for patch_name, corners, sign, wetted, waterplane in definitions:
        patches.append(
            OrientedSurfacePatch(
                bilinear_patch(corners, f"{name}_{patch_name}"),
                sign,
                wetted,
                waterplane,
                f"{name}_{patch_name}",
            )
        )
    return ClosedSurface(patches)
