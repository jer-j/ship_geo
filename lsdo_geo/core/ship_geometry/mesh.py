"""Differentiable analysis meshes and terminal neutral-file export."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import csdl_alpha as csdl
import numpy as np

from .closed_surface import ClosedSurface
from .surfaces import TensorProductSurface


@dataclass
class SurfaceMesh:
    """CSDL vertex coordinates with fixed triangular connectivity."""

    vertices: csdl.Variable
    triangles: np.ndarray
    patch_ranges: tuple[tuple[str, int, int], ...] = ()

    def current_vertices(self) -> np.ndarray:
        """Return the current numerical vertex coordinates."""
        return np.asarray(self.vertices.value, dtype=float)


@dataclass(frozen=True)
class WatertightMeshReport:
    """Topological edge counts after geometric vertex welding."""

    boundary_edges: int
    nonmanifold_edges: int
    welded_vertices: int
    triangles: int

    @property
    def is_watertight(self) -> bool:
        """Return whether every edge belongs to exactly two triangles."""
        return self.boundary_edges == 0 and self.nonmanifold_edges == 0


def tessellate_surface(
    surface: TensorProductSurface,
    resolution: tuple[int, int] = (31, 61),
    normal_sign: float = 1.0,
    name: str = "surface",
) -> SurfaceMesh:
    """Create a structured triangular mesh while retaining CSDL vertices."""
    if resolution[0] < 2 or resolution[1] < 2:
        raise ValueError("surface mesh resolution must be at least (2, 2).")
    if normal_sign not in (-1.0, 1.0):
        raise ValueError("normal_sign must be either +1 or -1.")
    u = np.linspace(0.0, 1.0, resolution[0])
    v = np.linspace(0.0, 1.0, resolution[1])
    u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
    vertices = surface.evaluate(coordinates)
    indices = np.arange(vertices.shape[0]).reshape(resolution)
    triangles: list[tuple[int, int, int]] = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            lower_left = int(indices[i, j])
            lower_right = int(indices[i, j + 1])
            upper_left = int(indices[i + 1, j])
            upper_right = int(indices[i + 1, j + 1])
            if normal_sign > 0.0:
                triangles.extend(
                    (
                        (lower_left, upper_left, upper_right),
                        (lower_left, upper_right, lower_right),
                    )
                )
            else:
                triangles.extend(
                    (
                        (lower_left, upper_right, upper_left),
                        (lower_left, lower_right, upper_right),
                    )
                )
    triangle_array = np.asarray(triangles, dtype=np.int64)
    return SurfaceMesh(
        vertices=vertices,
        triangles=triangle_array,
        patch_ranges=((name, 0, triangle_array.shape[0]),),
    )


def tessellate_closed_surface(
    closed_surface: ClosedSurface,
    resolutions: tuple[int, int] | Sequence[tuple[int, int]] = (31, 61),
) -> SurfaceMesh:
    """Tessellate all oriented patches without severing vertex derivatives."""
    if (
        isinstance(resolutions, tuple)
        and len(resolutions) == 2
        and all(isinstance(value, int) for value in resolutions)
    ):
        patch_resolutions = [resolutions] * len(closed_surface.patches)
    else:
        patch_resolutions = list(resolutions)
        if len(patch_resolutions) != len(closed_surface.patches):
            raise ValueError("one mesh resolution is required per patch.")

    vertex_blocks: list[csdl.Variable] = []
    triangle_blocks: list[np.ndarray] = []
    patch_ranges: list[tuple[str, int, int]] = []
    vertex_offset = 0
    triangle_offset = 0
    for patch, resolution in zip(closed_surface.patches, patch_resolutions):
        mesh = tessellate_surface(
            patch.surface,
            resolution=resolution,
            normal_sign=patch.normal_sign,
            name=patch.name,
        )
        vertex_blocks.append(mesh.vertices)
        triangle_blocks.append(mesh.triangles + vertex_offset)
        start = triangle_offset
        triangle_offset += mesh.triangles.shape[0]
        patch_ranges.append((patch.name, start, triangle_offset))
        vertex_offset += mesh.vertices.shape[0]
    vertices = (
        vertex_blocks[0]
        if len(vertex_blocks) == 1
        else csdl.concatenate(tuple(vertex_blocks), axis=0)
    )
    return SurfaceMesh(
        vertices=vertices,
        triangles=np.vstack(triangle_blocks),
        patch_ranges=tuple(patch_ranges),
    )


def evaluate_watertight_mesh(
    mesh: SurfaceMesh,
    tolerance: float = 1.0e-8,
) -> WatertightMeshReport:
    """Weld current coincident vertices and count edge incidences."""
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    vertices = mesh.current_vertices()
    quantized = np.round(vertices / tolerance).astype(np.int64)
    _, welded_indices = np.unique(quantized, axis=0, return_inverse=True)
    triangles = welded_indices[mesh.triangles]
    nondegenerate = np.logical_and.reduce(
        (
            triangles[:, 0] != triangles[:, 1],
            triangles[:, 1] != triangles[:, 2],
            triangles[:, 2] != triangles[:, 0],
        )
    )
    triangles = triangles[nondegenerate]
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return WatertightMeshReport(
        boundary_edges=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        welded_vertices=int(np.unique(welded_indices).size),
        triangles=int(triangles.shape[0]),
    )


def export_ascii_stl(
    mesh: SurfaceMesh, path: str | Path, name: str = "ship_geo"
) -> Path:
    """Export the current mesh state as an ASCII STL file."""
    output = Path(path)
    vertices = mesh.current_vertices()
    with output.open("w", encoding="utf-8") as stream:
        stream.write(f"solid {name}\n")
        for triangle in mesh.triangles:
            points = vertices[triangle]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            magnitude = np.linalg.norm(normal)
            if magnitude > 0.0:
                normal = normal / magnitude
            else:
                normal = np.zeros(3)
            stream.write(
                f"  facet normal {normal[0]:.16g} {normal[1]:.16g} {normal[2]:.16g}\n"
            )
            stream.write("    outer loop\n")
            for point in points:
                stream.write(
                    f"      vertex {point[0]:.16g} {point[1]:.16g} {point[2]:.16g}\n"
                )
            stream.write("    endloop\n  endfacet\n")
        stream.write(f"endsolid {name}\n")
    return output


def export_obj(mesh: SurfaceMesh, path: str | Path) -> Path:
    """Export the current mesh state as a Wavefront OBJ file."""
    output = Path(path)
    vertices = mesh.current_vertices()
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# ship_geo differentiable surface mesh\n")
        for point in vertices:
            stream.write(f"v {point[0]:.16g} {point[1]:.16g} {point[2]:.16g}\n")
        for triangle in mesh.triangles + 1:
            stream.write(f"f {triangle[0]} {triangle[1]} {triangle[2]}\n")
    return output
