"""Differentiable nested-space refinement and inverse spline fitting."""

from __future__ import annotations

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import numpy as np
import numpy.typing as npt

from lsdo_geo.core.splines.f_spline import FSplineCurve

from .surfaces import TensorProductSurface


def _greville(knots: np.ndarray, degree: int, count: int) -> np.ndarray:
    if degree == 0:
        return 0.5 * (knots[:count] + knots[1 : count + 1])
    return np.array(
        [np.mean(knots[index + 1 : index + degree + 1]) for index in range(count)]
    )


def _contains_knots(coarse: np.ndarray, refined: np.ndarray) -> bool:
    """Return whether every coarse knot occurs with sufficient multiplicity."""
    values, counts = np.unique(coarse, return_counts=True)
    for value, count in zip(values, counts):
        if np.count_nonzero(np.isclose(refined, value, atol=1.0e-14)) < count:
            return False
    return True


def refine_curve(
    curve: FSplineCurve,
    knots: npt.ArrayLike,
    name: str | None = None,
) -> FSplineCurve:
    """Insert knots into a curve with an exact differentiable linear map."""
    refined_knots = np.asarray(knots, dtype=float).reshape(-1)
    degree = int(curve.space.degree[0])
    coarse_knots = np.asarray(
        curve.space.knots[curve.space.knot_indices[0]], dtype=float
    )
    if not _contains_knots(coarse_knots, refined_knots):
        raise ValueError("refined knot vector must contain the original knots.")
    count = refined_knots.size - degree - 1
    if count <= degree:
        raise ValueError("invalid refined knot vector.")
    refined_space = lfs.BSplineSpace(
        num_parametric_dimensions=1,
        degree=(degree,),
        coefficients_shape=(count,),
        knots=refined_knots,
    )
    parameters = _greville(refined_knots, degree, count)
    values = curve.evaluate(parameters)
    basis = refined_space.compute_basis_matrix(parameters[:, None]).toarray()
    if np.linalg.matrix_rank(basis) < count:
        raise ValueError("refined Greville collocation matrix is singular.")
    transformation = np.linalg.solve(basis, np.eye(count))
    coefficients = transformation @ values
    return FSplineCurve(
        function=lfs.Function(
            refined_space,
            coefficients,
            name=name or f"{curve.function.name}_refined",
        ),
        quadrature_order=curve.quadrature_order,
    )


def refine_surface(
    surface: TensorProductSurface,
    u_knots: npt.ArrayLike,
    v_knots: npt.ArrayLike,
    name: str | None = None,
) -> TensorProductSurface:
    """Insert knots in both surface directions using CSDL operations."""
    new_u = np.asarray(u_knots, dtype=float).reshape(-1)
    new_v = np.asarray(v_knots, dtype=float).reshape(-1)
    degrees = tuple(int(value) for value in surface.space.degree)
    old_u = np.asarray(surface.space.knots[surface.space.knot_indices[0]], dtype=float)
    old_v = np.asarray(surface.space.knots[surface.space.knot_indices[1]], dtype=float)
    if not _contains_knots(old_u, new_u) or not _contains_knots(old_v, new_v):
        raise ValueError("refined knot vectors must contain the original knots.")
    shape = (new_u.size - degrees[0] - 1, new_v.size - degrees[1] - 1)
    refined_space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=degrees,
        coefficients_shape=shape,
        knots=np.concatenate((new_u, new_v)),
    )
    u_parameters = _greville(new_u, degrees[0], shape[0])
    v_parameters = _greville(new_v, degrees[1], shape[1])
    u_grid, v_grid = np.meshgrid(u_parameters, v_parameters, indexing="ij")
    coordinates = np.column_stack((u_grid.ravel(), v_grid.ravel()))
    values = surface.evaluate(coordinates)
    basis = refined_space.compute_basis_matrix(coordinates).toarray()
    if np.linalg.matrix_rank(basis) < basis.shape[0]:
        raise ValueError("refined surface collocation matrix is singular.")
    transformation = np.linalg.solve(basis, np.eye(basis.shape[0]))
    coefficients = (transformation @ values).reshape(shape + (3,))
    return TensorProductSurface(
        lfs.Function(
            refined_space,
            coefficients,
            name=name or f"{surface.function.name}_refined",
        ),
        quadrature_order=surface.quadrature_order,
    )


def fit_offset_surface(
    values: csdl.Variable | npt.ArrayLike,
    parametric_coordinates: npt.ArrayLike,
    degree: tuple[int, int],
    coefficients_shape: tuple[int, int],
    knots: npt.ArrayLike | None = None,
    regularization: float = 0.0,
    name: str = "offset_fit",
) -> TensorProductSurface:
    """Fit an offset table with a fixed CSDL linear least-squares solve.

    This utility is intended for approximation validation and inverse fitting,
    not as the primary first-principles hull definition.
    """
    coordinates = np.asarray(parametric_coordinates, dtype=float).reshape((-1, 2))
    if isinstance(values, csdl.Variable):
        fitting_values = values.reshape((coordinates.shape[0], 3))
    else:
        array = np.asarray(values, dtype=float).reshape((coordinates.shape[0], 3))
        fitting_values = csdl.Variable(value=array)
    if regularization < 0.0:
        raise ValueError("regularization must be nonnegative.")
    space = lfs.BSplineSpace(
        num_parametric_dimensions=2,
        degree=degree,
        coefficients_shape=coefficients_shape,
        knots=None if knots is None else np.asarray(knots, dtype=float),
    )
    basis = space.compute_basis_matrix(coordinates).toarray()
    normal = basis.T @ basis + regularization * np.eye(basis.shape[1])
    if np.linalg.matrix_rank(normal) < normal.shape[0]:
        raise ValueError("offset fitting system is rank deficient.")
    fitting_map = np.linalg.solve(normal, basis.T)
    coefficients = (fitting_map @ fitting_values).reshape(coefficients_shape + (3,))
    return TensorProductSurface(lfs.Function(space, coefficients, name=name))
