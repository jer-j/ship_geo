"""Whole-weather-deck F-splines and exact topside boundary replacement.

The deck coordinate runs from the stem tip to the transom, not from FP to
AP. Assembly registers two scalar curves with the caller's variational
system; it never starts a separate nonlinear solve.
"""

import csdl_alpha as csdl
import numpy as np

from .form_curves import FormCurveKind, FormCurveProblem, piecewise_open_knots


def assemble_deck_edge(system, parameters, half_breadths, heights, *,
                       num_control_points=24, fairness_weight=1.e-8,
                       name="weather_deck"):
    """Register breadth and sheer fits in one existing CSDL system.

    Parameters
    ----------
    system : VariationalSystem
        Shared, unsolved hull system.
    parameters : array_like
        Strictly increasing fixed deck coordinates, including 0 and 1.
    half_breadths, heights : csdl.Variable or array_like
        Normalized ordinates with shape (n,). Use one fixed length scale for
        both ordinates. CSDL targets retain their design derivatives.
    num_control_points : int
        Coefficients per cubic curve. Simple knots allocate extra resolution
        to the forward 25 percent, preserving C2 at every interior knot.
    fairness_weight : float
        Positive weight on the integral of squared second derivatives.

    Returns
    -------
    tuple[FormCurveAssembly, FormCurveAssembly]
        Breadth and height assemblies. Finalize both with the result of the
        caller's single solve. Endpoints are exact; interior data are soft.
    """
    u = np.asarray(parameters, dtype=float)
    if (u.ndim != 1 or u.size < 4 or not np.all(np.isfinite(u))
            or np.any(np.diff(u) <= 0) or u[0] != 0 or u[-1] != 1):
        raise ValueError("Deck parameters must increase strictly from 0 to 1.")
    if not np.isfinite(fairness_weight) or fairness_weight <= 0:
        raise ValueError("fairness_weight must be finite and positive.")
    targets = tuple(value if isinstance(value, csdl.Variable)
                    else csdl.Variable(value=np.asarray(value, dtype=float))
                    for value in (half_breadths, heights))
    if any(value.shape != (u.size,) for value in targets):
        raise ValueError("Each deck target must have shape (number of samples,).")
    knots = piecewise_open_knots(num_control_points, 3, (0.12, 0.25, 0.8),
                                 (0.25, 0.25, 0.3, 0.2))
    assemblies = []
    for kind, target in zip((FormCurveKind.DECK_EDGE, FormCurveKind.DECK_HEIGHT), targets):
        problem = FormCurveProblem(
            kind, num_control_points=num_control_points, knots=knots,
            fairness_weights={2: fairness_weight}, name=f"{name}_{kind.value}")
        problem.add_value_constraint(0., target[0])
        problem.add_value_constraint(1., target[-1])
        assembly = problem.assemble(system)
        residual = assembly.curve.evaluate(u).reshape((u.size,)) - target
        system.add_objective(csdl.sum(residual ** 2) / u.size)
        assemblies.append(assembly)
    return tuple(assemblies)


def apply_deck_edge(control_net, deck_coefficients, transverse_blend):
    r"""Set the deck boundary exactly while retaining the waterline two-jet.

    For compatible clamped spaces, return coefficients of
    ``S(u,v) + g(v) * (D(u) - S(u,1))``. Inputs have shapes (nu,nv,3),
    (nu,3), and (nv,). They are coefficients, never sampled coordinates.
    The caller must express all inputs in common spaces. A degree-five
    Bezier transverse blend [0,0,0,1,1,1] represents
    ``g(v)=10*v**3-15*v**4+6*v**5``. It has zero first and second derivatives
    at both ends, so the retained waterline two-jet is unchanged.

    This operation alone does not preserve an already replaced stem unless
    its deck corner agrees with D(0). Apply the deck first and then a
    corner-compatible stem, or enforce that compatibility explicitly.
    """
    if len(control_net.shape) != 3 or control_net.shape[-1] != 3:
        raise ValueError("control_net must have shape (nu,nv,3).")
    nu, nv, _ = control_net.shape
    if deck_coefficients.shape != (nu, 3) or transverse_blend.shape != (nv,):
        raise ValueError("Deck and blend coefficients must use compatible spaces.")
    delta = deck_coefficients - control_net[:, -1, :].reshape((nu, 3))
    return control_net + csdl.expand(delta, (nu, nv, 3), "ik->ijk") * csdl.expand(
        transverse_blend, (nu, nv, 3), "j->ijk")
