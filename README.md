# ship_geo

`ship_geo` extends [`lsdo_geo`](https://github.com/LSDOlab/lsdo_geo) into a
first-principles, differentiable geometry system for ship design. The first
implemented geometry primitive is the F-Spline: a B-spline whose control
points are obtained by minimizing a fairness functional subject to geometric
form constraints.

The package uses `csdl_alpha` to express the Karush-Kuhn-Tucker residual and
solve it with the CSDL Newton solver. Curves of form, compatible transverse
sections, and coupled surface-fairness terms can be assembled into one global
solve and remain in the same computational graph used by a multidisciplinary
design optimization.

## Implemented geometry

- Open, non-rational B-spline curves.
- First- and second-derivative fairness measures supported by the current
  Cython basis backend.
- Point, derivative, tangent-angle, curvature, area, and centroid constraints.
- Newton solution of the complete F-Spline KKT system.
- Analytic implicit derivatives through the CSDL graph.
- Scalar longitudinal curves of form with value, derivative, integral, and
  moment constraints.
- Round-bilge and hard-chine compatible section templates.
- One-solve section-family assembly and tensor-product surface lofting.
- Free F-Surface control nets with thin-plate fairness, geometric constraints,
  and multi-patch continuity in the same global Newton solve.
- Pointed boundaries, exact waterplane/transom caps, and a surface
  patch-connectivity graph.
- General oriented multi-patch hydrostatics, including displacement, centers,
  waterplane properties, sectional areas, and wetted area.
- Sampled CSDL validity fields, closure and watertightness diagnostics, exact
  knot refinement, residual indicators, and offset-table fitting.
- Differentiable structured meshes with terminal STL and OBJ export.
- Analytic box, prism, and Wigley verification.
- Regional DTMB 5415 convergence validation that explicitly retains and
  reports the sonar dome and dome-to-hull transition.

NURBS and McCulloch-style relational feasibility contraction remain outside
the current scope. The development sequence is documented in
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

## Installation

```bash
git clone https://github.com/jer-j/ship_geo.git
cd ship_geo
python -m pip install -e ".[test]"
```

The implementation reuses `lsdo_function_spaces.BSplineSpace` and its
`lsdo_b_splines_cython` basis backend. It does not duplicate basis evaluation
or use SciPy in the ship-specific F-Spline layer. Curve operations, form
constraints, the KKT system, Newton solution, and implicit derivatives are
CSDL operations. The current upstream Cython 0.29.28 build supports Python
3.10; `ship_geo` will broaden its Python range when that upstream build does.

## F-Spline example

```python
import numpy as np
import csdl_alpha as csdl

from lsdo_geo import FSplineProblem

recorder = csdl.Recorder(inline=True)
recorder.start()

problem = FSplineProblem(
    num_control_points=8,
    degree=3,
    physical_dimension=2,
    fairness_weights={2: 1.0},
)
problem.add_point_constraint(0.0, [0.0, 0.0])
problem.add_point_constraint(1.0, [4.0, 5.0])
problem.add_tangent_angle_constraint(0.0, np.deg2rad(80.0))
problem.add_tangent_angle_constraint(1.0, np.deg2rad(15.0))
problem.add_area_constraint(14.0)

curve = problem.solve()
section_points = curve.evaluate(np.linspace(0.0, 1.0, 101)).value

recorder.stop()
```

Run the complete demonstration with:

```bash
python examples/f_spline_ship_section.py
python examples/first_principles_fspline_hull.py
python examples/f_surface_global_solve.py
python examples/dtmb_5415_validation.py
```

The scripts write section-family, coupled-surface, complete-hull, and DTMB 5415
validation figures and print KKT, hydrostatic, derivative, validity, and
regional approximation diagnostics.

## Attribution

The general geometry, function-space, and CSDL infrastructure originated in
the LSDO Lab `lsdo_geo` project. The ship-specific F-Spline formulation follows
the constrained fairness approach developed by Harries and later documented
and extended by McCulloch.
