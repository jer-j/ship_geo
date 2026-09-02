# `ship_geo` implementation plan

## Scope

`ship_geo` will construct differentiable ship geometry from naval-architecture
form parameters. It will not begin from an imported baseline hull and will not
use free-form deformation as its primary representation.

The design is organized into three layers:

1. **Design topology:** dimensions, curves of form, section properties, and
   named hull features.
2. **Representation topology:** compatible B-spline curves and surface patches,
   including their shared edges and continuity conditions.
3. **Analysis geometry:** differentiable evaluation points and meshes supplied
   to hydrostatics, resistance, seakeeping, maneuvering, structures, and MDO.

Discrete topology is selected before a gradient-based optimization. Spline
coefficients and continuous form parameters remain differentiable within that
fixed topology.

## Milestone 1: F-Spline kernel

Status: implemented in the initial `ship_geo` milestone.

- Reuse `lsdo_function_spaces.BSplineSpace` for the open B-spline
  representation and `lsdo_b_splines_cython` for sparse basis evaluation.
- Assemble derivative fairness energies with CSDL operations and composite
  Gaussian quadrature.
- Represent form parameters as CSDL equality constraints.
- Form the Lagrangian and its stationarity residual using CSDL derivatives.
- Solve the KKT system with `csdl_alpha.nonlinear_solvers.Newton`.
- Support point, derivative, tangent angle, curvature, signed area, and centroid
  constraints.
- Verify constraint satisfaction, basis derivatives, fairness matrices, and
  derivatives of the implicit solution.
- Demonstrate a family of area-controlled transverse ship sections.

## Milestone 2: curves of form

Status: implemented for scalar longitudinal distributions.

- Add specialized semantics for the sectional-area curve, design waterline,
  centerplane profile, deck edge, chines, and longitudinal parameter
  distributions.
- Use a fixed nondimensional longitudinal coordinate with dimensional CSDL
  ordinates.
- Establish sign, orientation, and moment conventions for every curve type.
- Add deterministic input validation and useful initial-guess construction.

## Milestone 3: section engine

Status: round-bilge and hard-chine templates implemented.

- Define reusable round-bilge and chined section templates from F-Spline
  segments.
- Drive section area from the SAC and endpoints from the waterline and keel
  curves.
- Drive local tangent conditions from deadrise and flare distributions.
- Guarantee common degrees and knot vectors across compatible sections.
- Support flat-of-bottom, flat-of-side, bilge, knuckle, and transom features.

## Milestone 4: hull surfaces

Status: compatible loft, free surface-control-net variational states, global
surface fairness, pointed end closures, waterplane and transom caps, and patch
graph implemented. Every free surface is assembled into the shared
`VariationalSystem`, so several curves, sections, and surface patches can use
one Newton solve. `SectionLoftProblem` supports both the fixed compatible-loft
map and a variational free-surface mode coupled exactly to all generating
sections.

- Skin compatible section-control polygons into tensor-product B-spline
  surfaces.
- Add longitudinal fairness and guide-curve constraints.
- Implement stem, transom, deck, and centerplane caps.
- Introduce a patch graph with positional and tangent-continuity residuals.
- Add free `FSurfaceProblem` control nets to the same global KKT system when a
  section-derived compatible loft is not sufficiently expressive.

## Milestone 5: hydrostatics and validity

Status: general oriented multi-patch hydrostatics and sampled validity fields
implemented. The divergence-theorem formulation is verified against an exact
box, an exact Wigley surface, and a transom-ended triangular prism.

- Integrate volume and first moments over closed parametric surfaces.
- Compute displacement, centers of buoyancy and flotation, waterplane area,
  sectional areas, and wetted area in the CSDL graph.
- Check patch gaps, tangent discontinuities, reversed surface Jacobians,
  negative section breadths, and self-intersections.
- Validate against boxes, ellipsoids, and the Wigley hull before published hull
  forms are introduced.

## Milestone 6: approximation and refinement

Status: exact differentiable curve/surface knot insertion, CSDL offset-table
fitting, residual-driven span indicators, feature-aligned parameterization,
and DTMB 5415 regional convergence validation implemented. The sonar dome,
dome transition, and main hull remain distinct representation regions.

- Fit spline patches to offset tables for validation, not as the primary design
  mechanism.
- Add knot insertion and residual-based refinement.
- Demonstrate regional and global convergence on DTMB 5415, including its
  sonar dome, dome-to-hull transition, main hull, and transom termination.
- Hold representation topology fixed during each gradient-based optimization.

With the same fine control-net sizes, aligning the knot distribution with the
reference feature locations reduces the DTMB 5415 global RMS surface error
from $0.350\ \mathrm{mm}$ to $0.0116\ \mathrm{mm}$ and the maximum error from
$6.003\ \mathrm{mm}$ to $0.378\ \mathrm{mm}$ on the
$6.119\ \mathrm{m}$ reference model. The forward bow and sonar-dome face and
the main hull then reproduce to numerical precision; the remaining error is
localized to the deliberately reduced dome-transition patch.

## Milestone 7: downstream geometry

Status: structured CSDL surface meshes, seam-welded watertightness checks, and
terminal STL/OBJ export are implemented. Production B-rep sewing and STEP
export remain deferred.

- Produce analysis-specific point sets and surface meshes.
- Reuse the applicable `lsdo_geo` projection, refitting, and mesh interfaces.
- Add watertight export after the differentiable internal representation is
  stable.

## Verification policy

Every new primitive must include:

- an analytic or independently computed reference result,
- an absolute and relative residual tolerance,
- derivative verification for every exposed CSDL input,
- a regression test for orientation and sign conventions,
- a minimal runnable example that does not depend on proprietary geometry.
