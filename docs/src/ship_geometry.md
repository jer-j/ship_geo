# First-principles hull geometry

`ship_geo` now assembles longitudinal curves of form, compatible transverse
F-Spline sections, and a tensor-product B-spline hull surface. The geometry is
created from form constraints rather than by deforming an imported parent
hull.

## One global variational solve

Each curve or section contributes an implicit coefficient state, a fairness
term, and equality constraints to `VariationalSystem`. Cross-curve continuity
and surface-fairness terms may be added before calling `solve()`. The system
forms one Lagrangian,

$$
\mathcal L(\mathbf z,\boldsymbol\lambda;\mathbf q)
=\sum_a J_a(\mathbf z_a)
+J_S(\mathbf z)
+\boldsymbol\lambda^T\mathbf g(\mathbf z,\mathbf q),
$$

and one global KKT residual,

$$
\mathbf R=
\begin{bmatrix}
\nabla_{\mathbf z}\mathcal L\\
\mathbf g
\end{bmatrix}
=\mathbf 0.
$$

`csdl_alpha.Newton` is invoked once. The compatible loft is a fixed CSDL
linear map from the solved transverse control polygons to the surface control
net, so it creates no nested nonlinear solve.

## Constants and differentiable variables

Whether a quantity belongs in the CSDL graph depends on whether a legitimate
design derivative passes through it.

| Quantity | Representation | Reason |
|---|---|---|
| Gauss nodes and weights | NumPy constant | Fixed numerical integration rule |
| Knot vectors, degree, topology | NumPy/Python constant | Fixed discretization during one optimization |
| Evaluation parameters | NumPy constant | Fixed sampling locations |
| Principal dimensions | CSDL variable/expression | Design inputs |
| Section area, breadth, draft | CSDL variable/expression | Form parameters and curve outputs |
| Tangent, curvature, centroid targets | CSDL variable when design-dependent | Preserve implicit design derivatives |
| Curve and surface coefficients | CSDL implicit or explicit variables | Geometry states and outputs |
| Hydrostatic quantities | CSDL variables | Differentiable analysis outputs |

Quadrature data should only become design-dependent if the integration domain,
knot locations, or quadrature rule itself is optimized. That is intentionally
outside the fixed-topology formulation.

## Coordinate convention

A transverse half section stores `(z, y)` and runs from keel to waterline. Its
signed area is therefore

$$
A_{1/2}=\int_{-T}^{0} y(z)\,dz.
$$

The starboard hull surface stores `(x, y, z)`, with `u` increasing from keel to
waterline and `v` increasing from bow to stern. Compatible section control
polygons are fitted longitudinally to form

$$
\mathbf S(u,v)=
\sum_i\sum_j N_{i,p}(u)M_{j,q}(v)\mathbf P_{ij}.
$$

Pointed bow and stern sections are explicit collapsed centerplane curves
derived from the nearest interior section. Hard chines use an interior knot of
multiplicity equal to the spline degree, producing a $C^0$ join.

## Hydrostatics

For a symmetric starboard half surface with outward area vector
$\mathbf n\,dA=\mathbf S_u\times\mathbf S_v\,du\,dv$, volume is evaluated as

$$
\nabla=2\int_S y n_y\,dA.
$$

The implementation also computes LCB, VCB, waterplane area, LCF, transverse
and longitudinal waterplane moments of inertia, sectional areas, and wetted
surface area. The exact quadratic Wigley representation verifies

$$
\nabla=\frac{4}{9}LBT,
\qquad
z_B=-\frac{3}{8}T,
\qquad
A_{WP}=\frac{2}{3}LB.
$$

For a general closed patch collection, every patch carries an outward normal
sign and explicit `wetted` and `waterplane` roles. Volume and first moments use
the full divergence-theorem identities

$$
\nabla=\frac{1}{3}\int_{\partial V}\mathbf r\mathbin{\cdot}\mathbf n\,dA,
\qquad
\int_V x_i\,dV=\frac{1}{2}\int_{\partial V}x_i^2 n_i\,dA.
$$

`ClosedSurface.from_symmetric_starboard()` mirrors a starboard hull and can
construct exact waterplane and transom caps directly from compatible boundary
control rows. Analytic box and prismatic-transom tests verify orientation,
volume, centroids, waterplane inertias, wetted area, and CSDL derivatives.

## Approximation, topology, and validity

`refine_curve()` and `refine_surface()` perform exact nested-space knot
insertion through differentiable linear maps. `fit_offset_surface()` provides
a CSDL linear least-squares inverse fit for validation against offset tables.
The fixed patch topology is recorded by `PatchGraph`, which can add sampled
positional and tangent-continuity constraints to the same global variational
system.

`evaluate_surface_validity()` exposes sampled half-breadths, surface Jacobian
magnitudes, coordinate monotonicity, centerplane gaps, and waterline errors as
CSDL arrays. They may be used directly as optimizer constraints or inspected
numerically through `SurfaceValidity.report()`.

`evaluate_closed_surface_closure()` matches sampled patch boundaries in either
direction. `section_self_intersections()` detects crossings in transverse
polylines, while residual span indicators identify knot intervals that need
refinement. Structured CSDL meshes preserve derivatives at their vertices;
their current numerical state can be seam-welded for watertightness checks and
exported to STL or OBJ as a terminal, non-differentiable operation.

The next published validation hull is DTMB 5415. Its sonar dome is represented
as a separate connected region and receives its own approximation-error and
refinement report, rather than being smoothed into the main forebody.

![One-solve F-Spline hull](images/first_principles_fspline_hull.png)
