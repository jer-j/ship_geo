# F-Spline kernel

An F-Spline is a B-spline whose coefficients are found from a fairing problem,
rather than supplied directly. For a planar curve,

$$
\mathbf r(t)=\sum_{i=0}^{n-1}N_{i,p}(t)\mathbf P_i,\qquad 0\le t\le 1,
$$

`lsdo_function_spaces.BSplineSpace` owns the spline representation and
`lsdo_b_splines_cython` evaluates its sparse basis matrices. `ship_geo` adds
the variational problem

$$
\min_{\mathbf P}\ J(\mathbf P)
=\sum_k \alpha_k\int_0^1
\left\lVert\mathbf r^{(k)}(t)\right\rVert^2\,dt,
\qquad
\mathbf g(\mathbf P,\mathbf q)=\mathbf 0,
$$

where $\mathbf q$ contains form parameters such as endpoints, tangent angles,
area, and centroid. Composite Gauss-Legendre quadrature turns each fairness
integral into CSDL operations.

The implementation forms the Lagrangian

$$
\mathcal L(\mathbf P,\boldsymbol\lambda;\mathbf q)
=J(\mathbf P)+\boldsymbol\lambda^T\mathbf g(\mathbf P,\mathbf q)
$$

and solves its KKT residual with `csdl_alpha.nonlinear_solvers.Newton`:

$$
\mathbf F=
\begin{bmatrix}
\nabla_{\mathbf P}\mathcal L\\
\mathbf g
\end{bmatrix}
=\mathbf 0.
$$

Because the state and residual are CSDL variables, downstream quantities can
differentiate through the implicit solution. No ship-specific SciPy solve or
duplicate B-spline basis implementation is used.

## Implemented form constraints

| Constraint | Residual |
|---|---|
| Point | $\mathbf r(t_i)-\mathbf q_i$ |
| Derivative | $\mathbf r^{(k)}(t_i)-\mathbf q_i$ |
| Tangent angle | $y'\cos\theta-x'\sin\theta$ |
| Curvature | $\kappa(t_i)-\kappa_i$ |
| Signed area | $A(\mathbf r)-A_i$ |
| Area centroid | $\bar{\mathbf r}(\mathbf r)-\bar{\mathbf r}_i$ |

The first milestone uses open, non-rational curves. The current upstream
Cython backend safely exposes basis derivatives through order two, which is
sufficient for bending-energy fairing and curvature constraints.

![Area-controlled F-Spline ship sections](images/f_spline_ship_section.png)
