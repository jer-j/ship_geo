# ship_geo

`ship_geo` extends the LSDO geometry stack with differentiable,
first-principles ship geometry. Its first primitive is a CSDL-native F-Spline:
a fair B-spline determined from naval-architecture form constraints.

The project reuses `lsdo_function_spaces` and `lsdo_b_splines_cython` for
spline representation and fast sparse basis evaluation. The fairing problem,
constraints, KKT residual, Newton solve, and implicit derivatives live in the
CSDL graph.
