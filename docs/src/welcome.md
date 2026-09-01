# ship_geo

`ship_geo` extends the LSDO geometry stack with differentiable,
first-principles ship geometry. Its core primitives are CSDL-native F-Splines,
longitudinal curves of form, compatible transverse sections, tensor-product
hull surfaces, differentiable hydrostatics, and fixed-topology refinement.

The project reuses `lsdo_function_spaces` and `lsdo_b_splines_cython` for
spline representation and fast sparse basis evaluation. The fairing problem,
constraints, KKT residual, Newton solve, and implicit derivatives live in the
CSDL graph. A complete section family and its coupled surface-fairness terms
are assembled into one KKT system and one Newton solve.
