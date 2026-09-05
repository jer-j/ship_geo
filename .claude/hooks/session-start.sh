#!/bin/bash
# Rebuild the Python environment for a fresh session.
#
# Three constraints make this less obvious than "pip install -e .":
#
#   * lsdo_b_splines_cython ships a Cython 0.29.28 build that supports only
#     Python 3.10, so the venv is pinned to that interpreter rather than
#     whatever python3 happens to be.
#   * That same backend needs numpy 1.x. Installing jax[cpu] afterwards pulls
#     numpy 2.x, which breaks the basis evaluation at import, so numpy is
#     re-pinned last and the pin is verified.
#   * The JAX backend is what makes the hull solve tractable -- the inline
#     interpreter spent over an hour building a graph the compiler handles in
#     minutes -- so it is part of the environment, not an optional extra.
#   * The three LSDO dependencies are declared as git URLs, so a plain install
#     takes whatever HEAD is that day, and those heads are not always
#     compatible with each other. A build from HEAD on 2026-09-05 switched
#     lsdo_function_spaces.BSplineSpace to a non-Cython implementation that
#     imports csdl_alpha.experimental.CustomExplicitOperationBeta, which the
#     current csdl_alpha does not define -- the package fails at import. The
#     commits below are the set this work was done and validated against.
set -euo pipefail


cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

VENV=".venv"
PYTHON="$VENV/bin/python"

# Idempotent: an existing venv with a working import is left alone.
if [ -x "$PYTHON" ] && "$PYTHON" - <<'CHECK' >/dev/null 2>&1
import numpy, jax, csdl_alpha, lsdo_b_splines_cython, lsdo_geo
assert numpy.__version__.startswith("1."), numpy.__version__
CHECK
then
    echo "environment already present: $("$PYTHON" -c 'import numpy,jax;print("numpy",numpy.__version__,"jax",jax.__version__)')"
    exit 0
fi

INTERPRETER="$(command -v python3.10 || true)"
if [ -z "$INTERPRETER" ]; then
    echo "python3.10 not found; lsdo_b_splines_cython requires it" >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    INSTALL=(uv pip install --python "$PYTHON")
    [ -x "$PYTHON" ] || uv venv --python "$INTERPRETER" "$VENV"
else
    INSTALL=("$PYTHON" -m pip install)
    [ -x "$PYTHON" ] || "$INTERPRETER" -m venv "$VENV"
fi

# pyproject.toml pins the three LSDO commits, so this resolves to the
# validated set rather than to whatever HEAD is today.
"${INSTALL[@]}" -e ".[test]"
"${INSTALL[@]}" "jax[cpu]"
# Last, so it wins over the numpy 2.x that jax pulls in; the Cython backend
# needs numpy 1.x.
"${INSTALL[@]}" "numpy==1.26.4"

"$PYTHON" - <<'VERIFY'
import numpy, jax, csdl_alpha, lsdo_function_spaces, lsdo_geo
from csdl_alpha.experimental import JaxSimulator
assert numpy.__version__.startswith("1."), f"numpy {numpy.__version__} breaks the cython backend"
# The pin that matters: BSplineSpace must be the Cython-backed class. If the
# resolution drifted, this is where it shows up rather than mid-solve.
import lsdo_b_splines_cython.cython.basis_matrix_curve  # noqa: F401
space = lsdo_function_spaces.BSplineSpace(
    num_parametric_dimensions=1, degree=(3,), coefficients_shape=(6,)
)
assert space.compute_basis_matrix(numpy.array([[0.5]])).shape == (1, 6)
print("numpy", numpy.__version__, "| jax", jax.__version__, "| basis backend ok")
VERIFY
