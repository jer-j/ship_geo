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

# Idempotent: an existing venv with a working import is left alone. The check
# includes whether JAX can see the GPU when the machine has one, so a venv
# built before the hardware changed gets rebuilt rather than quietly running
# on the CPU.
WANT_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    WANT_CUDA=1
fi
if [ -x "$PYTHON" ] && WANT_CUDA="$WANT_CUDA" "$PYTHON" - <<'CHECK' >/dev/null 2>&1
import os
import numpy, jax, csdl_alpha, lsdo_b_splines_cython, lsdo_geo
assert numpy.__version__.startswith("1."), numpy.__version__
if os.environ.get("WANT_CUDA") == "1":
    assert any(d.platform == "gpu" for d in jax.devices()), "no GPU device"
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

# JAX flavour follows the hardware. On WSL the CUDA runtime comes from the
# Windows driver, so nvidia-smi answering inside the distribution is the
# thing to test -- there is no Linux display driver to look for. The cuda12
# wheels carry their own CUDA libraries, so nothing else needs installing.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    echo "CUDA device present: $(nvidia-smi -L | head -1)"
    "${INSTALL[@]}" "jax[cuda12]"
else
    "${INSTALL[@]}" "jax[cpu]"
fi
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
print(
    "numpy", numpy.__version__,
    "| jax", jax.__version__,
    "| devices", jax.devices(),
    "| basis backend ok",
)
VERIFY
