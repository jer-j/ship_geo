# Running the DTMB 5415 reconstruction at higher resolution

Everything below was measured on a 15 GB container. The construction is not
resolution-limited; the XLA compile is memory-limited, and the two ceilings
that matter are recorded here so a larger machine can go straight to the
configurations this one could not hold.

## What the hull is now

One F-Spline per station from keel to deck edge, lofted into one surface.
The waterline and the blend line are interior conditions on that curve, not
boundaries between patches, so the quick work, the dead work and the sonar
dome are a single patch and continuity across all of it is whatever the
curve has rather than what two separately faired curves happened to agree
on.

Displacement, LCB and the waterplane coefficient stay exact because the area
constraint is taken over the sub-arc below the waterline
(`FSplineProblem.add_area_constraint(..., parameter_range=...)`), verified
against a case with a closed-form answer: 2.000000 over a full arc and
0.500000 over its lower half.

## Memory, measured

XLA counts operations, not tensor elements, so the node count does not move
when the readback meshes shrink -- peak memory does, because the tensors
flowing through those operations get smaller. Both levers matter and they
are independent.

| configuration | nodes | outcome on 15 GB |
| --- | --- | --- |
| 19 stations, banded, partial lower band | 182k | compiled, peak 12.4 GB |
| 19 stations, banded, full-length lower band | 236k | killed |
| 19 stations, unified, no surface fairness | 148k | compiled |
| 19 stations, unified, thin-plate fairness | 175k | killed at mesh 0.6, 0.4 and 0.3 |
| 15 stations, unified, thin-plate fairness | 141k | compiled, peak ~12 GB |

A run that is killed leaves no traceback: the log stops mid-compile and the
container's free memory jumps back. That is the signature to look for.

## The fairing question, unresolved

Sarioz (2006) argues the fairness measure has to be attached to the surface,
and that the choice of functional decides the quality of the result. Three
configurations were measured on the same hull. Curvature variation is the
RMS of the change in discrete curvature along a curve; roughness is the RMS
second difference of a form curve, in mm.

| | none | form curves 0.05 | surface 0.02 |
| --- | --- | --- | --- |
| deck edge, curvature variation | 0.45 | **0.11** | 0.40 |
| deck-edge form curve | 4.51 | **1.97** | 4.19 |
| bulb section v=0.02 | **0.73** | 1.25 | 0.87 |
| holdout RMS | **4.14 mm** | 5.67 mm | 5.02 mm |

Note the last column is 15 stations where the others are 19, so it is not a
clean comparison -- run a matched pair before drawing conclusions from it.

What is established: the waviness is manufactured by the fit, not present in
the data. The forward deck observations are 0.37 mm rough and monotone and
the curve fitted through them is 11.81 mm rough, a factor of 32. A
compatible loft applies no fairing at all unless a weight is set, so the
surface interpolated whatever the sections did.

What is not established: a weight that fairs the deck edge without
distorting the bulb. Fairing every generating curve pulls the section
endpoints off the observations and the section fit, still at 250 to 1,
fights its own boundary conditions hardest where the hull changes fastest.
The thin-plate surface term is the better instrument in principle and was
never given enough weight to prove it here, because the weights that might
have worked did not fit in memory.

## What to run first on a larger machine

The matched pair the comparison above is missing, at full resolution:

```bash
python examples/dtmb_5415_accurate_reconstruction.py --cache base.npz \
    --backend jax --unify-bands \
    --num-section-control-points 16 --num-form-control-points 18

python examples/dtmb_5415_accurate_reconstruction.py --cache faired.npz \
    --backend jax --unify-bands \
    --num-section-control-points 16 --num-form-control-points 18 \
    --surface-fairness 0.2 --form-fairness 0.002
```

Then sweep `--surface-fairness` upward from 0.2. The runs here never got
above 0.02 at full resolution, so the useful range is unexplored. Score both
with the curvature measurements, not by eye.

`--mesh-scale` only affects the readback meshes. Control nets and knot
vectors are cached, so any figure can be rebuilt at any resolution offline
and the meshes can stay small.

## Where the exact functional would go

Eq. 3 of the paper is the integral of `(k1 + k2)^2` over the surface. What
is implemented is its quadratic surrogate, the thin-plate energy
`Quu^2 + 2 Quv^2 + Qvv^2`, which stays quadratic in the control points and
so leaves the KKT system's character unchanged. The paper is explicit that
the exact functional gives better surfaces at substantially higher cost. It
would enter as a new objective term in `TensorProductSurface`, alongside
`fairness_energy`, and would make the system nonlinear in a way the current
one is not -- worth trying only once memory is not the binding constraint.

## On WSL with an NVIDIA GPU

Check the memory cap first, because it is the thing this work was actually
limited by and WSL does not give you the machine's RAM by default. WSL2
takes half of Windows' memory, and on older builds 8 GB, whatever the box
has. In `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=48GB
swap=16GB
```

then `wsl --shutdown` and reopen. Confirm inside the distribution with
`free -g`. A 128 GB workstation that reports 24 GB to `free` will reproduce
every ceiling in the table above.

CUDA needs only the Windows driver; do not install a Linux display driver
inside the distribution. `nvidia-smi -L` answering inside WSL is the test,
and it is what the session hook keys on to choose `jax[cuda12]` over
`jax[cpu]`. Those wheels carry their own CUDA libraries.

Run the solve on the GPU with `--device gpu`; it defaults to CPU. Two things
are worth knowing before assuming that is the faster choice:

* The ceiling that killed runs here was **host** memory during XLA
  compilation, not device memory. More RAM removes it. A GPU does not.
* The solve depends on float64 -- the constraint residuals quoted throughout
  are at 1e-15, and float32 cannot hold them. Consumer GeForce parts run
  float64 at a small fraction of their float32 rate, so on such a card the
  compiled solve may well be slower on the GPU than on the CPU. Datacenter
  parts do not have that gap. Measure both rather than assuming; the run
  prints `[jax] devices:` so it is clear which one produced a number.

JAX preallocates most of the visible VRAM on first use. If that crowds out
anything else, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.8`.

## Two things that will bite

The three LSDO dependencies are declared as git URLs and their heads are not
always compatible. `pyproject.toml` now pins the commits this work was
validated against; a build from HEAD on 2026-09-05 fails at import because
`lsdo_function_spaces.BSplineSpace` moved to a non-Cython implementation
requiring a `csdl_alpha` API that does not exist in the matching commit.
Move the pins deliberately, together, and re-run the tests.

numpy must stay on 1.x for the Cython basis backend, and `jax[cpu]` will
upgrade it to 2.x if installed afterwards. `.claude/hooks/session-start.sh`
re-pins it last and then checks that the backend still evaluates.
