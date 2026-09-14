"""Fit the complete DTMB 5415 weather-deck edge with one CSDL Newton solve.

Run with --source pointing to the checksum-pinned reference IGES. The fit is
an edge calibration, not a claim of calibrated topside or sonar-dome shape.
Independent, offset face samples provide held-out geometric errors.
"""

import argparse
import hashlib
import json
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo.core.ship_geometry.deck_edge import assemble_deck_edge
from lsdo_geo.core.splines.variational import VariationalSystem
from lsdo_geo.validation.dtmb_5415 import load_dtmb_5415, DTMB5415_SOURCE_SHA256


def sample_edges(reference, count, held_out=False):
    """Sample exact active v=1 boundaries; retain physical x coordinates."""
    functions = reference.build_functions()
    u = (np.arange(count)+0.5)/count if held_out else np.linspace(0, 1, count)
    points = np.vstack([
        patch.evaluate(functions[region], np.column_stack((u, np.ones(count)))).value
        for region, patch in reference.patches.items()])
    return points[np.argsort(points[:, 0])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/src/images/whole_deck"))
    parser.add_argument("--num-control-points", type=int, default=24)
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != DTMB5415_SOURCE_SHA256:
        raise ValueError("Reference checksum does not match the pinned DTMB 5415 source.")
    recorder = csdl.Recorder(inline=True); recorder.start()
    reference = load_dtmb_5415(args.source)
    train = sample_edges(reference, 201)
    check = sample_edges(reference, 400, held_out=True)
    start, end = train[0, 0], train[-1, 0]
    length = end-start
    u = (train[:, 0]-start)/length
    # Tiny source seams are retained as independent observations, not welded.
    if np.any(np.diff(u) <= 0):
        raise ValueError("Reference deck must be single-valued in physical x.")
    system = VariationalSystem("dtmb_whole_deck")
    assemblies = assemble_deck_edge(
        system, u, train[:, 1]/length, train[:, 2]/length,
        num_control_points=args.num_control_points)
    result = system.solve()
    breadth, sheer = (assembly.finalize(result) for assembly in assemblies)
    uc = (check[:, 0]-start)/length
    predicted = np.column_stack((check[:, 0],
        length*breadth.evaluate(uc).value.ravel(), length*sheer.evaluate(uc).value.ravel()))
    error = np.linalg.norm(predicted-check, axis=1)
    dense = np.linspace(0, 1, 1001)
    x = start + length*dense
    y = length*breadth.evaluate(dense).value.ravel()
    z = length*sheer.evaluate(dense).value.ravel()
    knots = np.asarray(breadth.space.knots).ravel()
    interior = np.unique(knots[(knots > 0) & (knots < 1)])
    jump = {}
    for order in range(3):
        jump[str(order)] = float(max(np.max(np.abs(
            curve.evaluate(interior-1.e-10, order).value-
            curve.evaluate(interior+1.e-10, order).value)) for curve in (breadth, sheer)))
    report = {
        "source_sha256": DTMB5415_SOURCE_SHA256,
        "coordinate": "physical deck-tip-to-transom x, increasing aft; source z=0 waterplane",
        "num_control_points_per_curve": args.num_control_points,
        "training_samples": len(train), "held_out_samples": len(check),
        "deck_x_bounds_m": [float(start), float(end)],
        "degree": 3,
        "knots": knots.tolist(),
        "half_breadth_coefficients_normalized": breadth.coefficients.value.ravel().tolist(),
        "height_coefficients_normalized": sheer.coefficients.value.ravel().tolist(),
        "held_out_rms_m": float(np.sqrt(np.mean(error**2))),
        "held_out_max_m": float(np.max(error)),
        "sampled_min_half_breadth_m": float(np.min(y)),
        "normalized_near_knot_two_sided_difference": jump,
        "max_constraint_residual": float(np.max(np.abs(result.constraint_residual.value))),
        "max_stationarity_residual": float(np.max(np.abs(result.stationarity_residual.value))),
        "scope": "Edge fit only; whole hull and sonar dome not calibrated by this example."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for axis, ordinate, index, label in zip(axes[:2], (y, z), (1, 2),
                                          ("Half-breadth y [m]", "Sheer z [m]")):
        axis.plot(check[:, 0], check[:, index], color="0.55", lw=3, label="Reference IGES")
        axis.plot(x, ordinate, color="#0072b2", lw=1.5, label="Native CSDL F-spline")
        axis.set_ylabel(label); axis.grid(alpha=.2); axis.legend()
    axes[2].plot(check[:, 0], error*1000, color="#d55e00")
    axes[2].set_ylabel("Held-out error [mm]"); axes[2].set_xlabel("Source x [m], forward to aft")
    axes[2].grid(alpha=.2)
    figure.suptitle("DTMB 5415 complete weather-deck edge: stem tip to transom")
    figure.savefig(args.output.with_suffix(".png"), dpi=160)
    plt.close(figure)
    print(json.dumps(report, indent=2))
    recorder.stop()


if __name__ == "__main__":
    main()
