"""Run regional spline-convergence validation on DTMB 5415."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import csdl_alpha as csdl
import matplotlib.pyplot as plt
import numpy as np

from lsdo_geo.validation import (
    DTMB5415Region,
    download_dtmb_5415,
    fit_dtmb_5415,
    load_dtmb_5415,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/src/images/dtmb_5415_validation.png"),
    )
    arguments = parser.parse_args()
    source = arguments.source
    if source is None:
        source = Path(tempfile.gettempdir()) / "ship_geo_dtmb_5415.iges"
        if not source.exists():
            download_dtmb_5415(source)

    recorder = csdl.Recorder(inline=True)
    recorder.start()
    reference = load_dtmb_5415(source)
    reference_functions = reference.build_functions()
    approximations = [
        fit_dtmb_5415(reference, level) for level in ("coarse", "medium", "fine")
    ]

    print("DTMB 5415 reference dimensions:", reference.dimensions())
    for approximation in approximations:
        dome = approximation.sonar_dome
        print(
            f"{approximation.level:>6}: global RMS "
            f"{1e3 * approximation.global_rms_error:.4f} mm, "
            f"dome RMS {1e3 * dome.rms_error:.4f} mm, "
            f"global max {1e3 * approximation.global_maximum_error:.4f} mm"
        )

    colors = {
        DTMB5415Region.SONAR_DOME: "#d55e00",
        DTMB5415Region.SONAR_DOME_TRANSITION: "#e69f00",
        DTMB5415Region.MAIN_HULL: "#0072b2",
    }
    figure = plt.figure(figsize=(12.0, 6.8))
    axis = figure.add_subplot(111, projection="3d")
    for region, patch in reference.patches.items():
        _, values = patch.sample_grid(reference_functions[region], (45, 75))
        points = values.value.reshape((45, 75, 3))
        axis.plot_surface(
            points[:, :, 0],
            points[:, :, 1],
            points[:, :, 2],
            color=colors[region],
            alpha=0.72,
            linewidth=0.0,
        )
    fine = approximations[-1]
    for fit in fine.patches.values():
        coefficients = np.asarray(fit.surface.coefficients.value)
        axis.plot_wireframe(
            coefficients[:, :, 0],
            coefficients[:, :, 1],
            coefficients[:, :, 2],
            color="black",
            linewidth=0.45,
            alpha=0.7,
        )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("DTMB 5415: main hull, sonar-dome transition, and sonar dome")
    axis.set_box_aspect((6.2, 1.0, 1.0))
    axis.view_init(elev=18.0, azim=-67.0)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(arguments.output, dpi=180)
    print(f"wrote {arguments.output}")
    recorder.stop()


if __name__ == "__main__":
    main()
