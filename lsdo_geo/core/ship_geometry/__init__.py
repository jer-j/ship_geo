"""First-principles differentiable ship-geometry primitives."""

from .closed_surface import (
    ClosedSurface,
    OrientedSurfacePatch,
    bilinear_patch,
    create_transom_cap,
    create_waterplane_cap,
    mirror_surface_y,
    rectangular_box_surface,
)
from .diagnostics import (
    ClosureReport,
    curve_span_error_indicators,
    evaluate_closed_surface_closure,
    recommended_refinement_knots,
    section_self_intersections,
)
from .f_surface import FSurfaceAssembly, FSurfaceProblem
from .form_curves import (
    FormCurve,
    FormCurveAssembly,
    FormCurveKind,
    FormCurveProblem,
)
from .form_parameter_hull import (
    FormParameterHullAssembly,
    FormParameterHullGeometry,
    FormParameterHullProblem,
    LongitudinalFitTargets,
    NavalHullParameters,
)
from .hull import HullGeometry, SectionLoftAssembly, SectionLoftProblem
from .hydrostatics import (
    Hydrostatics,
    compute_closed_surface_hydrostatics,
    compute_hydrostatics,
    compute_sectional_areas,
)
from .mesh import (
    SurfaceMesh,
    WatertightMeshReport,
    evaluate_watertight_mesh,
    export_ascii_stl,
    export_obj,
    tessellate_closed_surface,
    tessellate_surface,
)
from .refinement import fit_offset_surface, refine_curve, refine_surface
from .sections import (
    SectionAssembly,
    SectionProblem,
    SectionTemplate,
    collapsed_section,
)
from .surfaces import (
    CompatibleLoft,
    LongitudinalLoftRegion,
    PatchConnection,
    PatchGraph,
    RegionalCompatibleLoft,
    TensorProductSurface,
    wigley_surface,
)
from .validity import SurfaceValidity, evaluate_surface_validity

__all__ = [
    "ClosedSurface",
    "ClosureReport",
    "CompatibleLoft",
    "FSurfaceAssembly",
    "FSurfaceProblem",
    "FormCurve",
    "FormCurveAssembly",
    "FormCurveKind",
    "FormCurveProblem",
    "FormParameterHullAssembly",
    "FormParameterHullGeometry",
    "FormParameterHullProblem",
    "HullGeometry",
    "Hydrostatics",
    "LongitudinalFitTargets",
    "LongitudinalLoftRegion",
    "NavalHullParameters",
    "OrientedSurfacePatch",
    "PatchConnection",
    "PatchGraph",
    "RegionalCompatibleLoft",
    "SectionAssembly",
    "SectionLoftAssembly",
    "SectionLoftProblem",
    "SectionProblem",
    "SectionTemplate",
    "SurfaceMesh",
    "SurfaceValidity",
    "TensorProductSurface",
    "WatertightMeshReport",
    "bilinear_patch",
    "collapsed_section",
    "compute_closed_surface_hydrostatics",
    "compute_hydrostatics",
    "compute_sectional_areas",
    "create_transom_cap",
    "create_waterplane_cap",
    "curve_span_error_indicators",
    "evaluate_closed_surface_closure",
    "evaluate_surface_validity",
    "evaluate_watertight_mesh",
    "export_ascii_stl",
    "export_obj",
    "fit_offset_surface",
    "mirror_surface_y",
    "recommended_refinement_knots",
    "rectangular_box_surface",
    "refine_curve",
    "refine_surface",
    "section_self_intersections",
    "tessellate_closed_surface",
    "tessellate_surface",
    "wigley_surface",
]
