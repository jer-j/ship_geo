from pathlib import Path

from .core.geometry.geometry import Geometry
from .core.geometry.geometry_functions import *
from .core.geometry.mesh import Mesh
from .core.parameterization.ffd_block import FFDBlock
from .core.parameterization.free_form_deformation_functions import *
from .core.parameterization.parameterization_solver import (
    GeometricVariables,
    ParameterizationSolver,
)
from .core.parameterization.sectional_parameterization import (
    SectionalParameterization,
    SectionalParameters,
)
from .core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs,
)
from .core.ship_geometry import (
    BasicCurveName,
    ClosedSurface,
    ClosureReport,
    CompatibleLoft,
    ControlCurveName,
    FormCurve,
    FormCurveAssembly,
    FormCurveKind,
    FormCurveProblem,
    FormParameterHullAssembly,
    FormParameterHullGeometry,
    FormParameterHullProblem,
    FSurfaceAssembly,
    FSurfaceProblem,
    HullGeometry,
    HullCurveNetwork,
    Hydrostatics,
    LongitudinalFitTargets,
    LongitudinalLoftRegion,
    NavalHullParameters,
    OrientedSurfacePatch,
    PatchConnection,
    PatchGraph,
    RegionalCompatibleLoft,
    SectionAssembly,
    SectionControlValues,
    SectionLoftAssembly,
    SectionLoftProblem,
    SectionProblem,
    SectionTemplate,
    SonarDomeSectionParameters,
    SurfaceMesh,
    SurfaceValidity,
    TensorProductSurface,
    WatertightMeshReport,
    bilinear_patch,
    collapsed_section,
    compute_closed_surface_hydrostatics,
    compute_hydrostatics,
    compute_sectional_areas,
    create_transom_cap,
    create_waterplane_cap,
    curve_span_error_indicators,
    evaluate_closed_surface_closure,
    evaluate_surface_validity,
    evaluate_watertight_mesh,
    feature_aligned_interpolation_knots,
    export_ascii_stl,
    export_obj,
    fit_offset_surface,
    mirror_surface_y,
    recommended_refinement_knots,
    rectangular_box_surface,
    refine_curve,
    refine_surface,
    section_self_intersections,
    tessellate_closed_surface,
    tessellate_surface,
    wigley_surface,
)
from .core.splines import (
    FSplineAssembly,
    FSplineCurve,
    FSplineProblem,
    VariationalResult,
    VariationalSystem,
)

_REPO_ROOT_FOLDER = Path(__file__).parents[0]
IMPORT_FOLDER = _REPO_ROOT_FOLDER / "core" / "stored_files" / "imports"
REFIT_FOLDER = _REPO_ROOT_FOLDER / "core" / "stored_files" / "refits"
PROJECTIONS_FOLDER = _REPO_ROOT_FOLDER / "core" / "stored_files" / "projections"
