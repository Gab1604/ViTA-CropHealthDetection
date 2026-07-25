"""ViTA crop vigor and anomaly monitoring package."""

from .analyzer import AnalysisConfig, CropHealthResult, analyze_crop_health
from .indices import calculate_indices
from .pipeline import (
    MissionPipelineResult,
    Phase1Config,
    Phase1Decision,
    evaluate_phase1,
    run_mission_pipeline,
)

__all__ = [
    "AnalysisConfig",
    "CropHealthResult",
    "MissionPipelineResult",
    "Phase1Config",
    "Phase1Decision",
    "analyze_crop_health",
    "calculate_indices",
    "evaluate_phase1",
    "run_mission_pipeline",
]

__version__ = "0.2.0"
