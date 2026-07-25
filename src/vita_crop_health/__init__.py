"""ViTA crop vigor and anomaly monitoring package."""

from .analyzer import AnalysisConfig, CropHealthResult, analyze_crop_health
from .indices import calculate_indices

__all__ = [
    "AnalysisConfig",
    "CropHealthResult",
    "analyze_crop_health",
    "calculate_indices",
]

__version__ = "0.1.0"
