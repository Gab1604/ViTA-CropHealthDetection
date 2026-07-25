"""Crop-condition scoring from Blue, Green, Red and NIR bands.

This module intentionally reports vigor/stress anomalies rather than diagnosing a disease or
claiming a specific cause. The available bands cannot uniquely identify drought, pests,
nutrient deficiency or thermal stress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np

from .indices import calculate_indices


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the rule-based MVP.

    Default thresholds are generic starting points, not agronomic ground truth. They should be
    calibrated for crop type, phenological stage, sensor and location when labelled field data
    becomes available.
    """

    reflectance_scale: float = 1.0
    savi_l: float = 0.5

    ndvi_low: float = 0.25
    ndvi_high: float = 0.75
    gndvi_low: float = 0.20
    gndvi_high: float = 0.65
    savi_low: float = 0.20
    savi_high: float = 0.70

    ndvi_weight: float = 0.50
    gndvi_weight: float = 0.30
    savi_weight: float = 0.20

    relative_penalty_strength: float = 0.35
    temporal_penalty_strength: float = 0.40
    temporal_free_drop: float = 0.03
    temporal_severe_drop: float = 0.20

    anomaly_health_threshold: float = 0.45
    relative_anomaly_threshold: float = 0.65
    temporal_anomaly_threshold: float = 0.50

    minimum_valid_pixels: int = 64
    minimum_valid_crop_fraction: float = 0.05

    def validate(self) -> None:
        if self.reflectance_scale <= 0 or not np.isfinite(self.reflectance_scale):
            raise ValueError("reflectance_scale must be positive and finite.")
        if self.savi_l < 0 or not np.isfinite(self.savi_l):
            raise ValueError("savi_l must be finite and non-negative.")
        for low_name, high_name in (
            ("ndvi_low", "ndvi_high"),
            ("gndvi_low", "gndvi_high"),
            ("savi_low", "savi_high"),
        ):
            low = getattr(self, low_name)
            high = getattr(self, high_name)
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                raise ValueError(f"{high_name} must be greater than {low_name}.")

        weights = self.ndvi_weight + self.gndvi_weight + self.savi_weight
        if not np.isclose(weights, 1.0, atol=1e-6):
            raise ValueError("NDVI, GNDVI and SAVI weights must sum to 1.")
        if min(self.ndvi_weight, self.gndvi_weight, self.savi_weight) < 0:
            raise ValueError("Index weights cannot be negative.")
        if self.temporal_severe_drop <= self.temporal_free_drop:
            raise ValueError("temporal_severe_drop must exceed temporal_free_drop.")
        if self.minimum_valid_pixels < 1:
            raise ValueError("minimum_valid_pixels must be at least 1.")
        if not 0 <= self.minimum_valid_crop_fraction <= 1:
            raise ValueError("minimum_valid_crop_fraction must be between 0 and 1.")


@dataclass
class CropHealthResult:
    """Analysis output, including machine-readable maps and a compact report."""

    condition: str
    score: float | None
    anomaly_fraction: float | None
    valid_crop_fraction: float
    metrics: dict[str, float | int | None]
    reasons: list[str] = field(default_factory=list)
    health_map: np.ndarray | None = None
    anomaly_mask: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    indices: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self, config: AnalysisConfig | None = None) -> dict[str, object]:
        """Return a JSON-serializable report without large raster arrays."""
        payload: dict[str, object] = {
            "condition": self.condition,
            "score": self.score,
            "anomaly_fraction": self.anomaly_fraction,
            "valid_crop_fraction": self.valid_crop_fraction,
            "metrics": self.metrics,
            "reasons": self.reasons,
            "interpretation": (
                "This is a multispectral vigor/anomaly estimate, not a diagnosis of a "
                "specific disease or stress cause."
            ),
        }
        if config is not None:
            payload["config"] = asdict(config)
        return payload


def _to_bool_mask(mask: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}.")
    return array.astype(bool, copy=False)


def _validate_bands(bands: Mapping[str, np.ndarray], prefix: str = "") -> tuple[int, int]:
    required = ("red", "green", "blue", "nir")
    missing = [name for name in required if name not in bands]
    if missing:
        raise ValueError(f"Missing {prefix}bands: {', '.join(missing)}.")

    arrays = {name: np.asarray(bands[name]) for name in required}
    shapes = {name: array.shape for name, array in arrays.items()}
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError(f"All {prefix}bands must be 2-D arrays, got {shapes}.")
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All {prefix}bands must share one shape, got {shapes}.")
    return arrays["red"].shape


def _linear_health(index: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((index - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _robust_deficit_penalty(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Measure how far each pixel falls below the field median using median absolute deviation."""
    penalty = np.zeros(values.shape, dtype=np.float32)
    sample = values[valid]
    if sample.size == 0:
        return penalty

    median = float(np.nanmedian(sample))
    mad = float(np.nanmedian(np.abs(sample - median)))
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale < 1e-4:
        return penalty

    deficit_z = np.maximum((median - values) / robust_scale, 0.0)
    penalty[valid] = np.clip(deficit_z[valid] / 3.0, 0.0, 1.0)
    return penalty


def _safe_stat(values: np.ndarray, valid: np.ndarray, operation: str) -> float | None:
    sample = values[valid]
    if sample.size == 0:
        return None
    if operation == "mean":
        return float(np.nanmean(sample))
    if operation == "median":
        return float(np.nanmedian(sample))
    if operation == "p25":
        return float(np.nanpercentile(sample, 25))
    raise ValueError(f"Unsupported operation: {operation}")


def _classify(score: float, anomaly_fraction: float) -> str:
    if score >= 75.0 and anomaly_fraction <= 0.15:
        return "Healthy"
    if score >= 60.0 and anomaly_fraction <= 0.30:
        return "Watch"
    if score >= 40.0:
        return "Moderate anomaly"
    return "High anomaly"


def analyze_crop_health(
    *,
    bands: Mapping[str, np.ndarray],
    crop_mask: np.ndarray,
    cloud_mask: np.ndarray | None = None,
    previous_bands: Mapping[str, np.ndarray] | None = None,
    previous_cloud_mask: np.ndarray | None = None,
    config: AnalysisConfig | None = None,
) -> CropHealthResult:
    """Calculate indices, a 0-100 crop-condition score and an anomaly map.

    Mask semantics:

    - ``crop_mask == True`` means the pixel belongs to crop.
    - ``cloud_mask == True`` means the pixel is cloud/cloud-shadow and must be excluded.

    ``previous_bands`` is optional. When supplied, negative NDVI/GNDVI changes contribute a
    temporal penalty. All arrays must already be co-registered and have the same pixel grid.
    """
    cfg = config or AnalysisConfig()
    cfg.validate()

    shape = _validate_bands(bands)
    crop = _to_bool_mask(crop_mask, "crop_mask", shape)
    cloudy = (
        np.zeros(shape, dtype=bool)
        if cloud_mask is None
        else _to_bool_mask(cloud_mask, "cloud_mask", shape)
    )

    indices = calculate_indices(
        red=bands["red"],
        green=bands["green"],
        blue=bands["blue"],
        nir=bands["nir"],
        scale=cfg.reflectance_scale,
        savi_l=cfg.savi_l,
    )

    finite = np.isfinite(indices["ndvi"]) & np.isfinite(indices["gndvi"]) & np.isfinite(indices["savi"])
    valid = crop & ~cloudy & finite
    crop_pixels = int(np.count_nonzero(crop))
    valid_pixels = int(np.count_nonzero(valid))
    valid_crop_fraction = valid_pixels / crop_pixels if crop_pixels else 0.0

    empty_health = np.full(shape, np.nan, dtype=np.float32)
    empty_anomaly = np.zeros(shape, dtype=bool)
    if crop_pixels == 0:
        return CropHealthResult(
            condition="Insufficient data",
            score=None,
            anomaly_fraction=None,
            valid_crop_fraction=0.0,
            metrics={"crop_pixels": 0, "valid_pixels": 0},
            reasons=["The crop mask contains no crop pixels."],
            health_map=empty_health,
            anomaly_mask=empty_anomaly,
            valid_mask=valid,
            indices=indices,
        )
    if valid_pixels < cfg.minimum_valid_pixels or valid_crop_fraction < cfg.minimum_valid_crop_fraction:
        return CropHealthResult(
            condition="Insufficient data",
            score=None,
            anomaly_fraction=None,
            valid_crop_fraction=float(valid_crop_fraction),
            metrics={
                "crop_pixels": crop_pixels,
                "valid_pixels": valid_pixels,
                "cloud_excluded_crop_fraction": float(np.count_nonzero(crop & cloudy) / crop_pixels),
            },
            reasons=["Too few valid, cloud-free crop pixels are available for a reliable estimate."],
            health_map=empty_health,
            anomaly_mask=empty_anomaly,
            valid_mask=valid,
            indices=indices,
        )

    ndvi_health = _linear_health(indices["ndvi"], cfg.ndvi_low, cfg.ndvi_high)
    gndvi_health = _linear_health(indices["gndvi"], cfg.gndvi_low, cfg.gndvi_high)
    savi_health = _linear_health(indices["savi"], cfg.savi_low, cfg.savi_high)
    absolute_health = (
        cfg.ndvi_weight * ndvi_health
        + cfg.gndvi_weight * gndvi_health
        + cfg.savi_weight * savi_health
    ).astype(np.float32)

    relative_penalty = (
        cfg.ndvi_weight * _robust_deficit_penalty(indices["ndvi"], valid)
        + cfg.gndvi_weight * _robust_deficit_penalty(indices["gndvi"], valid)
        + cfg.savi_weight * _robust_deficit_penalty(indices["savi"], valid)
    ).astype(np.float32)

    temporal_penalty = np.zeros(shape, dtype=np.float32)
    temporal_valid = np.zeros(shape, dtype=bool)
    mean_ndvi_change: float | None = None
    mean_gndvi_change: float | None = None

    if previous_bands is not None:
        previous_shape = _validate_bands(previous_bands, prefix="previous ")
        if previous_shape != shape:
            raise ValueError(f"Previous bands have shape {previous_shape}; expected {shape}.")
        previous_cloudy = (
            np.zeros(shape, dtype=bool)
            if previous_cloud_mask is None
            else _to_bool_mask(previous_cloud_mask, "previous_cloud_mask", shape)
        )
        previous_indices = calculate_indices(
            red=previous_bands["red"],
            green=previous_bands["green"],
            blue=previous_bands["blue"],
            nir=previous_bands["nir"],
            scale=cfg.reflectance_scale,
            savi_l=cfg.savi_l,
        )
        temporal_valid = (
            valid
            & ~previous_cloudy
            & np.isfinite(previous_indices["ndvi"])
            & np.isfinite(previous_indices["gndvi"])
        )
        ndvi_change = indices["ndvi"] - previous_indices["ndvi"]
        gndvi_change = indices["gndvi"] - previous_indices["gndvi"]
        combined_drop = np.maximum(-(0.6 * ndvi_change + 0.4 * gndvi_change), 0.0)
        denominator = cfg.temporal_severe_drop - cfg.temporal_free_drop
        temporal_penalty[temporal_valid] = np.clip(
            (combined_drop[temporal_valid] - cfg.temporal_free_drop) / denominator,
            0.0,
            1.0,
        )
        mean_ndvi_change = _safe_stat(ndvi_change, temporal_valid, "mean")
        mean_gndvi_change = _safe_stat(gndvi_change, temporal_valid, "mean")

    health = absolute_health.copy()
    health *= 1.0 - cfg.relative_penalty_strength * relative_penalty
    health *= 1.0 - cfg.temporal_penalty_strength * temporal_penalty
    health = np.clip(health, 0.0, 1.0).astype(np.float32)
    health_map = np.full(shape, np.nan, dtype=np.float32)
    health_map[valid] = health[valid]

    anomaly_mask = valid & (
        (health < cfg.anomaly_health_threshold)
        | (relative_penalty > cfg.relative_anomaly_threshold)
        | (temporal_penalty > cfg.temporal_anomaly_threshold)
    )
    anomaly_fraction = float(np.count_nonzero(anomaly_mask) / valid_pixels)

    median_health = float(np.nanmedian(health[valid]))
    p25_health = float(np.nanpercentile(health[valid], 25))
    score = float(np.clip(100.0 * (0.60 * median_health + 0.40 * p25_health), 0.0, 100.0))
    condition = _classify(score, anomaly_fraction)

    metrics: dict[str, float | int | None] = {
        "crop_pixels": crop_pixels,
        "valid_pixels": valid_pixels,
        "cloud_excluded_crop_fraction": float(np.count_nonzero(crop & cloudy) / crop_pixels),
        "mean_ndvi": _safe_stat(indices["ndvi"], valid, "mean"),
        "median_ndvi": _safe_stat(indices["ndvi"], valid, "median"),
        "mean_gndvi": _safe_stat(indices["gndvi"], valid, "mean"),
        "mean_savi": _safe_stat(indices["savi"], valid, "mean"),
        "mean_evi": _safe_stat(indices["evi"], valid & np.isfinite(indices["evi"]), "mean"),
        "surface_water_candidate_fraction": float(
            np.count_nonzero(valid & (indices["ndwi"] > 0.15)) / valid_pixels
        ),
        "mean_relative_penalty": _safe_stat(relative_penalty, valid, "mean"),
        "temporal_comparable_fraction": float(np.count_nonzero(temporal_valid) / valid_pixels),
        "mean_ndvi_change": mean_ndvi_change,
        "mean_gndvi_change": mean_gndvi_change,
    }

    reasons: list[str] = []
    if anomaly_fraction > 0.30:
        reasons.append(f"{anomaly_fraction:.1%} of valid crop pixels were flagged as anomalous.")
    mean_ndvi = metrics["mean_ndvi"]
    if isinstance(mean_ndvi, float) and mean_ndvi < cfg.ndvi_low:
        reasons.append("Mean NDVI is below the configured low-vigor reference.")
    if mean_ndvi_change is not None and mean_ndvi_change < -cfg.temporal_free_drop:
        reasons.append(f"Mean NDVI decreased by {abs(mean_ndvi_change):.3f} from the previous acquisition.")
    if not reasons:
        reasons.append("No dominant low-vigor or temporal-decline signal was found with the configured thresholds.")

    return CropHealthResult(
        condition=condition,
        score=score,
        anomaly_fraction=anomaly_fraction,
        valid_crop_fraction=float(valid_crop_fraction),
        metrics=metrics,
        reasons=reasons,
        health_map=health_map,
        anomaly_mask=anomaly_mask,
        valid_mask=valid,
        indices=indices,
    )
