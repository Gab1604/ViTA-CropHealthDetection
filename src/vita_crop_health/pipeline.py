"""End-to-end ViTA mission pipeline.

Phase 1 is intended for onboard execution:
1. cloud gate;
2. crop gate;
3. downlink/discard decision.

Phase 2 is intended for ground execution and calls the crop-health analyzer only for scenes
accepted by Phase 1. Detector models remain external: this module consumes their binary masks
or probability maps, so teammates can connect any cloud/crop detector without changing the
health-monitoring code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np

from .analyzer import AnalysisConfig, CropHealthResult, analyze_crop_health


@dataclass(frozen=True)
class Phase1Config:
    """Thresholds used by the onboard cloud and crop gates.

    Fractions are represented in the interval [0, 1]. Crop coverage is evaluated over clear
    pixels, not over the whole scene, because cloudy pixels cannot be classified reliably.
    """

    max_cloud_fraction: float = 0.40
    min_crop_fraction: float = 0.30
    cloud_probability_threshold: float = 0.50
    crop_probability_threshold: float = 0.50

    def validate(self) -> None:
        for name in (
            "max_cloud_fraction",
            "min_crop_fraction",
            "cloud_probability_threshold",
            "crop_probability_threshold",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1.")


@dataclass(frozen=True)
class Phase1Decision:
    """Machine-readable onboard decision."""

    action: str
    stage: str
    reason: str
    cloud_fraction: float
    clear_fraction: float
    crop_fraction_clear: float | None = None
    crop_fraction_total: float | None = None

    @property
    def should_downlink(self) -> bool:
        return self.action == "DOWNLINK_TO_GROUND"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["should_downlink"] = self.should_downlink
        return payload


@dataclass
class MissionPipelineResult:
    """Combined Phase 1 decision and optional Phase 2 crop-health output."""

    phase1: Phase1Decision
    health: CropHealthResult | None = None

    def to_dict(
        self,
        *,
        phase1_config: Phase1Config | None = None,
        health_config: AnalysisConfig | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "phase1": self.phase1.to_dict(),
            "phase2_executed": self.health is not None,
            "health": self.health.to_dict(health_config) if self.health is not None else None,
        }
        if phase1_config is not None:
            payload["phase1_config"] = asdict(phase1_config)
        return payload


def _to_detection_mask(
    output: np.ndarray,
    *,
    name: str,
    probability_threshold: float,
) -> np.ndarray:
    """Convert a detector output into a boolean mask.

    Supported detector outputs:
    - boolean masks;
    - integer binary masks containing only 0 and 1;
    - floating-point probability maps in [0, 1].
    """

    array = np.asarray(output)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D mask/probability map, got {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if array.dtype == np.bool_:
        return array

    finite = np.isfinite(array)
    if not np.all(finite):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if np.issubdtype(array.dtype, np.integer):
        unique = np.unique(array)
        if not np.all(np.isin(unique, (0, 1))):
            raise ValueError(
                f"{name} integer masks must contain only 0 and 1; got values {unique[:8]}."
            )
        return array.astype(bool)

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"{name} probability maps must be in [0, 1]; got range [{minimum}, {maximum}]."
        )
    return array >= probability_threshold


def evaluate_phase1(
    *,
    cloud_output: np.ndarray,
    crop_output: np.ndarray | None = None,
    config: Phase1Config | None = None,
) -> Phase1Decision:
    """Apply the sequential onboard cloud and crop gates.

    ``cloud_output`` and ``crop_output`` are outputs from the teammates' detector models. They
    may be binary masks or probability maps. When ``crop_output`` is omitted and the cloud gate
    passes, the returned action is ``RUN_CROP_DETECTION``. This allows a real onboard controller
    to avoid running the crop detector for scenes that are already too cloudy.
    """

    cfg = config or Phase1Config()
    cfg.validate()

    cloud_mask = _to_detection_mask(
        cloud_output,
        name="cloud_output",
        probability_threshold=cfg.cloud_probability_threshold,
    )
    cloud_fraction = float(np.mean(cloud_mask))
    clear_fraction = 1.0 - cloud_fraction

    if cloud_fraction >= cfg.max_cloud_fraction:
        return Phase1Decision(
            action="DISCARD_TOO_CLOUDY",
            stage="CLOUD_GATE",
            reason=(
                f"Cloud coverage {cloud_fraction:.1%} is greater than or equal to the "
                f"configured limit {cfg.max_cloud_fraction:.1%}."
            ),
            cloud_fraction=cloud_fraction,
            clear_fraction=clear_fraction,
        )

    if crop_output is None:
        return Phase1Decision(
            action="RUN_CROP_DETECTION",
            stage="CLOUD_GATE",
            reason=(
                f"Cloud coverage {cloud_fraction:.1%} is below the configured limit; "
                "the scene may proceed to crop detection."
            ),
            cloud_fraction=cloud_fraction,
            clear_fraction=clear_fraction,
        )

    crop_mask = _to_detection_mask(
        crop_output,
        name="crop_output",
        probability_threshold=cfg.crop_probability_threshold,
    )
    if crop_mask.shape != cloud_mask.shape:
        raise ValueError(
            f"crop_output has shape {crop_mask.shape}; expected {cloud_mask.shape}."
        )

    clear_mask = ~cloud_mask
    clear_pixels = int(np.count_nonzero(clear_mask))
    if clear_pixels == 0:
        return Phase1Decision(
            action="DISCARD_TOO_CLOUDY",
            stage="CROP_GATE",
            reason="No clear pixels remain for crop evaluation.",
            cloud_fraction=cloud_fraction,
            clear_fraction=clear_fraction,
            crop_fraction_clear=0.0,
            crop_fraction_total=0.0,
        )

    clear_crop = crop_mask & clear_mask
    crop_fraction_clear = float(np.count_nonzero(clear_crop) / clear_pixels)
    crop_fraction_total = float(np.mean(clear_crop))

    if crop_fraction_clear < cfg.min_crop_fraction:
        return Phase1Decision(
            action="DISCARD_LOW_CROP",
            stage="CROP_GATE",
            reason=(
                f"Crop coverage over clear pixels {crop_fraction_clear:.1%} is below the "
                f"configured minimum {cfg.min_crop_fraction:.1%}."
            ),
            cloud_fraction=cloud_fraction,
            clear_fraction=clear_fraction,
            crop_fraction_clear=crop_fraction_clear,
            crop_fraction_total=crop_fraction_total,
        )

    return Phase1Decision(
        action="DOWNLINK_TO_GROUND",
        stage="CROP_GATE",
        reason=(
            f"Cloud coverage {cloud_fraction:.1%} passed and crop coverage over clear pixels "
            f"{crop_fraction_clear:.1%} passed; send the scene and masks to Phase 2."
        ),
        cloud_fraction=cloud_fraction,
        clear_fraction=clear_fraction,
        crop_fraction_clear=crop_fraction_clear,
        crop_fraction_total=crop_fraction_total,
    )


def run_mission_pipeline(
    *,
    bands: Mapping[str, np.ndarray],
    cloud_output: np.ndarray,
    crop_output: np.ndarray,
    phase1_config: Phase1Config | None = None,
    health_config: AnalysisConfig | None = None,
    previous_bands: Mapping[str, np.ndarray] | None = None,
    previous_cloud_mask: np.ndarray | None = None,
) -> MissionPipelineResult:
    """Simulate the complete Phase 1 -> Phase 2 flow in one process.

    In deployment, Phase 1 can call :func:`evaluate_phase1` onboard and transmit the accepted
    bands plus cloud/crop masks. The ground service can then call :func:`analyze_crop_health`.
    This combined helper is mainly for integration tests, demonstrations and a single-machine
    prototype.
    """

    p1_cfg = phase1_config or Phase1Config()
    decision = evaluate_phase1(
        cloud_output=cloud_output,
        crop_output=crop_output,
        config=p1_cfg,
    )
    if not decision.should_downlink:
        return MissionPipelineResult(phase1=decision, health=None)

    cloud_mask = _to_detection_mask(
        cloud_output,
        name="cloud_output",
        probability_threshold=p1_cfg.cloud_probability_threshold,
    )
    crop_mask = _to_detection_mask(
        crop_output,
        name="crop_output",
        probability_threshold=p1_cfg.crop_probability_threshold,
    )

    health_result = analyze_crop_health(
        bands=bands,
        crop_mask=crop_mask,
        cloud_mask=cloud_mask,
        previous_bands=previous_bands,
        previous_cloud_mask=previous_cloud_mask,
        config=health_config,
    )
    return MissionPipelineResult(phase1=decision, health=health_result)
