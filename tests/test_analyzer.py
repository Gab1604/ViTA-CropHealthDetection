import numpy as np

from vita_crop_health import AnalysisConfig, analyze_crop_health


def _healthy_bands(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "red": np.full(shape, 0.15, dtype=np.float32),
        "green": np.full(shape, 0.25, dtype=np.float32),
        "blue": np.full(shape, 0.10, dtype=np.float32),
        "nir": np.full(shape, 0.75, dtype=np.float32),
    }


def test_uniform_vigorous_field_is_healthy() -> None:
    shape = (64, 64)
    result = analyze_crop_health(
        bands=_healthy_bands(shape),
        crop_mask=np.ones(shape, dtype=bool),
    )

    assert result.condition == "Healthy"
    assert result.score is not None and result.score >= 75.0
    assert result.anomaly_fraction == 0.0
    assert np.count_nonzero(result.anomaly_mask) == 0


def test_low_vigor_patch_is_flagged() -> None:
    shape = (64, 64)
    bands = _healthy_bands(shape)
    damaged = np.zeros(shape, dtype=bool)
    damaged[16:48, 16:48] = True

    bands["red"][damaged] = 0.40
    bands["green"][damaged] = 0.45
    bands["nir"][damaged] = 0.45

    result = analyze_crop_health(
        bands=bands,
        crop_mask=np.ones(shape, dtype=bool),
    )

    assert result.condition in {"Watch", "Moderate anomaly", "High anomaly"}
    assert result.anomaly_fraction is not None and result.anomaly_fraction >= 0.24
    assert np.mean(result.anomaly_mask[damaged]) > 0.95
    assert np.mean(result.anomaly_mask[~damaged]) < 0.05


def test_cloud_pixels_are_excluded() -> None:
    shape = (32, 32)
    cloud_mask = np.zeros(shape, dtype=bool)
    cloud_mask[:, :8] = True

    result = analyze_crop_health(
        bands=_healthy_bands(shape),
        crop_mask=np.ones(shape, dtype=bool),
        cloud_mask=cloud_mask,
    )

    assert result.valid_crop_fraction == 0.75
    assert np.count_nonzero(result.valid_mask & cloud_mask) == 0
    assert np.all(np.isnan(result.health_map[cloud_mask]))


def test_temporal_decline_adds_penalty() -> None:
    shape = (48, 48)
    previous = _healthy_bands(shape)
    current = _healthy_bands(shape)

    declining = np.zeros(shape, dtype=bool)
    declining[:, 24:] = True
    current["red"][declining] = 0.30
    current["green"][declining] = 0.38
    current["nir"][declining] = 0.55

    result = analyze_crop_health(
        bands=current,
        previous_bands=previous,
        crop_mask=np.ones(shape, dtype=bool),
    )

    assert result.metrics["mean_ndvi_change"] is not None
    assert result.metrics["mean_ndvi_change"] < 0
    assert np.nanmean(result.health_map[declining]) < np.nanmean(result.health_map[~declining])


def test_insufficient_data_when_crop_mask_is_empty() -> None:
    shape = (16, 16)
    result = analyze_crop_health(
        bands=_healthy_bands(shape),
        crop_mask=np.zeros(shape, dtype=bool),
        config=AnalysisConfig(minimum_valid_pixels=1),
    )

    assert result.condition == "Insufficient data"
    assert result.score is None
