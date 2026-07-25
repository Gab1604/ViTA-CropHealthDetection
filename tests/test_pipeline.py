import numpy as np

from vita_crop_health import (
    Phase1Config,
    evaluate_phase1,
    run_mission_pipeline,
)


def _bands(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "red": np.full(shape, 0.15, dtype=np.float32),
        "green": np.full(shape, 0.25, dtype=np.float32),
        "blue": np.full(shape, 0.10, dtype=np.float32),
        "nir": np.full(shape, 0.75, dtype=np.float32),
    }


def test_cloud_gate_discards_without_running_crop_detection() -> None:
    cloud = np.ones((20, 20), dtype=bool)
    cloud[:8, :] = False  # 60% cloud

    result = evaluate_phase1(
        cloud_output=cloud,
        config=Phase1Config(max_cloud_fraction=0.40),
    )

    assert result.action == "DISCARD_TOO_CLOUDY"
    assert result.stage == "CLOUD_GATE"
    assert not result.should_downlink


def test_clear_scene_requests_crop_detector_when_crop_output_is_missing() -> None:
    result = evaluate_phase1(
        cloud_output=np.zeros((20, 20), dtype=bool),
        crop_output=None,
    )

    assert result.action == "RUN_CROP_DETECTION"
    assert result.crop_fraction_clear is None


def test_crop_fraction_is_measured_over_clear_pixels() -> None:
    shape = (10, 10)
    cloud = np.zeros(shape, dtype=bool)
    cloud[:, :2] = True  # 20 cloudy pixels, 80 clear pixels
    crop = np.zeros(shape, dtype=bool)
    crop[:, 2:6] = True  # 40 crop pixels, all clear -> 50% of clear pixels

    result = evaluate_phase1(
        cloud_output=cloud,
        crop_output=crop,
        config=Phase1Config(max_cloud_fraction=0.40, min_crop_fraction=0.45),
    )

    assert result.action == "DOWNLINK_TO_GROUND"
    assert result.crop_fraction_clear == 0.5
    assert result.crop_fraction_total == 0.4


def test_low_crop_scene_is_discarded() -> None:
    shape = (20, 20)
    crop = np.zeros(shape, dtype=bool)
    crop[:4, :] = True  # 20%

    result = evaluate_phase1(
        cloud_output=np.zeros(shape, dtype=bool),
        crop_output=crop,
        config=Phase1Config(min_crop_fraction=0.30),
    )

    assert result.action == "DISCARD_LOW_CROP"
    assert not result.should_downlink


def test_phase2_runs_only_after_both_phase1_gates_pass() -> None:
    shape = (32, 32)
    cloud = np.zeros(shape, dtype=bool)
    crop = np.ones(shape, dtype=bool)

    accepted = run_mission_pipeline(
        bands=_bands(shape),
        cloud_output=cloud,
        crop_output=crop,
    )
    assert accepted.phase1.should_downlink
    assert accepted.health is not None
    assert accepted.health.condition == "Healthy"

    rejected_crop = np.zeros(shape, dtype=bool)
    rejected = run_mission_pipeline(
        bands=_bands(shape),
        cloud_output=cloud,
        crop_output=rejected_crop,
    )
    assert rejected.phase1.action == "DISCARD_LOW_CROP"
    assert rejected.health is None


def test_probability_maps_are_thresholded() -> None:
    cloud_probability = np.full((10, 10), 0.1, dtype=np.float32)
    crop_probability = np.full((10, 10), 0.8, dtype=np.float32)

    result = evaluate_phase1(
        cloud_output=cloud_probability,
        crop_output=crop_probability,
        config=Phase1Config(
            cloud_probability_threshold=0.5,
            crop_probability_threshold=0.6,
        ),
    )

    assert result.action == "DOWNLINK_TO_GROUND"
    assert result.cloud_fraction == 0.0
    assert result.crop_fraction_clear == 1.0
