# ViTA Phase 1 / Phase 2 integration

The cloud and crop neural networks are developed outside this repository. This package consumes
their outputs and provides the decision logic that connects both detectors to crop-health
monitoring.

## Intended mission flow

```text
MULTISPECTRAL SCENE (Blue, Green, Red, NIR)
                     |
                     v
PHASE 1 - ONBOARD
Cloud detector -> cloud mask/probability map
                     |
          cloud fraction >= limit?
               /              \
             yes               no
             |                  |
     DISCARD_TOO_CLOUDY         v
                         Run crop detector
                                  |
                    crop fraction over clear pixels
                                  |
                        below minimum threshold?
                           /              \
                         yes               no
                         |                  |
                 DISCARD_LOW_CROP    DOWNLINK_TO_GROUND
                                             |
                                             v
PHASE 2 - GROUND
Bands + cloud mask + crop mask
             |
             v
NDVI + GNDVI + SAVI + spatial/temporal anomaly analysis
             |
             v
Crop-condition score, label, heatmap and final report
```

## What the integration expects from the teammates' models

Both detector outputs must be two-dimensional and aligned with the multispectral bands.
Supported formats are:

- boolean mask;
- integer binary mask containing only `0` and `1`;
- floating-point probability map in `[0, 1]`.

Mask semantics:

- cloud output: `True` / high probability means cloud or cloud shadow;
- crop output: `True` / high probability means crop.

The default probability threshold is `0.5` for both models and is configurable.

## Onboard API

```python
from vita_crop_health import Phase1Config, evaluate_phase1

config = Phase1Config(
    max_cloud_fraction=0.40,
    min_crop_fraction=0.30,
    cloud_probability_threshold=0.50,
    crop_probability_threshold=0.50,
)

# Step 1: do not run the crop detector yet.
cloud_decision = evaluate_phase1(
    cloud_output=cloud_model_output,
    crop_output=None,
    config=config,
)

if cloud_decision.action == "RUN_CROP_DETECTION":
    crop_model_output = crop_detector(image)

    final_onboard_decision = evaluate_phase1(
        cloud_output=cloud_model_output,
        crop_output=crop_model_output,
        config=config,
    )
```

Possible Phase 1 actions:

- `DISCARD_TOO_CLOUDY`
- `RUN_CROP_DETECTION`
- `DISCARD_LOW_CROP`
- `DOWNLINK_TO_GROUND`

Crop percentage is calculated over **clear pixels**, not the full image:

```text
crop_fraction_clear = crop AND NOT cloud pixels / all NOT cloud pixels
```

This prevents clouds from artificially reducing the crop percentage.

## Ground API

Only scenes with `DOWNLINK_TO_GROUND` should reach Phase 2:

```python
from vita_crop_health import analyze_crop_health

health = analyze_crop_health(
    bands={
        "blue": blue,
        "green": green,
        "red": red,
        "nir": nir,
    },
    cloud_mask=cloud_mask,
    crop_mask=crop_mask,
)

print(health.condition)
print(health.score)
print(health.anomaly_fraction)
```

## Single-machine integration test

For development and demonstration, the entire flow can run in one process:

```python
from vita_crop_health import run_mission_pipeline

result = run_mission_pipeline(
    bands=bands,
    cloud_output=cloud_model_output,
    crop_output=crop_model_output,
)

report = result.to_dict()
```

The combined helper does not mean the health algorithm should run onboard. It simulates the
full mission flow; in deployment, Phase 1 transmits accepted bands and masks to the ground,
where Phase 2 performs crop-health analysis.

## Recommended downlink package

For every accepted scene, Phase 1 should send:

- Blue, Green, Red and NIR bands or the accepted crop region;
- cloud mask;
- crop mask;
- acquisition timestamp;
- geolocation / scene identifier;
- cloud fraction;
- crop fraction over clear pixels;
- detector versions and probability thresholds.

The exact cloud and crop percentage thresholds are mission parameters and should be calibrated
against real EnduroSat imagery and available downlink capacity.
