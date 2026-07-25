# ViTA Crop Health Detection

A lightweight **multispectral crop vigor and stress-anomaly monitoring MVP** for co-registered **Blue, Green, Red and NIR** imagery.

The system combines:

- cloud/cloud-shadow exclusion;
- an external crop mask;
- NDVI, GNDVI and SAVI vigor information;
- optional EVI and Green-NIR NDWI diagnostics;
- within-field anomaly detection;
- optional temporal decline detection;
- a final 0–100 crop-condition score and condition label.

> The output is an anomaly estimate, not a diagnosis. With Blue, Green, Red and NIR alone, the system cannot reliably identify a specific disease, drought, nutrient deficiency, pest attack, plant water content or thermal stress.

## Pipeline

```text
Blue + Green + Red + NIR
          ↓
Cloud / cloud-shadow mask
          ↓
Crop mask
          ↓
NDVI + GNDVI + SAVI
          ↓
Absolute vigor score
+ within-field relative anomaly penalty
+ optional temporal decline penalty
          ↓
Health map + anomaly mask + crop-condition report
```

## Implemented indices

```text
NDVI  = (NIR - Red)   / (NIR + Red)
GNDVI = (NIR - Green) / (NIR + Green)
SAVI  = ((NIR - Red) * (1 + L)) / (NIR + Red + L), default L = 0.5
EVI   = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
NDWI  = (Green - NIR) / (Green + NIR)
```

EVI and SAVI use additive constants, so they require reflectance-scaled data. Use the correct `--scale` value for your product.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -e ".[plots,dev]"
```

For GeoTIFF input:

```bash
pip install -e ".[all]"
```

## Quick demo

Generate a synthetic field containing a low-vigor patch and a cloud-covered region:

```bash
python examples/generate_demo_data.py

vita-crop-health \
  --red demo_data/red.npy \
  --green demo_data/green.npy \
  --blue demo_data/blue.npy \
  --nir demo_data/nir.npy \
  --crop-mask demo_data/crop_mask.npy \
  --cloud-mask demo_data/cloud_mask.npy \
  --scale 1 \
  --output outputs/demo
```

## Running on project data

All arrays must be:

- two-dimensional;
- co-registered;
- on the same pixel grid;
- radiometrically comparable;
- already corrected/aligned as required by the source product.

Mask semantics:

- `crop_mask != 0`: crop pixel;
- `cloud_mask != 0`: cloud or cloud-shadow pixel to exclude.

Example for common Sentinel-2 L2A reflectance values stored around `0–10000`:

```bash
vita-crop-health \
  --red B04.tif \
  --green B03.tif \
  --blue B02.tif \
  --nir B08.tif \
  --crop-mask crop_mask.tif \
  --cloud-mask cloud_mask.tif \
  --scale 10000 \
  --output outputs/acquisition_001
```

For already normalized reflectance in `[0, 1]`, use `--scale 1`. For 8-bit reflectance-like imagery, use `--scale 255`, but note that calibrated reflectance is preferable.

## Temporal analysis

Temporal comparison is stronger than interpreting a single acquisition. The previous scene must be aligned to the current scene.

```bash
vita-crop-health \
  --red current/red.npy \
  --green current/green.npy \
  --blue current/blue.npy \
  --nir current/nir.npy \
  --crop-mask current/crop_mask.npy \
  --cloud-mask current/cloud_mask.npy \
  --previous-red previous/red.npy \
  --previous-green previous/green.npy \
  --previous-blue previous/blue.npy \
  --previous-nir previous/nir.npy \
  --previous-cloud-mask previous/cloud_mask.npy \
  --scale 1 \
  --output outputs/temporal_run
```

A decrease in NDVI/GNDVI contributes a temporal penalty. The algorithm does not treat every small change as stress: a configurable free-drop margin is used before the penalty begins.

## Outputs

Each run writes:

```text
crop_health_report.json     final score, condition, metrics and explanations
health_map.npy              per-pixel score in [0,1], NaN outside valid crop
anomaly_mask.npy            binary anomaly mask
valid_crop_mask.npy         cloud-free crop pixels used in the analysis
ndvi.npy
gndvi.npy
savi.npy
evi.npy
ndwi.npy
health_quicklook.png        optional visualization when matplotlib is installed
```

Example report:

```json
{
  "condition": "Moderate anomaly",
  "score": 54.2,
  "anomaly_fraction": 0.187,
  "valid_crop_fraction": 0.91,
  "metrics": {
    "mean_ndvi": 0.58,
    "mean_gndvi": 0.49,
    "mean_savi": 0.55,
    "mean_ndvi_change": -0.08
  },
  "reasons": [
    "18.7% of valid crop pixels were flagged as anomalous.",
    "Mean NDVI decreased by 0.080 from the previous acquisition."
  ]
}
```

Possible condition labels:

- `Healthy`
- `Watch`
- `Moderate anomaly`
- `High anomaly`
- `Insufficient data`

## How the score works

The current MVP is deliberately transparent and rule-based:

1. NDVI, GNDVI and SAVI are converted to configurable 0–1 vigor scores.
2. They are combined with default weights of 50%, 30% and 20%.
3. Pixels significantly below the median behavior of the same field receive a robust relative-anomaly penalty.
4. When a previous acquisition is provided, persistent NDVI/GNDVI decline adds a temporal penalty.
5. The field score is derived from the median and lower quartile of valid pixel scores, so localized degradation affects the output without letting a few isolated pixels dominate it.

Defaults are **generic engineering starting points**, not universal crop-health truth. Real deployment should calibrate thresholds by:

- crop type;
- phenological stage;
- region and season;
- sensor/product level;
- verified agronomic observations.

## Custom configuration

Any `AnalysisConfig` field can be overridden using JSON:

```json
{
  "ndvi_low": 0.30,
  "ndvi_high": 0.78,
  "temporal_free_drop": 0.04,
  "temporal_severe_drop": 0.18,
  "anomaly_health_threshold": 0.48
}
```

Run with:

```bash
vita-crop-health ... --config config.json
```

A command-line `--scale` value overrides `reflectance_scale` from the JSON file.

## Python API

```python
import numpy as np
from vita_crop_health import AnalysisConfig, analyze_crop_health

result = analyze_crop_health(
    bands={
        "red": red,
        "green": green,
        "blue": blue,
        "nir": nir,
    },
    crop_mask=crop_mask,
    cloud_mask=cloud_mask,
    previous_bands=None,
    config=AnalysisConfig(reflectance_scale=1.0),
)

print(result.condition)
print(result.score)
print(result.anomaly_fraction)
```

## Tests

```bash
pytest
```

## Main limitations

This MVP cannot directly determine:

- crop water content, because no SWIR band is used;
- surface temperature or thermal stress, because no thermal/FIR band is used;
- the exact cause of low vigor;
- a specific plant disease;
- agronomic yield loss without external labels and validation.

The scientifically defensible interpretation is:

> **Reduced vegetation vigor or an anomalous crop response was detected in the available multispectral bands.**
