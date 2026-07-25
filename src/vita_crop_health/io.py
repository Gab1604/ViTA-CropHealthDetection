"""Input/output helpers for NumPy arrays and optional GeoTIFF files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .analyzer import AnalysisConfig, CropHealthResult


def load_array(path: str | Path, *, band: int = 1) -> np.ndarray:
    """Load a 2-D array from ``.npy``, ``.npz`` or GeoTIFF.

    GeoTIFF support is optional and requires installing ``vita-crop-health[geotiff]``.
    For ``.npz`` files containing multiple arrays, append ``::key`` to the path.
    """
    raw_path = str(path)
    npz_key: str | None = None
    if "::" in raw_path:
        raw_path, npz_key = raw_path.rsplit("::", 1)
    file_path = Path(raw_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        array = np.load(file_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(file_path, allow_pickle=False) as archive:
            if npz_key is None:
                if len(archive.files) != 1:
                    raise ValueError(
                        f"{file_path} contains {archive.files}; select one with path::key."
                    )
                npz_key = archive.files[0]
            if npz_key not in archive.files:
                raise KeyError(f"Key {npz_key!r} not found in {file_path}; available: {archive.files}")
            array = archive[npz_key]
    elif suffix in {".tif", ".tiff"}:
        try:
            import rasterio
        except ImportError as exc:
            raise RuntimeError(
                "GeoTIFF input requires rasterio. Install with: pip install -e '.[geotiff]'"
            ) from exc
        with rasterio.open(file_path) as dataset:
            if not 1 <= band <= dataset.count:
                raise ValueError(
                    f"Band index {band} is invalid for {file_path}; it has {dataset.count} band(s)."
                )
            array = dataset.read(band)
    else:
        raise ValueError(f"Unsupported input format for {file_path}. Use .npy, .npz, .tif or .tiff.")

    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{file_path} must contain a 2-D array, got shape {array.shape}.")
    return array


def save_result(
    output_directory: str | Path,
    result: CropHealthResult,
    *,
    config: AnalysisConfig,
    create_quicklook: bool = True,
) -> Path:
    """Write the report, index rasters and masks to an output directory."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "crop_health_report.json"
    report_path.write_text(
        json.dumps(result.to_dict(config), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if result.health_map is not None:
        np.save(output / "health_map.npy", result.health_map)
    if result.anomaly_mask is not None:
        np.save(output / "anomaly_mask.npy", result.anomaly_mask.astype(np.uint8))
    if result.valid_mask is not None:
        np.save(output / "valid_crop_mask.npy", result.valid_mask.astype(np.uint8))
    for name, values in result.indices.items():
        np.save(output / f"{name}.npy", values)

    if create_quicklook and result.health_map is not None:
        _save_quicklook(output / "health_quicklook.png", result)

    return report_path


def _save_quicklook(path: Path, result: CropHealthResult) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(result.health_map, vmin=0.0, vmax=1.0, cmap="RdYlGn")
    if result.anomaly_mask is not None and np.any(result.anomaly_mask):
        axis.contour(result.anomaly_mask.astype(float), levels=[0.5], linewidths=0.8)
    axis.set_title(f"Crop condition: {result.condition} | score: {result.score}")
    axis.set_axis_off()
    figure.colorbar(image, ax=axis, label="Health score (0-1)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def json_ready(value: Any) -> Any:
    """Convert NumPy scalar values to regular Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value
