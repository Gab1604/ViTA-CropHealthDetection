"""Multispectral vegetation-index calculations for Blue, Green, Red and NIR data."""

from __future__ import annotations

from typing import Mapping

import numpy as np


_EPSILON = 1e-8


def _as_float_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}.")
    return array


def _validate_same_shape(bands: Mapping[str, np.ndarray]) -> None:
    shapes = {name: array.shape for name, array in bands.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"All bands must have the same shape, got {shapes}.")


def normalize_reflectance(band: np.ndarray, scale: float) -> np.ndarray:
    """Convert a band to floating-point reflectance.

    ``scale`` is the denominator used by the source product. Examples:
    Sentinel-2 L2A commonly uses 10000, while already normalized reflectance uses 1.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number.")
    return np.asarray(band, dtype=np.float32) / np.float32(scale)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate ``(a - b) / (a + b)`` while avoiding invalid divisions."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = a + b
    result = np.full(a.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > _EPSILON)
    result[valid] = (a[valid] - b[valid]) / denominator[valid]
    return np.clip(result, -1.0, 1.0)


def calculate_indices(
    *,
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    nir: np.ndarray,
    scale: float = 1.0,
    savi_l: float = 0.5,
) -> dict[str, np.ndarray]:
    """Calculate the indices supported by the available four bands.

    Returned maps:

    - ``ndvi``: general vegetation vigor.
    - ``gndvi``: green-band variant, sensitive to chlorophyll-related variation.
    - ``savi``: soil-adjusted vegetation index.
    - ``evi``: enhanced vegetation index; most reliable with calibrated reflectance.
    - ``ndwi``: Green-NIR water index for open/surface water, not plant water content.
    """
    if not np.isfinite(savi_l) or savi_l < 0:
        raise ValueError("savi_l must be a finite non-negative number.")

    raw = {
        "red": _as_float_array(red, "red"),
        "green": _as_float_array(green, "green"),
        "blue": _as_float_array(blue, "blue"),
        "nir": _as_float_array(nir, "nir"),
    }
    _validate_same_shape(raw)

    bands = {name: normalize_reflectance(array, scale) for name, array in raw.items()}
    red_r = bands["red"]
    green_r = bands["green"]
    blue_r = bands["blue"]
    nir_r = bands["nir"]

    ndvi = normalized_difference(nir_r, red_r)
    gndvi = normalized_difference(nir_r, green_r)
    ndwi = normalized_difference(green_r, nir_r)

    savi_denominator = nir_r + red_r + savi_l
    savi = np.full(red_r.shape, np.nan, dtype=np.float32)
    savi_valid = (
        np.isfinite(nir_r)
        & np.isfinite(red_r)
        & (np.abs(savi_denominator) > _EPSILON)
    )
    savi[savi_valid] = (
        (nir_r[savi_valid] - red_r[savi_valid])
        * (1.0 + savi_l)
        / savi_denominator[savi_valid]
    )
    savi = np.clip(savi, -1.5, 1.5)

    evi_denominator = nir_r + 6.0 * red_r - 7.5 * blue_r + 1.0
    evi = np.full(red_r.shape, np.nan, dtype=np.float32)
    evi_valid = (
        np.isfinite(nir_r)
        & np.isfinite(red_r)
        & np.isfinite(blue_r)
        & (np.abs(evi_denominator) > _EPSILON)
    )
    evi[evi_valid] = (
        2.5
        * (nir_r[evi_valid] - red_r[evi_valid])
        / evi_denominator[evi_valid]
    )
    evi = np.clip(evi, -1.0, 2.0)

    return {
        "ndvi": ndvi.astype(np.float32, copy=False),
        "gndvi": gndvi.astype(np.float32, copy=False),
        "savi": savi.astype(np.float32, copy=False),
        "evi": evi.astype(np.float32, copy=False),
        "ndwi": ndwi.astype(np.float32, copy=False),
    }
