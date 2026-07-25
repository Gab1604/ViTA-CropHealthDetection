import numpy as np

from vita_crop_health.indices import calculate_indices, normalized_difference


def test_calculate_indices_known_values() -> None:
    shape = (3, 4)
    indices = calculate_indices(
        red=np.full(shape, 0.20),
        green=np.full(shape, 0.30),
        blue=np.full(shape, 0.10),
        nir=np.full(shape, 0.60),
    )

    np.testing.assert_allclose(indices["ndvi"], 0.50, atol=1e-6)
    np.testing.assert_allclose(indices["gndvi"], 1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(indices["ndwi"], -1.0 / 3.0, atol=1e-6)
    np.testing.assert_allclose(indices["savi"], 0.6 / 1.3, atol=1e-6)
    np.testing.assert_allclose(indices["evi"], 1.0 / 2.05, atol=1e-6)


def test_normalized_indices_are_scale_invariant() -> None:
    reflectance = calculate_indices(
        red=np.array([[0.2]], dtype=np.float32),
        green=np.array([[0.3]], dtype=np.float32),
        blue=np.array([[0.1]], dtype=np.float32),
        nir=np.array([[0.6]], dtype=np.float32),
        scale=1.0,
    )
    scaled = calculate_indices(
        red=np.array([[2000]], dtype=np.uint16),
        green=np.array([[3000]], dtype=np.uint16),
        blue=np.array([[1000]], dtype=np.uint16),
        nir=np.array([[6000]], dtype=np.uint16),
        scale=10000.0,
    )

    for name in ("ndvi", "gndvi", "savi", "evi", "ndwi"):
        np.testing.assert_allclose(reflectance[name], scaled[name], atol=1e-6)


def test_zero_denominator_produces_nan() -> None:
    result = normalized_difference(
        np.array([[1.0]], dtype=np.float32),
        np.array([[-1.0]], dtype=np.float32),
    )
    assert np.isnan(result[0, 0])
