"""Generate a small synthetic four-band scene for testing the CLI."""

from pathlib import Path

import numpy as np


OUTPUT = Path("demo_data")
OUTPUT.mkdir(exist_ok=True)
shape = (256, 256)
y, x = np.mgrid[: shape[0], : shape[1]]

# Reflectance-like values in [0, 1].
red = np.full(shape, 0.16, dtype=np.float32)
green = np.full(shape, 0.27, dtype=np.float32)
blue = np.full(shape, 0.11, dtype=np.float32)
nir = np.full(shape, 0.73, dtype=np.float32)

crop_mask = np.zeros(shape, dtype=np.uint8)
crop_mask[24:232, 20:236] = 1
cloud_mask = np.zeros(shape, dtype=np.uint8)
cloud_mask[30:85, 165:230] = 1

# A low-vigor elliptical patch.
anomaly = ((x - 92) / 45) ** 2 + ((y - 158) / 35) ** 2 <= 1
anomaly &= crop_mask.astype(bool)
red[anomaly] = 0.36
green[anomaly] = 0.40
nir[anomaly] = 0.48

for name, array in {
    "red": red,
    "green": green,
    "blue": blue,
    "nir": nir,
    "crop_mask": crop_mask,
    "cloud_mask": cloud_mask,
}.items():
    np.save(OUTPUT / f"{name}.npy", array)

print(f"Demo arrays written to {OUTPUT.resolve()}")
