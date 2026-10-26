"""Seeded synthetic inputs shared by benchmark setup and correctness tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from imzy import IMZMLWriter

LAYOUTS = ("continuous_profile", "processed_centroid")
PIXELS = (256, 4096)
POINTS = 2048
TARGETS = np.linspace(150.0, 950.0, 32)
TOLERANCE = 0.3


def spectra(layout: str, pixels: int, points: int = POINTS) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield varied spectra with a shared profile axis or independent centroid axes."""
    if layout not in LAYOUTS or pixels < 1 or points < 2:
        raise ValueError("Expected a supported layout, positive pixels, and at least two points.")
    rng = np.random.default_rng(20261026)
    axis = np.linspace(100.0, 1000.0, points)
    for _ in range(pixels):
        mzs = (
            axis
            if layout == "continuous_profile"
            else np.sort(rng.uniform(100.0, 1000.0, int(rng.integers(points // 2, points + 1))))
        )
        yield mzs, rng.lognormal(3.0, 1.2, size=len(mzs)).astype(np.float32)


def dataset_path(directory: Path, layout: str, pixels: int) -> Path:
    """Return the stable path used by ASV's shared setup cache."""
    return directory / f"{layout}-{pixels}.imzML"


def generate(directory: Path, layout: str, pixels: int, points: int = POINTS) -> Path:
    """Write seeded imzML data without including generation in measured workloads."""
    directory.mkdir(parents=True, exist_ok=True)
    path = dataset_path(directory, layout, pixels)
    continuous = layout == "continuous_profile"
    with IMZMLWriter(
        path,
        spectrum_type="profile" if continuous else "centroid",
        ibd_mode="continuous" if continuous else "processed",
    ) as writer:
        for i, (mzs, intensities) in enumerate(spectra(layout, pixels, points)):
            writer.add_spectrum(mzs, intensities, (i % 128 + 1, i // 128 + 1))
    return path


def consume(mzs: np.ndarray, intensities: np.ndarray) -> float:
    """Read every value with identical reductions for both backends."""
    return float(mzs.sum(dtype=np.float64) + intensities.sum(dtype=np.float64))
