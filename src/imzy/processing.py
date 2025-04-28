"""Processing."""

import typing as ty

import numba
import numpy as np
from ims_utils.spectrum import find_between_batch

IndicesType = numba.types.List(np.ndarray)


@numba.njit(cache=True, fastmath=True)
def accumulate_peaks_centroid(peaks_min: np.ndarray, peaks_max: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Sum intensities for specified number of peaks where each spectra and in centroid-mode."""
    indices = find_between_batch(x, peaks_min, peaks_max)
    result = np.zeros(len(peaks_min), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


def accumulate_peaks_profile(indices: list[np.ndarray], y: np.ndarray):
    """Sum intensities for specified number of peaks where spectra are in profile-mode."""
    return _accumulate_peaks_profile(numba.typed.List(indices), y)


@numba.njit(cache=True, fastmath=True)
def _accumulate_peaks_profile(indices: IndicesType, y: np.ndarray) -> np.ndarray:
    """Sum intensities for specified number of peaks where spectra are in profile-mode."""
    result = np.zeros(len(indices), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


# Precompile numba functions
def _precompute():
    import os

    if not os.environ.get("IMZY_PRECOMPUTE", "0") == "1":
        return
    x = np.arange(10, dtype=np.float32)
    y = np.random.randn(10)
    mins = np.array([1, 2], dtype=np.float32)
    maxs = np.array([3, 4], dtype=np.float32)
    accumulate_peaks_centroid(mins, maxs, x, y)
    mask = np.full(10, False, dtype=bool)
    mask[1:3] = True
    accumulate_peaks_profile([mask], y)


_precompute()
