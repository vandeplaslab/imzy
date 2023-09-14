"""Utility functions."""
import typing as ty

import numba
import numpy as np
from koyo.spectrum import find_between_batch
from scipy.interpolate import interp1d


def resample_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray):
    """Resample array at specified ppm."""
    mz_idx = np.digitize(mz_array, new_mz, True)

    # sum together values from multiple bins
    y_ppm = np.zeros_like(new_mz)
    for i, idx in enumerate(mz_idx):
        try:
            y_ppm[idx] += intensity_array[i]
        except IndexError:
            pass
    return y_ppm


def interpolate_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray):
    """Resample array at specified ppm."""
    func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
    return new_mz, func(new_mz)


@numba.njit(cache=True, fastmath=True)
def accumulate_peaks_centroid(peaks_min: np.ndarray, peaks_max: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Sum intensities for specified number of peaks where each spectra and in centroid-mode."""
    indices = find_between_batch(x, peaks_min, peaks_max)
    result = np.zeros(len(peaks_min), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


IndicesType = numba.types.List(np.ndarray)


def accumulate_peaks_profile(indices: ty.List[np.ndarray], y: np.ndarray):
    """Sum intensities for specified number of peaks where spectra are in profile-mode."""
    return _accumulate_peaks_profile(numba.typed.List(indices), y)


@numba.njit(cache=True, fastmath=True)
def _accumulate_peaks_profile(indices: IndicesType, y: np.ndarray):
    """Sum intensities for specified number of peaks where spectra are in profile-mode."""
    result = np.zeros(len(indices), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


def optimize_chunks_along_axis(
    axis: int,
    *,
    array: ty.Optional[np.ndarray] = None,
    shape: ty.Optional[ty.Tuple[int, ...]] = None,
    dtype=None,
    max_size: int = 1e6,
) -> ty.Optional[ty.Tuple[int, ...]]:
    """Optimize chunk size along specified axis."""
    if array is not None:
        dtype, shape = array.dtype, array.shape
    elif shape is None or dtype is None:
        raise ValueError("You must specify either an array or `shape` and `dtype`")
    assert len(shape) == 2, "Only supporting 2d arrays at the moment."
    assert axis <= 1, "Only supporting 2d arrays at the moment, use -1, 0 or 1 in the `axis` argument"
    assert hasattr(dtype, "itemsize"), "Data type must have the attribute 'itemsize'"
    item_size = np.dtype(dtype).itemsize

    if max_size == 0:
        return None

    n = 0
    max_n = shape[1] if axis == 0 else shape[0]
    while (n * item_size * shape[axis]) <= max_size and n < max_n:
        n += 1
    if n < 1:
        n = 1
    return (shape[0], n) if axis == 0 else (n, shape[1])
