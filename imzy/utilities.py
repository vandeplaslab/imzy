"""Utility functions."""
import typing as ty

import numba
import numpy as np
from scipy.interpolate import interp1d


def chunks(item_list, n_items: int):
    """Yield successive n-sized chunks from `item_list`."""
    for i in range(0, len(item_list), n_items):
        yield item_list[i : i + n_items]


def find_nearest_index(data: np.ndarray, value: ty.Union[int, float, np.ndarray, ty.Iterable]):
    """Find nearest index of asked value

    Parameters
    ----------
    data : np.array
        input array (e.g. m/z values)
    value : Union[int, float, np.ndarray]
        asked value

    Returns
    -------
    index :
        index value
    """
    data = np.asarray(data)
    if isinstance(value, ty.Iterable):
        return [np.argmin(np.abs(data - _value)) for _value in value]
    return np.argmin(np.abs(data - value))


def find_nearest_value(data: np.ndarray, value: ty.Union[int, float, np.ndarray, ty.Iterable]):
    """Find nearest value"""
    idx = find_nearest_index(data, value)
    return data[idx]


def find_nearest_index_sorted(array, value):
    """Much quicker implementation of `find_nearest_index` if the data is sorted"""
    return np.searchsorted(array, value, side="left")


@numba.njit(fastmath=True, cache=True)
def ppm_error(x: ty.Union[float, np.ndarray], y: ty.Union[float, np.ndarray]) -> ty.Union[float, np.ndarray]:
    """Calculate ppm error"""
    return ((x - y) / y) * 1e6


@numba.njit(fastmath=True, cache=True)
def get_window_for_ppm(mz: float, ppm: float) -> float:
    """Calculate window size for specified peak at specified ppm."""
    step = mz * 1e-6  # calculate appropriate step size for specified mz value
    peak_x_ppm = mz
    is_subtract = ppm < 0
    ppm = abs(ppm)
    while True:
        if ((peak_x_ppm - mz) / mz) * 1e6 >= ppm:
            break
        peak_x_ppm += step
    value = peak_x_ppm - mz
    return value if not is_subtract else -value


def resample_ppm(new_mz: np.ndarray, mz_array: np.ndarray, intensity_array: np.ndarray):
    """Resample array at specified ppm"""
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
    """Resample array at specified ppm"""
    func = interp1d(mz_array, intensity_array, fill_value=0, bounds_error=False)
    return new_mz, func(new_mz)


@numba.njit(cache=True, fastmath=True)
def find_between(data: np.ndarray, min_value: float, max_value: float):
    """Find indices between windows."""
    return np.where(np.logical_and(data >= min_value, data <= max_value))[0]


@numba.njit(cache=True, fastmath=True)
def find_between_ppm(data: np.ndarray, value: float, ppm: float):
    """Find indices between window and ppm."""
    window = get_window_for_ppm(value, abs(ppm))
    return find_between(data, value - window, value + window)


@numba.njit(cache=True, fastmath=True)
def find_between_batch(array: np.ndarray, min_value: np.ndarray, max_value: np.ndarray):
    """Find indices between specified boundaries for many items."""
    res = []
    for i in range(len(min_value)):
        res.append(find_between(array, min_value[i], max_value[i]))
    return res


@numba.njit()
def accumulate_peaks_centroid(peaks_min: np.ndarray, peaks_max: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Sum intensities for specified number of peaks where each spectra and in centroid-mode."""
    indices = find_between_batch(x, peaks_min, peaks_max)
    result = np.zeros(len(peaks_min), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


@numba.njit()
def accumulate_peaks_profile(indices: ty.List[np.ndarray], y: np.ndarray):
    """Sum intensities for specified number of peaks where spectra are in profile-mode."""
    result = np.zeros(len(indices), dtype=y.dtype)
    for i, mask in enumerate(indices):
        if mask.size > 0:
            result[i] = y[mask].sum()
    return result


def get_mzs_for_tol(mzs: np.ndarray, tol: float = None, ppm: float = None):
    """Get min/max values for specified tolerance or ppm."""
    if tol is None and ppm is None or tol == 0 and ppm == 0:
        raise ValueError("Please specify `tol` or `ppm`.")
    elif tol is not None and ppm is not None:
        raise ValueError("Please only specify `tol` or `ppm`.")

    mzs = np.asarray(mzs)
    if tol:
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    else:
        tol = np.asarray([get_window_for_ppm(mz, ppm) for mz in mzs])
        mzs_min = mzs - tol
        mzs_max = mzs + tol
    return mzs_min, mzs_max


def get_ppm_axis(mz_start: float, mz_end: float, ppm: float):
    """Compute sequence of m/z values at a particular ppm"""
    import math

    if mz_start == 0 or mz_end == 0 or ppm == 0:
        raise ValueError("Input values cannot be equal to 0.")
    length = (np.log(mz_end) - np.log(mz_start)) / np.log((1 + 1e-6 * ppm) / (1 - 1e-6 * ppm))
    length = math.floor(length) + 1
    mz = mz_start * np.power(((1 + 1e-6 * ppm) / (1 - 1e-6 * ppm)), (np.arange(length)))
    return mz


@numba.njit()
def trim_axis(x: np.ndarray, y: np.ndarray, min_val: float, max_val: float):
    """Trim axis to prevent accumulation of edges."""
    mask = np.where((x >= min_val) & (x <= max_val))
    return x[mask], y[mask]


@numba.njit()
def set_ppm_axis(mz_x: np.ndarray, mz_y: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Set values for axis."""
    mz_idx = np.digitize(x, mz_x, True)
    for i, idx in enumerate(mz_idx):
        mz_y[idx] += y[i]
    return mz_y
