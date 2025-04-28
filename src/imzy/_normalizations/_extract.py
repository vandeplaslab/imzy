"""Get normalization utilities."""

import warnings
from pathlib import Path

import numba

try:
    import hdf5plugin
except ImportError:

    class hdf5plugin:
        """Dummy class."""

        LZ4 = lambda *args, **kwargs: {}


import numpy as np
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._normalizations._hdf5_store import H5NormalizationStore


def get_normalizations() -> list[str]:
    """Get list of available normalizations."""
    return [
        "TIC",
        "RMS",
        "Median",
        "0-95% TIC",
        "0-90% TIC",
        "5-100% TIC",
        "10-100% TIC",
        "5-95% TIC",
        "10-90% TIC",
        "0-norm",
        "2-norm",
        "3-norm",
    ]


def _get_outlier_mask(norm: np.ndarray, n_orders: int = 2) -> np.ndarray:
    """Retrieve normalization and determine outliers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        norm = np.log10(norm + 1)
        mask = norm < (norm.max() - n_orders)
    return mask


def create_normalizations_hdf5(input_dir: PathLike, hdf_path: PathLike) -> Path:
    """Create group with datasets inside."""
    from imzy._readers import get_reader

    reader = get_reader(input_dir)
    n_pixels = reader.n_pixels

    compression = hdf5plugin.LZ4()

    if Path(hdf_path).suffix != ".h5":
        hdf_path = Path(hdf_path).with_suffix(".h5")

    store = H5NormalizationStore(hdf_path, mode="a")
    with store.open() as h5:
        group = store._add_group(h5, store.NORMALIZATION_KEY)
        for normalization in get_normalizations():
            store._add_array_to_group(group, normalization, None, shape=(n_pixels,), dtype=np.float32, **compression)
    store.flush()
    return hdf_path


def extract_normalizations_hdf5(input_dir: PathLike, hdf_path: PathLike, silent: bool = False) -> Path:
    """Extract normalizations from hdf5."""
    normalization_names = get_normalizations()
    normalizations = compute_normalizations(input_dir, clean=True, silent=silent)

    store = H5NormalizationStore(hdf_path, mode="a")
    with np.errstate(invalid="ignore", divide="ignore"):
        with store.open() as h5:
            group = store._get_group(h5, store.NORMALIZATION_KEY)
            for i, normalization in enumerate(normalization_names):
                group[normalization][:] = normalizations[:, i]
            store.flush()
    return hdf_path


def compute_normalizations(input_dir: Path, clean: bool = True, silent: bool = False) -> np.ndarray:
    """Calculate normalizations for a set of frames."""
    from imzy._readers import get_reader

    reader = get_reader(input_dir)

    # specify normalization names
    names = get_normalizations()

    # pre-assign array
    n_frames = reader.n_pixels
    framelist = reader.pixels
    # start_frame, norm_array = check_cache()
    norm_array = np.zeros((n_frames, len(names)), dtype=np.float32)
    for i, (_, y) in enumerate(
        tqdm(
            reader.spectra_iter(framelist, silent=True),
            disable=silent,
            miniters=100,
            mininterval=2,
            total=len(framelist),
            desc="Computing normalizations...",
        )
    ):
        y = y.astype(np.float32)
        try:
            norm_array[i] = calculate_normalizations_optimized(y)
        except Exception:
            norm_array[i] = calculate_normalizations_optimized.py_func(y)
    norm_array = np.nan_to_num(norm_array, nan=1.0)
    # clean-up normalizations
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(norm_array.shape[1]):
            # get normalization
            norm = norm_array[:, i]
            if clean:
                # remove outliers
                mask = _get_outlier_mask(norm, 2)
                norm[mask] = np.median(norm[mask])
                # save the normalizations as 'multiplier' version so it's easier to apply
            norm_array[:, i] = 1 / (norm / np.median(norm))
    return norm_array


@numba.njit(fastmath=True, cache=True)
def calculate_normalizations(spectrum: np.ndarray) -> np.ndarray:
    """Calculate various normalizations.

    This function expects float32 spectrum.
    """
    px_norms = np.zeros(12, dtype=np.float32)
    px_norms[0] = np.sum(spectrum)  # TIC
    if px_norms[0] == 0:
        return px_norms

    px_norms[1] = np.sqrt(np.mean(np.square(spectrum)))  # RMS
    px_norms[2] = np.median(spectrum[spectrum > 0])  # Median
    _spectrum = spectrum[spectrum > 0]
    if _spectrum.size > 1:
        q95, q90, q10, q5 = np.nanquantile(_spectrum, [0.95, 0.9, 0.1, 0.05])
    else:
        q95, q90, q10, q5 = 0, 0, 0, 0
    px_norms[3] = np.sum(spectrum[spectrum < q95])  # 0-95% TIC
    px_norms[4] = np.sum(spectrum[spectrum < q90])  # 0-90% TIC
    px_norms[5] = np.sum(spectrum[spectrum > q5])  # 5-100% TIC
    px_norms[6] = np.sum(spectrum[spectrum > q10])  # 10-100% TIC
    px_norms[7] = np.sum(spectrum[(spectrum > q5) & (spectrum < q95)])  # 5-95% TIC
    px_norms[8] = np.sum(spectrum[(spectrum > q10) & (spectrum < q90)])  # 10-90% TIC
    px_norms[9] = np.linalg.norm(spectrum, 0)  # 0-norm
    px_norms[10] = np.linalg.norm(spectrum, 2)  # 2-norm
    px_norms[11] = np.linalg.norm(spectrum, 3)  # 3-norm
    return px_norms


@numba.njit(fastmath=True, cache=True)
def calculate_normalizations_optimized(spectrum: np.ndarray) -> np.ndarray:
    """Calculate various normalizations, optimized version.

    This function expects a float32 spectrum.
    """
    px_norms = np.zeros(12, dtype=np.float32)

    # Direct computation without conditions
    px_norms[0] = np.sum(spectrum)  # TIC
    if px_norms[0] == 0:
        return px_norms

    # Compute these norms directly without extra conditions
    px_norms[1] = np.sqrt(np.mean(np.square(spectrum)))  # RMS
    px_norms[9] = np.linalg.norm(spectrum, 0)  # 0-norm
    px_norms[10] = np.linalg.norm(spectrum, 2)  # 2-norm
    px_norms[11] = np.linalg.norm(spectrum, 3)  # 3-norm

    # Filter positive values once and reuse
    positive_spectrum = spectrum[spectrum > 0]
    px_norms[2] = np.median(positive_spectrum) if positive_spectrum.size > 0 else 0  # Median

    # Calculating quantiles once for all needed
    if positive_spectrum.size > 1:
        q95, q90, q10, q5 = np.nanquantile(positive_spectrum, [0.95, 0.9, 0.1, 0.05])
    else:
        q95, q90, q10, q5 = 0, 0, 0, 0

    # Using logical indexing with boolean arrays might be faster due to numba optimization
    condition_q95 = spectrum < q95
    condition_q90 = spectrum < q90
    condition_q5 = spectrum > q5
    condition_q10 = spectrum > q10

    px_norms[3] = np.sum(spectrum[condition_q95])  # 0-95% TIC
    px_norms[4] = np.sum(spectrum[condition_q90])  # 0-90% TIC
    px_norms[5] = np.sum(spectrum[condition_q5])  # 5-100% TIC
    px_norms[6] = np.sum(spectrum[condition_q10])  # 10-100% TIC

    # For ranges, we can combine conditions
    px_norms[7] = np.sum(spectrum[condition_q5 & condition_q95])  # 5-95% TIC
    px_norms[8] = np.sum(spectrum[condition_q10 & condition_q90])  # 10-90% TIC
    return px_norms


# Precompile numba functions
def _precompute():
    import os

    if not os.environ.get("IMZY_PRECOMPUTE", "0") == "1":
        return
    calculate_normalizations(np.zeros(10, dtype=np.float32))
    calculate_normalizations_optimized(np.zeros(10, dtype=np.float32))


_precompute()
