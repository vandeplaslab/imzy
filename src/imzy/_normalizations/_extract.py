"""Get normalization utilities."""

import warnings
from pathlib import Path

import numba

try:
    import hdf5plugin
except ImportError:

    class hdf5plugin:
        """Dummy class."""

        def LZ4(*args, **kwargs):
            return {}


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
    return _compute_normalizations(reader, clean=clean, silent=silent)


def _compute_normalizations(reader, clean: bool = True, silent: bool = False) -> np.ndarray:
    """Calculate normalizations for a set of frames."""
    # specify normalization names
    names = get_normalizations()

    # pre-assign array
    n_frames = reader.n_pixels
    framelist = reader.pixels
    norm_array = np.zeros((n_frames, len(names)), dtype=np.float32)
    with reader._enable_faster_iter():
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


# quantiles of the positive intensities used by the trimmed TIC normalizations
_QUANTILES = (0.05, 0.1, 0.5, 0.9, 0.95)


@numba.njit(nogil=True, cache=True)
def _positive_quantiles(positive: np.ndarray) -> np.ndarray:
    """Linearly interpolated ``_QUANTILES`` (same as ``np.quantile``) without fully sorting the array."""
    n = positive.size
    ranks = np.empty(2 * len(_QUANTILES), dtype=np.int64)
    for j, q in enumerate(_QUANTILES):
        lo = int(np.floor(q * (n - 1)))
        ranks[2 * j] = lo
        ranks[2 * j + 1] = min(lo + 1, n - 1)
    part = np.partition(positive, ranks)

    out = np.empty(len(_QUANTILES), dtype=np.float64)
    for j, q in enumerate(_QUANTILES):
        lo, hi = ranks[2 * j], ranks[2 * j + 1]
        out[j] = part[lo] + (q * (n - 1) - lo) * (part[hi] - part[lo])
    return out


@numba.njit(nogil=True, cache=True)
def calculate_normalizations_optimized(spectrum: np.ndarray) -> np.ndarray:
    """Calculate various normalizations for a single spectrum.

    Accepts integer or floating spectra (no conversion needed) and accumulates in float64. Spectra are mostly
    zeros, so non-zero values are compacted first and all remaining work runs on those only. Quantiles are taken
    from the positive values.
    """
    px_norms = np.zeros(12, dtype=np.float32)
    n = spectrum.size

    # compact non-zero values (the branch is well predicted on sparse spectra)
    values = np.empty(n, dtype=np.float64)
    nnz = 0
    for i in range(n):
        v = spectrum[i]
        if v != 0:
            values[nnz] = v
            nnz += 1

    # moments over non-zeros; positives are compacted in place
    n_pos = 0
    s1, s2, s3, neg_sum = 0.0, 0.0, 0.0, 0.0
    for j in range(nnz):
        x = values[j]
        s1 += x
        x2 = x * x
        s2 += x2
        s3 += x2 * abs(x)
        if x > 0:
            values[n_pos] = x
            n_pos += 1
        else:
            neg_sum += x

    px_norms[0] = s1  # TIC
    if px_norms[0] == 0:
        return px_norms

    px_norms[1] = np.sqrt(s2 / n)  # RMS
    px_norms[9] = nnz  # 0-norm
    px_norms[10] = np.sqrt(s2)  # 2-norm
    px_norms[11] = np.cbrt(s3)  # 3-norm

    if n_pos == 0:
        return px_norms

    positive = values[:n_pos]
    q5, q10, median, q90, q95 = _positive_quantiles(positive)
    px_norms[2] = median  # Median
    if n_pos == 1:
        q5, q10, q90, q95 = 0.0, 0.0, 0.0, 0.0

    # thresholded sums over positives; negatives are always below q90/q95 and never above q5/q10
    lt95, lt90, gt5, gt10, mid95, mid90 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for x in positive:
        if x < q95:
            lt95 += x
            if x > q5:
                mid95 += x
        if x < q90:
            lt90 += x
            if x > q10:
                mid90 += x
        if x > q5:
            gt5 += x
        if x > q10:
            gt10 += x

    px_norms[3] = neg_sum + lt95  # 0-95% TIC (x < q95)
    px_norms[4] = neg_sum + lt90  # 0-90% TIC (x < q90)
    px_norms[5] = gt5  # 5-100% TIC (x > q5)
    px_norms[6] = gt10  # 10-100% TIC (x > q10)
    px_norms[7] = mid95  # 5-95% TIC (q5 < x < q95)
    px_norms[8] = mid90  # 10-90% TIC (q10 < x < q90)
    return px_norms


# Precompile numba functions
def _precompute():
    import os

    if not os.environ.get("IMZY_PRECOMPUTE", "0") == "1":
        return
    calculate_normalizations(np.zeros(10, dtype=np.float32))
    calculate_normalizations_optimized(np.zeros(10, dtype=np.float32))


_precompute()
