"""Normalizations."""
from pathlib import Path

import numba
import numpy as np
from koyo.typing import PathLike
from tqdm import tqdm

try:
    import hdf5plugin
except ImportError:

    class hdf5plugin:
        """Dummy class."""

        LZ4 = lambda *args, **kwargs: {}


from imzy._normalizations._extract import get_normalizations
from imzy._normalizations._hdf5_store import H5NormalizationStore


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
        group = store._get_group(h5, store.NORMALIZATIONS_KEY)
        for normalization in get_normalizations():
            group.create_dataset(normalization, shape=(n_pixels,), dtype=np.float32, **compression)
    store.flush()
    return hdf_path


def extract_normalizations_hdf5(input_dir: PathLike, hdf_path: PathLike, silent: bool = False) -> Path:
    """Extract normalizations from hdf5."""
    normalization_names = get_normalizations()
    normalizations = compute_normalizations(input_dir, silent=silent)

    store = H5NormalizationStore(hdf_path, mode="a")
    with store.open() as h5:
        group = store._get_group(h5, store.NORMALIZATIONS_KEY)
        for i, normalization in enumerate(normalization_names):
            norm = normalizations[:, i]
            group[normalization][:] = norm
    store.flush()
    return hdf_path


def compute_normalizations(input_dir: Path, silent: bool = False) -> np.ndarray:
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
        norm_array[i] = calculate_normalizations(y.astype(np.float32))
    return norm_array


@numba.njit(fastmath=True, cache=True)
def calculate_normalizations(spectrum: np.ndarray) -> np.ndarray:
    """Calculate various normalizations.

    This function expects spectrum in the form

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
