"""Base reader."""
import typing as ty
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..types import PathLike
from ..utilities import accumulate_peaks_centroid, find_between, find_between_ppm, get_window_for_ppm


class BaseReader:
    """Base reader class."""

    # private attributes
    _xyz_coordinates: ty.Optional[np.ndarray] = None
    _tic: ty.Optional[np.ndarray] = None
    _current = -1

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def __iter__(self):
        return self

    def __next__(self):
        """Get next spectrum"""
        if self._current < self.n_pixels - 1:
            self._current += 1
            return self[self._current]
        else:
            self._current = -1
            raise StopIteration

    def __getitem__(self, item: int):
        """Retrieve spectrum"""
        return self.get_spectrum(item)

    def _init(self):
        """Method which is called to initialize the reader."""
        raise NotImplementedError("Must implement method")

    @property
    def xyz_coordinates(self) -> np.ndarray:
        """Return xyz coordinates."""
        return self._xyz_coordinates

    @property
    def x_coordinates(self) -> np.ndarray:
        """Return x-axis coordinates/"""
        return self._xyz_coordinates[:, 0]

    @property
    def y_coordinates(self) -> np.ndarray:
        """Return y-axis coordinates."""
        return self._xyz_coordinates[:, 1]

    @property
    def z_coordinates(self) -> np.ndarray:
        """Return z-axis coordinates."""
        return self._xyz_coordinates[:, 2]

    @property
    def pixels(self) -> np.ndarray:
        """Iterable of pixels in the dataset."""
        return np.arange(self.n_pixels)

    @property
    def n_pixels(self):
        """Return the total number of pixels in the dataset."""
        return len(self.x_coordinates)

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether the data is in centroid or profile mode."""
        raise NotImplementedError("Must implement method")

    def get_spectrum(self, index: int):
        """Return mass spectrum."""
        return self._read_spectrum(index)

    def _read_spectrum(self, index: int) -> ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Must implement method")

    def _read_spectra(self, indices: ty.Optional[np.ndarray] = None) -> ty.Iterator[ty.Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("Must implement method")

    def get_chromatogram(self, indices: ty.Iterable[int]):
        """Return chromatogram."""

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return TIC image."""
        if self._tic is None:
            res = np.zeros(self.n_pixels)
            for i, (_, y) in enumerate(self.iter_spectra(silent)):
                res[i] = y.sum()
            self._tic = res
        return self._tic

    def get_ion_image(self, mz: float, tol: float = None, ppm: float = None, silent: bool = False):
        """Return ion image for specified m/z with tolerance or ppm."""
        if tol is None and ppm is None or tol == 0 and ppm == 0:
            raise ValueError("Please specify `tol` or `ppm`.")
        elif tol is not None and ppm is not None:
            raise ValueError("Please only specify `tol` or `ppm`.")
        func = find_between if tol else find_between_ppm
        val = tol if tol else ppm
        res = np.zeros(self.n_pixels)
        if self.is_centroid:
            for i, (x, y) in enumerate(self.iter_spectra(silent)):
                mask = func(x, mz, val)
                res[i] = y[mask].sum()
        else:
            x, _ = self[0]
            mask = func(x, mz, val)
            for i, (x, y) in enumerate(self.iter_spectra(silent)):
                res[i] = y[mask].sum()
        return self.reshape(res)

    def get_ion_images(self, mzs: ty.Iterable[float], tol: float = None, ppm: float = None, silent: bool = False):
        """Return many ion images for specified m/z values."""
        if tol is None and ppm is None or tol == 0 and ppm == 0:
            raise ValueError("Please specify `tol` or `ppm`.")
        elif tol is not None and ppm is not None:
            raise ValueError("Please only specify `tol` or `ppm`.")
        res = np.zeros((len(mzs), self.n_pixels))
        mzs = np.asarray(mzs)
        if tol:
            mzs_min = mzs - tol
            mzs_max = mzs + tol
        else:
            ppm = np.asarray([get_window_for_ppm(mz, ppm) for mz in mzs])
            mzs_min = mzs - tol
            mzs_max = mzs + tol

        if self.is_centroid:
            for i, (x, y) in enumerate(self.iter_spectra(silent)):
                res[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)

    def reshape(self, array: np.ndarray) -> np.ndarray:
        """Reshape vector into an image."""
        raise NotImplementedError("Must implement method")

    def iter_spectra(self, silent: bool = False):
        """Yield spectra."""
        yield from tqdm(self._read_spectra(), total=self.n_pixels, disable=silent, miniters=500)
