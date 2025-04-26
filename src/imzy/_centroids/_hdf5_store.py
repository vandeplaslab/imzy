"""Wrapper class for HDF5-based centroids."""

import typing as ty
from contextlib import contextmanager

import numpy as np
from yoki5.base import Store

try:
    import h5py
except ImportError:
    h5py = None

from koyo.typing import PathLike
from koyo.utilities import find_nearest_index

from imzy._centroids._base import BaseCentroids


def format_ppm(mz: float, ppm: float):
    """Create formatted ppm'¬s."""
    return f"{mz:.3f} Da ± {ppm:.1f}ppm"


def format_tol(mz: float, mda: float):
    """Create formatted label."""
    return f"{mz:.3f} Da ± {mda:.3f}Da"


class H5CentroidsStore(Store, BaseCentroids):
    """HDF5-centroids."""

    # Private attributes
    PEAKS_KEY = "Array"
    PEAKS_ARRAY_KEY = "Array/array"
    SPATIAL_KEY = "Misc/Spatial"

    # Cache attributes
    _chunk_info = None
    _xs = None
    _proxy = None
    _is_chunked = None
    _low_mem: bool = True
    _ion_labels = None
    _tol, _ppm = None, None
    _peaks = None
    unique_id: str = ""

    def __init__(
        self,
        path: PathLike,
        xyz_coordinates: ty.Optional[np.ndarray] = None,
        pixel_index: ty.Optional[np.ndarray] = None,
        image_shape: ty.Optional[tuple[int, int]] = None,
        mode: str = "a",
    ):
        Store.__init__(self, path=path, mode=mode)
        BaseCentroids.__init__(self, xyz_coordinates, pixel_index, image_shape)

    @property
    def is_low_mem(self) -> bool:
        """Get low memory flag."""
        return self._low_mem

    @is_low_mem.setter
    def is_low_mem(self, value: bool):
        self._low_mem = value
        if self._proxy:
            self._proxy.low_mem = value

    @property
    def is_chunked(self) -> bool:
        """Get chunked flag."""
        if self._is_chunked is None:
            with self.open() as h5:
                self._is_chunked = h5[self.PEAKS_KEY].attrs.get("is_chunked")
        return self._is_chunked

    @property
    def xs(self):
        """Get xs."""
        if self._xs is None:
            with self.open() as h5:
                self._xs = h5[self.PEAKS_KEY]["mzs"][:]
        return self._xs

    @property
    def chunk_info(self) -> ty.Optional[dict[int, np.ndarray]]:
        """Returns chunked data."""
        from natsort import natsorted

        if not self.is_chunked:
            return None

        if self._chunk_info is None:
            chunk_info = {}
            start, end = 0, 0
            with self.open("r") as h5:
                for key in natsorted(h5[self.PEAKS_KEY].keys()):
                    try:
                        key_int = int(key)
                    except ValueError:
                        continue
                    end += len(h5[f"{self.PEAKS_KEY}/{key}"])
                    chunk_info[key_int] = np.arange(start, end)
                    start = end
            self._chunk_info = chunk_info
        return self._chunk_info

    @contextmanager
    def lazy_peaks(self):
        """Get reference to the peak's data without actually loading it into memory."""
        if self.is_chunked:
            if self._proxy is None:
                self._proxy = LazyPeaksProxy(self)
            yield self._proxy
        else:
            with self.open() as h5:
                group = h5[self.PEAKS_KEY]
                yield group["array"]

    @property
    def peaks(self) -> np.ndarray:
        """Load peaks data.

        This function uses custom data loading to speed things up. Because we chunk the ion images along one dimension,
        it becomes very slow to read the entire data in one go. It can be optimized a little by read the data one image
        at a time and then building 2d array.
        """
        if self._peaks is None:
            if self.is_chunked:
                if self._proxy is None:
                    self._proxy = LazyPeaksProxy(self)
                self._peaks = self._proxy.peaks()
            else:
                array = np.zeros(self.shape, dtype=self.dtype)
                with self.open("r") as h5:
                    for sl in h5[self.PEAKS_ARRAY_KEY].iter_chunks():
                        array[sl] = h5[self.PEAKS_ARRAY_KEY][sl]
                self._peaks = array
        return self._peaks

    @property
    def tol(self) -> int:
        """Get Da tolerance the data was extracted at."""
        if self._tol is None:
            self._tol = self.get_attr(self.PEAKS_KEY, "tol")
        return self._tol

    @property
    def ppm(self) -> float:
        """Get ppm/bins the data was extracted at."""
        if self._ppm is None:
            self._ppm = self.get_attr(self.PEAKS_KEY, "ppm")
        return self._ppm

    def _setup_labels(self):
        """Setup labels."""
        if self._ion_labels is None:
            mzs = self.xs
            ppm = self.ppm
            tol = self.tol
            self._ion_labels = {
                "ppm": [format_ppm(mz, ppm) for mz in mzs] if ppm else [],
                "tol": [format_tol(mz, tol) for i, mz in enumerate(mzs)] if tol is not None else [],
            }
            if not self._ion_labels["ppm"]:
                self._ion_labels["ppm"] = self._ion_labels["tol"]

    def index_to_name(self, index: int, effective: bool = True, mz: bool = False):
        """Convert index to ion name."""
        if self._ion_labels is None:
            self._setup_labels()
        return self._ion_labels["ppm"][index]

    @property
    def labels(self) -> tuple[np.ndarray, ty.Optional[np.ndarray], ty.Optional[np.ndarray]]:
        """Return all available labels."""
        return self.xs, None, None

    def lazy_iter(self):
        """Lazily yield ion images."""
        with self.lazy_peaks() as peaks:
            for i, mz in enumerate(self.xs):
                yield mz, peaks[:, i]

    def get_ion(self, value: ty.Union[int, float]) -> np.ndarray:
        """Retrieve single ion."""
        if isinstance(value, float):
            value = find_nearest_index(self.xs, value)
        with self.lazy_peaks() as peaks:
            return peaks[:, value]

    def get_ions(self, mzs: np.ndarray):
        """Retrieve multiple ions."""
        _, indices = self.get_ion_indices(mzs)
        with self.lazy_peaks() as peaks:
            return peaks[:, indices]

    def update(self, array: np.ndarray, framelist: np.ndarray, chunk_id: ty.Optional[int] = None):
        """Update array."""
        if self.is_chunked:
            # TODO: check size of the array against the size of the chunk
            with self.open() as h5:
                h5[f"{self.PEAKS_KEY}/{chunk_id!s}"][:] = array
                h5.flush()
        else:
            with self.open() as h5:
                h5[self.PEAKS_ARRAY_KEY][framelist] = array
                h5.flush()


class LazyPeaksProxy:
    """Proxy class to enable similar interface to lazy-peaks."""

    _peaks = None

    def __init__(self, obj: H5CentroidsStore, low_mem: bool = True):
        self.obj = obj
        self.low_mem = low_mem

    def __getitem__(self, item):
        res = []
        if self.low_mem:
            for array in self.chunk_iter():
                try:
                    res.append(array[item[0], item[1]])
                except TypeError:
                    res.append(np.column_stack([array[item[0], i] for i in item[1]]))
        else:
            try:
                res.append(self.peaks()[item[0], item[1]])
            except TypeError:
                res.append(np.column_stack([self.peaks()[item[0], i] for i in item[1]]))
        return np.concatenate(res)

    @property
    def shape(self) -> tuple[int, int]:
        """Return shape."""
        chunk_info = self.obj.chunk_info
        n_px = sum(len(indices) for indices in chunk_info.values())
        return n_px, self.obj.n_peaks

    @property
    def dtype(self):
        """Return dtype."""
        with self.obj.open("r") as h5:
            return h5[f"{self.obj.PEAKS_KEY}/0"].dtype

    def chunk_iter(self):
        """Iterator of smaller arrays."""
        chunk_info = self.obj.chunk_info
        with self.obj.open("r") as h5:
            for chunk_id in chunk_info:
                yield h5[f"{self.obj.PEAKS_KEY}/{chunk_id!s}"]

    def peaks(self):
        """Dense version which is very inefficient."""
        if self._peaks is None:
            temp = np.zeros(self.shape, dtype=self.dtype)
            start = 0
            for chunk in self.chunk_iter():
                temp[start : start + chunk.shape[0], :] = chunk[:]
                start += chunk.shape[0]
            self._peaks = temp
        return self._peaks
