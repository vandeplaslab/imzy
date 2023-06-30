"""Wrapper class for HDF5-based centroids."""
import typing as ty
from contextlib import contextmanager

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

from koyo.typing import PathLike
from koyo.utilities import find_nearest_index

from imzy._centroids._base import BaseCentroids


class H5CentroidsStore(BaseCentroids):
    """HDF5-centroids."""

    # Private attributes
    PEAKS_KEY = "Array"
    PEAKS_ARRAY_KEY = "Array/array"

    # Cache attributes
    _chunk_info = None
    _xs = None
    _proxy = None
    _is_chunked = None
    _low_mem: bool = True

    def __init__(
        self,
        path: PathLike,
        xyz_coordinates: ty.Optional[np.ndarray] = None,
        pixel_index: ty.Optional[np.ndarray] = None,
        image_shape: ty.Optional[ty.Tuple[int, int]] = None,
        mode: str = "a",
    ):
        super().__init__(xyz_coordinates, pixel_index, image_shape)
        assert h5py is not None, "h5py is not installed."

        self.path = path
        self.mode = mode

    @property
    def is_low_mem(self):
        return self._low_mem

    @is_low_mem.setter
    def is_low_mem(self, value: bool):
        self._low_mem = value
        if self._proxy:
            self._proxy.low_mem = value

    @property
    def is_chunked(self) -> bool:
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
    def chunk_info(self) -> ty.Optional[ty.Dict[int, np.ndarray]]:
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

    @contextmanager
    def open(self, mode: ty.Optional[str] = None):
        """Safely open storage."""
        if mode is None:
            mode = self.mode
        try:
            f_ptr = h5py.File(self.path, mode=mode, rdcc_nbytes=1024 * 1024 * 4)
        except FileExistsError as err:
            raise err
        try:
            yield f_ptr
        finally:
            f_ptr.close()

    def flush(self):
        """Flush data to disk."""
        with self.open() as h5:
            h5.flush()

    @staticmethod
    def _get_group(hdf, group_name: str, flush: bool = True):
        try:
            group = hdf[group_name]
        except KeyError:
            group = hdf.create_group(group_name)
            if flush:
                hdf.flush()
        return group

    @staticmethod
    def _add_data_to_group(
        group_obj,
        dataset_name,
        data,
        dtype,
        chunks=None,
        maxshape=None,
        compression=None,
        compression_opts=None,
        shape=None,
    ):
        """Add data to group."""
        replaced_dataset = False

        if dtype is None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
        if shape is None:
            if hasattr(data, "shape"):
                shape = data.shape

        if dataset_name in list(group_obj.keys()):
            if group_obj[dataset_name].dtype == dtype:
                try:
                    group_obj[dataset_name][:] = data
                    replaced_dataset = True
                except TypeError:
                    del group_obj[dataset_name]
            else:
                del group_obj[dataset_name]

        if not replaced_dataset:
            group_obj.create_dataset(
                dataset_name,
                data=data,
                dtype=dtype,
                compression=compression,
                chunks=chunks,
                maxshape=maxshape,
                compression_opts=compression_opts,
                shape=shape,
            )


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
    def shape(self) -> ty.Tuple[int, int]:
        """Return shape."""
        chunk_info = self.obj.chunk_info
        n_px = chunk_info[max(chunk_info.keys())].max()
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
