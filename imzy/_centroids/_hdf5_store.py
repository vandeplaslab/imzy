"""Wrapper class for HDF5-based centroids."""
import typing as ty
import numpy as np
from contextlib import contextmanager
import h5py

from ..types import PathLike
from ._base import BaseCentroids


class H5CentroidsStore(BaseCentroids):
    """HDF5-centroids."""

    PEAKS_KEY = "Array"

    def __init__(self, path: PathLike, mode: str = "a"):
        self.path = path
        self.mode = mode

    @property
    def is_chunked(self) -> bool:
        return True

    @property
    def chunk_info(self) -> ty.Optional[ty.Dict[int, np.ndarray]]:
        return

    @contextmanager
    def open(self, mode: str = None):
        """Safely open storage"""
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


class LazyPeaksProxy:
    """Proxy class to enable similar interface to lazy-peaks."""

    def __init__(self, obj: H5CentroidsStore):
        self.obj = obj

    def __getitem__(self, item):
        res = []
        for array in self.chunk_iter():
            try:
                res.append(array[item[0], item[1]])
            except TypeError:
                res.append(np.column_stack([array[item[0], i] for i in item[1]]))
        return np.concatenate(res)

    @property
    def shape(self) -> ty.Tuple[int, int]:
        """Return shape."""
        chunk_info = self.obj.chunk_info
        n_px = chunk_info[max(chunk_info.keys())].max()  # noqa
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
                yield h5[f"{self.obj.PEAKS_KEY}/{str(chunk_id)}"]
