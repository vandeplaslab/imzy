"""Mixin class for HDF5 files."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager

try:
    import h5py
except ImportError:
    h5py = None
from koyo.typing import PathLike


def parse_from_attribute(attribute):
    """Parse attribute from cache."""
    if isinstance(attribute, str) and attribute == "__NONE__":
        attribute = None
    return attribute


def parse_to_attribute(attribute):
    """Parse attribute to cache."""
    if attribute is None:
        attribute = "__NONE__"
    return attribute


def check_hdf5() -> None:
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import h5py
        import hdf5plugin
    except ImportError:
        raise ImportError("Please install `h5py` and `hdf5plugins` to continue. You can do `pip install imzy[hdf5]")


class HDF5Mixin:
    """Mixin class for HDF5 files."""

    path: PathLike
    mode: str

    def _init_hdf5(self, path: PathLike, mode: str = "a"):
        assert h5py is not None, "h5py is not installed."

        self.path = path
        self.mode = mode

    @contextmanager
    def open(self, mode: ty.Optional[str] = None) -> h5py.File:
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

    def get_attr(self, dataset_name, attr: str, default=None):
        """Safely retrieve 1 attribute."""
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            value = parse_from_attribute(group.attrs.get(attr, default))
            return value

    def get_array(self, dataset_name: str, key: str) -> np.ndarray:
        """Safely retrieve 1 array."""
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            return group[key][:]
