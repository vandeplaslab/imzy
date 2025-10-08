"""Mixin class for HDF5 files."""

from __future__ import annotations


def check_hdf5() -> None:
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import h5py
        import hdf5plugin
        import yoki5
    except ImportError:
        raise ImportError("Please install `yoki5` to continue. You can do `pip install imzy[hdf5]")
