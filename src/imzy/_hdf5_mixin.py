"""Mixin class for HDF5 files."""


def check_hdf5() -> None:
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import h5py
        import hdf5plugin
    except ImportError:
        raise ImportError("Please install `h5py` and `hdf5plugins` to continue. You can do `pip install imzy[hdf5]")
