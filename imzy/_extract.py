"""Methods to handle data extraction from any reader."""
import typing as ty
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ._readers import get_reader
from .types import PathLike
from .utilities import accumulate_peaks_centroid, accumulate_peaks_profile, chunks, find_between_batch


def check_zarr():
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import dask  # noqa
        import rechunker  # noqa
        import zarr  # noqa
    except ImportError:
        raise ImportError(
            "Please install `zarr`, `dask` and `rechunker` to continue. You can do `pip install imzy[zarr]"
        )


def create_centroids_zarr(
    input_dir: PathLike,
    zarr_path: PathLike,
    n_peaks: int,
    mzs: np.ndarray = None,
    mzs_min: np.ndarray = None,
    mzs_max: np.ndarray = None,
    tol: float = 0,
    ppm: float = 0,
    ys: np.ndarray = None,
):
    """Create group with datasets inside."""
    import zarr

    reader = get_reader(input_dir)

    store = zarr.DirectoryStore(str(zarr_path))
    group = zarr.group(store=store)
    # add metadata
    group.attrs.update({"ppm": ppm, "tol": tol})

    n_pixels = 250
    if n_pixels > reader.n_pixels:
        n_pixels = reader.n_pixels

    array = group.require_dataset(
        "array_temp",
        shape=(int(reader.n_pixels), int(n_peaks)),
        chunks=(int(n_pixels), int(n_peaks)),
        dtype=np.float32,
    )
    if mzs is not None:
        group.create_dataset("mzs", data=mzs, overwrite=True)
    if mzs_min is not None:
        group.create_dataset("mzs_min", data=mzs_min, overwrite=True)
    if mzs_max is not None:
        group.create_dataset("mzs_max", data=mzs_max, overwrite=True)
    if ys is not None:
        group.create_dataset("ys", data=ys, overwrite=True)
    return array


def extract_centroids_zarr(
    input_dir: PathLike,
    zarr_path: PathLike,
    indices: np.ndarray,
    mzs_min: np.ndarray,
    mzs_max: np.ndarray,
    silent: bool = False,
    sync_path: str = None,
):
    """Extract peaks for particular subset of frames."""
    import zarr

    reader = get_reader(input_dir)
    synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path is not None else None
    ds = zarr.open(str(zarr_path), mode="a", synchronizer=synchronizer)
    chunk_size = ds.chunks[0]

    # profile-mode data is easier to handle so we can create mask once and then use the same mask for every pixel
    extract_indices = None
    if not reader.is_centroid:
        x, _ = reader.get_spectrum(0)
        extract_indices = find_between_batch(x, mzs_min, mzs_max)

    chunked_indices = list(chunks(indices, chunk_size))
    for indices in tqdm(chunked_indices, disable=silent, desc="Extracting image chunks..."):
        # to reduce the number of writes to disk, we accumulate data using temporary array
        temp = np.zeros((len(indices), len(mzs_min)), dtype=np.float32)
        for i, index in enumerate(indices):
            x, y = reader[index]
            if reader.is_centroid:
                temp[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)
            else:
                temp[i] = accumulate_peaks_profile(extract_indices, y)
        ds[indices[0] : indices[-1] + 1] = temp


def rechunk_zarr_array(
    input_dir: PathLike,
    zarr_path: PathLike,
    target_path: PathLike,
    chunk_size: ty.Optional[ty.Tuple[int, int]] = None,
    silent: bool = False,
):
    """Re-chunk zarr array to more optional format.

    The re-chunking will basically rotate the array to allow much quicker retrieval of ion images at the cost of
    reducing performance of retrieving spectra.
    """
    import zarr
    from dask.diagnostics import ProgressBar
    from rechunker import rechunk

    mobj = get_reader(input_dir)
    ds = zarr.open(str(zarr_path), mode="r")
    temp_path = str(Path(zarr_path) / "intermediate")

    if chunk_size is None:
        n_im = 25  # each chynk will have 25 images
        if n_im >= ds.shape[1]:
            n_im = ds.shape[1]
        chunk_size = (int(mobj.n_pixels), n_im)

    # create re-chunking plan and execute it immediately.
    rechunk_plan = rechunk(
        ds,
        target_chunks=chunk_size,
        target_store=target_path,
        temp_store=temp_path,
        max_mem="512MB",
    )
    if silent:
        rechunk_plan.execute()
    else:
        with ProgressBar():
            rechunk_plan.execute()

    # clean-up old array
    _safe_rmtree(temp_path)  # remove the intermediate array
    _safe_rmtree(zarr_path)  # remove the temporary array


def check_hdf5():
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import h5py  # noqa
        import hdf5plugins  # noqa
    except ImportError:
        raise ImportError("Please install `h5py` and `hdf5plugins` to continue. You can do `pip install imzy[hdf5]")


def create_centroids_hdf5(
    input_dir: PathLike,
    zarr_path: PathLike,
    n_peaks: int,
    mzs: np.ndarray = None,
    mzs_min: np.ndarray = None,
    mzs_max: np.ndarray = None,
    tol: float = 0,
    ppm: float = 0,
    ys: np.ndarray = None,
):
    """Create group with datasets inside."""
    import h5py
    import hdf5plugins  # ensures LZ4 compression is available

    reader = get_reader(input_dir)

    # store = zarr.DirectoryStore(str(zarr_path))
    # group = zarr.group(store=store)
    # # add metadata
    # group.attrs.update({"ppm": ppm, "tol": tol})
    #
    # n_pixels = 250
    # if n_pixels > reader.n_pixels:
    #     n_pixels = reader.n_pixels
    #
    # array = group.require_dataset(
    #     "array_temp",
    #     shape=(int(reader.n_pixels), int(n_peaks)),
    #     chunks=(int(n_pixels), int(n_peaks)),
    #     dtype=np.float32,
    # )
    # if mzs is not None:
    #     group.create_dataset("mzs", data=mzs, overwrite=True)
    # if mzs_min is not None:
    #     group.create_dataset("mzs_min", data=mzs_min, overwrite=True)
    # if mzs_max is not None:
    #     group.create_dataset("mzs_max", data=mzs_max, overwrite=True)
    # if ys is not None:
    #     group.create_dataset("ys", data=ys, overwrite=True)
    # return array


def extract_centroids_hdf5(
    input_dir: PathLike,
    zarr_path: PathLike,
    indices: np.ndarray,
    mzs_min: np.ndarray,
    mzs_max: np.ndarray,
    silent: bool = False,
    sync_path: str = None,
):
    """Extract peaks for particular subset of frames."""
    # import zarr
    #
    # reader = get_reader(input_dir)
    # synchronizer = zarr.ProcessSynchronizer(sync_path) if sync_path is not None else None
    # ds = zarr.open(str(zarr_path), mode="a", synchronizer=synchronizer)
    # chunk_size = ds.chunks[0]
    #
    # # profile-mode data is easier to handle so we can create mask once and then use the same mask for every pixel
    # extract_indices = None
    # if not reader.is_centroid:
    #     x, _ = reader.get_spectrum(0)
    #     extract_indices = find_between_batch(x, mzs_min, mzs_max)
    #
    # chunked_indices = list(chunks(indices, chunk_size))
    # for indices in tqdm(chunked_indices, disable=silent, desc="Extracting image chunks..."):
    #     # to reduce the number of writes to disk, we accumulate data using temporary array
    #     temp = np.zeros((len(indices), len(mzs_min)), dtype=np.float32)
    #     for i, index in enumerate(indices):
    #         x, y = reader[index]
    #         if reader.is_centroid:
    #             temp[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)
    #         else:
    #             temp[i] = accumulate_peaks_profile(extract_indices, y)
    #     ds[indices[0] : indices[-1] + 1] = temp


def _safe_rmtree(path):
    from shutil import rmtree

    try:
        rmtree(path)
    except (OSError, FileNotFoundError):
        pass
