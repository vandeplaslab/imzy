"""Methods to handle data extraction from any reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.spectrum import find_between_batch
from koyo.typing import PathLike
from koyo.utilities import chunks
from tqdm import tqdm

try:
    import hdf5plugin
except ImportError:
    pass

from imzy._readers import get_reader
from imzy.utilities import accumulate_peaks_centroid, accumulate_peaks_profile


def check_zarr() -> None:
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import dask
        import rechunker
        import zarr
    except ImportError:
        raise ImportError(
            "Please install `zarr`, `dask` and `rechunker` to continue. You can do `pip install imzy[zarr]"
        )


def create_centroids_zarr(
    input_dir: PathLike,
    zarr_path: PathLike,
    n_peaks: int,
    mzs: ty.Optional[np.ndarray] = None,
    mzs_min: ty.Optional[np.ndarray] = None,
    mzs_max: ty.Optional[np.ndarray] = None,
    tol: ty.Optional[float] = None,
    ppm: ty.Optional[float] = None,
    ys: ty.Optional[np.ndarray] = None,
):
    """Create group with datasets inside."""
    import zarr

    if tol is None and ppm is None:
        raise ValueError("Either `tol` or `ppm` should be specified.")

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
    sync_path: ty.Optional[str] = None,
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


def check_hdf5() -> None:
    """Check whether Zarr, dask and rechunker are installed."""
    try:
        import h5py
        import hdf5plugin
    except ImportError:
        raise ImportError("Please install `h5py` and `hdf5plugins` to continue. You can do `pip install imzy[hdf5]")


def get_chunk_info(n_pixels: int, n_peaks: int, max_mem: float = 512) -> ty.Dict[int, np.ndarray]:
    """Get chunk size information for particular dataset."""
    import math

    _max_mem = (float(n_pixels) * n_peaks * 4) / (1024**2)  # assume 4 bytes per element
    n_tasks = math.ceil(_max_mem / max_mem) or 1
    return dict(enumerate(list(chunks(np.arange(n_pixels), n_tasks=n_tasks))))


def create_centroids_hdf5(
    input_dir: PathLike,
    hdf_path: PathLike,
    n_peaks: int,
    mzs: ty.Optional[np.ndarray] = None,
    mzs_min: ty.Optional[np.ndarray] = None,
    mzs_max: ty.Optional[np.ndarray] = None,
    tol: ty.Optional[float] = None,
    ppm: ty.Optional[float] = None,
    ys: ty.Optional[np.ndarray] = None,
    chunk_info: ty.Optional[ty.Dict[int, np.ndarray]] = None,
) -> Path:
    """Create group with datasets inside."""
    from imzy._centroids import H5CentroidsStore
    from imzy.utilities import optimize_chunks_along_axis

    if tol is None and ppm is None:
        raise ValueError("Either `tol` or `ppm` should be specified.")

    reader = get_reader(input_dir)
    n_pixels = reader.n_pixels
    array_shape = (n_pixels, n_peaks)

    compression = hdf5plugin.LZ4()

    if Path(hdf_path).suffix != ".h5":
        hdf_path = Path(hdf_path).with_suffix(".h5")

    store = H5CentroidsStore(hdf_path, mode="a")
    with store.open() as h5:
        group = store._get_group(h5, store.PEAKS_KEY)
        # add attributes
        group.attrs["n_peaks"] = parse_to_attribute(n_peaks)
        group.attrs["tol"] = parse_to_attribute(tol)
        group.attrs["ppm"] = parse_to_attribute(ppm)
        group.attrs["is_chunked"] = chunk_info is not None
        # add data
        if mzs is not None:
            store._add_data_to_group(group, "mzs", mzs, maxshape=(None,), dtype=mzs.dtype)
        if mzs_min is not None:
            store._add_data_to_group(group, "mzs_min", mzs_min, maxshape=(None,), dtype=mzs_min.dtype)
        if mzs_max is not None:
            store._add_data_to_group(group, "mzs_max", mzs_max, maxshape=(None,), dtype=mzs_max.dtype)
        if ys is not None:
            store._add_data_to_group(group, "ys", ys, maxshape=(None,), dtype=ys.dtype)
        if chunk_info is None:
            store._add_data_to_group(
                group,
                "array",
                None,
                chunks=optimize_chunks_along_axis(0, shape=array_shape, dtype=np.float32),
                maxshape=(n_peaks, None),  # to enable resizing
                shape=array_shape,
                dtype=np.float32,
                **compression,
            )
        else:
            for chunk_id, framelist in chunk_info.items():
                array_shape = (len(framelist), array_shape[1])
                store._add_data_to_group(
                    group,
                    str(chunk_id),
                    None,
                    chunks=optimize_chunks_along_axis(0, shape=array_shape, dtype=np.float32),
                    maxshape=(len(framelist), None),  # to enable resizing
                    shape=array_shape,
                    dtype=np.float32,
                    **compression,
                )
    return Path(hdf_path)


def extract_centroids_hdf5(
    input_dir: PathLike,
    hdf_path: PathLike,
    indices: np.ndarray,
    mzs_min: np.ndarray,
    mzs_max: np.ndarray,
    silent: bool = False,
):
    """Extract peaks for particular subset of frames."""
    from imzy._centroids import H5CentroidsStore

    reader = get_reader(input_dir)
    store = H5CentroidsStore(hdf_path, mode="a")

    # profile-mode data is easier to handle so we can create mask once and then use the same mask for every pixel
    extract_indices = None
    if not reader.is_centroid:
        x, _ = reader.get_spectrum(0)
        extract_indices = find_between_batch(x, mzs_min, mzs_max)

    if store.is_chunked:
        chunked_indices = list(store.chunk_info.values())
    else:
        chunked_indices = [indices]

    n_peaks = len(mzs_min)
    n_chunks = len(chunked_indices)
    for chunk_id, indices in enumerate(chunked_indices):
        # to reduce the number of writes to disk, we accumulate data using temporary array
        temp = np.zeros((len(indices), len(mzs_min)), dtype=np.float32)
        for i, index in enumerate(
            tqdm(
                indices,
                disable=silent,
                desc=f"Extracting {n_peaks} peaks (chunk={chunk_id+1}/{n_chunks})",
                miniters=25,
                mininterval=0.2,
            )
        ):
            x, y = reader[index]
            if reader.is_centroid:
                temp[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)
            else:
                temp[i] = accumulate_peaks_profile(extract_indices, y)
        store.update(temp, indices, chunk_id)


def _safe_rmtree(path):
    from shutil import rmtree

    try:
        rmtree(path)
    except (OSError, FileNotFoundError):
        pass


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
