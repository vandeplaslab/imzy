from __future__ import annotations

import importlib
import typing as ty
from math import prod
from pathlib import Path

import numpy as np
from ims_utils.spectrum import find_between_batch
from koyo.typing import PathLike
from koyo.utilities import chunks
from tqdm import tqdm
from yoki5.utilities import optimize_chunks_along_axis, parse_to_attribute

from imzy import get_reader
from imzy._typing import SpatialInfo
from imzy.processing import accumulate_peaks_centroid, accumulate_peaks_profile
from imzy.utilities import _safe_rmtree

if ty.TYPE_CHECKING:
    import zarr

try:
    import hdf5plugin
except ImportError:

    class hdf5plugin:
        """Dummy class."""

        def LZ4(*args, **kwargs):
            return {}


def check_zarr() -> None:
    """Check whether Zarr and Dask are installed."""
    try:
        importlib.import_module("dask.array")
        importlib.import_module("zarr")
    except ImportError:
        raise ImportError('Zarr support requires Python 3.11+ and `pip install "imzy[zarr]"`.') from None


def create_centroids_zarr(
    input_dir: PathLike,
    zarr_path: PathLike,
    n_peaks: int,
    mzs: np.ndarray | None = None,
    mzs_min: np.ndarray | None = None,
    mzs_max: np.ndarray | None = None,
    tol: float | None = None,
    ppm: float | None = None,
    ys: np.ndarray | None = None,
    reader_kwargs: dict[str, ty.Any] | None = None,
) -> zarr.Array:
    """Create a Zarr 3 group containing centroid metadata and a temporary array."""
    import zarr

    if tol is None and ppm is None:
        raise ValueError("Either `tol` or `ppm` should be specified.")

    reader = get_reader(input_dir, **(reader_kwargs or {}))
    group = zarr.open_group(str(zarr_path), mode="a", zarr_format=3)
    # add metadata
    group.attrs.update({"ppm": ppm, "tol": tol})

    n_pixels = 250
    if n_pixels > reader.n_pixels:
        n_pixels = reader.n_pixels

    array = group.require_array(
        "array_temp",
        shape=(int(reader.n_pixels), int(n_peaks)),
        chunks=(int(n_pixels), int(n_peaks)),
        dtype=np.float32,
    )
    if mzs is not None:
        group.create_array("mzs", data=mzs, overwrite=True)
    if mzs_min is not None:
        group.create_array("mzs_min", data=mzs_min, overwrite=True)
    if mzs_max is not None:
        group.create_array("mzs_max", data=mzs_max, overwrite=True)
    if ys is not None:
        group.create_array("ys", data=ys, overwrite=True)
    return array


def extract_centroids_zarr(
    input_dir: PathLike,
    zarr_path: PathLike,
    indices: np.ndarray,
    mzs_min: np.ndarray,
    mzs_max: np.ndarray,
    silent: bool = False,
    sync_path: str | None = None,
    reader_kwargs: dict[str, ty.Any] | None = None,
) -> None:
    """Extract centroid rows with a single writer; process synchronizers are unsupported."""
    import zarr

    if sync_path is not None:
        raise ValueError("Zarr 3 does not support process synchronizers.")
    reader = get_reader(input_dir, **(reader_kwargs or {}))
    ds = zarr.open_array(str(zarr_path), mode="r+")
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
    chunk_size: tuple[int, int] | None = None,
    silent: bool = False,
    reader_kwargs: dict[str, ty.Any] | None = None,
) -> None:
    """Rechunk a temporary Zarr array with Dask and a 512 MiB block planning limit.

    The re-chunking will basically rotate the array to allow much quicker retrieval of ion images at the cost of
    reducing performance of retrieving spectra.
    """
    import dask.array as dsa
    import zarr
    from dask.diagnostics import ProgressBar

    ds = zarr.open_array(str(zarr_path), mode="r")

    if chunk_size is None:
        chunk_size = (int(ds.shape[0]), min(25, int(ds.shape[1])))

    max_bytes = 512 * 1024**2
    if prod(chunk_size) * ds.dtype.itemsize > max_bytes:
        raise ValueError("Target chunks must not exceed 512 MiB.")
    target = zarr.create_array(
        store=str(target_path),
        shape=ds.shape,
        chunks=chunk_size,
        dtype=ds.dtype,
        zarr_format=3,
        overwrite=True,
    )
    source = dsa.from_zarr(ds).rechunk(chunk_size, block_size_limit=max_bytes)
    copy = dsa.store(source, target, compute=False, lock=False)
    if silent:
        copy.compute()
    else:
        with ProgressBar():
            copy.compute()

    _safe_rmtree(zarr_path)


def get_chunk_info(n_pixels: int, n_peaks: int, max_mem: float = 512) -> dict[int, np.ndarray]:
    """Get chunk size information for particular dataset."""
    import math

    _max_mem = (float(n_pixels) * n_peaks * 4) / (1024**2)  # assume 4 bytes per element
    n_tasks = math.ceil(_max_mem / max_mem) or 1
    chunk_info = dict(enumerate(list(chunks(np.arange(n_pixels), n_tasks=n_tasks))))
    assert sum(len(v) for v in chunk_info.values()) == n_pixels, "Chunking failed"
    return chunk_info


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
    chunk_info: ty.Optional[dict[int, np.ndarray]] = None,
    spatial_info: ty.Optional[SpatialInfo] = None,
    reader_kwargs: ty.Optional[dict[str, ty.Any]] = None,
) -> Path:
    """Create group with datasets inside."""
    from imzy._centroids import H5CentroidsStore

    if tol is None and ppm is None:
        raise ValueError("Either `tol` or `ppm` should be specified.")

    reader = get_reader(input_dir, **(reader_kwargs or {}))
    n_pixels = reader.n_pixels
    array_shape = (n_pixels, n_peaks)

    compression = hdf5plugin.LZ4()

    if Path(hdf_path).suffix != ".h5":
        hdf_path = Path(hdf_path).with_suffix(".h5")

    store = H5CentroidsStore(hdf_path, mode="a")
    with store.open() as h5:
        group = store._add_group(h5, store.PEAKS_KEY)
        # add attributes
        group.attrs["n_peaks"] = parse_to_attribute(n_peaks)
        group.attrs["tol"] = parse_to_attribute(tol)
        group.attrs["ppm"] = parse_to_attribute(ppm)
        group.attrs["is_chunked"] = chunk_info is not None
        # add data
        if mzs is not None:
            store._add_array_to_group(group, "mzs", mzs, maxshape=(None,), dtype=mzs.dtype)
        if mzs_min is not None:
            store._add_array_to_group(group, "mzs_min", mzs_min, maxshape=(None,), dtype=mzs_min.dtype)
        if mzs_max is not None:
            store._add_array_to_group(group, "mzs_max", mzs_max, maxshape=(None,), dtype=mzs_max.dtype)
        if ys is not None:
            store._add_array_to_group(group, "ys", ys, maxshape=(None,), dtype=ys.dtype)
        if chunk_info is None:
            store._add_array_to_group(
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
                store._add_array_to_group(
                    group,
                    str(chunk_id),
                    None,
                    chunks=optimize_chunks_along_axis(0, shape=array_shape, dtype=np.float32),
                    maxshape=(len(framelist), None),  # to enable resizing
                    shape=array_shape,
                    dtype=np.float32,
                    **compression,
                )
        group = store._add_group(h5, store.SPATIAL_KEY)
        if spatial_info is not None:
            group.attrs["pixel_size"] = spatial_info.pop("pixel_size", 1.0)
            for key, value in spatial_info.items():
                store._add_array_to_group(group, key, value, dtype=value.dtype)
    return Path(hdf_path)


def extract_centroids_hdf5(
    input_dir: PathLike,
    hdf_path: PathLike,
    indices: np.ndarray,
    mzs_min: np.ndarray,
    mzs_max: np.ndarray,
    silent: bool = False,
    reader_kwargs: ty.Optional[dict[str, ty.Any]] = None,
):
    """Extract peaks for particular subset of frames."""
    from imzy._centroids import H5CentroidsStore

    reader = get_reader(input_dir, **(reader_kwargs or {}))
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
                desc=f"Extracting {n_peaks} peaks (chunk={chunk_id + 1}/{n_chunks})",
                miniters=250,
                mininterval=2,
            )
        ):
            x, y = reader[index]
            if reader.is_centroid:
                temp[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)
            else:
                temp[i] = accumulate_peaks_profile(extract_indices, y)
        store.update(temp, indices, chunk_id)
