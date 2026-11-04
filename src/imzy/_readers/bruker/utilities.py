"""Various utility functions for Bruker readers."""

from __future__ import annotations

import sqlite3
import typing as ty
from contextlib import contextmanager
from pathlib import Path

import numba
import numpy as np
from koyo.typing import PathLike


@numba.njit(cache=True, nogil=True)
def scan_buffer_to_csc(
    buffer: np.ndarray, n_scans: int, mz_min_idx: int, mz_max_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a `tims_read_scans_v2` buffer straight into CSC arrays of shape (mz_max_idx - mz_min_idx, n_scans).

    The buffer layout is: number of peaks of each scan, followed by the TOF indices and then the intensities of each
    scan in turn (all uint32). Because the buffer is scan-major, the CSC matrix (columns = mobility scans) can be
    filled directly in two passes (count, then fill) without any intermediate COO matrix.

    Parameters
    ----------
    buffer : np.ndarray[np.uint32]
        scan buffer of a single frame
    n_scans : int
        number of scans contained in the buffer
    mz_min_idx : int
        first TOF index (inclusive) to keep; it becomes row 0 of the output
    mz_max_idx : int
        last TOF index (exclusive) to keep

    Returns
    -------
    data : np.ndarray[np.int32]
        intensities
    indices : np.ndarray[np.int32]
        row indices, relative to `mz_min_idx`
    indptr : np.ndarray[np.int32]
        column pointers (length `n_scans + 1`)
    """
    indptr = np.zeros(n_scans + 1, dtype=np.int32)
    pos = n_scans
    total = 0
    for scan in range(n_scans):
        n_peaks = np.int64(buffer[scan])
        if pos + 2 * n_peaks > buffer.size:
            raise ValueError("Scan buffer is shorter than the peak counts it declares.")
        for i in range(pos, pos + n_peaks):
            if mz_min_idx <= buffer[i] < mz_max_idx:
                total += 1
        pos += 2 * n_peaks
        indptr[scan + 1] = total

    data = np.empty(total, dtype=np.int32)
    indices = np.empty(total, dtype=np.int32)
    pos = n_scans
    k = 0
    for scan in range(n_scans):
        n_peaks = np.int64(buffer[scan])
        for i in range(n_peaks):
            tof = np.int64(buffer[pos + i])
            if mz_min_idx <= tof < mz_max_idx:
                indices[k] = tof - mz_min_idx
                data[k] = buffer[pos + n_peaks + i]
                k += 1
        pos += 2 * n_peaks
    return data, indices, indptr


@numba.njit(cache=True, nogil=True)
def peaks_to_csc(
    scan_offsets: np.ndarray,
    tof_indices: np.ndarray,
    intensities: np.ndarray,
    n_scans: int,
    mz_min_idx: int,
    mz_max_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as `scan_buffer_to_csc`, but for frames already decoded into peak arrays (e.g. by `pytsf`).

    The peaks of scan `s` are `tof_indices[scan_offsets[s]:scan_offsets[s + 1]]`. Scans beyond `n_scans` are ignored.
    """
    n_avail = min(n_scans, scan_offsets.size - 1)
    indptr = np.zeros(n_scans + 1, dtype=np.int32)
    total = 0
    for scan in range(n_avail):
        for i in range(scan_offsets[scan], scan_offsets[scan + 1]):
            if mz_min_idx <= tof_indices[i] < mz_max_idx:
                total += 1
        indptr[scan + 1] = total
    indptr[n_avail + 1 :] = total

    data = np.empty(total, dtype=np.int32)
    indices = np.empty(total, dtype=np.int32)
    k = 0
    for scan in range(n_avail):
        for i in range(scan_offsets[scan], scan_offsets[scan + 1]):
            tof = np.int64(tof_indices[i])
            if mz_min_idx <= tof < mz_max_idx:
                indices[k] = tof - mz_min_idx
                data[k] = intensities[i]
                k += 1
    return data, indices, indptr


@numba.njit(cache=True, nogil=True)
def scan_buffer_to_profile(buffer: np.ndarray, n_scans: int, n_mz_bins: int) -> np.ndarray:
    """Sum all scans of a `tims_read_scans_v2` buffer into a profile spectrum of length `n_mz_bins`."""
    out = np.zeros(n_mz_bins, dtype=np.int64)
    pos = n_scans
    for scan in range(n_scans):
        n_peaks = np.int64(buffer[scan])
        if pos + 2 * n_peaks > buffer.size:
            raise ValueError("Scan buffer is shorter than the peak counts it declares.")
        for i in range(n_peaks):
            tof = np.int64(buffer[pos + i])
            if tof < n_mz_bins:
                out[tof] += buffer[pos + n_peaks + i]
        pos += 2 * n_peaks
    return out


@numba.njit(cache=True, nogil=True)
def peaks_to_profile(tof_indices: np.ndarray, intensities: np.ndarray, n_mz_bins: int) -> np.ndarray:
    """Sum decoded peaks (all scans) into a profile spectrum of length `n_mz_bins`."""
    out = np.zeros(n_mz_bins, dtype=np.int64)
    for i in range(tof_indices.size):
        tof = np.int64(tof_indices[i])
        if tof < n_mz_bins:
            out[tof] += intensities[i]
    return out


def get_sparse_data_from_buffer(
    buffer: np.ndarray, mz_min_idx: int, mz_max_idx: int, scan_begin: int, scan_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get COO (data, rows, cols) arrays from a `tims_read_scans_v2` buffer containing scans `scan_begin:scan_end`.

    Kept for backwards compatibility - prefer `scan_buffer_to_csc`, which avoids the COO step.

    Returns
    -------
    data : np.ndarray
        intensities
    rows : np.ndarray
        absolute TOF indices
    cols : np.ndarray
        absolute scan numbers
    """
    data, indices, indptr = scan_buffer_to_csc(buffer, scan_end - scan_begin, mz_min_idx, mz_max_idx)
    rows = indices.astype(np.int64) + mz_min_idx
    cols = np.repeat(np.arange(scan_begin, scan_end), np.diff(indptr))
    return data, rows, cols


def guess_sql_filename(path: Path) -> str:
    """Guess the SQL filename based on the path."""
    if path.suffix == ".d":
        sql_filename = "analysis.tsf" if (path / "analysis.tsf").exists() else "analysis.tdf"
    elif path.suffix in [".tsf", ".tdf"]:
        sql_filename = path.name
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return sql_filename


@contextmanager
def sql_reader(path: Path, sql_filename: str | None = None) -> ty.Generator[sqlite3.Connection, None, None]:
    """SQL reader context manager."""
    if sql_filename is None:
        sql_filename = guess_sql_filename(path)
    conn = sqlite3.connect(path / sql_filename)
    try:
        yield conn
    finally:
        conn.close()


def get_pixel_size(path: Path, sql_filename: str | None = None, cache_dir_name: str | None = None) -> float:
    """Get the pixel size from the Bruker file."""
    from ims_utils.coordinates import get_spacing_consecutive

    from imzy.utilities import read_cache, write_cache

    data = read_cache(path, "pixel_size_cache", ["pixel_size", "motor_positions"], cache_dir_name=cache_dir_name)
    if data["pixel_size"] is not None:
        return float(data["pixel_size"])
    else:
        motor_positions = None
        if data["motor_positions"] is not None:
            motor_positions = data["motor_positions"]
        else:
            with sql_reader(path, sql_filename) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT MotorPositionX, MotorPositionY FROM MaldiFrameInfo")
                motor_positions = np.array(cursor.fetchall())

        if motor_positions is None:
            raise ValueError("Failed to extract motor positions from the file!")

        x_px, y_px = get_spacing_consecutive(motor_positions[:, 0], motor_positions[:, 1])
        if np.isnan(x_px) and np.isnan(y_px):
            raise ValueError(f"Failed to calculate pixel size: both x_px and y_px are NaN")
        if not np.isclose(x_px, y_px) and not (np.isnan(x_px) or np.isnan(y_px)):
            raise ValueError(f"Pixel size is not equal in both dimensions: {x_px} vs {y_px}")
        if np.isnan(x_px):
            x_px = y_px
        if np.isnan(y_px):
            y_px = x_px
        pixel_size = float(x_px)
        write_cache(
            path,
            "pixel_size_cache",
            data={"pixel_size": pixel_size, "motor_positions": motor_positions},
            cache_dir_name=cache_dir_name,
        )
        return pixel_size


def get_coordinates_and_roi(path: Path, sql_filename: str | None = None, cache_dir_name: str | None = None):
    """Get the coordinates and ROI from the Bruker file."""
    from imzy.utilities import read_cache, write_cache

    data = read_cache(path, "frame_index_cache", ["frame_index_position"], cache_dir_name=cache_dir_name)
    # load data from cache
    if data["frame_index_position"] is not None:
        frame_index_position = data["frame_index_position"]
    else:
        with sql_reader(path, sql_filename) as conn:
            try:
                cursor = conn.execute("SELECT Frame, XIndexPos, YIndexPos, RegionNumber FROM MaldiFrameInfo")
            except sqlite3.OperationalError:
                # non-imaging (e.g. LC) acquisition: treat the frames as a single line in one region
                cursor = conn.execute("SELECT Id, Id - 1, 0, 0 FROM Frames ORDER BY Id")
            frame_index_position = np.array(cursor.fetchall()).reshape(-1, 4)
        write_cache(
            path,
            "frame_index_cache",
            data={"frame_index_position": frame_index_position},
            cache_dir_name=cache_dir_name,
        )
    return frame_index_position


def get_tic(path: Path, sql_filename: str | None = None, cache_dir_name: str | None = None) -> np.ndarray:
    """Get the total ion current (TIC) from the Bruker file."""
    from imzy.utilities import read_cache, write_cache

    data = read_cache(path, "tic_cache", ["tic", "region_frames"], cache_dir_name=cache_dir_name)
    if data["tic"] is not None and not isinstance(data["region_frames"], str):
        tic = np.ravel(data["tic"])
    else:
        with sql_reader(path, sql_filename) as conn:
            try:
                cursor = conn.execute("SELECT SummedIntensities, RegionNumber FROM Spectra")
                tic_data = np.array(cursor.fetchall())
                region_frames = tic_data[:, 1]
                tic = tic_data[:, 0]
            except sqlite3.OperationalError:
                # this is required to handle TSF files (and maybe TDF?)
                cursor = conn.execute("SELECT RegionNumber FROM MaldiFrameInfo")
                region_frames = np.array(cursor.fetchall())
                cursor = conn.execute("SELECT SummedIntensities FROM Frames")
                tic = np.array(cursor.fetchall())
            tic = np.ravel(tic)
        write_cache(path, "tic_cache", data={"tic": tic, "region_frames": region_frames}, cache_dir_name=cache_dir_name)
    return tic


def is_tsf(path: PathLike) -> bool:
    """Check if path is Bruker .d/tsf acquired on a timsTOF (non-neofleX) instrument."""
    path = Path(path)
    return (
        path.suffix.lower() == ".d"
        and path.is_dir()
        and (path / "analysis.tsf").exists()
        and (path / "analysis.tsf_bin").exists()
        and _is_timstof_instrument(path)
    )


def is_tdf(path: PathLike) -> bool:
    """Check if path is Bruker .d/tdf."""
    path = Path(path)
    return (
        path.suffix.lower() == ".d"
        and path.is_dir()
        and (path / "analysis.tdf").exists()
        and (path / "analysis.tdf_bin").exists()
    )


def get_instrument_name(path: Path, sql_filename: str = "analysis.tsf") -> str:
    """Return the `InstrumentName` stored in the dataset metadata, or an empty string if it is missing."""
    with sql_reader(path, sql_filename) as conn:
        row = conn.execute("SELECT Value FROM GlobalMetadata WHERE Key='InstrumentName'").fetchone()
    return str(row[0]) if row and row[0] is not None else ""


def _is_timstof_instrument(path: Path) -> bool:
    """Check if the instrument is not neofleX."""
    return "neoflex" not in get_instrument_name(path).lower()
