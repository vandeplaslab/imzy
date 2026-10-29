"""Various utility functions for Bruker readers."""

from __future__ import annotations

import sqlite3
import typing as ty
from contextlib import contextmanager
from pathlib import Path

import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def get_sparse_data_from_buffer(
    buffer: np.ndarray, mz_min_idx: int, mz_max_idx: int, scan_begin: int, scan_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the instantiation values for COO array by iterating over the data extracted from the Bruker frame.

    Parameters
    ----------
    buffer : np.ndarray[np.uint32]
        array containing data extracted for particular frame
    mz_min_idx : int
        m/z start index of what data should be included in the COO array
    mz_max_idx : int
        m/z end index of what data should be included in the COO array
    scan_begin : int
        start mobility bin
    scan_end : int
        end mobility bin

    Returns
    -------
    data : np.ndarray
        standard list containing `data` information to be used by the COO sparse matrix
    rows : np.ndarray
        standard list containing `rows` information to be used by the COO sparse matrix
    cols : np.ndarray
        standard list containing `cols` information to be used by the COO sparse matrix
    """
    data = []
    rows = []
    cols = []

    d = scan_end - scan_begin
    for col in range(scan_begin, scan_end):
        n_peaks = buffer[col - scan_begin]
        indices = buffer[d : d + n_peaks]
        d += n_peaks
        intensities = buffer[d : d + n_peaks]
        d += n_peaks
        for i in range(indices.shape[0]):
            mz_val = indices[i]
            mz_int = intensities[i]
            if mz_min_idx <= mz_val < mz_max_idx:
                data.append(mz_int)
                rows.append(mz_val)
                cols.append(col)
    return np.asarray(data), np.asarray(rows), np.asarray(cols)


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
    yield conn
    conn.close()


def get_pixel_size(path: Path, sql_filename: str | None = None, cache_dir_name: str | None = None) -> float:
    """Get the pixel size from the Bruker file."""
    from ims_utils.coordinates import get_spacing_consecutive

    from imzy.utilities import read_cache, write_cache

    data = read_cache(path, "pixel_size_cache", ["pixel_size", "motor_positions"], cache_dir_name=cache_dir_name)
    if data["pixel_size"] is not None:
        return data["pixel_size"]
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
        if not np.isclose(x_px, y_px):
            raise ValueError(f"Pixel size is not equal in both dimensions: {x_px} vs {y_px}")
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
            cursor = conn.cursor()
            cursor.execute("SELECT Frame, XIndexPos, YIndexPos, RegionNumber FROM MaldiFrameInfo")
            frame_index_position = np.array(cursor.fetchall())
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
