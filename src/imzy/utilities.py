"""Utility functions."""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger


def get_rois(path: PathLike) -> list[int]:
    """Get ROIs from file."""
    path = Path(path)
    if path.suffix == ".imzML":
        return [0]
    elif path.suffix == ".d":
        return get_rois_from_bruker_d(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def get_rois_from_bruker_d(path: PathLike) -> list[int]:
    """Get ROIs from Bruker .d file."""
    import sqlite3

    path = Path(path)
    if path.suffix == ".d":
        if (path / "analysis.tdf").exists():
            path = path / "analysis.tdf"
        elif (path / "analysis.tsf").exists():
            path = path / "analysis.tsf"
        else:
            path = path / "peaks.sqlite"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # make sqlite connection
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    # cursor.execute("SELECT RegionNumber FROM MaldiFrameInfo ORDER BY RegionNumber DESC LIMIT 1")
    try:
        cursor.execute("SELECT RegionNumber FROM MaldiFrameInfo ORDER BY ROWID DESC LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("SELECT RegionNumber FROM Spectra ORDER BY ROWID DESC LIMIT 1")

    last_roi = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return list(range(0, last_roi + 1))


def _safe_rmtree(path: PathLike) -> None:
    from contextlib import suppress
    from shutil import rmtree

    with suppress(FileNotFoundError, OSError, PermissionError):
        rmtree(path)


def _auto_guess_ppm(resolution: int, mz_ppm: float | str) -> float:
    """Get m/z ppm spacing."""
    if mz_ppm == "auto":
        if resolution >= 200_000:
            mz_ppm = 1.0
        elif resolution >= 120_000:
            mz_ppm = 2.5
        elif resolution >= 60_000:
            mz_ppm = 3.5
        elif resolution >= 50_000:
            mz_ppm = 5.0
        elif resolution >= 30_000:
            mz_ppm = 7.0
        else:
            mz_ppm = 10.0
    return mz_ppm  # type: ignore[return-value]


def write_cache(path: PathLike, filename: str, data: dict) -> None:
    """Sometimes, reading data from raw data can be very slow, so we can cache it instead.

    Cache data is usually written inside the raw directory (e.g. inside Bruker .d or Waters .raw) or next to it
    (e.g. when dealing with imzML).

    Parameters
    ----------
    path : PathLike
        Path to the directory containing the .icache folder
    filename : str
        name of the cache file without the .npz suffix
    data : dict
        dictionary containing cache data
    """
    cache_dir_path = Path(path) / ".icache"
    cache_dir_path.mkdir(exist_ok=True)
    tmp_filename = cache_dir_path / (filename + ".tmp.npz")
    filename = cache_dir_path / (filename + ".npz")
    np.savez(tmp_filename, **data)
    try:
        tmp_filename.rename(filename)
    except OSError:
        with suppress(FileNotFoundError):
            filename.unlink()
        tmp_filename.rename(filename)
    except PermissionError:
        logger.warning("Could not write cache file.")
    finally:
        with suppress(FileNotFoundError):
            tmp_filename.unlink()


def read_cache(path: PathLike, filename: str, keys: list[str]) -> dict[str, np.ndarray | None]:
    """Load cache metadata.

    Parameters
    ----------
    path : PathLike
        Path to the directory containing the .icache folder
    filename : str
        Name of the cache file without the .npz suffix
    keys : list
        Keys to be read when cache file is loaded
    """
    cache_file_path = Path(path) / ".icache" / (filename + ".npz")

    data = {}.fromkeys(keys)
    if cache_file_path.exists():
        with np.load(cache_file_path, mmap_mode="r") as f_ptr:
            for key in keys:
                try:
                    data[key] = f_ptr[key]
                except KeyError:
                    data[key] = None
    return data
