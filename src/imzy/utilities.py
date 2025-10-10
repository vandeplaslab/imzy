"""Utility functions."""

from __future__ import annotations

from pathlib import Path

from koyo.typing import PathLike


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
        else:
            path = path / "analysis.tsf"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # make sqlite connection
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    # cursor.execute("SELECT RegionNumber FROM MaldiFrameInfo ORDER BY RegionNumber DESC LIMIT 1")
    cursor.execute("SELECT RegionNumber FROM MaldiFrameInfo ORDER BY ROWID DESC LIMIT 1")
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
