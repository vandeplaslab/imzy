"""Utility functions."""
import typing as ty
from pathlib import Path

from koyo.typing import PathLike


def get_rois(path: PathLike) -> ty.List[int]:
    """Get ROIs from file."""
    path = Path(path)
    if path.suffix == ".imzML":
        return [0]
    elif path.suffix == ".d":
        return get_rois_from_bruker_d(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def get_rois_from_bruker_d(path: PathLike) -> ty.List[int]:
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
