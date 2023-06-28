"""Readers."""
import typing as ty
from pathlib import Path

from koyo.system import IS_MAC
from koyo.typing import PathLike

from imzy._readers.imzml import IMZMLReader

if not IS_MAC:
    from imzy._readers.bruker import TDFReader, TSFReader
else:
    TDFReader = TSFReader = None


if ty.TYPE_CHECKING:
    from imzy._readers._base import BaseReader

__all__ = ("IMZMLReader", "get_reader")


def get_reader(path: PathLike, **kwargs) -> "BaseReader":
    """Get reader based on it's file extension.

    Parameters
    ----------
    path : PathLike
        Path to the dataset.
    kwargs : dict
        Dictionary of extra keyword-arguments that should be passed to the reader. For definition of what arguments are
        supported, please see individual readers.
    """
    path = Path(path)
    if path.suffix.lower() == ".imzml":
        return IMZMLReader(path, **kwargs)
    elif path.suffix.lower() == ".d" and (path / "analysis.tdf").exists() and not IS_MAC:
        return TDFReader(path, **kwargs)
    elif path.suffix.lower() == ".d" and (path / "analysis.tsf").exists() and not IS_MAC:
        return TSFReader(path, **kwargs)
    raise NotImplementedError("Reader for dataset with specified path has not been implemented yet.")
