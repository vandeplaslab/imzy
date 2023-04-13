"""Readers."""
import typing as ty
from pathlib import Path

from koyo.typing import PathLike

from imzy._readers.imzml import IMZMLReader

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
    raise NotImplementedError("Reader for dataset with specified path has not been implemented yet.")
