"""Readers."""

import typing as ty
from pathlib import Path

from koyo.system import IS_MAC
from koyo.typing import PathLike

from imzy._readers._base import BaseReader
from imzy._readers.imzml import IMZMLReader

if not IS_MAC:
    from imzy._readers.bruker import NeoFlexReader, TDFReader, TSFReader, is_neoflex, is_tdf, is_tsf
else:
    TDFReader = TSFReader = NeoFlexReader = is_tdf = is_tsf = is_neoflex = None  # type: ignore[misc,assignment]


__all__ = ("BaseReader", "IMZMLReader", "NeoFlexReader", "TDFReader", "TSFReader", "get_reader")


def get_reader(path: PathLike, **kwargs: ty.Any) -> BaseReader:
    """Get reader based on its file extension.

    Parameters
    ----------
    path : PathLike
        Path to the dataset.
    kwargs : dict
        Dictionary of extra keyword-arguments that should be passed to the reader. For definition of what arguments are
        supported, please see individual readers.
    """
    import imzy

    path = Path(path)
    pm = imzy.discover_plugins()
    return pm.get_reader(path, **kwargs)
