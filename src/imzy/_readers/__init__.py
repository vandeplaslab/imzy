"""Readers."""

import typing as ty
from pathlib import Path

from koyo.system import IS_WIN
from koyo.typing import PathLike

from imzy._readers._base import BaseReader
from imzy._readers.bruker import (
    BAFReader,
    NeoFlexReader,
    RapiflexReader,
    TDFReader,
    TSFReader,
    is_neoflex,
    is_tdf,
    is_tsf,
)
from imzy._readers.imzml import IMZMLReader
from imzy._readers.scils import SCiLSReader

if IS_WIN:
    from imzy._readers.waters import WatersReader, is_waters
else:
    WatersReader = is_waters = None  # type: ignore[misc,assignment]


__all__ = (
    "BAFReader",
    "BaseReader",
    "IMZMLReader",
    "NeoFlexReader",
    "RapiflexReader",
    "SCiLSReader",
    "TDFReader",
    "TSFReader",
    "WatersReader",
    "get_reader",
)


def get_reader(path: PathLike, **kwargs: ty.Any) -> BaseReader:
    """Get a file reader based on its extension and file contents.

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
