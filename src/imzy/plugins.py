"""Plugin manager."""

from __future__ import annotations

import typing as ty

from koyo.system import IS_MAC, IS_WIN
from koyo.typing import PathLike
from pluggy import PluginManager

from imzy import hookspec
from imzy._readers.imzml import _imzml

if not IS_MAC:
    from imzy._readers.bruker import _neoflex, _tdf, _tsf
else:
    _tdf = _tsf = _neoflex = None  # type: ignore[assignment]
if IS_WIN:
    from imzy._readers.waters import _raw
else:
    _raw = None  # type: ignore[assignment]


if ty.TYPE_CHECKING:
    from imzy._readers import BaseReader


class ImzyPluginManager(PluginManager):
    """Plugin manager."""

    def __init__(self):
        super().__init__("imzy")
        self.add_hookspecs(hookspec)
        # register own plugins
        self.register(_imzml)
        if _tdf is not None:
            self.register(_tdf)
        if _tsf is not None:
            self.register(_tsf)
        if _neoflex is not None:
            self.register(_neoflex)
        if _raw is not None:
            self.register(_raw)
        # add entry hooks
        self.load_setuptools_entrypoints("imzy.plugins")

    def get_reader(self, path: PathLike, **kwargs: ty.Any) -> BaseReader:
        """Get reader for the specified path."""
        for reader in self.hook.imzy_reader(path=path, **kwargs):
            if reader is not None:
                return reader
        raise NotImplementedError(f"Reader for dataset with specified path has not been implemented yet. (path={path})")
