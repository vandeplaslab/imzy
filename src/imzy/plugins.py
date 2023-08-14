"""Plugin manager."""
import typing as ty

from koyo.system import IS_MAC
from pluggy import PluginManager

from imzy import hookspec
from imzy._readers.imzml import _imzml

if not IS_MAC:
    from imzy._readers.bruker import _tdf, _tsf
else:
    _tdf = _tsf = None

if ty.TYPE_CHECKING:
    from imzy._readers import BaseReader


class ImzyPluginManager(PluginManager):
    """Plugin manager."""

    def __init__(self):
        super().__init__("imzy")
        self.add_hookspecs(hookspec)
        # register own plugins
        self.register(_imzml)
        if not IS_MAC:
            self.register(_tdf)
            self.register(_tsf)
        # add entry hooks
        self.load_setuptools_entrypoints("imzy.plugins")

    def get_reader(self, path, **kwargs) -> "BaseReader":
        """Get reader for specified path."""
        for reader in self.hook.imzy_reader(path=path, **kwargs):
            if reader is not None:
                return reader
        raise NotImplementedError(f"Reader for dataset with specified path has not been implemented yet. (path={path})")
