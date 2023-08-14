"""imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats."""
from functools import lru_cache
from importlib import metadata as importlib_metadata

from imzy._centroids import H5CentroidsStore, InMemoryStore, ZarrCentroidsStore
from imzy._readers import BaseReader, IMZMLReader, TDFReader, TSFReader, get_reader

# Global instance of plugin manager
_plugin_manager = None


def get_version() -> str:
    """Get version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


@lru_cache
def discover_plugins():
    """Initialize plugins."""
    global _plugin_manager

    if _plugin_manager is None:
        from imzy.plugins import ImzyPluginManager

        _plugin_manager = ImzyPluginManager()
    return _plugin_manager


__version__: str = get_version()
__all__ = (
    "BaseReader",
    "IMZMLReader",
    "TDFReader",
    "TSFReader",
    "get_reader",
    "H5CentroidsStore",
    "ZarrCentroidsStore",
    "InMemoryStore",
)
