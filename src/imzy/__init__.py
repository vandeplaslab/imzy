"""imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats."""

from __future__ import annotations

import importlib
import typing as ty
from functools import lru_cache
from importlib import metadata as importlib_metadata

from loguru import logger

# disable loguru logger
logger.disable("imzy")

# Global instance of plugin manager
_plugin_manager = None


def get_version() -> str:
    """Get version."""
    try:
        return importlib_metadata.version(__name__)
    except (importlib_metadata.PackageNotFoundError, KeyError):  # pragma: no cover
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
    "EmptySpectrumWarning",
    "H5CentroidsStore",
    "H5NormalizationStore",
    "IMZMLReader",
    "IMZMLWriter",
    "IMZMLWriterWarning",
    "InMemoryStore",
    "NeoFlexReader",
    "SkippedSpectrumWarning",
    "TDFReader",
    "TSFReader",
    "WatersReader",
    "ZarrCentroidsStore",
    "get_normalizations",
    "get_reader",
    "write_imzml",
)
# Map exported names -> (module, attribute)
_LAZY_IMPORTS = {
    # Centroids
    "H5CentroidsStore": ("imzy._centroids", "H5CentroidsStore"),
    "InMemoryStore": ("imzy._centroids", "InMemoryStore"),
    "ZarrCentroidsStore": ("imzy._centroids", "ZarrCentroidsStore"),
    # Readers
    "BaseReader": ("imzy._readers", "BaseReader"),
    "IMZMLReader": ("imzy._readers", "IMZMLReader"),
    "NeoFlexReader": ("imzy._readers", "NeoFlexReader"),
    "TDFReader": ("imzy._readers", "TDFReader"),
    "TSFReader": ("imzy._readers", "TSFReader"),
    "WatersReader": ("imzy._readers", "WatersReader"),
    "get_reader": ("imzy._readers", "get_reader"),
    # Normalizations
    "get_normalizations": ("imzy._normalizations", "get_normalizations"),
    "H5NormalizationStore": ("imzy._normalizations", "H5NormalizationStore"),
    # Writers
    "EmptySpectrumWarning": ("imzy._writers", "EmptySpectrumWarning"),
    "IMZMLWriter": ("imzy._writers", "IMZMLWriter"),
    "IMZMLWriterWarning": ("imzy._writers", "IMZMLWriterWarning"),
    "SkippedSpectrumWarning": ("imzy._writers", "SkippedSpectrumWarning"),
    "write_imzml": ("imzy._writers", "write_imzml"),
}


def __getattr__(name: str) -> ty.Any:
    """Lazily import objects when accessed."""
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)

    # Cache so future access is fast
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))


# --- typing support for IDEs ---
if ty.TYPE_CHECKING:
    from imzy._centroids import H5CentroidsStore, InMemoryStore, ZarrCentroidsStore
    from imzy._normalizations import H5NormalizationStore, get_normalizations
    from imzy._readers import (
        IMZMLReader,
        NeoFlexReader,
        TDFReader,
        TSFReader,
        WatersReader,
        get_reader,
    )
    from imzy._writers import (
        EmptySpectrumWarning,
        IMZMLWriter,
        IMZMLWriterWarning,
        SkippedSpectrumWarning,
        write_imzml,
    )
