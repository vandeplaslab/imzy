# type: ignore[attr-defined]
"""imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats."""

import sys

from ._centroids import H5CentroidsStore, InMemoryStore, ZarrCentroidsStore  # noqa: F401
from ._readers import IMZMLReader, get_reader  # noqa: F401

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    """Get version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
__all__ = ("IMZMLReader", "get_reader", "H5CentroidsStore", "ZarrCentroidsStore", "InMemoryStore")
