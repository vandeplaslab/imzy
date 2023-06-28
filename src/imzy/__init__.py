# type: ignore[attr-defined]
"""imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats."""


from importlib import metadata as importlib_metadata

from imzy._centroids import H5CentroidsStore, InMemoryStore, ZarrCentroidsStore
from imzy._readers import IMZMLReader, TDFReader, TSFReader, get_reader


def get_version() -> str:
    """Get version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
__all__ = (
    "IMZMLReader",
    "TDFReader",
    "TSFReader",
    "get_reader",
    "H5CentroidsStore",
    "ZarrCentroidsStore",
    "InMemoryStore",
)
