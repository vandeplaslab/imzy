"""Centroids."""
try:
    from ._hdf5_store import H5CentroidsStore
except ImportError:
    H5CentroidsStore = None
try:
    from ._zarr_store import ZarrCentroidsStore
except ZarrCentroidsStore:
    ZarrCentroidsStore = None

from ._memory_store import InMemoryStore  # noqa
