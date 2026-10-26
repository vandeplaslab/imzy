"""Zarr peaks store."""

from pathlib import Path

import numpy as np

try:
    import dask.array as dsa
    import zarr
except ImportError:
    dsa, zarr = None, None

from koyo.typing import PathLike
from koyo.utilities import find_nearest_index_single

from imzy._centroids._base import BaseCentroids


class ZarrCentroidsStore(BaseCentroids):
    """Convenience wrapper for Zarr-directory."""

    def __init__(
        self,
        path: PathLike,
        xyz_coordinates: np.ndarray | None = None,
        pixel_index: np.ndarray | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> None:
        """Open a centroid group for reading."""
        super().__init__(xyz_coordinates, pixel_index, image_shape)
        assert dsa is not None and zarr is not None, "dask and zarr are not installed."

        self.path = Path(path)
        self.z_store = zarr.open_group(str(path), mode="a")
        try:
            self.xs = dsa.from_zarr(self.z_store["xs"]).compute()  # load array into memory
        except KeyError:
            self.xs = dsa.from_zarr(self.z_store["mzs"]).compute()  # load array into memory
        self.peaks = dsa.from_zarr(self.z_store["array"])
        self.filtered = self.z_store.get("filtered")
        if self.filtered is not None:
            self.filtered = dsa.from_zarr(self.filtered)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<filename={self.path.name}; no. images={len(self.xs)}>"

    def get_ion(self, value: int | float) -> np.ndarray:
        """Get ion array."""
        if isinstance(value, float):
            value = find_nearest_index_single(self.xs, value)
        return self.peaks[:, value].compute()

    def get_ions(self, mzs: np.ndarray, filtered: bool = False) -> np.ndarray:
        """Retrieve ions."""
        _, indices = self.get_ion_indices(mzs)
        if filtered:
            if self.filtered is None:
                raise ValueError("Please cache filtered images.")
            return self.filtered[:, indices].compute()
        return self.peaks[:, indices].compute()
