"""Zarr peaks store."""
import typing as ty
from pathlib import Path

import dask.array as dsa
import numpy as np
import zarr

from ..types import PathLike
from ..utilities import find_nearest_index_single
from ._base import BaseCentroids


class ZarrCentroidsStore(BaseCentroids):
    """Convenience wrapper for Zarr-directory."""

    def __init__(
        self,
        path: PathLike,
        xyz_coordinates: ty.Optional[np.ndarray] = None,
        pixel_index: ty.Optional[np.ndarray] = None,
        image_shape: ty.Optional[ty.Tuple[int, int]] = None,
    ):
        super(ZarrCentroidsStore, self).__init__(xyz_coordinates, pixel_index, image_shape)
        self.path = Path(path)
        self.z_store = zarr.open(str(path))
        try:
            self.xs = dsa.from_zarr(self.z_store["xs"]).compute()  # load array into memory
        except KeyError:
            self.xs = dsa.from_zarr(self.z_store["mzs"]).compute()  # load array into memory
        self.peaks = dsa.from_zarr(self.z_store["array"])
        self.filtered = self.z_store.get("filtered")
        if self.filtered:
            self.filtered = dsa.from_zarr(self.filtered)

    def __repr__(self):
        return f"{self.__class__.__name__}<filename={self.path.name}; no. images={len(self.xs)}>"

    def get_ion(self, value: ty.Union[int, float]) -> np.ndarray:
        """Get ion array"""
        if isinstance(value, float):
            value = find_nearest_index_single(self.xs, value)
        return self.peaks[:, value].compute()

    def get_ions(self, indices: np.ndarray, filtered: bool = False):
        """Retrieve ions."""
        if filtered:
            if not self.filtered:
                raise ValueError("Please cache filtered images.")
            return self.filtered[:, indices].compute()
        return self.peaks[:, indices].compute()
