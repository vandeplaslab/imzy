"""Zarr peaks store."""
import typing as ty
from pathlib import Path

import dask.array as dsa
import numpy as np
import zarr

from ..types import PathLike
from ..utilities import (
    find_nearest_index_single,
    reshape_array,
    reshape_array_batch,
    reshape_array_batch_from_coordinates,
    reshape_array_from_coordinates,
)
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
        self.xyz_coordinates = np.asarray(xyz_coordinates) if xyz_coordinates is not None else None
        self.pixel_index = pixel_index
        self.image_shape = image_shape
        self._reshape_by_coordinates = self.xyz_coordinates is not None
        # update image shape
        if self.xyz_coordinates is not None and self.image_shape is None:
            self.image_shape = np.asarray(
                (
                    self.xyz_coordinates[:, 1].max(),  # y-dim first
                    self.xyz_coordinates[:, 0].max(),  # x-dim second
                )
            )

    def __repr__(self):
        return f"{self.__class__.__name__}<filename={self.path.name}; no. images={len(self.xs)}>"

    def _reshape_single(self, array, fill_value=np.nan):
        """Reshape single image."""
        if self._reshape_by_coordinates:
            return reshape_array_from_coordinates(array, self.image_shape, self.xyz_coordinates, fill_value=fill_value)
        return reshape_array(array, self.image_shape, self.pixel_index, fill_value=fill_value)

    def _reshape_multiple(self, array, fill_value=np.nan):
        if self._reshape_by_coordinates:
            return reshape_array_batch_from_coordinates(
                array, self.image_shape, self.xyz_coordinates, fill_value=fill_value
            )
        return reshape_array_batch(array, self.image_shape, self.pixel_index, fill_value=fill_value)

    def get_ion(self, name: ty.Union[int, float]) -> np.ndarray:
        """Get ion array"""
        if isinstance(name, str):
            raise ValueError("Cannot parse string.")
        if isinstance(name, float):
            name = find_nearest_index_single(self.xs, name)
        return self.peaks[:, name].compute()

    def get_ions(self, indices: np.ndarray, filtered: bool = False):
        """Retrieve ions."""
        if filtered:
            if not self.filtered:
                raise ValueError("Please cache filtered images.")
            return self.filtered[:, indices].compute()
        return self.peaks[:, indices].compute()
