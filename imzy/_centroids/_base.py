"""Base class for all Centroids stores."""
import typing as ty

import numpy as np

from ..utilities import find_nearest_index_array


class BaseCentroids:
    """Base class."""

    xs = None
    peaks = None

    def __getitem__(self, item):
        """Get ion image."""
        # label or index provided
        if isinstance(item, (int, str)):
            return self.get_ion_image(item)
        # iterable of label or indices provided
        elif isinstance(item, ty.Iterable):
            return self.get_ion_images(item)
        # slice provided - will call self again
        elif isinstance(item, slice):
            if item.step is None:
                item = list(range(item.start, item.stop))
            else:
                item = list(range(item.start, item.stop, item.step))
            return self[item]
        raise KeyError(f"Did not know how to interpret {item} of type {type(item)}")

    def _reshape_single(self, array, fill_value=np.nan):
        """Reshape single image."""
        raise NotImplementedError("Must implement method")

    def _reshape_multiple(self, array, fill_value=np.nan):
        """Reshape multiple images."""
        raise NotImplementedError("Must implement method")

    def get_ion(self, name: ty.Union[int, float]) -> np.ndarray:
        """Get ion array"""
        raise NotImplementedError("Must implement method")

    def get_ions(self, indices: np.ndarray):
        """Retrieve ions."""
        raise NotImplementedError("Must implement method")

    @property
    def n_peaks(self) -> int:
        return len(self.xs)

    def get_ion_indices(self, xs: np.ndarray) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Find ions"""
        # we can calculate this once and then cache it for the future
        indices = find_nearest_index_array(self.xs, xs)
        return self.xs[indices], indices

    def get_ion_image(self, name: ty.Union[int, float], fill_value=np.nan):
        """Get ion image that has been reshaped."""
        data = self.get_ion(name)
        return self._reshape_single(data, fill_value)

    def get_ion_images(self, indices: ty.Iterable[int], fill_value=np.nan):
        """Get ion image that has been reshaped."""
        data = self.get_ions(np.asarray(indices))
        return self._reshape_multiple(data, fill_value)
