"""In-memory store."""
import typing as ty

import numpy as np

from imzy._centroids._base import BaseCentroids

if ty.TYPE_CHECKING:
    from imzy._readers import BaseReader


class InMemoryStore(BaseCentroids):
    """Class that implements in-memory store."""

    def __init__(
        self,
        reader: "BaseReader",
        xyz_coordinates: ty.Optional[np.ndarray] = None,
        pixel_index: ty.Optional[np.ndarray] = None,
        image_shape: ty.Optional[ty.Tuple[int, int]] = None,
    ):
        super().__init__(xyz_coordinates, pixel_index, image_shape)
        self.reader = reader

    def get_ion(self, mz: float, fill_value=np.nan, ppm: float = 3.0) -> np.ndarray:
        """Get ion."""
        return self.reader.get_ion_image(mz, ppm=ppm, fill_value=fill_value)

    def get_ions(self, mzs: ty.Iterable[float], fill_value=np.nan, ppm: float = 3.0):
        """Get ion images."""
        return self.reader.get_ion_images(mzs, ppm=ppm, fill_value=fill_value)
