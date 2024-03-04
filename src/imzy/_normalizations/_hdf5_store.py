"""HDF5 store for normalizations."""
import numpy as np

from imzy._hdf5_mixin import HDF5Mixin


class H5NormalizationStore(HDF5Mixin):
    """HDF5 store for normalizations."""

    NORMALIZATIONS_KEY: str = "Normalizations"

    def __init__(self, path: str, mode: str = "a"):
        self._init_hdf5(path, mode)

    def get_normalization(self) -> np.ndarray:
        """Get normalization data."""

    def get_normalization_list(self) -> list[str]:
        """Get list of available normalizations."""
        with self.open("r") as h5:
            normalizations = list(h5[self.NORMALIZATIONS_KEY].keys())
        return normalizations
