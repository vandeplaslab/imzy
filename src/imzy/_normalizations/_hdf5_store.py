"""HDF5 store for normalizations."""

import typing as ty

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from yoki5.base import Store


class H5NormalizationStore(Store):
    """HDF5 store for normalizations."""

    NORMALIZATION_KEY: str = "Normalization"

    def __init__(self, path: PathLike, mode: str = "a"):
        super().__init__(path=path, mode=mode)

    def __getitem__(self, item: str) -> np.ndarray:
        """Retrieve item."""
        return self.get_normalization(item)

    def __call__(self, array: np.ndarray, *args, **kwargs):
        """Apply normalization."""
        return self.normalize(array, *args, **kwargs)

    def get_normalization_list(self) -> list[str]:
        """Get list of available normalizations."""
        with self.open("r") as h5:
            normalizations = list(h5[self.NORMALIZATION_KEY].keys())
        return normalizations

    def get_normalization(self, name: str, as_median: bool = True, as_multiplier: bool = True) -> np.ndarray:
        """Get normalization array."""
        if not name.startswith(f"{self.NORMALIZATION_KEY}/"):
            name = f"{self.NORMALIZATION_KEY}/{name}"
        try:
            normalization = self.get_array(name, "normalization")
        except ValueError:
            _, name = name.split("/")
            normalization = self.get_array(self.NORMALIZATION_KEY, name)
        except KeyError:
            raise KeyError(f"Normalization '{name}' not found in store.")
        return normalization

    def normalize(self, array: np.ndarray, name: str, **kwargs) -> np.ndarray:
        """Normalize array."""
        array = np.asanyarray(array)
        if array.ndim < 2:  # or is_single_2d(self.mobj, array):
            return self._single_normalize(array, name, **kwargs)
        else:
            return self._batch_normalize(array, name, **kwargs)

    def _single_normalize(self, array: np.ndarray, norm: str, **kwargs: ty.Any) -> np.ndarray:
        """Apply normalization to an array.

        Parameters
        ----------
        array : np.ndarray
            array to be reshaped
        norm : str
            name of the normalization to be applied to the array

        Returns
        -------
        array : np.ndarray
            normalize array. If 'reshape' is True it will be automatically reshaped to a 2D representation
        """
        if array.ndim > 2:
            raise ValueError("Array should be 2D or 1D")
        try:
            norm_array = self.get_normalization(norm, as_multiplier=True)
            array = np.multiply(array, norm_array)
        except ValueError:
            logger.warning(f"Normalization '{norm}' not found. Skipping normalization.")
        return _postprocess(array)

    def _batch_normalize(
        self,
        array: np.ndarray,
        norm: str,
        **kwargs: ty.Any,
    ):
        if array.ndim < 2:
            raise ValueError("Expected two-dimensional array of 'N pixels * M peaks'.")

        norm_array = self.get_normalization(
            norm,
            as_multiplier=True,
        )
        if array.ndim == 2:
            if norm_array.shape[0] != array.shape[0]:
                raise ValueError(
                    f"The input array does not have the same number of pixels as the normalization vector."
                    f" (norm={norm_array.shape}; array={array.shape})"
                )
        array = np.multiply(array.T, norm_array).T
        return _postprocess(array)


def _postprocess(array: np.ndarray):
    # array[np.isnan(array)] = 0
    array[np.isinf(array)] = 0
    return array
