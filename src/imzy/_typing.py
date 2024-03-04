"""Typing for the package."""
import typing as ty

import numpy as np


class SpatialInfo(ty.TypedDict):
    """Spatial data."""

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    shape: ty.Tuple[int, int]
