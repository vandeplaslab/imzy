"""Typing for the package."""

import typing as ty

import numpy as np


class SpatialInfo(ty.TypedDict):
    """Spatial data."""

    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    image_shape: tuple[int, int]
    pixel_size: float
