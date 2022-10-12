"""Utility functions."""
import typing as ty

import numpy as np


def optimize_chunks_along_axis(
    axis: int,
    *,
    array: ty.Optional[np.ndarray] = None,
    shape: ty.Optional[ty.Tuple[int, ...]] = None,
    dtype=None,
    max_size: int = 1e6,
) -> ty.Optional[ty.Tuple[int, ...]]:
    """Optimize chunk size along specified axis"""
    if array is not None:
        dtype, shape = array.dtype, array.shape
    elif shape is None or dtype is None:
        raise ValueError("You must specify either an array or `shape` and `dtype`")
    assert len(shape) == 2, "Only supporting 2d arrays at the moment."
    assert axis <= 1, "Only supporting 2d arrays at the moment, use -1, 0 or 1 in the `axis` argument"
    assert hasattr(dtype, "itemsize"), "Data type must have the attribute 'itemsize'"
    item_size = np.dtype(dtype).itemsize

    if max_size == 0:
        return None

    n = 0
    max_n = shape[1] if axis == 0 else shape[0]
    while (n * item_size * shape[axis]) <= max_size and n < max_n:
        n += 1
    if n < 1:
        n = 1
    return (shape[0], n) if axis == 0 else (n, shape[1])
