"""Types."""
import typing as ty
from pathlib import Path

import numpy as np

PathLike = ty.Union[str, Path]
ArrayLike = ty.TypeVar("ArrayLike", ty.List, np.ndarray, ty.Iterable)
