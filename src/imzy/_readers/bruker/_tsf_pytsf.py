"""TSF reader backed by `pytsf`, used on macOS where Bruker ships no timsdata library."""

from __future__ import annotations

import typing as ty
import warnings
from contextlib import contextmanager

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from pytsf import TsfData, TsfError

from imzy._readers.bruker._mixin import BrukerBaseReader
from imzy._readers.bruker.utilities import is_tsf
from imzy.hookspec import hook_impl


@contextmanager
def forward_pytsf_warnings() -> ty.Generator[None, None, None]:
    """Re-emit warnings raised by `pytsf` through the logger instead of the warnings module."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    for warning in caught:
        logger.warning(f"pytsf: {warning.message}")


def log_ignored_recalibration(use_recalibrated_state: bool) -> None:
    """Log that the pytsf backend always uses the raw acquisition calibration."""
    if use_recalibrated_state:
        logger.info("pytsf backend ignores 'use_recalibrated_state' - raw acquisition calibration is used.")


class TSFReader(BrukerBaseReader):
    """Bruker .d/tsf reader using the pure-Python `pytsf` backend."""

    sql_filename = "analysis.tsf"
    _tsf: TsfData | None = None

    def __init__(
        self, path: PathLike, use_recalibrated_state: bool = False, auto_profile: bool = True, roi: int | None = None
    ):
        self.use_recalibrated_state = use_recalibrated_state
        self._is_centroid: bool | None = None
        super().__init__(path, auto_profile=auto_profile, roi=roi)

    def _init(self) -> None:
        super()._init()
        log_ignored_recalibration(self.use_recalibrated_state)
        self._tsf = TsfData(self.path)
        # full-range m/z axes extrapolate past the calibrants by design, same as the vendor library
        self._tsf.calibration.warn_outside_range = False

        # data attributes
        self.n_mz_bins = int(np.round(self.mz_to_index(1, [self.mz_max]))[0])

    def close(self) -> None:
        """Close file."""
        if self._tsf is not None:
            self._tsf.close()
            self._tsf = None

    def has_recalibrated_state(self) -> bool:
        """Recalibrated states are not supported by `pytsf`."""
        return False

    @property
    def is_centroid(self) -> bool:
        """Determine whether the data is centroided."""
        if self._is_centroid is None:
            self._is_centroid = not self._tsf.meta.has_profile_spectra
        return self._is_centroid

    def index_to_mz(self, frame_id: int, indices: ty.Iterable[int]) -> np.ndarray:
        """Convert m/z index to m/z."""
        return self._tsf.index_to_mz(frame_id, np.asarray(indices, dtype=np.float64))

    def mz_to_index(self, frame_id: int, mzs: ty.Iterable[float]) -> np.ndarray:
        """Convert m/z to index."""
        return self._tsf.mz_to_index(frame_id, np.asarray(mzs, dtype=np.float64))

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.is_centroid:
            return self.read_centroid_spectrum(index)
        return self.mz_x, self.read_profile_spectrum(index)

    def read_centroid_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read centroid spectrum (m/zs, intensities) for a pixel index."""
        return self.read_centroid_frame(self._frame_id_for_index(index))

    def read_centroid_spectrum_with_width(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read centroid spectrum (m/zs, intensities, widths) for a pixel index."""
        return self.read_centroid_frame_with_width(self._frame_id_for_index(index))

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Read profile spectrum (intensities) for a pixel index."""
        return self.read_profile_frame(self._frame_id_for_index(index))

    def read_centroid_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Read centroid spectrum (m/zs, intensities) for a raw Bruker frame id."""
        # frame ids must be python ints - sqlite3 binds numpy integers as blobs
        frame_id = int(frame_id)
        try:
            indices, intensities = self._tsf.read_line_spectrum(frame_id)
        except TsfError as exc:
            raise RuntimeError(str(exc)) from exc
        mzs = self.index_to_mz(frame_id, indices)
        return mzs.astype(np.float64, copy=False), intensities.astype(np.float32, copy=False)

    def read_centroid_frame_with_width(self, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read centroid spectrum (m/zs, intensities, widths) for a raw Bruker frame id."""
        frame_id = int(frame_id)
        try:
            indices, intensities, widths = self._tsf.read_line_spectrum_with_width(frame_id)
        except TsfError as exc:
            raise RuntimeError(str(exc)) from exc
        mzs = self.index_to_mz(frame_id, indices)
        return (
            mzs.astype(np.float64, copy=False),
            intensities.astype(np.float32, copy=False),
            widths.astype(np.float32, copy=False),
        )

    def read_profile_frame(self, frame_id: int) -> np.ndarray:
        """Read profile spectrum (intensities) for a raw Bruker frame id."""
        try:
            return self._tsf.read_profile_spectrum(int(frame_id)).astype(np.uint32, copy=False)
        except TsfError as exc:
            # vendor library raises RuntimeError when there are no profile spectra
            raise RuntimeError(str(exc)) from exc


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> TSFReader | None:
    """Return TSFReader if path is Bruker .d/tsf."""
    if is_tsf(path):
        return TSFReader(path, **kwargs)
    return None
