"""Reader for PHI SmartSoft-TOF `.raw` files (ToF-SIMS imaging, PHI nanoTOF instruments).

The file records individual ion arrivals rather than per-pixel spectra, so on initialization the reader bins every
event onto the detector time-channel grid and stores sparse per-pixel spectra (occupied channels and ion counts).
See `imzy._readers.phi._format` for the file layout. No vendor library is needed.
"""

from __future__ import annotations

import logging
import math
import typing as ty
from pathlib import Path

import numpy as np
from ims_utils.spectrum import get_mzs_for_tol
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy._readers.phi._format import (
    SOFH_MAGIC,
    BlockIndex,
    PhiHeader,
    PhiMassAxis,
    build_mass_axis,
    iter_events,
    parse_header,
    scan_blocks,
)
from imzy.hookspec import hook_impl

logger = logging.getLogger(__name__)


def is_phi(path: PathLike) -> bool:
    """Return True if the path is a PHI SmartSoft-TOF `.raw` file (a file, unlike Waters `.raw` directories)."""
    path = Path(path)
    if path.suffix.lower() != ".raw" or not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(len(SOFH_MAGIC)) == SOFH_MAGIC
    except OSError:
        return False


class PhiReader(BaseReader):
    """Reader for PHI SmartSoft-TOF `.raw` ToF-SIMS imaging data.

    Every pixel of the (square, possibly mosaic) raster is a pixel of the reader, in raster order, so pixels that
    recorded no ions return empty spectra. Spectra are centroid-like: the m/z of each occupied detector channel and
    the number of ions recorded in it, summed over all frames.

    Parameters
    ----------
    path : PathLike
        Path to the `.raw` file.
    auto_profile : bool
        Unused, kept for API compatibility with other readers.
    use_appended_calibration : bool
        Use a recalibration appended to the file after acquisition (default) instead of the acquisition header's
        `Mass/Time` and `MassOffset`. The appended values describe the stored flight times, so only disable this to
        reproduce a conversion done with the original calibration.
    pixel_size : float, optional
        Override the pixel size in micrometers. By default it is `Raster Size (um) / ImagePixels` from the header.
    """

    def __init__(
        self,
        path: PathLike,
        auto_profile: bool = True,
        use_appended_calibration: bool = True,
        pixel_size: float | None = None,
    ):
        super().__init__(path, auto_profile=auto_profile)
        self.use_appended_calibration = use_appended_calibration
        self._pixel_size_override = pixel_size
        self._init()

    # ------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------
    def _init(self) -> None:
        if not self.path.is_file():
            raise ValueError(f"PHI .raw must be a file, not a directory: '{self.path}'.")
        header = parse_header(self.path)
        index = scan_blocks(self.path, header)

        slope, offset = header.mass_slope, header.mass_offset
        self._calibration_source = "header"
        appended = index.appended_calibration() if self.use_appended_calibration else None
        if appended is not None:
            slope, offset = float(appended["Mass/Time"]), float(appended["MassOffset"])
            if not (slope > 0 and math.isfinite(slope)):
                raise ValueError(f"PHI appended recalibration declares a non-positive Mass/Time ({slope}).")
            self._calibration_source = "appended"
        self._calibration = {"Mass/Time": slope, "MassOffset": offset}

        self._header: PhiHeader = header
        self._index: BlockIndex = index
        self._axis: PhiMassAxis = build_mass_axis(
            slope=slope,
            offset=offset,
            start_flight_time_us=header.start_flight_time_us,
            stop_flight_time_us=header.stop_flight_time_us,
            bin_size_ns=header.bin_size_ns,
            start_mz=header.start_mz,
            stop_mz=header.stop_mz,
        )
        self._mz = self._axis.mz
        self._mz.flags.writeable = False

        tiles_x, tiles_y = index.tile_grid
        self._nx, self._ny = header.image_pixels * tiles_x, header.image_pixels * tiles_y
        grid = np.arange(self._nx * self._ny)
        self._xyz_coordinates = np.column_stack((grid % self._nx, grid // self._nx, np.zeros_like(grid)))
        self._aggregate()

    def _aggregate(self) -> None:
        """Bin all ion events into sparse per-pixel spectra stored in CSR form (`_indptr`, `_channels`, `_counts`).

        Events are keyed by `pixel * n_channels + channel`, sorted once and run-length encoded. Peak memory is about
        8 bytes per recorded event.
        """
        n_x, n_y = self._nx, self._ny
        n_channels = len(self._mz)
        n_records = self._index.n_records
        keys = np.empty(n_records, dtype=np.int64)
        filled = invalid = off_raster = off_axis = 0
        for x, y, tof, n_raw in iter_events(self.path, self._index, self._header.image_pixels):
            invalid += n_raw - tof.size
            on_grid = (x >= 0) & (x < n_x) & (y >= 0) & (y < n_y)
            channel, ok = self._axis.to_channel(tof)
            off_raster += int(np.count_nonzero(~on_grid))
            off_axis += int(np.count_nonzero(on_grid & ~ok))
            ok &= on_grid
            n_ok = int(ok.sum())
            keys[filled : filled + n_ok] = (y[ok] * n_x + x[ok]) * n_channels + channel[ok]
            filled += n_ok
        self._dropped = {"invalid": invalid, "off_raster": off_raster, "off_axis": off_axis}
        if filled == 0:
            raise ValueError(f"No usable ion events in '{self.path}' (dropped: {self._dropped}).")
        if off_raster:
            logger.warning(
                f"{off_raster} of {n_records} ion events lie outside the {n_x}x{n_y} raster and were dropped."
            )

        # sort in place and run-length encode (avoids the extra copies of np.unique)
        keys = keys[:filled]
        keys.sort()
        change = np.empty(keys.size, dtype=bool)
        change[0] = True
        np.not_equal(keys[1:], keys[:-1], out=change[1:])
        starts = np.flatnonzero(change)
        unique = keys[starts]
        counts = np.diff(np.append(starts, keys.size))
        del keys, change, starts
        pixel_of = unique // n_channels
        self._channels = (unique - pixel_of * n_channels).astype(np.int32)
        self._counts = counts.astype(np.int32)
        self._peak_pixels = pixel_of.astype(np.int32)
        self._indptr = np.zeros(self.n_pixels + 1, dtype=np.int64)
        np.cumsum(np.bincount(pixel_of, minlength=self.n_pixels), out=self._indptr[1:])
        logger.debug(
            f"Aggregated {filled} ion events into {self._channels.size} peaks; dropped {self._dropped} events."
        )

    # ------------------------------------------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------------------------------------------
    @property
    def mz_min(self) -> float:
        """Minimum m/z value of the mass axis."""
        return float(self._mz[0])

    @property
    def mz_max(self) -> float:
        """Maximum m/z value of the mass axis."""
        return float(self._mz[-1])

    @property
    def mz_axis(self) -> np.ndarray:
        """m/z of every detector channel within the acquired mass range."""
        return self._mz

    @property
    def tof_axis(self) -> np.ndarray:
        """Flight time (microseconds) of every channel of `mz_axis`."""
        return self._axis.tof_us

    @property
    def is_centroid(self) -> bool:
        """Spectra only contain the occupied channels."""
        return True

    @property
    def rois(self) -> list[int]:
        """Single region."""
        return [0]

    def get_n_pixels_for_roi(self, roi: int) -> int:
        """Return the number of pixels in the ROI."""
        return self.n_pixels if roi == 0 else 0

    def get_pixels_for_roi(self, roi: int) -> np.ndarray:
        """Return pixel indices of the ROI."""
        return self.pixels if roi == 0 else np.asarray([], dtype=int)

    def _get_reader_kwargs(self) -> dict[str, ty.Any]:
        kwargs: dict[str, ty.Any] = {"use_appended_calibration": self.use_appended_calibration}
        if self._pixel_size_override is not None:
            kwargs["pixel_size"] = self._pixel_size_override
        return kwargs

    @property
    def x_pixel_size(self) -> float:
        """X pixel size in micrometers (1.0 if it cannot be determined)."""
        if self._pixel_size_override is not None:
            return float(self._pixel_size_override)
        return self._header.pixel_size_um or 1.0

    @property
    def y_pixel_size(self) -> float:
        """Y pixel size in micrometers (1.0 if it cannot be determined)."""
        return self.x_pixel_size

    @property
    def polarity(self) -> str:
        """Secondary ion polarity ('positive' or 'negative')."""
        return self._header.polarity

    @property
    def n_frames(self) -> int:
        """Number of acquisition frames found in the file."""
        return self._index.n_frames

    @property
    def tile_grid(self) -> tuple[int, int]:
        """Number of mosaic tiles along x and y."""
        return self._index.tile_grid

    @property
    def calibration_source(self) -> str:
        """'appended' if a recalibration appended to the file was used, otherwise 'header'."""
        return self._calibration_source

    @property
    def calibration(self) -> dict[str, float]:
        """`Mass/Time` and `MassOffset` used to compute the m/z axis."""
        return dict(self._calibration)

    @property
    def dropped_events(self) -> dict[str, int]:
        """Number of ion events dropped as invalid, outside the raster or outside the mass axis."""
        return dict(self._dropped)

    @property
    def metadata(self) -> dict[str, ty.Any]:
        """Header entries ('root'), header sections ('sections') and appended info blocks ('appended')."""
        return {
            "root": dict(self._header.entries),
            "sections": {key: dict(value) for key, value in self._header.sections.items()},
            "appended": [dict(info) for info in self._index.appended],
        }

    # ------------------------------------------------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------------------------------------------------
    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(m/z, ion counts)` of the occupied channels of a pixel."""
        if not 0 <= index < self.n_pixels:
            raise IndexError(f"Pixel index {index} is out of bounds for {self.n_pixels} pixels.")
        start, end = self._indptr[index], self._indptr[index + 1]
        return self._mz[self._channels[start:end]], self._counts[start:end].astype(np.float32)

    def _read_spectra(self, indices: np.ndarray | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = self.pixels if indices is None else np.asarray(indices, dtype=np.int64)
        for index in indices:
            yield self._read_spectrum(int(index))

    def _pixel_weights(self, indices: np.ndarray, scales: np.ndarray | None = None) -> np.ndarray:
        """Return per-pixel weights (repeated indices count multiple times)."""
        if len(indices) and (indices.min() < 0 or indices.max() >= self.n_pixels):
            raise IndexError(f"Pixel indices must be in the range [0, {self.n_pixels}).")
        weights = np.zeros(self.n_pixels, dtype=np.float64)
        np.add.at(weights, indices, 1.0 if scales is None else np.asarray(scales, dtype=np.float64)[indices])
        return weights

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce a summed mass spectrum of the occupied channels.

        Parameters
        ----------
        indices : iterable of int
            Pixel indices.
        scales : np.ndarray, optional
            Multiplier for each pixel (indexed by pixel index, i.e. of length `n_pixels`).
        silent : bool
            Unused, kept for API compatibility.
        """
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")
        weights = self._pixel_weights(indices, scales)[self._peak_pixels]
        mask = weights != 0
        summed = np.bincount(self._channels[mask], weights=self._counts[mask] * weights[mask], minlength=len(self._mz))
        occupied = np.flatnonzero(summed)
        return self._mz[occupied].copy(), summed[occupied]

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return total ion count per pixel."""
        if self._tic is None:
            self._tic = np.bincount(self._peak_pixels, weights=self._counts, minlength=self.n_pixels)
        return self._tic

    # ------------------------------------------------------------------------------------------------------------
    # ion images
    # ------------------------------------------------------------------------------------------------------------
    def _get_ions(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> np.ndarray:
        """Sum ion counts within each m/z window for every pixel (pixels without ions get 0)."""
        mzs_min, mzs_max = get_mzs_for_tol(np.asarray(mzs), tol, ppm)
        lo = np.searchsorted(self._mz, mzs_min, side="left")
        hi = np.searchsorted(self._mz, mzs_max, side="right")
        res = np.zeros((self.n_pixels, len(lo)), dtype=np.float32)
        for i in tqdm(range(len(lo)), disable=silent, desc="Extracting ions..."):
            mask = (self._channels >= lo[i]) & (self._channels < hi[i])
            res[:, i] = np.bincount(self._peak_pixels[mask], weights=self._counts[mask], minlength=self.n_pixels)
        return res

    def get_ion_image(
        self,
        mz: float,
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> np.ndarray:
        """Return ion image for specified m/z with tolerance or ppm."""
        return self.reshape(self._get_ions([mz], tol, ppm, fill_value, silent)[:, 0])


@hook_impl
def imzy_reader(path: PathLike, reader_kwargs: dict[str, ty.Any], **kwargs: ty.Any) -> PhiReader | None:
    """Return PhiReader if path is a PHI SmartSoft-TOF `.raw` file."""
    if is_phi(path):
        return PhiReader(path, **(reader_kwargs | kwargs))
    return None
