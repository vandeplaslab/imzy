"""Reader for TOFWERK TofDAQ HDF5 files with laser-ablation imaging data (e.g. LA-icpTOF).

File format
-----------
TofDAQ writes a single HDF5 file. The data cubes are indexed by `[write, buf, segment, ...]`:

- `PeakData/PeakData` - (writes, bufs, segments, n_peaks) integrated (fitted) peak intensities, with
  `PeakData/PeakTable` holding the label and mass of each peak.
- `FullSpectra/TofData` - (writes, bufs, segments, n_samples) full TOF spectra, with `FullSpectra/MassAxis` holding the
  m/z value of each sample.

The laser software adds the `ImageData` group which maps laser spots to the data cubes:

- `ImageData/SpotData/Data` - one row per laser spot: valid flag, X, Y, Z stage coordinates (micrometers), write index
  and buf index.
- `ImageData/ScanData/Data` - one row per scan line: scan type, sample type (0 = sample, 1 = reference standard), scan
  mode, offset into SpotData, number of spots and layer. Scan names are stored as `ScanNameNNNNN` attributes.
"""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ims_utils.spectrum import get_mzs_for_tol
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy.hookspec import hook_impl

SUFFIXES = (".h5", ".hdf5")
MODES = ("peaks", "profile")
SPOT_DATA = "ImageData/SpotData/Data"
SCAN_DATA = "ImageData/ScanData/Data"
PEAK_DATA = "PeakData/PeakData"
PEAK_TABLE = "PeakData/PeakTable"
TOF_DATA = "FullSpectra/TofData"
MASS_AXIS = "FullSpectra/MassAxis"


def _decode(value: ty.Any) -> ty.Any:
    """Decode HDF5 attribute value (byte strings and 1-element arrays) into a Python value."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode(value.ravel()[0])
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_tofwerk(path: PathLike) -> bool:
    """Return True if the path is a TOFWERK TofDAQ HDF5 file with laser imaging and peak data."""
    path = Path(path)
    if path.suffix.lower() not in SUFFIXES or not path.is_file():
        return False
    try:
        import h5py

        # TofDAQ writes an HDF5 user block, so the HDF5 signature is not at the start of the file
        if not h5py.is_hdf5(path):
            return False
        with h5py.File(path, "r") as f:
            return "TofDAQ Version" in f.attrs and SPOT_DATA in f and PEAK_DATA in f
    except Exception:
        return False


class TofwerkReader(BaseReader):
    """Reader for TOFWERK TofDAQ HDF5 files with laser-ablation imaging data.

    Each valid laser spot of the selected scans becomes one pixel. Spot coordinates (micrometers) are snapped onto a
    regular grid using the spot spacing as the pixel size.

    Parameters
    ----------
    path : PathLike
        Path to the `.h5` file.
    auto_profile : bool
        Unused, kept for API compatibility with other readers.
    roi : int
        Sample type of the scans to load: 0 = sample (default), 1 = reference standards.
    mode : str
        'peaks' (default) returns the integrated peak intensities (centroid, one value per entry of the peak table);
        'profile' returns the full TOF spectra on the mass axis.
    """

    def __init__(self, path: PathLike, auto_profile: bool = True, roi: int = 0, mode: str = "peaks"):
        super().__init__(path, auto_profile=auto_profile)
        if mode not in MODES:
            raise ValueError(f"Unknown mode '{mode}'. Available modes: {MODES}")
        self.roi = int(roi)
        self.mode = mode
        self._init()

    @contextmanager
    def _open(self) -> ty.Iterator[ty.Any]:
        """Open the file for reading."""
        import h5py

        with h5py.File(self.path, "r") as f:
            yield f

    # ------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------
    def _init(self) -> None:
        with self._open() as f:
            for key in (SPOT_DATA, SCAN_DATA):
                if key not in f:
                    raise ValueError(f"'{self.path}' does not look like a TOFWERK imaging file (missing '{key}').")
            self._data_key = PEAK_DATA if self.mode == "peaks" else TOF_DATA
            if self._data_key not in f:
                raise ValueError(f"'{self.path}' does not contain '{self._data_key}' needed for mode '{self.mode}'.")
            spots = f[SPOT_DATA][:]
            scans = f[SCAN_DATA][:]
            scan_attrs = f["ImageData/ScanData"].attrs
            self._scan_names = [str(_decode(scan_attrs.get(f"ScanName{i:05d}", ""))) for i in range(len(scans))]
            self._metadata = {
                "root": {key: _decode(value) for key, value in f.attrs.items()},
                "image": {key: _decode(value) for key, value in f["ImageData"].attrs.items()},
            }
            if self.mode == "peaks":
                table = f[PEAK_TABLE][:]
                self._mz = table["mass"].astype(np.float64)
                self._labels = [label.decode("utf-8", errors="replace") for label in table["label"]]
            else:
                self._mz = f[MASS_AXIS][:].astype(np.float64)
                self._labels = []
        if np.any(np.diff(self._mz) < 0):
            raise ValueError("The m/z axis of the file is not sorted.")
        self._mz.flags.writeable = False

        # select spots of the requested sample type
        sample_types = scans[:, 1].astype(int)
        self._rois = np.unique(sample_types).tolist()
        if self.roi not in self._rois:
            raise ValueError(f"ROI {self.roi} was not found. Available ROIs: {self._rois}")
        scan_indices = np.flatnonzero(sample_types == self.roi)
        selected = np.concatenate(
            [np.arange(int(offset), int(offset) + int(n)) for offset, n in scans[scan_indices, 3:5]]
        )
        spots = spots[selected]
        valid = (spots[:, 0] == 1) & (spots[:, 4] >= 0) & (spots[:, 5] >= 0)
        spots = spots[valid]
        if len(spots) == 0:
            raise ValueError(f"ROI {self.roi} does not contain any valid laser spots.")
        self._writes = spots[:, 4].astype(np.int64)
        self._bufs = spots[:, 5].astype(np.int64)
        self._stage_coordinates = spots[:, 1:4].copy()

        # snap stage coordinates onto a regular grid
        pixel_size = float(self._metadata["image"].get("SpotSpacing", 0) or 0)
        if not pixel_size > 0:
            steps = np.diff(np.unique(np.round(spots[:, 1:3], 3)))
            steps = steps[steps > 0]
            pixel_size = float(np.median(steps)) if steps.size else 1.0
        self._pixel_size = pixel_size
        x = np.round((spots[:, 1] - spots[:, 1].min()) / pixel_size).astype(np.int64)
        y = np.round((spots[:, 2] - spots[:, 2].min()) / pixel_size).astype(np.int64)
        if len(np.unique(y * (x.max() + 1) + x)) != len(x):
            raise ValueError("Several laser spots map onto the same pixel - cannot build an image grid.")
        self._xyz_coordinates = np.column_stack((x, y, np.zeros_like(x)))

    # ------------------------------------------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------------------------------------------
    @property
    def mz_min(self) -> float:
        """Minimum m/z value."""
        return float(self._mz[0])

    @property
    def mz_max(self) -> float:
        """Maximum m/z value."""
        return float(self._mz[-1])

    @property
    def mz_axis(self) -> np.ndarray:
        """m/z axis shared by all pixels (peak masses or TOF mass axis, depending on mode)."""
        return self._mz

    @property
    def is_centroid(self) -> bool:
        """Peak data is centroided, full spectra are in profile mode."""
        return self.mode == "peaks"

    @property
    def peak_labels(self) -> list[str]:
        """Labels of the peaks, e.g. '[63Cu]+' (empty in profile mode)."""
        return list(self._labels)

    @property
    def rois(self) -> list[int]:
        """Sample types present in the file (0 = sample, 1 = reference standards)."""
        return list(self._rois)

    def get_n_pixels_for_roi(self, roi: int) -> int:
        """Return the number of pixels in the ROI (only the loaded ROI has pixels)."""
        return self.n_pixels if roi == self.roi else 0

    def get_pixels_for_roi(self, roi: int) -> np.ndarray:
        """Return pixel indices of the ROI (only the loaded ROI has pixels)."""
        return self.pixels if roi == self.roi else np.asarray([], dtype=int)

    def _get_reader_kwargs(self) -> dict[str, ty.Any]:
        return {"roi": self.roi, "mode": self.mode}

    @property
    def x_pixel_size(self) -> float:
        """X pixel size in micrometers."""
        return self._pixel_size

    @property
    def y_pixel_size(self) -> float:
        """Y pixel size in micrometers."""
        return self._pixel_size

    @property
    def stage_coordinates(self) -> np.ndarray:
        """X, Y, Z stage coordinates (micrometers) of each pixel."""
        return self._stage_coordinates

    @property
    def scan_names(self) -> list[str]:
        """Names of all scans in the file (sample lines and reference standards)."""
        return list(self._scan_names)

    @property
    def metadata(self) -> dict[str, ty.Any]:
        """File ('root') and laser imaging ('image') attributes."""
        return self._metadata

    @property
    def polarity(self) -> str | None:
        """Ion polarity ('positive'/'negative') if known."""
        polarity = self._metadata["root"].get("IonMode")
        return str(polarity) if polarity else None

    # ------------------------------------------------------------------------------------------------------------
    # spectra
    # ------------------------------------------------------------------------------------------------------------
    def _iter_blocks(self, indices: np.ndarray) -> ty.Iterator[tuple[slice, np.ndarray]]:
        """Yield `(positions, intensities)` for runs of consecutive pixels that were recorded in the same write.

        `positions` is a slice into `indices` and `intensities` is a 2-D array with one row per pixel in that slice.
        """
        if len(indices) and (indices.min() < 0 or indices.max() >= self.n_pixels):
            raise IndexError(f"Pixel indices must be in the range [0, {self.n_pixels}).")
        writes, bufs = self._writes[indices], self._bufs[indices]
        boundaries = np.flatnonzero(np.diff(writes)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(indices)]))
        with self._open() as f:
            ds = f[self._data_key]
            for start, end in zip(starts, ends, strict=True):
                run = bufs[start:end]
                lo, hi = int(run.min()), int(run.max())
                slab = ds[int(writes[start]), lo : hi + 1, 0, :]
                yield slice(int(start), int(end)), slab[run - lo]

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return `(m/z, intensities)` of a pixel."""
        if not 0 <= index < self.n_pixels:
            raise IndexError(f"Pixel index {index} is out of bounds for {self.n_pixels} pixels.")
        with self._open() as f:
            return self._mz, f[self._data_key][self._writes[index], self._bufs[index], 0, :]

    def _read_spectra(self, indices: np.ndarray | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = self.pixels if indices is None else np.asarray(indices, dtype=np.int64)
        for _, block in self._iter_blocks(indices):
            for y in block:
                yield self._mz, y

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum.

        Parameters
        ----------
        indices : iterable of int
            Pixel indices.
        scales : np.ndarray, optional
            Multiplier for each pixel (indexed by pixel index, i.e. of length `n_pixels`).
        silent : bool
            Disable the progress bar.
        """
        indices = np.asarray(list(indices), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")
        summed = np.zeros(len(self._mz), dtype=np.float64)
        with tqdm(total=len(indices), disable=silent, desc="Summing spectra...") as progress:
            for positions, block in self._iter_blocks(indices):
                if scales is None:
                    summed += block.sum(axis=0, dtype=np.float64)
                else:
                    summed += np.asarray(scales)[indices[positions]] @ block.astype(np.float64)
                progress.update(len(block))
        return self._mz.copy(), summed

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return total intensity per pixel."""
        if self._tic is None:
            tic = np.zeros(self.n_pixels)
            with tqdm(total=self.n_pixels, disable=silent, desc="Computing TIC...") as progress:
                for positions, block in self._iter_blocks(self.pixels):
                    tic[positions] = block.sum(axis=1, dtype=np.float64)
                    progress.update(len(block))
            self._tic = tic
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
        """Sum intensities within each m/z window; the m/z axis is shared, so windows are resolved only once."""
        mzs_min, mzs_max = get_mzs_for_tol(np.asarray(mzs), tol, ppm)
        lo = np.searchsorted(self._mz, mzs_min, side="left")
        hi = np.searchsorted(self._mz, mzs_max, side="right")
        res = np.full((self.n_pixels, len(lo)), dtype=np.float32, fill_value=fill_value)
        with tqdm(total=self.n_pixels, disable=silent, desc="Extracting ions...") as progress:
            for positions, block in self._iter_blocks(self.pixels):
                cumsum = np.zeros((len(block), block.shape[1] + 1), dtype=np.float64)
                np.cumsum(block, axis=1, out=cumsum[:, 1:])
                res[positions] = cumsum[:, hi] - cumsum[:, lo]
                progress.update(len(block))
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
def imzy_reader(path: PathLike, reader_kwargs: dict[str, ty.Any], **kwargs: ty.Any) -> TofwerkReader | None:
    """Return TofwerkReader if path is a TOFWERK TofDAQ imaging file."""
    if is_tofwerk(path):
        return TofwerkReader(path, **(reader_kwargs | kwargs))
    return None
