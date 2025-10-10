"""Waters reader."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ims_utils.spectrum import get_ppm_axis
from koyo.system import IS_WIN
from koyo.typing import PathLike
from koyo.utilities import get_min_max
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy._readers.waters.MassLynxRawChromatogramReader import MassLynxRawChromatogramReader
from imzy._readers.waters.MassLynxRawInfoReader import MassLynxRawInfoReader
from imzy._readers.waters.MassLynxRawReader import MassLynxException, MassLynxRawReader
from imzy._readers.waters.MassLynxRawScanReader import MassLynxRawScanReader
from imzy.hookspec import hook_impl
from imzy.utilities import _auto_guess_ppm


class WatersReader(BaseReader):
    """Waters reader.

    Parameters
    ----------
    path : PathLike
        Path to the Waters data directory.
    auto_profile : bool, optional
        Automatically convert centroid data to profile data, by default True.
    resolution : int | str, optional
        IF "auto", use the instrument resolution from the metadata.
    mz_ppm : float | str, optional
        Specify the m/z ppm spacing, by default "auto". If "auto", use a reasonable value based on the instrument
        resolution.
    enable_ion_mobility : bool, optional
        Enable ion mobility dimension, by default True. If the data does not have ion mobility, this will be ignored.
        If the data does have ion mobility, this will enable reading the ion mobility dimension.
    """

    # mass offset when determining the mz range (can't remember why...)
    mz_offset: float = 5
    _mz_min: float
    _mz_max: float
    _mz_ppm: float | None = 5.0  # default m/z ppm spacing
    _mz_grid: np.ndarray | None = None

    # Reader objects
    _reader: MassLynxRawReader | None = None
    _info_reader: MassLynxRawInfoReader | None = None
    _data_reader: MassLynxRawScanReader | None = None
    _chromatogram_reader: MassLynxRawChromatogramReader | None = None

    def __init__(
        self,
        path: PathLike,
        auto_profile: bool = True,
        mz_ppm: float | str = "auto",
        resolution: int | str = "auto",
        enable_ion_mobility: bool = True,
    ) -> None:
        super().__init__(path, auto_profile=auto_profile)
        self.enable_ion_mobility = enable_ion_mobility
        self.resolution = _auto_guess_resolution(resolution)
        self.mz_ppm = _auto_guess_ppm(self.resolution, mz_ppm)
        self._init()

    def _init(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Method which is called to initialize the reader."""
        (
            self.n_functions,
            self.stats_in_functions,
            self.num_scans,
            self._mz_min,
            self._mz_max,
            self.mass_range,
            self.instrument_type,
        ) = self._setup_functions()
        self._setup_coordinates()
        self.n_dt_bins = self.get_n_mobility_bins()
        self._x_pixel_size, self._y_pixel_size = get_pixel_spacing(self.path)

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether data is in centroid or profile mode.

        Waters data is always in centroid mode, even if it's stored as profile, you can't read it as such using their
        SDK.
        """
        return True

    @property
    def mz_min(self) -> float:
        """Return minimum m/z value."""
        return self._mz_min

    @property
    def mz_max(self) -> float:
        """Return maximum m/z value."""
        return self._mz_max

    @property
    def mz_ppm(self) -> float:
        """Return m/z ppm spacing."""
        return self._mz_ppm

    @mz_ppm.setter
    def mz_ppm(self, value: float | str) -> None:
        """Set m/z ppm spacing."""
        if isinstance(value, str) and value == "auto":
            self._mz_ppm = _auto_guess_ppm(self.resolution, value)
        else:
            self._mz_ppm = float(value)
        self._mz_grid = None  # reset mz grid

    @property
    def reader(self) -> MassLynxRawReader:
        """Create the file parser."""
        if self._reader is None:
            self._reader = MassLynxRawReader(self.path, 1)  # type: ignore[no-untyped-call]
        return self._reader

    @property
    def info_reader(self) -> MassLynxRawInfoReader:
        """Create the info parser."""
        if self._info_reader is None:
            self._info_reader = MassLynxRawInfoReader(self.reader)  # type: ignore[no-untyped-call]
        return self._info_reader

    @property
    def data_reader(self) -> MassLynxRawScanReader:
        """Create the data parser."""
        if self._data_reader is None:
            self._data_reader = MassLynxRawScanReader(self.reader)  # type: ignore[no-untyped-call]
        return self._data_reader

    @property
    def chromatogram_reader(self) -> MassLynxRawChromatogramReader:
        """Create the chromatogram parser."""
        if self._chromatogram_reader is None:
            self._chromatogram_reader = MassLynxRawChromatogramReader(self.reader)  # type: ignore[no-untyped-call]
        return self._chromatogram_reader

    @contextmanager
    def _enable_faster_iter(self) -> ty.Generator[None, None, None]:
        """Context manager to temporarily enable faster iteration mode."""
        auto_profile = self.auto_profile
        self.auto_profile = False
        enable_faster_iter = True
        self.enable_ion_mobility = False
        try:
            yield
        finally:
            self.auto_profile = auto_profile
            self.enable_ion_mobility = enable_faster_iter

    def _setup_functions(self) -> tuple[int, dict, int, float, float, list[float], int]:
        """Get stats for each function."""
        n_functions = self.info_reader.GetNumberofFunctions()

        stats_in_functions = {}
        n_scans = 0
        mass_range = []  # type: ignore[var-annotated]
        for fcn in range(n_functions):
            _scans = self.info_reader.GetScansInFunction(fcn)
            _mass_range = self.info_reader.GetAcquisitionMassRange(fcn)
            ion_mode = self.info_reader.GetIonModeString(self.info_reader.GetIonMode(fcn))
            fcn_type = self.info_reader.GetFunctionTypeString(self.info_reader.GetFunctionType(fcn))

            stats_in_functions[fcn] = {
                "n_scans": _scans,
                "mass_range": mass_range,
                "ion_mode": ion_mode,
                "fcn_type": fcn_type,
            }
            n_scans += _scans
            mass_range.extend(_mass_range)
        return (
            n_functions,
            stats_in_functions,
            n_scans,
            min(mass_range) - self.mz_offset,
            max(mass_range) + self.mz_offset,
            mass_range,
            stats_in_functions[0]["ion_mode"],
        )

    def _get_frame_coordinates(self, fcn: int, scan: int) -> tuple[float, float]:
        """Get frame coordinates."""
        # retrieve x and y coordinate
        x_pos, y_pos = self.info_reader.GetScanItems(fcn, scan, [9, 10])
        return float(x_pos), float(y_pos)

    def _setup_coordinates(self) -> None:
        """Collect metadata.

        Collect image dimension information and ensure the x/y coordinate indexing is matching that of the library.
        """

        def _reindex(coordinates: np.ndarray) -> np.ndarray:
            """Waters coordinates are floats that can be positive or negative.

            We simply convert the floats to increasing integers that are a lot easier to deal with
            """
            coordinates = coordinates + np.abs(np.min(coordinates))
            spacing = np.diff(np.sort(coordinates)).max()
            coordinates = np.round(coordinates / spacing).astype(np.int32)
            return coordinates

        x_coordinates_, y_coordinates_, frame_indices = [], [], []
        scan_id = 0
        frame_to_fcn = {}
        for fcn in self.stats_in_functions:
            for _scan_id in range(self.stats_in_functions[fcn]["n_scans"]):
                x, y = self._get_frame_coordinates(fcn, _scan_id)
                x_coordinates_.append(x)
                y_coordinates_.append(y)
                frame_indices.append(scan_id)
                frame_to_fcn[scan_id] = (fcn, _scan_id)
                scan_id += 1

        self.frame_to_fcn = frame_to_fcn
        x_coordinates = _reindex(np.asarray(x_coordinates_))
        y_coordinates = _reindex(np.asarray(y_coordinates_))
        self._xyz_coordinates = np.column_stack((x_coordinates, y_coordinates, np.ones_like(x_coordinates)))
        self.frame_indices = np.asarray(frame_indices)
        self.x_min, self.x_max = get_min_max(x_coordinates)  # type: ignore[assignment]
        self.y_min, self.y_max = get_min_max(y_coordinates)  # type: ignore[assignment]

    @property
    def mz_x(self) -> np.ndarray:
        """Get m/z grid."""
        if self._mz_grid is None:
            self._mz_grid = get_ppm_axis(self.mz_min, self.mz_max, self.mz_ppm)
        return self._mz_grid

    def get_n_mobility_bins(self) -> int:
        """Get the number of bins."""
        try:
            # if this fails, its most likely (certainly?) because the dataset does not have ion mobility dimension
            _ = self.data_reader.ReadDriftScan(  # type: ignore[no-untyped-call]
                0,
                0,
                1,
            )
            n_dt_bins = 200
        except MassLynxException:
            n_dt_bins = 1
        return n_dt_bins

    @property
    def rois(self) -> list[int]:
        """Return a list of ROI indices."""
        return [0]  # waters files always have a single ROI

    @property
    def x_pixel_size(self) -> float:
        """Return x pixel size in micrometers."""
        return self._x_pixel_size

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size in micrometers."""
        return self._y_pixel_size

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return TIC image."""
        if self._tic is None:
            tic_y = []
            for fcn in self.stats_in_functions:
                tic_y_ = self.chromatogram_reader.ReadTIC(fcn)  # type: ignore[no-untyped-call]
                tic_y.extend(tic_y_)
            self._tic = np.asarray(tic_y)
        return self._tic

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a single spectrum."""
        if self.n_dt_bins == 1 or not self.enable_ion_mobility:
            fcn, scan = self.frame_to_fcn[index]
            x, y = self._read_scan_buffer_mz(fcn, scan)
            if self.auto_profile:
                x, y, _ = self._centroid_to_profile(x, y, resolution=self.resolution, mz_grid=self.mz_x)
            return x, y
        else:
            fcn, scan = self.frame_to_fcn[index]
            res = self._read_scan_buffer_dt(fcn, scan, 0, self.n_dt_bins)
            x = self.mz_x
            if self.auto_profile:
                # convert each drift time spectrum to profile
                y_profile = np.zeros((self.mz_x.size, self.n_dt_bins), dtype=np.float32)
                for drift_id, (x_, y_) in enumerate(res):
                    if np.any(y_ > 0):
                        _, y_profile_, _ = self._centroid_to_profile(
                            x_, y_, resolution=self.resolution, mz_grid=self.mz_x
                        )
                        y_profile[:, drift_id] = y_profile_
                y = y_profile
            else:
                y = res  # this will actually be a list of (x, y) tuples
            return x, y

    def _read_scan_buffer_mz(self, fcn: int, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.data_reader.ReadScan(fcn, frame_id)  # type: ignore[no-untyped-call]
        indices = np.where((x > self.mz_min) & (x < self.mz_max) & (y > 0))[0]
        x, y = x[indices], y[indices]
        return x, y  # type: ignore[no-any-return]

    def _read_scan_buffer_dt(
        self, fcn: int, frame_id: int, scan_begin: int = 0, scan_end: int = 200
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        res = []
        for drift_id in range(scan_begin, scan_end):
            x_, y_ = self.data_reader.ReadDriftScan(fcn, frame_id, drift_id)
            if x_.size > 0:
                indices = np.where((x_ > self.mz_min) & (x_ < self.mz_max) & (y_ > 0))[0]
                x_, y_ = x_[indices], y_[indices]
                res.append((x_, y_))
        return res

    def _read_spectra(self, indices: ty.Iterable[int] | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Read spectra without constantly opening and closing the file handle."""
        if indices is None:
            indices = self.pixels
        for index in indices:
            yield self._read_spectrum(index)

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum."""
        indices = np.asarray(indices)
        if np.any(indices >= self.n_pixels):
            raise ValueError("You cannot specify indices that are greater than the total number of pixels.")
        if scales is None:
            scales = np.ones(self.n_pixels, dtype=np.float32)

        mz_x = self.mz_x
        mz_y = np.zeros_like(mz_x, dtype=np.float64)
        for index in tqdm(indices, total=len(indices), disable=silent, desc="Summing spectra..."):
            fcn, scan = self.frame_to_fcn[index]
            x, y = self._read_scan_buffer_mz(fcn, scan)
            y = y * scales[index]
            x, y, _ = self._centroid_to_profile(x, y, resolution=self.resolution, mz_grid=mz_x)
            mz_y += y
        return mz_x, mz_y


def is_waters(path: PathLike) -> bool:
    """Check if the path is a Waters raw data directory."""
    path = Path(path)
    return (
        path.suffix.lower() == ".raw"
        and path.is_dir()
        and (path / "_extern.inf").exists()
        and (path / "_header.txt").exists()
        and IS_WIN
    )


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> WatersReader | None:
    """Return TDFReader if path is Bruker .d/tdf."""
    if is_waters(path):
        return WatersReader(path, **kwargs)
    return None


def _auto_guess_resolution(resolution: int | str) -> int:
    """Get resolution from metadata or use a provided value."""
    if resolution == "auto":
        # try to guess resolution based on the instrument type - ideally we would find out what instrument was used,
        # but this metadata is difficult to obtain. If data was collected on Synapt, then 40-60k is reasonable guess,
        # if it was collected on Xevo, then 60-120k is reasonable guess, if it was collected on MRT, then
        # 120-200k is a reasonable guess.
        resolution = 50_000
    return resolution  # type: ignore[return-value]


def get_pixel_spacing(path: PathLike) -> tuple[float, float]:
    """Get pixel spacing in micrometers.

    The pixel spacing is stored in the _extern.inf file in the Waters raw data directory.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    extern_file = path / "_extern.inf"
    if not extern_file.exists():
        return -1.0, -1.0

    x_step, y_step = -1.0, -1.0
    with open(extern_file) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("DesiXStep"):
                x_step = float(line.split()[1])
            if line.startswith("DesiYStep"):
                y_step = float(line.split()[1])
    return x_step * 1e6, y_step * 1e6  # convert to micrometers
