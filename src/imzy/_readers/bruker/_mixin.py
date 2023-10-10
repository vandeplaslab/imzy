"""Base class for file readers."""
import sqlite3
import typing as ty
from contextlib import contextmanager
from ctypes import CDLL, POINTER, c_double

import numpy as np
from koyo.typing import PathLike
from koyo.utilities import get_min_max
from tqdm import tqdm

from imzy._readers._base import BaseReader


class BrukerBaseReader(BaseReader):
    """Base class for TSF/TDF file readers."""

    # class attributes
    dll: ty.Optional[CDLL] = None
    handle = None
    sql_filename: str
    _mz_x: ty.Optional[np.ndarray] = None
    _rois: ty.Optional[ty.List[int]] = None
    _pixel_size: ty.Optional[float] = None

    # DLL functions
    _dll_close_func: ty.Callable
    _dll_index_to_mz_func: ty.Callable
    _dll_mz_to_index_func: ty.Callable

    def __init__(self, path: PathLike):
        super().__init__(path)
        self._init()

    def _init(self) -> None:
        """Extra initialization."""
        assert (self.path / self.sql_filename).exists(), f"Could not find {self.sql_filename} file."
        self._mz_min, self._mz_max = self.get_acquisition_mass_range()
        self.set_region_information()

    @property
    def mz_min(self) -> float:
        """Return minimum m/z value."""
        return self._mz_min

    @property
    def mz_max(self) -> float:
        """Return maximum m/z value."""
        return self._mz_max

    @property
    def rois(self) -> ty.List[int]:
        """Return list of ROIs."""
        if self._rois is None:
            self._rois = np.unique(self.region_number).tolist()
        return self._rois

    @property
    def x_pixel_size(self) -> float:
        """Return x pixel size in micrometers."""
        return self.pixel_size

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size in micrometers."""
        return self.pixel_size

    @property
    def pixel_size(self) -> float:
        """Return pixel size.

        This method will throw an error if the pixel size is not equal in both dimensions.
        """
        if self._pixel_size is None:
            with self.sql_reader() as conn:
                cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")
                self._pixel_size = float(cursor.fetchone()[0])
        return self._pixel_size

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether the data is in centroid or profile mode."""
        try:
            self._read_spectrum(1)
            return True
        except RuntimeError:
            return False

    def get_summed_spectrum(self, indices: ty.Iterable[int], silent: bool = False) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum."""
        indices = np.asarray(indices)
        if np.any(indices >= self.n_pixels):
            raise ValueError("You cannot specify indices that are greater than the total number of pixels.")
        mz_y = np.zeros_like(self.mz_x, dtype=np.float64)
        for index in tqdm(indices, disable=silent, total=len(indices), desc="Summing spectra..."):
            mz_y += self._read_spectrum(index)[1]
        return self.mz_x, mz_y

    def _read_spectrum(self, index: int) -> ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Must implement method")

    def _read_spectra(self, indices: ty.Optional[np.ndarray] = None) -> ty.Iterator[ty.Tuple[np.ndarray, np.ndarray]]:
        if indices is None:
            indices = self.pixels
        for index in indices:
            yield self._read_spectrum(index)

    def __enter__(self) -> "BrukerBaseReader":
        return self

    def __exit__(self, type, value, traceback) -> None:  # type: ignore
        self.close()

    def __del__(self) -> None:
        self.close()

    @contextmanager
    def sql_reader(self) -> ty.Generator[sqlite3.Connection, None, None]:
        """SQL reader context manager."""
        conn = sqlite3.connect(self.path / self.sql_filename)
        yield conn
        conn.close()

    def close(self) -> None:
        """Close file."""
        if hasattr(self, "handle") and self.handle is not None:
            self._dll_close_func(self.handle)
            self.handle = None

    def _call_conversion_func(self, frame_id: int, input_data: np.ndarray, func: ty.Callable) -> np.ndarray:
        raise NotImplementedError("Must implement method")

    def _call_conversion_func_base(
        self, frame_id: int, input_data: np.ndarray, func: ty.Callable
    ) -> ty.Tuple[int, np.ndarray]:
        if type(input_data) is np.ndarray and input_data.dtype == np.float64:
            # already "native" format understood by DLL -> avoid extra copy
            in_array = input_data
        else:
            # convert data to format understood by DLL:
            in_array = np.array(input_data, dtype=np.float64)

        cnt = len(in_array)
        out = np.empty(shape=cnt, dtype=np.float64)
        success = func(
            self.handle,
            frame_id,
            in_array.ctypes.data_as(POINTER(c_double)),
            out.ctypes.data_as(POINTER(c_double)),
            cnt,
        )
        return success, out

    @property
    def mz_index(self) -> np.ndarray:
        """Return index."""
        bruker_mz_max = self.read_profile_spectrum(1).shape[0]
        return np.arange(0, bruker_mz_max)

    @property
    def mz_x(self) -> np.ndarray:
        """Get x-axis of the mass spectrum."""
        if self._mz_x is None:
            self._mz_x = self.index_to_mz(1, self.mz_index)
        return self._mz_x

    def index_to_mz(self, frame_id: int, indices: ty.Iterable[int]) -> np.ndarray:
        """Convert m/z index to m/z."""
        return self._call_conversion_func(frame_id, indices, self._dll_index_to_mz_func)

    def mz_to_index(self, frame_id: int, mzs: ty.Iterable[float]) -> np.ndarray:
        """Convert m/z to index."""
        return self._call_conversion_func(frame_id, mzs, self._dll_mz_to_index_func)

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Read profile spectrum/."""
        raise NotImplementedError("Must implement method")

    def get_n_pixels(self) -> int:
        """Retrieve number of frames from the file."""
        with self.sql_reader() as conn:
            q = conn.execute("SELECT Max(Frame) FROM MaldiFrameInfo")
            n_frames = int(q.fetchone()[0])
        return n_frames

    def get_acquisition_mass_range(self) -> ty.Tuple[float, float]:
        """Retrieve acquisition mass range from the file."""
        with self.sql_reader() as conn:
            cursor = conn.execute("SELECT Key, Value FROM GlobalMetadata")
            fetch_all = cursor.fetchall()
        mz_min, mz_max = None, None
        for key, value in fetch_all:
            if key == "MzAcqRangeLower":
                mz_min = float(value)
            if key == "MzAcqRangeUpper":
                mz_max = float(value)

        if mz_min is None or mz_max is None:
            raise ValueError("Failed to extract experimental mass range from the file!")
        return mz_min, mz_max

    # noinspection PyAttributeOutsideInit
    def set_region_information(self, roi: ty.Optional[int] = None) -> None:
        """Collect file information."""
        data = self._read_cache("frame_index_cache", ["frame_index_position"])
        # load data from cache
        if data["frame_index_position"] is not None:
            frame_index_position = data["frame_index_position"]
        else:
            with self.sql_reader() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT Frame, XIndexPos, YIndexPos, RegionNumber FROM MaldiFrameInfo")
                frame_index_position = np.array(cursor.fetchall())
            self._write_cache("frame_index_cache", data={"frame_index_position": frame_index_position})

        # get ROI
        self.region_number = frame_index_position[:, 3]
        self.region_frames = np.arange(0, len(self.region_number))
        if self.region_number.size > 0:
            self._rois = np.arange(0, self.region_number[-1] + 1).tolist()

        # select only those frames that match specific region of interest
        if roi not in [None, "None"]:
            self.region_frames = np.where(self.region_number == roi)[0]

        # apply ROI restriction
        self.region_len = len(self.region_frames)
        self.frame_indices = frame_index_position[self.region_frames, 0]
        # self.x_coordinates_all = frame_index_position[:, 1]
        # self.y_coordinates_all = frame_index_position[:, 2]
        x_coordinates = frame_index_position[self.region_frames, 1]
        x_min, x_max = get_min_max(x_coordinates)
        y_coordinates = frame_index_position[self.region_frames, 2]
        y_min, y_max = get_min_max(y_coordinates)
        x_coordinates = x_coordinates - x_min
        y_coordinates = y_coordinates - y_min
        self._xyz_coordinates = np.column_stack((x_coordinates, y_coordinates, np.zeros_like(x_coordinates)))

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Get TIC data."""
        if self._tic is None:
            data = self._read_cache("tic", ["tic", "region_frames"])
            if data["tic"] is not None and not isinstance(data["region_frames"], str):
                tic = data["tic"]
            else:
                with self.sql_reader() as conn:
                    try:
                        cursor = conn.execute("SELECT SummedIntensities, RegionNumber FROM Spectra")
                        tic_data = np.array(cursor.fetchall())
                        region_frames = tic_data[:, 1]
                        tic = tic_data[:, 0]
                    except sqlite3.OperationalError:
                        # this is required to handle TSF files (and maybe TDF?)
                        cursor = conn.execute("SELECT RegionNumber FROM MaldiFrameInfo")
                        region_frames = np.array(cursor.fetchall())
                        cursor = conn.execute("SELECT SummedIntensities FROM Frames")
                        tic = np.array(cursor.fetchall())
                    tic = np.ravel(tic)
                self._write_cache("tic", data={"tic": tic, "region_frames": region_frames})
            self._tic = tic
        return self._tic
