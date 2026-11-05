"""Reader for Bruker .d datasets stored in the BAF format (``analysis.baf``).

Spectra are read through Bruker's ``baf2sql`` library, which ships for Windows and Linux only. Pixel coordinates and
regions are read from the ``*peaks.sqlite`` database written next to the BAF file.

Note
----
On first use, ``baf2sql`` creates the ``analysis.sqlite`` cache inside the dataset directory.
"""

from __future__ import annotations

import sqlite3
import typing as ty
from ctypes import CDLL, POINTER, c_char_p, c_double, c_int, c_uint32, c_uint64, cdll, create_string_buffer
from functools import lru_cache
from pathlib import Path

import numpy as np
from koyo.system import IS_MAC, IS_WIN
from koyo.typing import PathLike
from loguru import logger

from imzy._readers._base import BaseReader
from imzy._readers.bruker._mixin import BrukerBaseReader
from imzy.hookspec import hook_impl

# Bruker ships baf2sql for Windows and Linux only
HAS_BAF_BACKEND = not IS_MAC
BAF_LIB_NAME = "baf2sql_c.dll" if IS_WIN else "libbaf2sql_c.so"


@lru_cache(maxsize=1)
def get_baf2sql_dll() -> CDLL:
    """Load the ``baf2sql`` library (bundled copy first, then the system search path)."""
    if not HAS_BAF_BACKEND:
        raise OSError("Bruker baf2sql library is not available for macOS.")
    path = Path(__file__).parent / BAF_LIB_NAME
    dll = cdll.LoadLibrary(path.as_posix() if path.exists() else BAF_LIB_NAME)

    dll.baf2sql_array_close_storage.argtypes = [c_uint64]
    dll.baf2sql_array_close_storage.restype = None
    dll.baf2sql_array_get_num_elements.argtypes = [c_uint64, c_uint64, POINTER(c_uint64)]
    dll.baf2sql_array_get_num_elements.restype = c_int
    dll.baf2sql_array_open_storage.argtypes = [c_int, c_char_p]
    dll.baf2sql_array_open_storage.restype = c_uint64
    dll.baf2sql_array_read_double.argtypes = [c_uint64, c_uint64, POINTER(c_double)]
    dll.baf2sql_array_read_double.restype = c_int
    dll.baf2sql_get_last_error_string.argtypes = [c_char_p, c_uint32]
    dll.baf2sql_get_last_error_string.restype = c_uint32
    dll.baf2sql_get_sqlite_cache_filename_v2.argtypes = [c_char_p, c_uint32, c_char_p, c_int]
    dll.baf2sql_get_sqlite_cache_filename_v2.restype = c_uint32
    return dll


def throw_last_baf_error(dll: CDLL) -> ty.NoReturn:
    """Raise the last error reported by ``baf2sql``."""
    length = dll.baf2sql_get_last_error_string(None, 0)
    buf = create_string_buffer(length)
    dll.baf2sql_get_last_error_string(buf, length)
    raise RuntimeError(buf.value.decode("utf-8", errors="replace"))


def read_double(dll: CDLL, handle: int, array_id: int) -> np.ndarray:
    """Read a ``baf2sql`` array as float64."""
    n = c_uint64(0)
    if not dll.baf2sql_array_get_num_elements(handle, array_id, n):
        throw_last_baf_error(dll)
    buf = np.empty(n.value, dtype=np.float64)
    if not dll.baf2sql_array_read_double(handle, array_id, buf.ctypes.data_as(POINTER(c_double))):
        throw_last_baf_error(dll)
    return buf


def get_sqlite_cache_filename(dll: CDLL, path: Path, all_variables: bool = True) -> Path:
    """Return the ``analysis.sqlite`` cache for a BAF dataset, creating it if it does not exist yet."""
    u8path = str(path / "analysis.baf").encode("utf-8")
    length = dll.baf2sql_get_sqlite_cache_filename_v2(None, 0, u8path, all_variables)
    if length == 0:
        throw_last_baf_error(dll)
    buf = create_string_buffer(length)
    dll.baf2sql_get_sqlite_cache_filename_v2(buf, length, u8path, all_variables)
    return Path(buf.value.decode("utf-8"))


class BAFReader(BrukerBaseReader):
    """Bruker .d/baf reader.

    Parameters
    ----------
    path : PathLike
        Path to the Bruker .d directory.
    use_recalibrated_state : bool, optional
        Use the recalibrated mass axis if it is available, by default True.
    mode : str, optional
        Read 'profile' or 'centroid' (line) spectra, by default 'profile'.
    roi : int | None, optional
        Region of interest to load. If None, all regions are loaded.
    """

    # column index in `_array_ids` for (m/z, intensity) arrays of each mode
    _MODE_COLUMNS: ty.ClassVar[dict[str, tuple[int, int]]] = {"centroid": (0, 1), "profile": (2, 3)}

    def __init__(
        self,
        path: PathLike,
        use_recalibrated_state: bool = True,
        mode: str = "profile",
        roi: int | None = None,
        **kwargs: ty.Any,
    ):
        if mode not in self._MODE_COLUMNS:
            raise ValueError(f"Unsupported mode '{mode}' - use one of {list(self._MODE_COLUMNS)}.")
        self.use_recalibrated_state = use_recalibrated_state
        self.mode = mode
        self.sql_filename = find_peaks_sqlite(Path(path)).name
        super().__init__(path, auto_profile=False, roi=roi)

    def _init(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        self.dll = get_baf2sql_dll()
        self._dll_close_func = self.dll.baf2sql_array_close_storage
        cache_path = self.path / "analysis.sqlite"
        if not cache_path.exists():
            cache_path = get_sqlite_cache_filename(self.dll, self.path)
        self._read_array_ids(cache_path)

        handle = self.dll.baf2sql_array_open_storage(int(self.use_recalibrated_state), str(self.path).encode("utf-8"))
        if handle == 0:
            throw_last_baf_error(self.dll)
        self.handle = handle
        super()._init()

    def _read_array_ids(self, cache_path: Path) -> None:
        """Read the array IDs of every spectrum from the ``analysis.sqlite`` cache."""
        conn = sqlite3.connect(cache_path)
        try:
            rows = conn.execute(
                "SELECT Id, LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId FROM Spectra ORDER BY Id"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            raise ValueError(f"No spectra found in {cache_path}")
        # missing arrays (e.g. no profile data) are stored as NULL
        data = np.array([[-1 if value is None else value for value in row] for row in rows], dtype=np.int64)
        self._spectrum_ids = data[:, 0]
        self._array_ids = data[:, 1:]
        self._row_for_id = {int(spectrum_id): i for i, spectrum_id in enumerate(self._spectrum_ids)}

    def _get_frame_index_position(self) -> np.ndarray:
        """Return (frame id, x index, y index, region number) from the ``peaks.sqlite`` Spectra table."""
        with self.sql_reader() as conn:
            rows = conn.execute("SELECT Id, XIndexPos, YIndexPos, RegionNumber FROM Spectra ORDER BY Id").fetchall()
        return np.array(rows, dtype=np.int64).reshape(-1, 4)

    def get_acquisition_mass_range(self) -> tuple[float, float]:
        """Retrieve acquisition mass range from the ``Properties`` table."""
        with self.sql_reader() as conn:
            rows = dict(conn.execute("SELECT Key, Value FROM Properties").fetchall())
        if "MzAcqRangeLower" not in rows or "MzAcqRangeUpper" not in rows:
            raise ValueError("Failed to extract experimental mass range from the file!")
        return float(rows["MzAcqRangeLower"]), float(rows["MzAcqRangeUpper"])

    @property
    def polarity(self) -> str:
        """Return acquisition polarity ('Positive' or 'Negative')."""
        with self.sql_reader() as conn:
            value = conn.execute("SELECT Polarity FROM AcquisitionKeys").fetchone()[0]
        if isinstance(value, int):
            return "Negative" if value == 1 else "Positive"
        return "Positive" if str(value).startswith("+") else "Negative"

    @property
    def pixel_size(self) -> float:
        """Return pixel size estimated from motor positions, or 1.0 if they are not available."""
        if self._pixel_size is None:
            from ims_utils.coordinates import get_spacing_consecutive

            try:
                with self.sql_reader() as conn:
                    positions = np.array(conn.execute("SELECT MotorPositionX, MotorPositionY FROM Spectra").fetchall())
                x_px, y_px = get_spacing_consecutive(positions[:, 0], positions[:, 1])
                self._pixel_size = float(y_px if np.isnan(x_px) else x_px)
                if np.isnan(self._pixel_size):
                    raise ValueError("Pixel size is NaN.")
            except (sqlite3.OperationalError, IndexError, ValueError) as exc:
                logger.warning(f"Could not determine BAF pixel size ({exc}) - defaulting to 1.0.")
                self._pixel_size = 1.0
        return self._pixel_size

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether centroid (line) spectra are read."""
        return self.mode == "centroid"

    @property
    def mz_x(self) -> np.ndarray:
        """Return the m/z axis of the first profile spectrum (shared by all profile spectra)."""
        if self._mz_x is None:
            self._mz_x = self.read_frame(int(self._spectrum_ids[0]))[0]
        return self._mz_x

    def read_frame(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Read (m/z, intensity) arrays for a raw Bruker spectrum ID."""
        row = self._row_for_id.get(int(frame_id))
        if row is None:
            raise IndexError(f"Spectrum {frame_id} does not exist.")
        mz_col, int_col = self._MODE_COLUMNS[self.mode]
        mz_id, int_id = int(self._array_ids[row, mz_col]), int(self._array_ids[row, int_col])
        if mz_id < 0 or int_id < 0:
            raise RuntimeError(f"Spectrum {frame_id} has no {self.mode} data.")
        if self.dll is None or self.handle is None:
            raise RuntimeError("Reader has been closed.")
        mzs = read_double(self.dll, self.handle, mz_id)
        intensities = read_double(self.dll, self.handle, int_id).astype(np.float32)
        return mzs, intensities

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return self.read_frame(self._frame_id_for_index(index))

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Read intensities for a pixel index."""
        return self._read_spectrum(index)[1]

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return TIC computed from the spectra (peaks.sqlite has no summed intensities)."""
        return BaseReader.get_tic(self, silent=silent)


def find_peaks_sqlite(path: Path) -> Path:
    """Return the ``*peaks.sqlite`` file of a BAF dataset."""
    files = sorted(path.glob("*peaks.sqlite"))
    if not files:
        raise ValueError(f"No *peaks.sqlite file found in {path}")
    if len(files) > 1:
        raise ValueError(f"Cannot determine which Bruker *peaks.sqlite file to use in {path}: {files}")
    return files[0]


def is_baf(path: PathLike) -> bool:
    """Check if path is a Bruker .d/baf imaging dataset."""
    path = Path(path)
    return (
        path.suffix.lower() == ".d"
        and path.is_dir()
        and (path / "analysis.baf").exists()
        and not (path / "analysis.tdf").exists()
        and not (path / "analysis.tsf").exists()
        and any(path.glob("*peaks.sqlite"))
    )


@hook_impl
def imzy_reader(path: PathLike, **kwargs: ty.Any) -> BAFReader | None:
    """Return BAFReader if path is Bruker .d/baf."""
    if is_baf(path):
        return BAFReader(path, **kwargs)
    return None
