"""Python wrapper for timsdata.dll."""

from __future__ import annotations

import typing as ty
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
    cdll,
    create_string_buffer,
)
from enum import Enum
from pathlib import Path

import numpy as np
from koyo.system import IS_LINUX, IS_WIN
from koyo.typing import PathLike
from scipy import sparse

from imzy._readers.bruker._mixin import BrukerBaseReader
from imzy._readers.bruker.utilities import is_tdf, scan_buffer_to_csc, scan_buffer_to_profile
from imzy.hookspec import hook_impl

_base_path = Path(__file__).parent.absolute()
if IS_WIN:
    libname = "timsdata.dll"
elif IS_LINUX:
    libname = "libtimsdata.so"
else:
    raise Exception("Unsupported platform.")

dll_path = _base_path / libname
if dll_path.exists():
    DLL: CDLL = cdll.LoadLibrary(str(dll_path))
else:
    DLL: CDLL = cdll.LoadLibrary(libname)

DLL.tims_open_v2.argtypes = [c_char_p, c_uint32, c_uint32]
DLL.tims_open_v2.restype = c_uint64
DLL.tims_close.argtypes = [c_uint64]
DLL.tims_close.restype = None
DLL.tims_get_last_error_string.argtypes = [c_char_p, c_uint32]
DLL.tims_get_last_error_string.restype = c_uint32
DLL.tims_has_recalibrated_state.argtypes = [c_uint64]
DLL.tims_has_recalibrated_state.restype = c_uint32
DLL.tims_read_scans_v2.argtypes = [c_uint64, c_int64, c_uint32, c_uint32, c_void_p, c_uint32]
DLL.tims_read_scans_v2.restype = c_uint32
MSMS_SPECTRUM_FUNCTOR = CFUNCTYPE(None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float))
DLL.tims_read_pasef_msms.argtypes = [c_uint64, POINTER(c_int64), c_uint32, MSMS_SPECTRUM_FUNCTOR]
DLL.tims_read_pasef_msms.restype = c_uint32
DLL.tims_read_pasef_msms_for_frame.argtypes = [c_uint64, c_int64, MSMS_SPECTRUM_FUNCTOR]
DLL.tims_read_pasef_msms_for_frame.restype = c_uint32
MSMS_PROFILE_SPECTRUM_FUNCTOR = CFUNCTYPE(None, c_int64, c_uint32, POINTER(c_int32))
DLL.tims_read_pasef_profile_msms.argtypes = [c_uint64, POINTER(c_int64), c_uint32, MSMS_PROFILE_SPECTRUM_FUNCTOR]
DLL.tims_read_pasef_profile_msms.restype = c_uint32
DLL.tims_read_pasef_profile_msms_for_frame.argtypes = [c_uint64, c_int64, MSMS_PROFILE_SPECTRUM_FUNCTOR]
DLL.tims_read_pasef_profile_msms_for_frame.restype = c_uint32

DLL.tims_extract_centroided_spectrum_for_frame_v2.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    MSMS_SPECTRUM_FUNCTOR,
    c_void_p,
]
DLL.tims_extract_centroided_spectrum_for_frame_v2.restype = c_uint32
DLL.tims_extract_centroided_spectrum_for_frame_ext.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    c_double,
    MSMS_SPECTRUM_FUNCTOR,
    c_void_p,
]
DLL.tims_extract_centroided_spectrum_for_frame_ext.restype = c_uint32
DLL.tims_extract_profile_for_frame.argtypes = [
    c_uint64,
    c_int64,
    c_uint32,
    c_uint32,
    MSMS_PROFILE_SPECTRUM_FUNCTOR,
    c_void_p,
]
DLL.tims_extract_profile_for_frame.restype = c_uint32


class ChromatogramJob(Structure):
    _fields_: ty.ClassVar[list] = [
        ("id", c_int64),
        ("time_begin", c_double),
        ("time_end", c_double),
        ("mz_min", c_double),
        ("mz_max", c_double),
        ("ook0_min", c_double),
        ("ook0_max", c_double),
    ]


CHROMATOGRAM_JOB_GENERATOR = CFUNCTYPE(c_uint32, POINTER(ChromatogramJob), c_void_p)
CHROMATOGRAM_TRACE_SINK = CFUNCTYPE(c_uint32, c_int64, c_uint32, POINTER(c_int64), POINTER(c_uint64), c_void_p)
DLL.tims_extract_chromatograms.argtypes = [c_uint64, CHROMATOGRAM_JOB_GENERATOR, CHROMATOGRAM_TRACE_SINK, c_void_p]
DLL.tims_extract_chromatograms.restype = c_uint32

convfunc_argtypes: list = [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32]

DLL.tims_index_to_mz.argtypes = convfunc_argtypes
DLL.tims_index_to_mz.restype = c_uint32
DLL.tims_mz_to_index.argtypes = convfunc_argtypes
DLL.tims_mz_to_index.restype = c_uint32

DLL.tims_scannum_to_oneoverk0.argtypes = convfunc_argtypes
DLL.tims_scannum_to_oneoverk0.restype = c_uint32
DLL.tims_oneoverk0_to_scannum.argtypes = convfunc_argtypes
DLL.tims_oneoverk0_to_scannum.restype = c_uint32

DLL.tims_scannum_to_voltage.argtypes = convfunc_argtypes
DLL.tims_scannum_to_voltage.restype = c_uint32
DLL.tims_voltage_to_scannum.argtypes = convfunc_argtypes
DLL.tims_voltage_to_scannum.restype = c_uint32

DLL.tims_oneoverk0_to_ccs_for_mz.argtypes = [c_double, c_int32, c_double]
DLL.tims_oneoverk0_to_ccs_for_mz.restype = c_double

DLL.tims_ccs_to_oneoverk0_for_mz.argtypes = [c_double, c_int32, c_double]
DLL.tims_ccs_to_oneoverk0_for_mz.restype = c_double


def _throw_last_error(dll_handle: CDLL | None) -> None:
    """Throw last TimsData error string as an exception."""
    if dll_handle:
        n = dll_handle.tims_get_last_error_string(None, 0)
        buf = create_string_buffer(n)
        dll_handle.tims_get_last_error_string(buf, n)
        raise RuntimeError(buf.value)


def ook0_to_ccs_for_mz(ook0: float | np.ndarray, charge: int, mz: float) -> float | np.ndarray:
    """Convert 1/K0 to CCS for a given charge and mzn."""
    return DLL.tims_oneoverk0_to_ccs_for_mz(ook0, charge, mz)


def ccs_to_ook0_for_mz(ccs: float | np.ndarray, charge: int, mz: float) -> float | np.ndarray:
    """Convert CCS to 1/K0 for a given charge and mz."""
    return DLL.tims_ccs_to_oneoverk0_for_mz(ccs, charge, mz)


class PressureStrategy(Enum):
    """Pressure compensation strategy."""

    NoCompensation = 0
    GlobalCompensation = 1
    PerFrameCompensation = 2


#: arbitrary upper limit of a single frame (in bytes), same as Bruker's reference implementation
MAX_FRAME_BYTES = 16777216


class TDFReader(BrukerBaseReader):
    """Bruker .d/tdf reader using the vendor `timsdata` library."""

    sql_filename = "analysis.tdf"

    def __init__(
        self,
        path: PathLike,
        use_recalibrated_state: bool = False,
        pressure_compensation_strategy: PressureStrategy = PressureStrategy.NoCompensation,
        roi: int | None = None,
    ):
        self.use_recalibrated_state = use_recalibrated_state
        self.pressure_compensation_strategy = pressure_compensation_strategy
        # reusable scan buffer (uint32); grows in `_read_scans_into_buffer`
        self._scan_buffer = np.empty(1024, dtype=np.uint32)
        super().__init__(path, roi=roi)

    def _init(self) -> None:
        super()._init()
        self.dll = DLL
        # dll functions
        self._dll_close_func = self.dll.tims_close
        self._dll_index_to_mz_func = self.dll.tims_index_to_mz
        self._dll_mz_to_index_func = self.dll.tims_mz_to_index
        # init handle
        self.handle = self.dll.tims_open_v2(
            str(self.path).encode("utf-8"),
            1 if self.use_recalibrated_state else 0,
            self.pressure_compensation_strategy.value,
        )
        if self.handle == 0:
            _throw_last_error(self.dll)

        # data attributes
        self.n_dt_bins = self.get_n_mobility_bins()
        self.n_mz_bins = int(np.round(self.mz_to_index(1, [self.mz_max]))[0])
        self.frame_shape = (self.n_mz_bins, self.n_dt_bins)

    def _call_conversion_func(self, frame_id: int, input_data: np.ndarray, func: ty.Callable) -> np.ndarray:
        success, out = self._call_conversion_func_base(frame_id, input_data, func)
        if not success:
            _throw_last_error(self.dll)
        return out

    def scan_num_to_ook0(self, frame_id: int, scan_nums: np.ndarray) -> np.ndarray:
        """Convert scan numbers to 1/K0."""
        return self._call_conversion_func(frame_id, scan_nums, self.dll.tims_scannum_to_oneoverk0)

    def ook0_to_scan_num(self, frame_id: int, mobilities: np.ndarray) -> np.ndarray:
        """Convert 1/K0 to scan numbers."""
        return self._call_conversion_func(frame_id, mobilities, self.dll.tims_oneoverk0_to_scannum)

    def scan_num_to_voltage(self, frame_id: int, scan_nums: np.ndarray) -> np.ndarray:
        """Convert scan numbers to voltages."""
        return self._call_conversion_func(frame_id, scan_nums, self.dll.tims_scannum_to_voltage)

    def voltage_to_scan_num(self, frame_id: int, voltages: np.ndarray) -> np.ndarray:
        """Convert voltages to scan numbers."""
        return self._call_conversion_func(frame_id, voltages, self.dll.tims_voltage_to_scannum)

    def has_recalibrated_state(self) -> bool:
        """Check whether the dataset has a recalibrated state."""
        return bool(self.dll.tims_has_recalibrated_state(self.handle))

    def ook0_to_ccs(self, ook0: float, charge: int, mz: float) -> float:
        """Convert 1/K0 to CCS (Angstrom^2) using the Mason-Schamp equation."""
        return self.dll.tims_oneoverk0_to_ccs_for_mz(ook0, charge, mz)

    def read_frame_csc(self, frame_id: int, mz_min_idx: int = 0, mz_max_idx: int | None = None) -> sparse.csc_matrix:
        """Read raw Bruker frame as CSC matrix of shape (mz_max_idx - mz_min_idx, n_dt_bins).

        Row 0 corresponds to TOF index `mz_min_idx`; columns are mobility scans.
        """
        if mz_max_idx is None:
            mz_max_idx = self.n_mz_bins
        buffer = self._read_scans_into_buffer(frame_id, 0, self.n_dt_bins)
        data, indices, indptr = scan_buffer_to_csc(buffer, self.n_dt_bins, mz_min_idx, mz_max_idx)
        return sparse.csc_matrix((data, indices, indptr), shape=(mz_max_idx - mz_min_idx, self.n_dt_bins), copy=False)

    def read_frame(self, frame_id: int) -> sparse.csr_matrix:
        """Read raw Bruker frame as CSR matrix of shape (n_mz_bins, n_dt_bins)."""
        return self.read_frame_csc(frame_id).tocsr()

    def read_frame_profile(self, frame_id: int) -> np.ndarray:
        """Read raw Bruker frame summed over the mobility dimension."""
        buffer = self._read_scans_into_buffer(frame_id, 0, self.n_dt_bins)
        return scan_buffer_to_profile(buffer, self.n_dt_bins, self.n_mz_bins)

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return profile spectrum."""
        return self.mz_x, self.read_profile_spectrum(index)

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Return profile spectrum."""
        return self.read_frame_profile(self._frame_id_for_index(index))

    def read_scans(self, frame_id: int, scan_begin: int, scan_end: int) -> np.ndarray:
        """Read a range of scans from a frame in the 'tims_read_scans_v2' buffer layout (see 'timsdata.h')."""
        return self._read_scans_into_buffer(frame_id, scan_begin, scan_end).copy()

    def _read_scans_into_buffer(self, frame_id: int, scan_begin: int, scan_end: int) -> np.ndarray:
        """Read scans into the reusable buffer and return a view of it.

        The view is overwritten by the next call, so it must be consumed (or copied) straight away.
        """
        while True:
            buf = self._scan_buffer
            n_bytes = 4 * buf.size
            required_len = self.dll.tims_read_scans_v2(
                self.handle, frame_id, scan_begin, scan_end, buf.ctypes.data_as(POINTER(c_uint32)), n_bytes
            )
            if required_len == 0:
                _throw_last_error(self.dll)
            if required_len > n_bytes:
                if required_len > MAX_FRAME_BYTES:
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self._scan_buffer = np.empty(required_len // 4 + 1, dtype=np.uint32)  # grow buffer
            else:
                break
        return buf[: required_len // 4]

    def read_pasef_msms(self, precursor_list: ty.Sequence[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Read peak-picked MS/MS spectra for a list of precursors -> {precursor_id: (mz_values, area_values)}."""
        precursors_for_dll = np.array(precursor_list, dtype=np.int64)
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_peaks, mz_values, area_values):  # type: ignore[no-untyped-def]
            result[precursor_id] = (np.array(mz_values[0:num_peaks]), np.array(area_values[0:num_peaks]))

        rc = self.dll.tims_read_pasef_msms(
            self.handle, precursors_for_dll.ctypes.data_as(POINTER(c_int64)), len(precursor_list), callback_for_dll
        )
        if rc == 0:
            _throw_last_error(self.dll)
        return result

    def read_pasef_msms_for_frame(self, frame_id: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Read peak-picked MS/MS spectra for a frame -> {precursor_id: (mz_values, area_values)}."""
        result: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        @MSMS_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_peaks, mz_values, area_values):  # type: ignore[no-untyped-def]
            result[precursor_id] = (np.array(mz_values[0:num_peaks]), np.array(area_values[0:num_peaks]))

        rc = self.dll.tims_read_pasef_msms_for_frame(self.handle, frame_id, callback_for_dll)
        if rc == 0:
            _throw_last_error(self.dll)
        return result

    def read_pasef_profile_msms(self, precursor_list: ty.Sequence[int]) -> dict[int, np.ndarray]:
        """Read "quasi profile" MS/MS spectra for a list of precursors -> {precursor_id: intensity_values}."""
        precursors_for_dll = np.array(precursor_list, dtype=np.int64)
        result: dict[int, np.ndarray] = {}

        @MSMS_PROFILE_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_points, intensity_values):  # type: ignore[no-untyped-def]
            result[precursor_id] = np.array(intensity_values[0:num_points])

        rc = self.dll.tims_read_pasef_profile_msms(
            self.handle, precursors_for_dll.ctypes.data_as(POINTER(c_int64)), len(precursor_list), callback_for_dll
        )
        if rc == 0:
            _throw_last_error(self.dll)
        return result

    def read_pasef_profile_msms_for_frame(self, frame_id: int) -> dict[int, np.ndarray]:
        """Read "quasi profile" MS/MS spectra for a frame -> {precursor_id: intensity_values}."""
        result: dict[int, np.ndarray] = {}

        @MSMS_PROFILE_SPECTRUM_FUNCTOR
        def callback_for_dll(precursor_id, num_points, intensity_values):  # type: ignore[no-untyped-def]
            result[precursor_id] = np.array(intensity_values[0:num_points])

        rc = self.dll.tims_read_pasef_profile_msms_for_frame(self.handle, frame_id, callback_for_dll)
        if rc == 0:
            _throw_last_error(self.dll)
        return result

    def get_n_mobility_bins(self, quick: bool = True) -> int:
        """Get the number of ion mobility bins in the file."""
        with self.sql_reader() as conn:
            tims_db_cursor = conn.cursor()
            tims_db_cursor.execute("SELECT Distinct(NumScans) FROM Frames")
            if quick:
                n_dt_bins = tims_db_cursor.fetchone()[0]
            else:
                n_dt_bins = tims_db_cursor.fetchall()
                if len(n_dt_bins) > 1:
                    raise ValueError("Number of mobility bins is not consistent amongst all frames")
                n_dt_bins = int(n_dt_bins[0][0])
        return n_dt_bins


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> TDFReader | None:
    """Return TDFReader if path is Bruker .d/tdf."""
    if is_tdf(path):
        return TDFReader(path, **kwargs)
    return None
