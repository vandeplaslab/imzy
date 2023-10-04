"""Python wrapper for timsdata.dll."""
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
from imzy._readers.bruker.utilities import get_sparse_data_from_buffer
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
    _fields_: ty.ClassVar[ty.List] = [
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

convfunc_argtypes: ty.List = [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32]

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


def _throw_last_error(dll_handle: ty.Optional[CDLL]) -> None:
    """Throw last TimsData error string as an exception."""
    if dll_handle:
        n = dll_handle.tims_get_last_error_string(None, 0)
        buf = create_string_buffer(n)
        dll_handle.tims_get_last_error_string(buf, n)
        raise RuntimeError(buf.value)


def ook0_to_ccs_for_mz(ook0: ty.Union[float, np.ndarray], charge: int, mz: float) -> ty.Union[float, np.ndarray]:
    """Convert 1/K0 to CCS for a given charge and mzn."""
    return DLL.tims_oneoverk0_to_ccs_for_mz(ook0, charge, mz)


def ccs_to_ook0_for_mz(ccs: ty.Union[float, np.ndarray], charge: int, mz: float) -> ty.Union[float, np.ndarray]:
    """Convert CCS to 1/K0 for a given charge and mz."""
    return DLL.tims_ccs_to_oneoverk0_for_mz(ccs, charge, mz)


class PressureStrategy(Enum):
    """Pressure compensation strategy."""

    NoCompensation = 0
    GlobalCompensation = 1
    PerFrameCompensation = 2


class TDFReader(BrukerBaseReader):
    sql_filename = "analysis.tdf"

    def __init__(
        self,
        path: PathLike,
        use_recalibrated_state: bool = False,
        pressure_compensation_strategy: PressureStrategy = PressureStrategy.NoCompensation,
    ):
        self.use_recalibrated_state = use_recalibrated_state
        self.pressure_compensation_strategy = pressure_compensation_strategy
        self.initial_frame_buffer_size = 1024  # may grow in _read_scans()
        super().__init__(path)

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

    # noinspection PyMissingOrEmptyDocstring
    def read_frame(self, frame_id: int) -> sparse.csr_matrix:
        buffer = self._read_scan_buffer(frame_id, 0, self.n_dt_bins)

        # get COO matrix data
        data, rows, cols = get_sparse_data_from_buffer(buffer, 0, self.n_mz_bins, 0, self.n_dt_bins)

        # build COO matrix
        out_arr = sparse.coo_matrix((data, (rows, cols)), shape=self.frame_shape, dtype=np.int32)
        del data, rows, cols

        # convert to appropriate format and apply m/z restrictions
        return out_arr.asformat("csr")

    def _read_spectrum(self, index: int) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Return profile spectrum."""
        return self.mz_x, self.read_profile_spectrum(index)

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Return profile spectrum."""
        return self.read_frame(index + 1).sum(axis=1).A.flatten()

    def _read_scan_buffer(self, index: int, scan_begin: int, scan_end: int) -> np.ndarray:
        """Read a range of scans from a frame.

        Returning the data in the low-level buffer format defined for the 'tims_read_scans_v2' DLL function
         (see documentation in 'timsdata.h').
        """
        # buffer-growing loop
        while True:
            cnt = int(self.initial_frame_buffer_size)  # necessary cast to run with python 3.5
            buf = np.empty(shape=cnt, dtype=np.uint32)
            n = 4 * cnt

            required_len = self.dll.tims_read_scans_v2(
                self.handle, index, scan_begin, scan_end, buf.ctypes.data_as(POINTER(c_uint32)), n
            )
            if required_len == 0:
                _throw_last_error(self.dll)

            if required_len > n:
                if required_len > 16777216:
                    # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.initial_frame_buffer_size = required_len / 4 + 1  # grow buffer
            else:
                break
        return buf

    # def _read_scans(self, index, scan_begin, scan_end):
    #     """Read a range of scans from a frame, returning a list of scans.

    #     Each scan being represented as a tuple (index_array, intensity_array).

    #     """
    #     buf = self._read_scan_buffer(index, scan_begin, scan_end)

    #     result = []
    #     d = scan_end - scan_begin
    #     for i in range(scan_begin, scan_end):
    #         n_peaks = buf[i - scan_begin]
    #         indices = buf[d : d + n_peaks]
    #         d += n_peaks
    #         intensities = buf[d : d + n_peaks]
    #         d += n_peaks
    #         result.append((indices, intensities))
    #     return result

    # # read peak-picked spectra for a tims frame;
    # # returns a pair of arrays (mz_values, area_values).
    # def extract_centroid_spectrum_for_frame(self, index, scan_begin, scan_end, peak_picker_resolution=None):
    #     result = None

    #     @MSMS_SPECTRUM_FUNCTOR
    #     def callback_for_dll(precursor_id, num_peaks, mz_values, area_values):
    #         nonlocal result
    #         result = (mz_values[0:num_peaks], area_values[0:num_peaks])

    #     if peak_picker_resolution is None:
    #         rc = self.dll.tims_extract_centroided_spectrum_for_frame_v2(
    #             self.handle, index, scan_begin, scan_end, callback_for_dll, None
    #         )  # python dos not need the additional context, we have nonlocal
    #     else:
    #         rc = self.dll.tims_extract_centroided_spectrum_for_frame_ext(
    #             self.handle, index, scan_begin, scan_end, peak_picker_resolution, callback_for_dll, None
    #         )  # python dos not need the additional context, we have nonlocal

    #     if rc == 0:
    #         _throw_last_error(self.dll)
    #     return result

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


def is_tdf(path: PathLike) -> bool:
    """Check if path is Bruker .d/tdf."""
    from koyo.system import IS_MAC

    path = Path(path)
    return (
        path.suffix.lower() == ".d"
        and (path / "analysis.tdf").exists()
        and (path / "analysis.tdf_bin").exists()
        and not IS_MAC
    )


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> ty.Optional[TDFReader]:
    """Return TDFReader if path is Bruker .d/tdf."""
    if is_tdf(path):
        return TDFReader(path, **kwargs)
    return None
