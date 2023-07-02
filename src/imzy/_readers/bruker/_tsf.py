"""Python wrapper for timsdata.dll for reading tsf."""
from ctypes import (
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_uint32,
    c_uint64,
    cdll,
    create_string_buffer,
)
from pathlib import Path

import numpy as np
from koyo.system import IS_LINUX, IS_WIN
from koyo.typing import PathLike

from imzy._readers.bruker._mixin import BrukerBaseReader

_base_path = Path(__file__).parent.absolute()
if IS_WIN:
    libname = "timsdata.dll"
elif IS_LINUX:
    libname = "libtimsdata.so"
else:
    raise Exception("Unsupported platform.")

dll_path = _base_path / libname
if dll_path.exists():
    dll = cdll.LoadLibrary(str(dll_path))
else:
    dll = cdll.LoadLibrary(libname)

dll.tsf_open.argtypes = [c_char_p, c_uint32]
dll.tsf_open.restype = c_uint64
dll.tsf_close.argtypes = [c_uint64]
dll.tsf_close.restype = None
dll.tsf_get_last_error_string.argtypes = [c_char_p, c_uint32]
dll.tsf_get_last_error_string.restype = c_uint32
dll.tsf_has_recalibrated_state.argtypes = [c_uint64]
dll.tsf_has_recalibrated_state.restype = c_uint32
dll.tsf_read_line_spectrum_v2.argtypes = [c_uint64, c_int64, POINTER(c_double), POINTER(c_float), c_int32]
dll.tsf_read_line_spectrum_v2.restype = c_int32
dll.tsf_read_line_spectrum_with_width_v2.argtypes = [
    c_uint64,
    c_int64,
    POINTER(c_double),
    POINTER(c_float),
    POINTER(c_float),
    c_int32,
]
dll.tsf_read_line_spectrum_with_width_v2.restype = c_int32
dll.tsf_read_profile_spectrum_v2.argtypes = [c_uint64, c_int64, POINTER(c_uint32), c_int32]
dll.tsf_read_profile_spectrum_v2.restype = c_int32

convfunc_argtypes = [c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32]

dll.tsf_index_to_mz.argtypes = convfunc_argtypes
dll.tsf_index_to_mz.restype = c_uint32
dll.tsf_mz_to_index.argtypes = convfunc_argtypes
dll.tsf_mz_to_index.restype = c_uint32


def _throw_last_error(dll_handle):
    """Throw last TimsData error string as an exception."""
    n = dll_handle.tsf_get_last_error_string(None, 0)
    buf = create_string_buffer(n)
    dll_handle.tsf_get_last_error_string(buf, n)
    raise RuntimeError(buf.value)


class TSFReader(BrukerBaseReader):
    sql_filename = "analysis.tsf"

    def __init__(self, path: PathLike, use_recalibrated_state=False):
        self.use_recalibrated_state = use_recalibrated_state
        super().__init__(path)

    def _init(self):
        super()._init()
        self.dll = dll
        # dll functions
        self._dll_close_func = dll.tsf_close
        self._dll_index_to_mz_func = self.dll.tsf_index_to_mz
        self._dll_mz_to_index_func = self.dll.tsf_mz_to_index
        # init handle
        self.handle = self.dll.tsf_open(str(self.path).encode("utf-8"), 1 if self.use_recalibrated_state else 0)
        if self.handle == 0:
            _throw_last_error(self.dll)

        self.line_buffer_size = 1024  # may grow in read...Spectrum()
        self.profile_buffer_size = 1024  # may grow in read...Spectrum()

        # data attributes
        self.n_mz_bins = int(np.round(self.mz_to_index(1, [self.mz_max])))

    def _call_conversion_func(self, index, input_data, func):
        success, out = self._call_conversion_func_base(index, input_data, func)
        if not success:
            _throw_last_error(self.dll)
        return out

    # Output: tuple of lists (indices, intensities)
    def read_centroid_spectrum(self, index):
        """Read centroid spectrum."""
        # buffer-growing loop
        while True:
            cnt = int(self.profile_buffer_size)  # necessary cast to run with python 3.5
            index_buf = np.empty(shape=cnt, dtype=np.float64)
            intensity_buf = np.empty(shape=cnt, dtype=np.float32)

            required_len = self.dll.tsf_read_line_spectrum_v2(
                self.handle,
                index,
                index_buf.ctypes.data_as(POINTER(c_double)),
                intensity_buf.ctypes.data_as(POINTER(c_float)),
                self.profile_buffer_size,
            )

            if required_len < 0:
                _throw_last_error(self.dll)

            if required_len > self.profile_buffer_size:
                if required_len > 16777216:
                    # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.line_buffer_size = required_len  # grow buffer
            else:
                break
        return index_buf[0:required_len], intensity_buf[0:required_len]

    def _read_spectrum(self, index: int):
        return self.mz_x, self.read_profile_spectrum(index)

    # Output intensities
    def read_profile_spectrum(self, index: int):
        """Read profile spectrum."""
        # buffer-growing loop
        while True:
            cnt = int(self.profile_buffer_size)  # necessary cast to run with python 3.5
            intensity_buf = np.empty(shape=cnt, dtype=np.uint32)

            required_len = self.dll.tsf_read_profile_spectrum_v2(
                self.handle, index + 1, intensity_buf.ctypes.data_as(POINTER(c_uint32)), self.profile_buffer_size
            )

            if required_len < 0:
                _throw_last_error(self.dll)

            if required_len > self.profile_buffer_size:
                if required_len > 16777216:
                    # arbitrary limit for now...
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.profile_buffer_size = required_len  # grow buffer
            else:
                break

        return intensity_buf[0:required_len]
