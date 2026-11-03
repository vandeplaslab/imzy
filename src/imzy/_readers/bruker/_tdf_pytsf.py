"""TDF reader backed by `pytsf`, used on macOS where Bruker ships no timsdata library.

Differences from the vendor-backed reader:
- intensities are returned as stored in the file (the vendor library applies an undocumented per-file factor)
- 1/K0 is calibrated by `pytsf` from reference ions when present (~0.3%), otherwise from the acquisition range
- scan number <-> voltage conversion is not available
"""

from __future__ import annotations

import typing as ty
import warnings
from enum import Enum

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from pytsf import MobilityCalibration, TdfData, ccs_to_oneoverk0, oneoverk0_to_ccs
from scipy import sparse

from imzy._readers.bruker._mixin import BrukerBaseReader
from imzy._readers.bruker._tsf_pytsf import forward_pytsf_warnings, log_ignored_recalibration
from imzy._readers.bruker.utilities import is_tdf
from imzy.hookspec import hook_impl


def ook0_to_ccs_for_mz(ook0: float | np.ndarray, charge: int, mz: float) -> float | np.ndarray:
    """Convert 1/K0 to CCS for a given charge and m/z."""
    return oneoverk0_to_ccs(ook0, mz, charge)


def ccs_to_ook0_for_mz(ccs: float | np.ndarray, charge: int, mz: float) -> float | np.ndarray:
    """Convert CCS to 1/K0 for a given charge and m/z."""
    return ccs_to_oneoverk0(ccs, mz, charge)


class PressureStrategy(Enum):
    """Pressure compensation strategy."""

    NoCompensation = 0
    GlobalCompensation = 1
    PerFrameCompensation = 2


class TDFReader(BrukerBaseReader):
    """Bruker .d/tdf reader using the pure-Python `pytsf` backend."""

    sql_filename = "analysis.tdf"
    _tdf: TdfData | None = None
    _mobility: MobilityCalibration | None = None

    def __init__(
        self,
        path: PathLike,
        use_recalibrated_state: bool = False,
        pressure_compensation_strategy: PressureStrategy = PressureStrategy.NoCompensation,
        roi: int | None = None,
    ):
        self.use_recalibrated_state = use_recalibrated_state
        # only affects mobility conversion in the vendor library; not supported by pytsf
        self.pressure_compensation_strategy = pressure_compensation_strategy
        super().__init__(path, roi=roi)

    def _init(self) -> None:
        super()._init()
        log_ignored_recalibration(self.use_recalibrated_state)
        # mobility is calibrated lazily (see `mobility`), so the placeholder calibration warning is not relevant
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._tdf = TdfData(self.path, mobility="acquisition_range")
        # full-range m/z axes extrapolate past the calibrants by design, same as the vendor library
        self._tdf.calibration.warn_outside_range = False

        # data attributes
        self.n_dt_bins = self.get_n_mobility_bins()
        self.n_mz_bins = int(np.round(self.mz_to_index(1, [self.mz_max]))[0])
        self.frame_shape = (self.n_mz_bins, self.n_dt_bins)

    def close(self) -> None:
        """Close file."""
        if self._tdf is not None:
            self._tdf.close()
            self._tdf = None

    @property
    def mobility(self) -> MobilityCalibration:
        """Mobility calibration, preferring reference ions recorded in the file."""
        if self._mobility is None:
            with forward_pytsf_warnings(), TdfData(self.path, mobility="auto") as tdf:
                self._mobility = tdf.mobility
            self._mobility.warn_outside_range = False
            logger.info(f"pytsf mobility calibration: {self._mobility.method} (1/K0 is approximate on macOS)")
        return self._mobility

    def index_to_mz(self, frame_id: int, indices: ty.Iterable[int]) -> np.ndarray:
        """Convert m/z index to m/z."""
        return self._tdf.index_to_mz(np.asarray(indices, dtype=np.float64))

    def mz_to_index(self, frame_id: int, mzs: ty.Iterable[float]) -> np.ndarray:
        """Convert m/z to index."""
        return self._tdf.mz_to_index(np.asarray(mzs, dtype=np.float64))

    def scan_num_to_ook0(self, frame_id: int, scan_nums: np.ndarray) -> np.ndarray:
        """Convert scan numbers to 1/K0."""
        return self.mobility.scan_to_oneoverk0(np.asarray(scan_nums, dtype=np.float64))

    def ook0_to_scan_num(self, frame_id: int, mobilities: np.ndarray) -> np.ndarray:
        """Convert 1/K0 to scan numbers."""
        return self.mobility.oneoverk0_to_scan(np.asarray(mobilities, dtype=np.float64))

    def scan_num_to_voltage(self, frame_id: int, scan_nums: np.ndarray) -> np.ndarray:
        """Convert scan numbers to voltages."""
        raise NotImplementedError("Scan number to voltage conversion is not available with the pytsf backend.")

    def voltage_to_scan_num(self, frame_id: int, voltages: np.ndarray) -> np.ndarray:
        """Convert voltages to scan numbers."""
        raise NotImplementedError("Voltage to scan number conversion is not available with the pytsf backend.")

    def read_frame(self, frame_id: int) -> sparse.csr_matrix:
        """Read frame as sparse (m/z bins, mobility bins) matrix."""
        frame = self._tdf.read_frame(int(frame_id))
        rows, cols, data = frame.tof_indices, frame.scan, frame.intensities
        mask = (rows < self.n_mz_bins) & (cols < self.n_dt_bins)
        out_arr = sparse.coo_matrix((data[mask], (rows[mask], cols[mask])), shape=self.frame_shape, dtype=np.int32)
        return out_arr.asformat("csr")

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return profile spectrum."""
        return self.mz_x, self.read_profile_spectrum(index)

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Return profile spectrum."""
        return self.read_frame(self._frame_id_for_index(index)).sum(axis=1).A.flatten()

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
