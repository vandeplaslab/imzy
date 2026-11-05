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
from imzy._readers.bruker.utilities import is_tdf, peaks_to_csc, peaks_to_profile
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

    def has_recalibrated_state(self) -> bool:
        """Recalibrated states are not supported by `pytsf`."""
        return False

    def ook0_to_ccs(self, ook0: float, charge: int, mz: float) -> float:
        """Convert 1/K0 to CCS (Angstrom^2) using the Mason-Schamp equation (305 K, as Bruker)."""
        return float(oneoverk0_to_ccs(ook0, mz, charge))

    def read_frame_csc(self, frame_id: int, mz_min_idx: int = 0, mz_max_idx: int | None = None) -> sparse.csc_matrix:
        """Read raw Bruker frame as CSC matrix of shape (mz_max_idx - mz_min_idx, n_dt_bins).

        Row 0 corresponds to TOF index `mz_min_idx`; columns are mobility scans.
        """
        if mz_max_idx is None:
            mz_max_idx = self.n_mz_bins
        # frame ids must be python ints - sqlite3 binds numpy integers as blobs
        frame = self._tdf.read_frame(int(frame_id))
        data, indices, indptr = peaks_to_csc(
            frame.scan_offsets, frame.tof_indices, frame.intensities, self.n_dt_bins, mz_min_idx, mz_max_idx
        )
        return sparse.csc_matrix((data, indices, indptr), shape=(mz_max_idx - mz_min_idx, self.n_dt_bins), copy=False)

    def read_frame(self, frame_id: int) -> sparse.csr_matrix:
        """Read raw Bruker frame as CSR matrix of shape (n_mz_bins, n_dt_bins)."""
        return self.read_frame_csc(frame_id).tocsr()

    def read_scans(self, frame_id: int, scan_begin: int, scan_end: int) -> np.ndarray:
        """Read a range of scans from a frame in the vendor 'tims_read_scans_v2' buffer layout.

        The layout is the number of peaks of each scan, followed by the TOF indices and then the intensities of each
        scan in turn, all as uint32.
        """
        frame = self._tdf.read_frame(int(frame_id))
        scan_end = min(scan_end, frame.num_scans)
        offsets = frame.scan_offsets[scan_begin : scan_end + 1]
        n_scans = len(offsets) - 1
        counts = np.diff(offsets)
        start, end = int(offsets[0]), int(offsets[-1])
        local_offsets = offsets[:-1] - start

        # position of every peak's TOF index in the buffer; its intensity follows the scan's indices
        scan_of_peak = np.repeat(np.arange(n_scans), counts)
        rank = np.arange(end - start) - local_offsets[scan_of_peak]
        index_pos = n_scans + 2 * local_offsets[scan_of_peak] + rank
        buffer = np.empty(n_scans + 2 * (end - start), dtype=np.uint32)
        buffer[:n_scans] = counts
        buffer[index_pos] = frame.tof_indices[start:end]
        buffer[index_pos + counts[scan_of_peak]] = frame.intensities[start:end]
        return buffer

    def read_frame_profile(self, frame_id: int) -> np.ndarray:
        """Read raw Bruker frame summed over the mobility dimension."""
        frame = self._tdf.read_frame(int(frame_id))
        end = frame.scan_offsets[min(self.n_dt_bins, frame.scan_offsets.size - 1)]
        return peaks_to_profile(frame.tof_indices[:end], frame.intensities[:end], self.n_mz_bins)

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return profile spectrum."""
        return self.mz_x, self.read_profile_spectrum(index)

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        """Return profile spectrum."""
        return self.read_frame_profile(self._frame_id_for_index(index))

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
