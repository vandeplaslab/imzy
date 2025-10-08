"""NeoFlex reader for Bruker files."""

import sqlite3
import typing as ty
from pathlib import Path

import numpy as np
from ims_utils.profile import centroid_to_profile
from koyo.typing import PathLike

from imzy._readers.bruker._tsf import TSFReader
from imzy.hookspec import hook_impl


class NeoFlexReader(TSFReader):
    """NeoFlex reader for Bruker files."""

    def __init__(
        self, path: PathLike, use_recalibrated_state: bool = False, auto_profile: bool = True, resolution: int = 30_000
    ):
        super().__init__(path, use_recalibrated_state)
        self.auto_profile = auto_profile
        self.resolution = resolution
        self._mz_grid: np.ndarray | None = None

    @property
    def mz_index(self) -> np.ndarray:
        """Get m/z index."""
        if self.is_centroid:
            min_index = self.mz_to_index(1, [self.mz_min])[0]
            max_index = self.mz_to_index(1, [self.mz_max])[0]
            mz_index = np.arange(0, int(np.round(max_index)))[int(np.round(min_index)) :]
            return mz_index
        bruker_mz_max = self.read_profile_spectrum(1).shape[0]
        return np.arange(0, bruker_mz_max)

    def _read_spectrum(self, frame_id: int, scan_begin: int = 0, scan_end: int = -1) -> tuple[np.ndarray, np.ndarray]:
        """Read scan data."""
        if self.is_centroid:
            x, y = self.read_centroid_spectrum(frame_id)
            if self.auto_profile:
                mzs, intensities, _ = centroid_to_profile(x, y, resolving_power=self.resolution, mz_grid=self.mz_x)
                return mzs, intensities
            return x, y
        return self.mz_x, self.read_profile_spectrum(frame_id)


def is_neoflex(path: PathLike) -> bool:
    """Check if path is Bruker .d/tsf."""
    from koyo.system import IS_MAC

    path = Path(path)
    return (
        path.suffix.lower() == ".d"
        and (path / "analysis.tsf").exists()
        and (path / "analysis.tsf_bin").exists()
        and not IS_MAC
        and _is_neoflex_instrument(path)
    )


def _is_neoflex_instrument(path: Path) -> bool:
    """Check if the instrument is neofleX."""
    conn = sqlite3.connect(path / "analysis.tsf", check_same_thread=False)
    cursor = conn.execute("SELECT Key, Value FROM GlobalMetadata WHERE Key='InstrumentName'")
    key, value = cursor.fetchone()
    is_neo = key == "InstrumentName" and "neoflex" in value.lower()
    conn.close()
    return is_neo


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> ty.Optional[NeoFlexReader]:
    """Return TDFReader if path is Bruker .d/tdf."""
    if is_neoflex(path):
        return NeoFlexReader(path, **kwargs)
    return None
