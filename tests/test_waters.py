"""Tests for the Waters reader logic, using a stub of the (Windows-only) MassLynx SDK."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest

SDK_MODULES = (
    "MassLynxRawReader",
    "MassLynxRawInfoReader",
    "MassLynxRawScanReader",
    "MassLynxRawChromatogramReader",
)


class MassLynxException(Exception):
    """Stub exception."""


class FakeInfoReader:
    """Two functions; positions are floats with an offset and a missing line."""

    def __init__(self) -> None:
        # function 0: 2 x 3 raster, function 1: one more row; y=2 (0.2) is missing -> gap in the acquisition
        self.positions = {
            0: [(1.0, 0.0), (1.1, 0.0), (1.2, 0.0), (1.0, 0.1), (1.1, 0.1), (1.2, 0.1)],
            1: [(1.0, 0.3), (1.1, 0.3), (1.2, 0.3000001)],
        }
        self.mass_ranges = {0: (100.0, 1000.0), 1: (150.0, 1200.0)}

    def GetNumberofFunctions(self) -> int:
        return len(self.positions)

    def GetScansInFunction(self, fcn: int) -> int:
        return len(self.positions[fcn])

    def GetAcquisitionMassRange(self, fcn: int) -> tuple[float, float]:
        return self.mass_ranges[fcn]

    def GetIonMode(self, fcn: int) -> int:
        return 1

    def GetIonModeString(self, mode: int) -> str:
        return "ES+"

    def GetFunctionType(self, fcn: int) -> int:
        return 0

    def GetFunctionTypeString(self, fcn_type: int) -> str:
        return "MS"

    def GetScanItems(self, fcn: int, scan: int, items: list[int]) -> list[str]:
        x, y = self.positions[fcn][scan]
        return [str(x), str(y)]


class FakeChromatogramReader:
    def ReadTIC(self, fcn: int) -> tuple[list[float], list[float]]:
        times = [float(i) for i in range(3 * (fcn + 1))]
        return times, [100.0 * (fcn + 1) + t for t in times]


class FakeScanReader:
    def ReadDriftScan(self, fcn: int, scan: int, drift: int) -> tuple[np.ndarray, np.ndarray]:
        raise MassLynxException("no ion mobility")


@pytest.fixture
def raw_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import `imzy._readers.waters._raw` with the MassLynx SDK replaced by stubs."""
    for name in SDK_MODULES:
        module = types.ModuleType(f"imzy._readers.waters.{name}")
        module.MassLynxException = MassLynxException  # type: ignore[attr-defined]
        setattr(module, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.delitem(sys.modules, "imzy._readers.waters._raw", raising=False)
    module = importlib.import_module("imzy._readers.waters._raw")
    yield module
    sys.modules.pop("imzy._readers.waters._raw", None)


@pytest.fixture
def reader(raw_module: types.ModuleType, tmp_path):
    raw_dir = tmp_path / "example.raw"
    raw_dir.mkdir()
    (raw_dir / "_extern.inf").write_text("DesiXStep\t0.0001\nDesiYStep\t0.0001\n")
    cls = raw_module.WatersReader
    cls._info_reader = FakeInfoReader()
    cls._chromatogram_reader = FakeChromatogramReader()
    cls._data_reader = FakeScanReader()
    try:
        yield cls(raw_dir)
    finally:
        cls._info_reader = cls._chromatogram_reader = cls._data_reader = None


def test_reindex_coordinates(raw_module):
    reindex = raw_module.reindex_coordinates
    # positive offset must be removed, not doubled
    np.testing.assert_array_equal(reindex(np.array([1.0, 1.1, 1.2])), [0, 1, 2])
    # negative positions
    np.testing.assert_array_equal(reindex(np.array([-0.2, -0.1, 0.0])), [0, 1, 2])
    # a gap must not inflate the step
    np.testing.assert_array_equal(reindex(np.array([0.0, 0.1, 0.2, 0.5])), [0, 1, 2, 5])
    # float jitter of identical positions is ignored
    np.testing.assert_array_equal(reindex(np.array([0.0, 0.1, 0.1000001, 0.2])), [0, 1, 1, 2])
    # single position
    np.testing.assert_array_equal(reindex(np.array([3.0, 3.0])), [0, 0])


def test_functions_and_coordinates(reader):
    assert reader.n_functions == 2
    assert reader.num_scans == 9
    # each function keeps its own mass range
    assert reader.stats_in_functions[0]["mass_range"] == [100.0, 1000.0]
    assert reader.stats_in_functions[1]["mass_range"] == [150.0, 1200.0]
    assert reader.mz_min == 100.0 - reader.mz_offset
    assert reader.mz_max == 1200.0 + reader.mz_offset
    assert reader.frame_to_fcn[6] == (1, 0)
    np.testing.assert_array_equal(reader.x_coordinates, [0, 1, 2, 0, 1, 2, 0, 1, 2])
    np.testing.assert_array_equal(reader.y_coordinates, [0, 0, 0, 1, 1, 1, 3, 3, 3])
    assert reader.n_dt_bins == 1
    assert reader.x_pixel_size == pytest.approx(100.0)


def test_tic_contains_intensities_only(reader):
    tic = reader.get_tic()
    assert tic.shape == (9,)
    np.testing.assert_allclose(tic, [100.0, 101.0, 102.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0])


@pytest.mark.parametrize("enable_ion_mobility", [True, False])
def test_enable_faster_iter_restores_state(reader, enable_ion_mobility):
    reader.enable_ion_mobility = enable_ion_mobility
    reader.auto_profile = True
    with reader._enable_faster_iter():
        assert not reader.enable_ion_mobility
        assert not reader.auto_profile
    assert reader.enable_ion_mobility is enable_ion_mobility
    assert reader.auto_profile is True
