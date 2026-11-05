"""Tests for the Bruker BAF reader, using a stub of the baf2sql library and synthetic SQLite files."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from koyo.system import IS_MAC

from imzy._readers.bruker import _baf

# array id -> values, as returned by baf2sql
MZ = np.array([100.0, 101.0, 102.0, 103.0])
ARRAYS = {
    10: MZ,
    11: np.array([1.0, 2.0, 3.0, 4.0]),
    20: MZ,
    21: np.array([5.0, 6.0, 7.0, 8.0]),
    30: MZ,
    31: np.array([0.0, 0.0, 9.0, 0.0]),
    40: np.array([100.5, 102.5]),
    41: np.array([7.0, 8.0]),
}
# Id, LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId
SPECTRA = [(1, 40, 41, 10, 11), (2, 40, 41, 20, 21), (3, None, None, 30, 31)]
# Id, XIndexPos, YIndexPos, RegionNumber, MotorPositionX, MotorPositionY
POSITIONS = [(1, 5, 7, 0, 0.0, 0.0), (2, 6, 7, 0, 50.0, 0.0), (3, 5, 8, 1, 0.0, 50.0)]


class FakeBaf2Sql:
    """Minimal stand-in for the ctypes `baf2sql` library."""

    def __init__(self) -> None:
        self.closed: list[int] = []

    def baf2sql_array_open_storage(self, recalibrated: int, path: bytes) -> int:
        return 42

    def baf2sql_array_close_storage(self, handle: int) -> None:
        self.closed.append(handle)

    def baf2sql_array_get_num_elements(self, handle: int, array_id: int, n) -> int:  # type: ignore[no-untyped-def]
        n.value = ARRAYS[array_id].size
        return 1

    def baf2sql_array_read_double(self, handle: int, array_id: int, buf) -> int:  # type: ignore[no-untyped-def]
        for i, value in enumerate(ARRAYS[array_id]):
            buf[i] = value
        return 1


def _write_db(path: Path, statements: list[tuple[str, list[tuple]]]) -> None:
    conn = sqlite3.connect(path)
    for statement, rows in statements:
        if rows:
            conn.executemany(statement, rows)
        else:
            conn.execute(statement)
    conn.commit()
    conn.close()


@pytest.fixture
def fake_dll(monkeypatch: pytest.MonkeyPatch) -> FakeBaf2Sql:
    dll = FakeBaf2Sql()
    monkeypatch.setattr(_baf, "get_baf2sql_dll", lambda: dll)
    return dll


@pytest.fixture
def baf_path(tmp_path: Path) -> Path:
    path = tmp_path / "example.d"
    path.mkdir()
    (path / "analysis.baf").write_bytes(b"")
    _write_db(
        path / "analysis.sqlite",
        [
            ("CREATE TABLE Spectra (Id, LineMzId, LineIntensityId, ProfileMzId, ProfileIntensityId)", []),
            ("INSERT INTO Spectra VALUES (?, ?, ?, ?, ?)", SPECTRA),
        ],
    )
    _write_db(
        path / "example_peaks.sqlite",
        [
            ("CREATE TABLE Spectra (Id, XIndexPos, YIndexPos, RegionNumber, MotorPositionX, MotorPositionY)", []),
            ("INSERT INTO Spectra VALUES (?, ?, ?, ?, ?, ?)", POSITIONS),
            ("CREATE TABLE Properties (Key, Value)", []),
            ("INSERT INTO Properties VALUES (?, ?)", [("MzAcqRangeLower", "100.0"), ("MzAcqRangeUpper", "103.0")]),
            ("CREATE TABLE AcquisitionKeys (Polarity)", []),
            ("INSERT INTO AcquisitionKeys VALUES (?)", [(1,)]),
        ],
    )
    return path


def test_is_baf(baf_path: Path, tmp_path: Path) -> None:
    assert _baf.is_baf(baf_path)
    (baf_path / "analysis.tsf").write_bytes(b"")
    assert not _baf.is_baf(baf_path)
    assert not _baf.is_baf(tmp_path)


def test_profile(fake_dll: FakeBaf2Sql, baf_path: Path) -> None:
    reader = _baf.imzy_reader(baf_path)
    assert isinstance(reader, _baf.BAFReader)
    assert reader.n_pixels == 3
    assert reader.rois == [0, 1]
    assert (reader.mz_min, reader.mz_max) == (100.0, 103.0)
    assert not reader.is_centroid
    assert reader.polarity == "Negative"
    np.testing.assert_array_equal(reader.x_coordinates, [0, 1, 0])
    np.testing.assert_array_equal(reader.y_coordinates, [0, 0, 1])
    np.testing.assert_array_equal(reader.mz_x, MZ)

    mzs, intensities = reader.get_spectrum(1)
    np.testing.assert_array_equal(mzs, MZ)
    np.testing.assert_array_equal(intensities, ARRAYS[21])
    assert intensities.dtype == np.float32
    np.testing.assert_allclose(reader.get_tic(silent=True), [10, 26, 9])
    _, summed = reader.get_summed_spectrum([0, 1, 2], silent=True)
    np.testing.assert_allclose(summed, [6, 8, 19, 12])

    reader.close()
    assert fake_dll.closed == [42]


def test_centroid_and_roi(fake_dll: FakeBaf2Sql, baf_path: Path) -> None:
    reader = _baf.BAFReader(baf_path, mode="centroid", roi=0)
    assert reader.is_centroid
    assert reader.n_pixels == 2
    assert reader.get_n_pixels_for_roi(1) == 1
    mzs, intensities = reader.get_spectrum(0)
    np.testing.assert_array_equal(mzs, ARRAYS[40])
    np.testing.assert_array_equal(intensities, ARRAYS[41])

    # spectrum 3 has no line spectrum
    reader = _baf.BAFReader(baf_path, mode="centroid", roi=1)
    with pytest.raises(RuntimeError, match="no centroid data"):
        reader.get_spectrum(0)


def test_invalid_mode(baf_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported mode"):
        _baf.BAFReader(baf_path, mode="raw")


@pytest.mark.skipif(importlib.util.find_spec("ims_utils.coordinates") is None, reason="needs newer ims-utils")
def test_pixel_size(fake_dll: FakeBaf2Sql, baf_path: Path) -> None:
    assert _baf.BAFReader(baf_path).pixel_size == 50.0


@pytest.mark.skipif(IS_MAC, reason="baf2sql is not available for macOS")
def test_get_reader(fake_dll: FakeBaf2Sql, baf_path: Path) -> None:
    from imzy import BAFReader, get_reader

    assert isinstance(get_reader(baf_path), BAFReader)


@pytest.mark.skipif(not IS_MAC, reason="macOS only")
def test_unavailable_on_mac() -> None:
    import imzy

    assert imzy.BAFReader is None
    with pytest.raises(OSError, match="not available for macOS"):
        _baf.get_baf2sql_dll.__wrapped__()
