"""Tests for Bruker ROI metadata handling."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_bruker_base_reader():
    """Load the Bruker mixin without importing Bruker DLL-backed modules."""
    module_path = Path(__file__).parents[1] / "src" / "imzy" / "_readers" / "bruker" / "_mixin.py"
    spec = importlib.util.spec_from_file_location("_imzy_test_bruker_mixin", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Bruker mixin module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.BrukerBaseReader


BrukerBaseReader = _load_bruker_base_reader()


class DummyBrukerReader(BrukerBaseReader):
    """Minimal Bruker reader that only exercises metadata and frame mapping."""

    sql_filename = "analysis.tsf"

    def read_profile_spectrum(self, index: int) -> np.ndarray:
        return np.asarray([self._frame_id_for_index(index)])

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray([100.0]), self.read_profile_spectrum(index)

    def _call_conversion_func(self, frame_id: int, input_data: np.ndarray, func) -> np.ndarray:
        return np.asarray(input_data)


@pytest.fixture()
def bruker_path(tmp_path: Path) -> Path:
    db_path = tmp_path / DummyBrukerReader.sql_filename
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT)")
    conn.executemany(
        "INSERT INTO GlobalMetadata (Key, Value) VALUES (?, ?)",
        [("MzAcqRangeLower", "100"), ("MzAcqRangeUpper", "1000")],
    )
    conn.execute(
        "CREATE TABLE MaldiFrameInfo (Frame INTEGER, XIndexPos INTEGER, YIndexPos INTEGER, RegionNumber INTEGER)"
    )
    conn.executemany(
        "INSERT INTO MaldiFrameInfo (Frame, XIndexPos, YIndexPos, RegionNumber) VALUES (?, ?, ?, ?)",
        [
            (1, 10, 20, 0),
            (2, 11, 20, 0),
            (3, 100, 50, 2),
            (4, 101, 50, 2),
            (5, 200, 80, 4),
            (6, 100, 51, 2),
        ],
    )
    conn.execute("CREATE TABLE Frames (Id INTEGER, SummedIntensities INTEGER)")
    conn.executemany(
        "INSERT INTO Frames (Id, SummedIntensities) VALUES (?, ?)",
        [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)],
    )
    conn.commit()
    conn.close()
    return tmp_path


def test_bruker_roi_crops_spatial_metadata_and_pixel_indices(bruker_path: Path):
    reader = DummyBrukerReader(bruker_path, roi=2)

    assert reader.rois == [0, 2, 4]
    assert reader.region_number.tolist() == [2, 2, 2]
    assert reader.n_pixels == 3
    assert reader.pixels.tolist() == [0, 1, 2]
    assert reader.frame_indices.tolist() == [3, 4, 6]
    assert reader.get_pixels_for_roi(2).tolist() == [0, 1, 2]
    assert reader.get_pixels_for_roi(0).tolist() == []
    assert reader._get_reader_kwargs() == {"roi": 2}
    np.testing.assert_array_equal(
        reader.xyz_coordinates,
        np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
    )
    assert reader.image_shape == (2, 2)
    assert reader.x_min == 0
    assert reader.y_min == 0
    assert reader.x_min_raw == 100
    assert reader.y_min_raw == 50
    assert reader.get_n_pixels_for_roi(0) == 2
    assert reader.get_n_pixels_for_roi(2) == 3


def test_bruker_roi_maps_logical_pixels_to_raw_frame_ids(bruker_path: Path):
    reader = DummyBrukerReader(bruker_path, roi=2)

    assert reader.read_profile_spectrum(0).item() == 3
    assert reader.read_profile_spectrum(1).item() == 4
    assert reader.read_profile_spectrum(2).item() == 6
    with pytest.raises(IndexError, match="out of bounds"):
        reader.read_profile_spectrum(3)


def test_bruker_roi_crops_tic_but_reuses_full_cache(bruker_path: Path):
    reader = DummyBrukerReader(bruker_path, roi=2)
    np.testing.assert_array_equal(reader.get_tic(), np.asarray([30, 40, 60]))

    reader = DummyBrukerReader(bruker_path, roi=4)
    np.testing.assert_array_equal(reader.get_tic(), np.asarray([50]))

    reader = DummyBrukerReader(bruker_path)
    np.testing.assert_array_equal(reader.get_tic(), np.asarray([10, 20, 30, 40, 50, 60]))
    assert reader.get_pixels_for_roi(2).tolist() == [2, 3, 5]
    assert reader._get_reader_kwargs() == {}


def test_bruker_roi_rejects_unknown_roi(bruker_path: Path):
    with pytest.raises(ValueError, match="ROI 99 was not found"):
        DummyBrukerReader(bruker_path, roi=99)
