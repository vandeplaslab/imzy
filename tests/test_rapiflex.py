"""Tests for the Bruker rapifleX reader using a synthetic dataset."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from imzy import RapiflexReader, get_reader
from imzy._readers.bruker._rapiflex import is_rapiflex

SPECTRA = {
    0: np.array([1, 2, 3, 4, 5], dtype=np.float32),
    2: np.array([10, 20, 30, 40, 50], dtype=np.float32),
    4: np.array([2, 0, 0, 0, 8], dtype=np.float32),
}
WIDTH, HEIGHT, N_DATAPOINTS = 3, 2, 5


def _write_dataset(data_dir: Path, truncate: int = 0, info_extra: tuple[str, ...] = ()) -> Path:
    data_dir.mkdir()
    header = struct.pack("<12I", 48, 0, 0, 0, WIDTH, HEIGHT, N_DATAPOINTS, 0, 0, 0, 0, 0)
    data_start = 48 + WIDTH * HEIGHT * 4
    offsets = np.zeros(WIDTH * HEIGHT, dtype=np.uint32)
    current_offset = data_start
    for raster_idx in sorted(SPECTRA):
        offsets[raster_idx] = current_offset
        current_offset += SPECTRA[raster_idx].nbytes

    payload = header + offsets.tobytes() + b"".join(SPECTRA[i].tobytes() for i in sorted(SPECTRA))
    (data_dir / "synthetic.dat").write_bytes(payload[: len(payload) - truncate])
    (data_dir / "synthetic_info.txt").write_text(
        "\n".join(["Mass Start: 100", "Mass End: 104", "Polarity: +", "Sample Name: synthetic", *info_extra]),
        encoding="utf-8",
    )
    (data_dir / "synthetic_poslog.txt").write_text(
        "\n".join(
            [
                "# comment",
                "0 0 __ 0.0 0.0 0.0",
                "0 0 R0X0Y0 0.0 0.0 0.0",
                "0 0 R1X2Y0 40.0 0.0 0.0",
                "0 0 R1X1Y1 20.0 20.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture
def rapiflex_path(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path / "synthetic_rapiflex", info_extra=("Raster: 20,25",))


def test_metadata_and_axes(rapiflex_path: Path) -> None:
    assert is_rapiflex(rapiflex_path)
    with RapiflexReader(rapiflex_path) as reader:
        assert reader.n_pixels == 3
        assert reader.rois == [0, 1]
        assert reader.image_shape == (2, 3)
        assert not reader.is_centroid
        assert (reader.mz_min, reader.mz_max) == (100.0, 104.0)
        np.testing.assert_allclose(reader.mz_x, [100, 101, 102, 103, 104])
        assert reader.x_pixel_size == 20
        assert reader.y_pixel_size == 25
        assert reader.polarity == "Positive"
        np.testing.assert_array_equal(reader.x_coordinates, [0, 2, 1])
        np.testing.assert_array_equal(reader.y_coordinates, [0, 0, 1])
        np.testing.assert_array_equal(reader.valid_raster_indices, [0, 2, 4])
        assert reader.info_metadata["Sample Name"] == "synthetic"


def test_spectra(rapiflex_path: Path) -> None:
    reader = RapiflexReader(rapiflex_path)
    for index, raster_idx in enumerate(sorted(SPECTRA)):
        mzs, intensities = reader.get_spectrum(index)
        np.testing.assert_allclose(mzs, reader.mz_x)
        np.testing.assert_array_equal(intensities, SPECTRA[raster_idx])
    # empty raster cell
    np.testing.assert_array_equal(reader.read_raster_spectrum(1), np.zeros(N_DATAPOINTS))
    np.testing.assert_allclose(reader.get_tic(silent=True), [15, 150, 10])
    _, summed = reader.get_summed_spectrum([0, 1, 2], silent=True)
    np.testing.assert_allclose(summed, [13, 22, 33, 44, 63])
    image = reader.get_ion_image(100.5, tol=0.6, fill_value=0, silent=True)
    np.testing.assert_array_equal(image, [[3, 0, 30], [0, 2, 0]])
    with pytest.raises(IndexError):
        reader.get_spectrum(3)


def test_roi(rapiflex_path: Path) -> None:
    reader = RapiflexReader(rapiflex_path, roi=1)
    assert reader.n_pixels == 2
    assert reader.get_n_pixels_for_roi(0) == 1
    assert reader.get_n_pixels_for_roi(1) == 2
    np.testing.assert_array_equal(reader.x_coordinates, [1, 0])
    np.testing.assert_array_equal(reader.y_coordinates, [0, 1])
    np.testing.assert_array_equal(reader.get_spectrum(0)[1], SPECTRA[2])
    assert reader._get_reader_kwargs() == {"roi": 1}
    with pytest.raises(ValueError, match="ROI 5 was not found"):
        RapiflexReader(rapiflex_path, roi=5)


def test_get_reader(rapiflex_path: Path) -> None:
    assert isinstance(get_reader(rapiflex_path), RapiflexReader)
    assert isinstance(get_reader(rapiflex_path, roi=0), RapiflexReader)


def test_truncated_file(tmp_path: Path) -> None:
    data_dir = _write_dataset(tmp_path / "truncated", truncate=4)
    with pytest.raises(ValueError, match="truncated: spectrum at raster position x=1, y=1"):
        RapiflexReader(data_dir)


def test_missing_pixel_size(tmp_path: Path) -> None:
    reader = RapiflexReader(_write_dataset(tmp_path / "no_raster"))
    assert reader.x_pixel_size == 1.0
    assert reader.y_pixel_size == 1.0


def test_not_rapiflex(tmp_path: Path) -> None:
    assert not is_rapiflex(tmp_path / "missing")
    (tmp_path / "only.dat").write_bytes(b"")
    assert not is_rapiflex(tmp_path)
