"""Tests for the PHI SmartSoft-TOF (ToF-SIMS) reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imzy import PhiReader, get_reader
from imzy._readers.phi import is_phi

from .utilities import PHI_SLOPE, SyntheticPhi, write_synthetic_phi


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticPhi:
    return write_synthetic_phi(tmp_path_factory.mktemp("phi"))


@pytest.fixture(scope="module")
def reader(synthetic: SyntheticPhi) -> PhiReader:
    return PhiReader(synthetic.path)


def _expected_spectrum(data: SyntheticPhi, index: int) -> tuple[np.ndarray, np.ndarray]:
    counts = data.dense().get(index, {})
    channels = np.asarray(sorted(counts), dtype=np.int64)
    return data.mz(channels), np.asarray([counts[c] for c in channels], dtype=np.float64)


def _dense_matrix(data: SyntheticPhi, mz_axis: np.ndarray) -> np.ndarray:
    """(n_pixels, n_channels) ion counts."""
    dense = np.zeros((data.nx * data.ny, len(mz_axis)))
    for pixel, counts in data.dense().items():
        for channel, count in counts.items():
            dense[pixel, channel] = count
    return dense


# --- detection --------------------------------------------------------------------------------------------------------
def test_is_phi(synthetic: SyntheticPhi, tmp_path: Path) -> None:
    assert is_phi(synthetic.path)
    other = tmp_path / "other.raw"
    other.write_bytes(b"NOPE" + b"\x00" * 32)
    assert not is_phi(other)
    waters = tmp_path / "waters.raw"
    waters.mkdir()
    assert not is_phi(waters)
    assert not is_phi(tmp_path / "missing.raw")
    renamed = tmp_path / "synthetic.bin"
    renamed.write_bytes(synthetic.path.read_bytes())
    assert not is_phi(renamed)


def test_get_reader(synthetic: SyntheticPhi) -> None:
    reader = get_reader(synthetic.path)
    assert isinstance(reader, PhiReader)
    assert reader.use_appended_calibration
    reader = get_reader(synthetic.path, pixel_size=5.0)
    assert reader.pixel_size == 5.0
    assert reader._get_reader_kwargs() == {"use_appended_calibration": True, "pixel_size": 5.0}


# --- geometry and metadata --------------------------------------------------------------------------------------------
def test_geometry(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    assert reader.n_pixels == 16
    assert reader.image_shape == (4, 4)
    assert reader.tile_grid == (1, 1)
    np.testing.assert_array_equal(reader.x_coordinates, np.tile(np.arange(4), 4))
    np.testing.assert_array_equal(reader.y_coordinates, np.repeat(np.arange(4), 4))
    assert reader.pixel_size == 25.0
    assert reader.rois == [0]
    assert reader.get_n_pixels_for_roi(0) == 16
    assert reader.get_n_pixels_for_roi(1) == 0
    assert reader._get_reader_kwargs() == {"use_appended_calibration": True}


def test_metadata(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    assert reader.polarity == "negative"
    assert reader.n_frames == 2
    assert reader.is_centroid
    assert reader.calibration_source == "header"
    assert reader.calibration == {"Mass/Time": PHI_SLOPE, "MassOffset": 0.0}
    assert reader.metadata["root"]["Calibrated"] == "no"
    assert reader.metadata["sections"]["PHI LMIG"]["Raster Size (um)"] == "100.0"
    assert reader.dropped_events == {
        "invalid": synthetic.n_invalid,
        "off_raster": synthetic.n_off_raster,
        "off_axis": synthetic.n_off_axis,
    }


def test_mass_axis(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    mz = reader.mz_axis
    assert np.all(np.diff(mz) > 0)
    np.testing.assert_allclose(mz[:10], synthetic.mz(np.arange(10)))
    assert reader.mz_min == pytest.approx(1.0)
    assert reader.mz_max <= 300.0
    np.testing.assert_allclose(reader.tof_axis, 0.5 + np.arange(len(mz)) * 0.001)
    np.testing.assert_allclose(np.sqrt(mz), PHI_SLOPE * reader.tof_axis)


# --- data -------------------------------------------------------------------------------------------------------------
def test_spectra(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    for i in range(reader.n_pixels):
        x, y = reader.get_spectrum(i)
        ex, ey = _expected_spectrum(synthetic, i)
        np.testing.assert_allclose(x, ex)
        np.testing.assert_array_equal(y, ey)
    # pixel (3, 3) has no ions
    x, y = reader.get_spectrum(15)
    assert x.size == y.size == 0
    order = np.arange(reader.n_pixels)[::-1]
    for i, (_, y) in zip(order, reader.spectra_iter(order, silent=True), strict=True):
        np.testing.assert_array_equal(y, _expected_spectrum(synthetic, int(i))[1])
    with pytest.raises(IndexError):
        reader.get_spectrum(reader.n_pixels)
    assert sum(sum(c.values()) for c in synthetic.dense().values()) == len(synthetic.events)


def test_tic_and_summed(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    dense = _dense_matrix(synthetic, reader.mz_axis)
    np.testing.assert_array_equal(reader.get_tic(silent=True), dense.sum(axis=1))
    assert reader.get_tic().sum() == len(synthetic.events)

    mz, summed = reader.get_summed_spectrum([0, 2, 4], silent=True)
    expected = dense[[0, 2, 4]].sum(axis=0)
    np.testing.assert_allclose(mz, reader.mz_axis[expected > 0])
    np.testing.assert_array_equal(summed, expected[expected > 0])

    scales = np.arange(1, reader.n_pixels + 1, dtype=np.float64)
    mz, summed = reader.get_summed_spectrum(reader.pixels, scales=scales, silent=True)
    expected = (dense * scales[:, None]).sum(axis=0)
    np.testing.assert_allclose(mz, reader.mz_axis[expected > 0])
    np.testing.assert_allclose(summed, expected[expected > 0])
    # repeated indices count twice
    _, twice = reader.get_summed_spectrum([1, 1], silent=True)
    _, once = reader.get_summed_spectrum([1], silent=True)
    np.testing.assert_array_equal(twice, 2 * once)
    with pytest.raises(ValueError):
        reader.get_summed_spectrum([], silent=True)
    with pytest.raises(IndexError):
        reader.get_summed_spectrum([reader.n_pixels], silent=True)


def test_ion_images(reader: PhiReader, synthetic: SyntheticPhi) -> None:
    dense = _dense_matrix(synthetic, reader.mz_axis)
    mz = reader.mz_axis
    targets = [synthetic.mz(10), synthetic.mz(500), synthetic.mz(3000), 150.0]
    tol = 0.05
    expected = np.stack([dense[:, np.abs(mz - t) <= tol].sum(axis=1) for t in targets], axis=1)
    assert expected[:, 0].sum() > 0 and expected[:, 3].sum() == 0

    np.testing.assert_array_equal(reader.to_table(targets, tol=tol, silent=True), expected)
    images = reader.get_ion_images(targets, tol=tol, silent=True)
    assert images.shape == (len(targets), 4, 4)
    np.testing.assert_array_equal(images.reshape(len(targets), -1).T, expected)
    image = reader.get_ion_image(targets[1], tol=tol, silent=True)
    np.testing.assert_array_equal(image.ravel(), expected[:, 1])
    assert image[3, 3] == 0

    # ppm windows agree with brute force too
    ppm = 100.0
    table = reader.to_table(targets, ppm=ppm, silent=True)
    for i, t in enumerate(targets):
        window = t * ppm * 1e-6
        np.testing.assert_array_equal(table[:, i], dense[:, np.abs(mz - t) <= window].sum(axis=1))


# --- variants ---------------------------------------------------------------------------------------------------------
def test_appended_calibration(tmp_path: Path) -> None:
    data = write_synthetic_phi(tmp_path, recalibrated=True)
    reader = PhiReader(data.path)
    assert reader.calibration_source == "appended"
    assert reader.calibration == {"Mass/Time": data.slope, "MassOffset": data.offset}
    np.testing.assert_allclose(reader.mz_axis[:5], data.mz(np.arange(5)))
    for i in range(reader.n_pixels):
        x, _ = reader.get_spectrum(i)
        np.testing.assert_allclose(x, _expected_spectrum(data, i)[0])

    original = PhiReader(data.path, use_appended_calibration=False)
    assert original.calibration_source == "header"
    np.testing.assert_allclose(original.mz_axis[:5], data.mz(np.arange(5), slope=PHI_SLOPE, offset=0.0))
    np.testing.assert_array_equal(original.get_tic(), reader.get_tic())


def test_mosaic(tmp_path: Path) -> None:
    data = write_synthetic_phi(tmp_path, mosaic=True)
    reader = PhiReader(data.path)
    assert reader.tile_grid == (2, 1)
    assert reader.image_shape == (4, 8)
    assert reader.n_pixels == 32
    assert data.events[:, 0].max() >= 4  # second tile is populated
    np.testing.assert_array_equal(reader.get_tic(), _dense_matrix(data, reader.mz_axis).sum(axis=1))
    assert reader.dropped_events["off_raster"] == data.n_off_raster


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"header_overrides": {"ImagePixels": None}}, "missing required field 'ImagePixels'"),
        ({"header_overrides": {"Mass/Time": "0"}}, "non-positive Mass/Time"),
        ({"header_overrides": {"StopFlightTime": "0.1"}}, "flight-time range is empty"),
        ({"terminate_header": False}, "no EOFH terminator"),
    ],
)
def test_malformed_header(tmp_path: Path, kwargs: dict, match: str) -> None:
    data = write_synthetic_phi(tmp_path, **kwargs)
    assert is_phi(data.path)
    with pytest.raises(ValueError, match=match):
        PhiReader(data.path)


def test_no_events(tmp_path: Path) -> None:
    data = write_synthetic_phi(tmp_path)
    raw = data.path.read_bytes()
    # truncate right after the header and terminate the block chain
    data.path.write_bytes(raw[:2048] + b"\x02" + b"\x00" * 15 + b"\x00" * 16)
    with pytest.raises(ValueError, match="No ion event blocks"):
        PhiReader(data.path)
