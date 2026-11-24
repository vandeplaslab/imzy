"""Tests for the TOFWERK TofDAQ (LA-ICP-TOF) reader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from imzy import TofwerkReader, get_reader
from imzy._readers.tofwerk import is_tofwerk

from .utilities import SyntheticTofwerk, write_synthetic_tofwerk


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticTofwerk:
    return write_synthetic_tofwerk(tmp_path_factory.mktemp("tofwerk"))


def _dense(data: SyntheticTofwerk, roi: int, mode: str) -> np.ndarray:
    """Expected (n_pixels, n_mz) intensities."""
    cube = data.peaks if mode == "peaks" else data.tof
    return np.stack([cube[w, b, 0] for w, b in data.write_bufs[roi]])


def _mz(data: SyntheticTofwerk, mode: str) -> np.ndarray:
    return data.masses if mode == "peaks" else data.mass_axis


# --- detection --------------------------------------------------------------------------------------------------------
def test_is_tofwerk(synthetic: SyntheticTofwerk, tmp_path: Path) -> None:
    import h5py

    assert is_tofwerk(synthetic.path)
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("array", data=np.zeros(3))
    assert not is_tofwerk(other)
    text = tmp_path / "text.h5"
    text.write_text("not hdf5")
    assert not is_tofwerk(text)
    assert not is_tofwerk(tmp_path / "missing.h5")


def test_is_tofwerk_userblock(tmp_path: Path) -> None:
    data = write_synthetic_tofwerk(tmp_path, userblock=True)
    assert data.path.read_bytes()[:8] == b"========"
    assert is_tofwerk(data.path)
    assert isinstance(get_reader(data.path), TofwerkReader)


def test_get_reader(synthetic: SyntheticTofwerk) -> None:
    reader = get_reader(synthetic.path)
    assert isinstance(reader, TofwerkReader)
    assert reader.mode == "peaks" and reader.roi == 0
    reader = get_reader(synthetic.path, roi=1, mode="profile")
    assert reader.roi == 1 and reader.mode == "profile"


# --- geometry and metadata --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("roi", [0, 1])
def test_geometry(synthetic: SyntheticTofwerk, roi: int) -> None:
    reader = TofwerkReader(synthetic.path, roi=roi)
    expected = np.asarray(synthetic.xy[roi])
    assert reader.n_pixels == len(expected)
    np.testing.assert_array_equal(reader.x_coordinates, expected[:, 0])
    np.testing.assert_array_equal(reader.y_coordinates, expected[:, 1])
    assert reader.image_shape == (expected[:, 1].max() + 1, expected[:, 0].max() + 1)
    assert reader.pixel_size == 20.0
    assert reader.rois == [0, 1]
    assert reader.get_n_pixels_for_roi(roi) == reader.n_pixels
    assert reader.get_n_pixels_for_roi(1 - roi) == 0
    assert reader._get_reader_kwargs() == {"roi": roi, "mode": "peaks"}


def test_metadata(synthetic: SyntheticTofwerk) -> None:
    reader = TofwerkReader(synthetic.path)
    assert reader.polarity == "positive"
    assert reader.peak_labels == synthetic.labels
    assert reader.scan_names == ["STD", "Line0", "Line1"]
    assert reader.metadata["image"]["SampleName"] == "synthetic"
    assert reader.is_centroid
    assert not TofwerkReader(synthetic.path, mode="profile").is_centroid
    assert reader.stage_coordinates.shape == (reader.n_pixels, 3)


def test_invalid_arguments(synthetic: SyntheticTofwerk) -> None:
    with pytest.raises(ValueError, match="ROI 5 was not found"):
        TofwerkReader(synthetic.path, roi=5)
    with pytest.raises(ValueError, match="Unknown mode"):
        TofwerkReader(synthetic.path, mode="other")


def test_duplicate_positions(tmp_path: Path) -> None:
    data = write_synthetic_tofwerk(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="same pixel"):
        TofwerkReader(data.path)


# --- data -------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["peaks", "profile"])
@pytest.mark.parametrize("roi", [0, 1])
def test_spectra(synthetic: SyntheticTofwerk, mode: str, roi: int) -> None:
    reader = TofwerkReader(synthetic.path, roi=roi, mode=mode)
    dense = _dense(synthetic, roi, mode)
    mz = _mz(synthetic, mode)
    for i in range(reader.n_pixels):
        x, y = reader.get_spectrum(i)
        np.testing.assert_allclose(x, mz)
        np.testing.assert_array_equal(y, dense[i])
    for i, (x, y) in enumerate(reader.spectra_iter(silent=True)):
        np.testing.assert_allclose(x, mz)
        np.testing.assert_array_equal(y, dense[i])
    # arbitrary order
    order = np.arange(reader.n_pixels)[::-1]
    ys = [y for _, y in reader.spectra_iter(order, silent=True)]
    np.testing.assert_array_equal(np.stack(ys), dense[order])
    with pytest.raises(IndexError):
        reader.get_spectrum(reader.n_pixels)


@pytest.mark.parametrize("mode", ["peaks", "profile"])
def test_tic_and_summed(synthetic: SyntheticTofwerk, mode: str) -> None:
    reader = TofwerkReader(synthetic.path, mode=mode)
    dense = _dense(synthetic, 0, mode).astype(np.float64)
    np.testing.assert_allclose(reader.get_tic(silent=True), dense.sum(axis=1), rtol=1e-6)
    mz, summed = reader.get_summed_spectrum([0, 2, 4], silent=True)
    np.testing.assert_allclose(mz, _mz(synthetic, mode))
    np.testing.assert_allclose(summed, dense[[0, 2, 4]].sum(axis=0), rtol=1e-6)
    scales = np.arange(1, reader.n_pixels + 1, dtype=np.float64)
    _, summed = reader.get_summed_spectrum(reader.pixels, scales=scales, silent=True)
    np.testing.assert_allclose(summed, (dense * scales[:, None]).sum(axis=0), rtol=1e-6)
    with pytest.raises(ValueError):
        reader.get_summed_spectrum([], silent=True)


@pytest.mark.parametrize("mode", ["peaks", "profile"])
def test_ion_images(synthetic: SyntheticTofwerk, mode: str) -> None:
    reader = TofwerkReader(synthetic.path, mode=mode)
    dense = _dense(synthetic, 0, mode).astype(np.float64)
    mz = _mz(synthetic, mode)
    targets = [20.0, 30.0, 100.0]
    expected = np.stack([dense[:, np.abs(mz - t) <= 5.1].sum(axis=1) for t in targets], axis=1)

    table = reader.to_table(targets, tol=5.1, silent=True)
    np.testing.assert_allclose(table, expected, rtol=1e-5)
    images = reader.get_ion_images(targets, tol=5.1, silent=True)
    assert images.shape == (len(targets), *reader.image_shape)
    np.testing.assert_allclose(images[:, reader.y_coordinates, reader.x_coordinates], expected.T, rtol=1e-5)
    image = reader.get_ion_image(30.0, tol=5.1, silent=True)
    np.testing.assert_allclose(image[reader.y_coordinates, reader.x_coordinates], expected[:, 1], rtol=1e-5)
    # the pixel of the invalid spot is empty (BaseReader.reshape fills with 0)
    assert image[2, 1] == 0
