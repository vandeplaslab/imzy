"""Tests for the SCiLS Lab (.sbd/.slx) reader."""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import numpy as np
import pytest

import imzy
from imzy import SCiLSReader, get_reader
from imzy._readers.scils import decode_spot_indices, is_scils, parse_length_prefixed, read_slx_metadata

from .utilities import SyntheticSCiLS, encode_length_prefixed, encode_spot_indices, write_synthetic_scils


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticSCiLS:
    return write_synthetic_scils(tmp_path_factory.mktemp("scils"))


@pytest.fixture
def reader(synthetic: SyntheticSCiLS) -> SCiLSReader:
    return SCiLSReader(synthetic.sbd_path)


def _expected(data: SyntheticSCiLS, mz_min: float, mz_max: float) -> np.ndarray:
    mask = (data.mz_axis >= mz_min) & (data.mz_axis <= mz_max)
    return data.dense[:, mask].sum(axis=1)


# --- decoding ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("gap", [10, 300, 70_000])
def test_decode_spot_indices_widths(gap: int) -> None:
    spots = np.cumsum(np.random.default_rng(0).integers(1, gap, size=50))
    np.testing.assert_array_equal(decode_spot_indices(encode_spot_indices(spots), len(spots)), spots)


def test_decode_single_spot() -> None:
    np.testing.assert_array_equal(decode_spot_indices(encode_spot_indices(np.array([7759])), 1), [7759])


def test_decode_spot_indices_wrong_size() -> None:
    with pytest.raises(ValueError, match="Unexpected size"):
        decode_spot_indices(zlib.compress(b"\x00" * 7), 2)


# --- SLX --------------------------------------------------------------------------------------------------------------
def test_parse_length_prefixed_multibyte() -> None:
    items = {"Date": "25-10-09 09:37:23", "CCS [Å²]": "µm", "empty": ""}
    assert parse_length_prefixed(encode_length_prefixed(items)) == items
    assert parse_length_prefixed(encode_length_prefixed(items).encode()) == items


@pytest.mark.parametrize("data", [None, "", "abc", "5:ab", "4:Date9999:x"])
def test_parse_length_prefixed_malformed(data: str | None) -> None:
    assert parse_length_prefixed(data) == {}


def test_read_slx_metadata(synthetic: SyntheticSCiLS) -> None:
    meta = read_slx_metadata(synthetic.slx_path)
    assert meta["master"] == {"Version": "10"}
    assert "secure" not in meta["meta"]
    (source,) = meta["data_sources"]
    assert source["polarity"] == "positive"
    assert source["meta"]["Source/Description"] == "Bruker timsTOF fleX .tsf file"
    assert source["meta"]["Unit/µm²"] == "Å²"
    assert meta["rasters"][0]["x_offset"] == 2513 and meta["rasters"][0]["y_offset"] == 721


def test_read_slx_metadata_not_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "x.slx"
    path.write_bytes(b"not a database")
    assert read_slx_metadata(path)["data_sources"] == []
    assert read_slx_metadata(tmp_path / "missing.slx")["rasters"] == []


# --- plugin discovery -------------------------------------------------------------------------------------------------
def test_get_reader_from_sbd_and_slx(synthetic: SyntheticSCiLS) -> None:
    for path in (synthetic.sbd_path, synthetic.slx_path):
        assert is_scils(path)
        r = get_reader(path)
        assert isinstance(r, SCiLSReader)
        assert r.path == synthetic.sbd_path
        assert r.slx_path == synthetic.slx_path
    assert imzy.SCiLSReader is SCiLSReader


def test_is_scils_negative(tmp_path: Path, synthetic: SyntheticSCiLS) -> None:
    assert not is_scils(tmp_path / "missing.sbd")
    bad = tmp_path / "bad.sbd"
    bad.write_bytes(b"not hdf5 at all")
    assert not is_scils(bad)
    assert not is_scils(synthetic.sbd_path.with_suffix(".h5"))
    orphan = tmp_path / "orphan.slx"  # SLX without SBD
    orphan.write_bytes(synthetic.slx_path.read_bytes())
    assert not is_scils(orphan)


def test_sbd_without_slx(tmp_path: Path) -> None:
    data = write_synthetic_scils(tmp_path, "solo")
    data.slx_path.unlink()
    r = get_reader(data.sbd_path)
    assert r.slx_path is None and r.slx_metadata == {}
    assert r.polarity == "negative"  # falls back to the Source attributes in the SBD file


# --- geometry / metadata ----------------------------------------------------------------------------------------------
def test_geometry(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    assert reader.n_pixels == len(synthetic.x)
    np.testing.assert_array_equal(reader.x_coordinates, synthetic.x - synthetic.x.min())
    np.testing.assert_array_equal(reader.y_coordinates, synthetic.y - synthetic.y.min())
    assert reader.image_shape == (reader.y_coordinates.max() + 1, reader.x_coordinates.max() + 1)
    assert reader.pixel_size == 10.0
    assert reader.is_centroid
    assert (reader.mz_min, reader.mz_max) == (300.0, 600.0)
    assert reader.rois == [0]
    assert reader.polarity == "positive"
    assert reader.metadata["sbd"]["SpectraType"] == "Peaks"


# --- ion images -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mz", [310.2, 450.0, 599.7])
def test_ion_image_tol(reader: SCiLSReader, synthetic: SyntheticSCiLS, mz: float) -> None:
    image = reader.get_ion_image(mz, tol=0.5)
    assert image.shape == reader.image_shape
    np.testing.assert_allclose(reader.flatten(image), _expected(synthetic, mz - 0.5, mz + 0.5), rtol=1e-5)


def test_ion_image_ppm(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    mz, ppm = 400.0, 1500.0
    delta = mz * ppm / 1e6
    image = reader.get_ion_image(mz, ppm=ppm)
    np.testing.assert_allclose(reader.flatten(image), _expected(synthetic, mz - delta, mz + delta), rtol=1e-5)


def test_ion_image_range_is_inclusive(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    lo, hi = synthetic.mz_axis[100], synthetic.mz_axis[104]
    image = reader.get_ion_image_range(lo, hi)
    np.testing.assert_allclose(reader.flatten(image), synthetic.dense[:, 100:105].sum(axis=1), rtol=1e-5)


def test_pixels_without_spot_are_zero(reader: SCiLSReader) -> None:
    image = reader.get_ion_image_range(300, 600)
    occupied = np.zeros(reader.image_shape, dtype=bool)
    occupied[reader.y_coordinates, reader.x_coordinates] = True
    assert np.all(image[~occupied] == 0)


def test_empty_windows(reader: SCiLSReader) -> None:
    assert reader.get_ion_image_range(100, 200).sum() == 0
    assert reader.get_ion_image_range(700, 800).sum() == 0
    assert reader.get_ion_image_range(450.0001, 450.0002).sum() == 0  # between two bins


def test_ion_image_requires_tol_or_ppm(reader: SCiLSReader) -> None:
    with pytest.raises(ValueError):
        reader.get_ion_image(400.0)
    with pytest.raises(ValueError):
        reader.get_ion_image(400.0, tol=0.1, ppm=10)
    with pytest.raises(ValueError):
        reader.get_ion_image_range(500, 400)


def test_ion_images_batch(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    mzs = [320.0, 410.5, 590.0]
    images = reader.get_ion_images(mzs, tol=0.4)
    assert images.shape == (3, *reader.image_shape)
    for image, mz in zip(images, mzs, strict=True):
        np.testing.assert_allclose(reader.flatten(image), _expected(synthetic, mz - 0.4, mz + 0.4), rtol=1e-5)


def test_ion_images_yield(reader: SCiLSReader) -> None:
    steps = list(reader._get_ions_yield([320.0, 410.5], tol=0.4))
    assert [s[:2] for s in steps[:-1]] == [(0, 2), (1, 2)]
    assert steps[-1][2].shape == (reader.n_pixels, 2)


def test_sum_of_all_bins_is_tic(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    full = reader.flatten(reader.get_ion_image_range(300, 600))
    np.testing.assert_allclose(full, synthetic.dense.sum(axis=1), rtol=1e-5)
    np.testing.assert_allclose(reader.get_tic(), synthetic.dense.sum(axis=1), rtol=1e-5)


# --- normalizations ---------------------------------------------------------------------------------------------------
def test_normalized_ion_image(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    assert reader.normalization_names == ["Total Ion Count", "Root Mean Square"]
    for name in ("TIC", "total ion count"):
        # after normalization, the per-pixel TIC equals the target value
        image = reader.get_ion_image_range(300, 600, normalization=name)
        np.testing.assert_allclose(reader.flatten(image), synthetic.norm_targets["Total Ion Count"], rtol=1e-4)
    raw = reader.get_ion_image(400.0, tol=1.0)
    norm = reader.get_ion_image(400.0, tol=1.0, normalization="RMS")
    np.testing.assert_allclose(
        reader.flatten(norm), reader.flatten(raw) / synthetic.norm_values["Root Mean Square"], rtol=1e-4
    )
    batch = reader.get_ion_images([400.0], tol=1.0, normalization="RMS")
    np.testing.assert_allclose(batch[0], norm, rtol=1e-6)


def test_unknown_normalization(reader: SCiLSReader) -> None:
    with pytest.raises(ValueError, match="not found"):
        reader.get_ion_image(400.0, tol=1.0, normalization="median")


def test_get_normalizations_are_multipliers(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    norms = reader.get_normalizations()
    assert set(norms) == {"TIC", "RMS"}
    values = synthetic.norm_values["Total Ion Count"]
    np.testing.assert_allclose(norms["TIC"], np.median(values) / values, rtol=1e-5)
    assert norms["TIC"].shape == (reader.n_pixels,)


def test_tic_without_tic_normalization(tmp_path: Path) -> None:
    data = write_synthetic_scils(tmp_path, with_tic_normalization=False)
    r = SCiLSReader(data.sbd_path)
    assert r.normalization_names == ["Root Mean Square"]
    np.testing.assert_allclose(r.get_tic(), data.dense.sum(axis=1), rtol=1e-5)


# --- rasters ----------------------------------------------------------------------------------------------------------
def test_rois(tmp_path: Path) -> None:
    data = write_synthetic_scils(tmp_path, n_rasters=2)
    full = SCiLSReader(data.sbd_path)
    assert full.rois == [0, 1]
    assert full.get_n_pixels_for_roi(0) + full.get_n_pixels_for_roi(1) == full.n_pixels
    for roi in (0, 1):
        r = SCiLSReader(data.sbd_path, roi=roi)
        mask = data.rasters == roi
        assert r.n_pixels == mask.sum()
        assert r.x_coordinates.min() == 0 and r.y_coordinates.min() == 0
        np.testing.assert_allclose(
            r.flatten(r.get_ion_image_range(300, 450)), data.dense[mask][:, data.mz_axis <= 450].sum(axis=1), rtol=1e-5
        )
        np.testing.assert_allclose(r.get_tic(), data.dense[mask].sum(axis=1), rtol=1e-5)
        assert r._get_reader_kwargs() == {"roi": roi}
    with pytest.raises(ValueError, match="ROI 5 was not found"):
        SCiLSReader(data.sbd_path, roi=5)


# --- robustness -------------------------------------------------------------------------------------------------------
def test_open_falls_back_for_flagged_files(synthetic: SyntheticSCiLS, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCiLS/HDFView can leave the 'open for write' flag set, which makes a plain read-only open fail."""
    import h5py

    original = h5py.File
    calls: list[dict] = []

    def flaky(*args, **kwargs):
        calls.append(kwargs)
        if not kwargs.get("swmr"):
            raise OSError("Unable to synchronously open file (file is already open for write)")
        return original(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", flaky)
    r = SCiLSReader(synthetic.sbd_path)
    assert r.get_ion_image_range(300, 600).sum() > 0
    assert any(call.get("swmr") for call in calls)


def test_open_raises_original_error(synthetic: SyntheticSCiLS, monkeypatch: pytest.MonkeyPatch) -> None:
    import h5py

    def broken(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(h5py, "File", broken)
    with pytest.raises(OSError, match="boom"):
        SCiLSReader(synthetic.sbd_path)


def test_not_a_scils_hdf5(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "other.sbd"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=[1])
    with pytest.raises(ValueError, match="SpectraGroup"):
        SCiLSReader(path)


def test_spectra_not_supported(reader: SCiLSReader) -> None:
    with pytest.raises(NotImplementedError):
        reader.get_spectrum(0)
    with pytest.raises(NotImplementedError):
        reader.get_summed_spectrum([0, 1])


# --- real data (optional) ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("SCILS_TEST_FILE"), reason="Set SCILS_TEST_FILE to run against a real dataset.")
def test_real_file() -> None:
    r = get_reader(os.environ["SCILS_TEST_FILE"])
    assert isinstance(r, SCiLSReader)
    full = r.flatten(r.get_ion_image_range(r.mz_min, r.mz_max))
    np.testing.assert_allclose(full, r.get_tic(), rtol=1e-4)
    image = r.get_ion_image(r.mz_axis[len(r.mz_axis) // 2], ppm=20)
    assert image.shape == r.image_shape and image.min() >= 0
