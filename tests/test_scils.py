"""Tests for the SCiLS Lab (.sbd/.slx) reader."""

from __future__ import annotations

import os
import zlib
from pathlib import Path

import numpy as np
import pytest

import imzy
from imzy import SCiLSReader, get_reader
from imzy._readers.scils import decode_mz, decode_spot_indices, is_scils, parse_length_prefixed, read_slx_metadata
from imzy._readers.scils._scils import _SPECTRA_CHUNK

from .utilities import (
    SyntheticSCiLS,
    encode_length_prefixed,
    encode_mz,
    encode_spot_indices,
    write_synthetic_scils,
)


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


# --- spectra ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("width", [None, 3, 4, 8])
def test_decode_mz_widths(width: int | None) -> None:
    rng = np.random.default_rng(0)
    mz = 323.1618 + np.cumsum(rng.uniform(0.001, 3.0, size=40))
    decoded = decode_mz(encode_mz(mz, width=width), len(mz))
    np.testing.assert_allclose(decoded, mz, atol=1e-6)  # 2 micro-Da resolution
    assert decoded[0] == mz[0]


@pytest.mark.parametrize("scale", [500_000.0, 1_000.0])
def test_decode_mz_scale(scale: float) -> None:
    mz = np.array([400.0, 400.5, 700.123])
    np.testing.assert_allclose(decode_mz(encode_mz(mz, scale=scale), 3), mz, atol=1 / scale)


def test_decode_mz_edge_cases() -> None:
    assert decode_mz(b"", 0).size == 0
    np.testing.assert_array_equal(decode_mz(encode_mz(np.array([555.5])), 1), [555.5])
    block = encode_mz(np.array([400.0, 401.0]))
    with pytest.raises(ValueError, match="trailer"):
        decode_mz(block[:-1], 2)
    with pytest.raises(ValueError, match="Unexpected SCiLS m/z block"):
        decode_mz(block, 3)  # number of peaks does not match


def test_decode_spot_indices_three_byte_width() -> None:
    spots = np.array([5, 100_000, 300_000, 900_000])  # gaps need 3 bytes
    np.testing.assert_array_equal(decode_spot_indices(encode_spot_indices(spots), len(spots)), spots)


def test_get_spectrum(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    for index in (0, 1, 200, reader.n_pixels - 1):
        mz, y = reader.get_spectrum(index)
        expected_y = synthetic.dense[index][synthetic.dense[index] > 0]
        assert mz.dtype == np.float64 and y.dtype == np.float32
        np.testing.assert_array_equal(mz, synthetic.spectra_mz[index])
        np.testing.assert_array_equal(y, expected_y)
        assert np.all(np.diff(mz) > 0)
        np.testing.assert_array_equal(reader[index][0], mz)
        assert reader.get_intensities(index).shape == y.shape
    np.testing.assert_allclose(reader.get_spectrum(0)[1].sum(), reader.get_tic()[0], rtol=1e-5)


def test_get_spectrum_out_of_bounds(reader: SCiLSReader) -> None:
    for index in (-1, reader.n_pixels):
        with pytest.raises(IndexError):
            reader.get_spectrum(index)


def test_read_spectra_all_chunks(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    """Iterating reads several chunks (more spots than the chunk size) and must keep the order."""
    assert reader.n_pixels > _SPECTRA_CHUNK
    spectra = list(reader._read_spectra())
    assert len(spectra) == reader.n_pixels
    for index in (0, _SPECTRA_CHUNK - 1, _SPECTRA_CHUNK, reader.n_pixels - 1):
        np.testing.assert_array_equal(spectra[index][0], synthetic.spectra_mz[index])
    assert sum(len(mz) for mz, _ in spectra) == int((synthetic.dense > 0).sum())


@pytest.mark.parametrize(
    "indices",
    [
        [3, 1, 2],  # contiguous, unordered
        [0, 400],  # sparse -> one read per spectrum
        [5, 5, 5],  # repeated
        [],  # nothing
    ],
)
def test_read_spectra_arbitrary_indices(reader: SCiLSReader, synthetic: SyntheticSCiLS, indices: list[int]) -> None:
    spectra = list(reader._read_spectra(np.asarray(indices, dtype=int)))
    assert len(spectra) == len(indices)
    for index, (mz, y) in zip(indices, spectra, strict=True):
        np.testing.assert_array_equal(mz, synthetic.spectra_mz[index])
        np.testing.assert_array_equal(y, synthetic.dense[index][synthetic.dense[index] > 0])


def test_spectra_iter(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    for i, (mz, _) in zip(range(3), reader, strict=False):
        np.testing.assert_array_equal(mz, synthetic.spectra_mz[i])
    list(reader.spectra_iter([0, 1], silent=True))


def test_empty_spectrum(tmp_path: Path) -> None:
    data = write_synthetic_scils(tmp_path, empty_spots=(3,), with_tic_normalization=False)
    r = SCiLSReader(data.sbd_path)
    mz, y = r.get_spectrum(3)
    assert mz.size == 0 and y.size == 0
    np.testing.assert_array_equal(r.get_spectrum(4)[0], data.spectra_mz[4])  # neighbours are unaffected
    assert r.get_tic()[3] == 0
    np.testing.assert_allclose(r.get_tic(), data.dense.sum(axis=1), rtol=1e-5)
    assert [len(mz) for mz, _ in r._read_spectra([2, 3, 4])][1] == 0
    assert r.flatten(r.get_ion_image_range(300, 600))[3] == 0


def test_summed_spectrum(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    indices = np.array([0, 7, 300, 301])
    mz, summed = reader.get_summed_spectrum(indices, silent=True)
    np.testing.assert_array_equal(mz, synthetic.mz_axis)
    np.testing.assert_allclose(summed, synthetic.dense[indices].sum(axis=0), rtol=1e-6)
    # result must be a copy
    mz[:] = 0
    assert reader.mz_axis[0] == 300.0


def test_summed_spectrum_all_pixels_and_scales(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    """Summing more spots than the chunk size, with a per-pixel multiplier (indexed by pixel)."""
    scales = np.random.default_rng(1).uniform(0.5, 2.0, reader.n_pixels).astype(np.float32)
    _, summed = reader.get_summed_spectrum(range(reader.n_pixels), scales=scales, silent=True)
    np.testing.assert_allclose(summed, (synthetic.dense * scales[:, None]).sum(axis=0), rtol=1e-4)
    # normalization multipliers from the file can be used directly
    tic_scales = reader.get_normalizations()["TIC"]
    _, summed = reader.get_summed_spectrum([2, 9], scales=tic_scales, silent=True)
    np.testing.assert_allclose(summed, (synthetic.dense[[2, 9]] * tic_scales[[2, 9], None]).sum(axis=0), rtol=1e-4)


def test_summed_spectrum_errors(reader: SCiLSReader) -> None:
    with pytest.raises(ValueError, match="at least one index"):
        reader.get_summed_spectrum([], silent=True)
    for bad in ([reader.n_pixels], [-1]):
        with pytest.raises(ValueError, match="greater than the total number of pixels"):
            reader.get_summed_spectrum(bad, silent=True)


def test_nearest_mz_bins(reader: SCiLSReader, synthetic: SyntheticSCiLS) -> None:
    axis = synthetic.mz_axis
    spacing = axis[1] - axis[0]
    mz = np.array([299.0, axis[0], axis[10] + 0.4 * spacing, axis[10] + 0.6 * spacing, axis[-1], 700.0])
    np.testing.assert_array_equal(reader.nearest_mz_bins(mz), [0, 0, 10, 11, len(axis) - 1, len(axis) - 1])
    # every synthetic peak falls into its own bin
    bins = np.flatnonzero(synthetic.dense[5])
    np.testing.assert_array_equal(reader.nearest_mz_bins(synthetic.spectra_mz[5]), bins)


def test_spectra_for_roi(tmp_path: Path) -> None:
    data = write_synthetic_scils(tmp_path, n_rasters=2)
    r = SCiLSReader(data.sbd_path, roi=1)
    offset = int((data.rasters == 0).sum())
    for index in (0, 5, r.n_pixels - 1):
        np.testing.assert_array_equal(r.get_spectrum(index)[0], data.spectra_mz[offset + index])
    mask = data.rasters == 1
    _, summed = r.get_summed_spectrum(range(r.n_pixels), silent=True)
    np.testing.assert_allclose(summed, data.dense[mask].sum(axis=0), rtol=1e-6)


def test_spectra_missing_from_file(tmp_path: Path) -> None:
    import h5py

    data = write_synthetic_scils(tmp_path, with_tic_normalization=False)
    with h5py.File(data.sbd_path, "a") as f:
        for key in ("mzOffsets", "compressedMz"):
            del f["SpectraGroup"][key]
    r = SCiLSReader(data.sbd_path)  # ion images still work
    assert r.get_ion_image_range(300, 600).sum() > 0
    with pytest.raises(ValueError, match="does not contain per-spot spectra"):
        r.get_spectrum(0)
    with pytest.raises(ValueError, match="TIC cannot be determined"):
        r.get_tic()


def test_get_spectrum_consistent_with_ion_images(reader: SCiLSReader) -> None:
    """Peaks from the spectra, binned onto the global axis, must give the same ion image as the image-major layout."""
    window = (reader.mz_axis[200], reader.mz_axis[230])
    from_spectra = np.zeros(reader.n_pixels)
    for i, (mz, y) in enumerate(reader._read_spectra()):
        bins = reader.nearest_mz_bins(mz)
        from_spectra[i] = y[(bins >= 200) & (bins <= 230)].sum()
    np.testing.assert_allclose(reader.flatten(reader.get_ion_image_range(*window)), from_spectra, rtol=1e-5)


# --- real data (optional) ---------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("SCILS_TEST_FILE"), reason="Set SCILS_TEST_FILE to run against a real dataset.")
def test_real_file() -> None:
    r = get_reader(os.environ["SCILS_TEST_FILE"])
    assert isinstance(r, SCiLSReader)
    full = r.flatten(r.get_ion_image_range(r.mz_min, r.mz_max))
    np.testing.assert_allclose(full, r.get_tic(), rtol=1e-4)
    image = r.get_ion_image(r.mz_axis[len(r.mz_axis) // 2], ppm=20)
    assert image.shape == r.image_shape and image.min() >= 0


@pytest.mark.skipif(not os.environ.get("SCILS_TEST_FILE"), reason="Set SCILS_TEST_FILE to run against a real dataset.")
def test_real_file_spectra() -> None:
    r = get_reader(os.environ["SCILS_TEST_FILE"])
    tic = r.get_tic()
    for index in (0, 1, r.n_pixels // 2, r.n_pixels - 1):
        mz, y = r.get_spectrum(index)
        assert len(mz) == len(y) > 0
        assert np.all(np.diff(mz) > 0) and r.mz_min - 0.01 <= mz[0] and mz[-1] <= r.mz_max + 0.01
        np.testing.assert_allclose(y.sum(), tic[index], rtol=1e-4)
    mz_all, summed = r.get_summed_spectrum(range(0, r.n_pixels, 50), silent=True)
    assert mz_all.shape == summed.shape == r.mz_axis.shape and summed.sum() > 0
