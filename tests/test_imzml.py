"""Tests for imzml files."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from koyo.system import is_installed

from imzy import IMZMLReader, get_reader
from imzy._readers.imzml._imzml import (
    SPECTRUM_MODE_CENTROID,
    SPECTRUM_MODE_PROFILE,
    choose_iterparse,
    init_metadata,
)

from .utilities import get_imzml_data


def _copy_imzml_dataset(path: Path, tmp_path: Path, *, copy_cache: bool = True) -> Path:
    """Copy an imzML dataset to a temporary directory for cache-writing tests."""
    imzml_path = tmp_path / path.name
    shutil.copyfile(path, imzml_path)
    shutil.copyfile(path.with_suffix(".ibd"), imzml_path.with_suffix(".ibd"))
    cache_path = path.with_suffix(".icache")
    if copy_cache and cache_path.exists():
        shutil.copyfile(cache_path, imzml_path.with_suffix(".icache"))
    return imzml_path


@pytest.mark.parametrize("path", get_imzml_data())
def test_init(path, tmp_path):
    path = _copy_imzml_dataset(path, tmp_path)
    reader = get_reader(path)
    assert isinstance(reader, IMZMLReader)
    assert reader.n_pixels > 0
    assert reader.xyz_coordinates.shape[0] == reader.n_pixels
    assert reader.xyz_coordinates.shape[1] == 3  # x, y and z

    # get spectrum
    x, y = reader.get_spectrum(0)
    assert x.shape == y.shape
    # also possible by getter
    x, y = reader[1]
    assert x.shape == y.shape

    # get ROIs
    assert reader.rois == [0]

    # get pixel size
    assert reader.pixel_size == reader.x_pixel_size == reader.y_pixel_size

    # get tic
    tic = reader.get_tic()
    assert len(tic) == reader.n_pixels
    assert tic.ndim == 1
    assert len(tic) == reader.n_pixels
    tic_2d = reader.reshape(tic)
    assert tic_2d.ndim == 2

    # get image
    image = reader.get_ion_image(500, tol=0.5)
    assert image.ndim == 2

    image = reader.get_ion_image(500, ppm=5)
    assert image.ndim == 2

    images = reader.get_ion_images([500, 600], tol=0.5)
    assert images.ndim == 3
    assert len(images) == 2

    images = reader.get_ion_images([500, 600], ppm=3)
    assert images.ndim == 3
    assert len(images) == 2

    for x, y in reader.spectra_iter():
        assert x.shape == y.shape

    # get summed spectrum
    mz_min, mz_max = reader._estimate_mass_range()
    assert mz_min != mz_max
    mz_x, mz_y = reader.get_summed_spectrum(reader.pixels)
    assert mz_x.shape == mz_y.shape


@pytest.mark.parametrize("path", get_imzml_data())
def test_init_metadata_elementtree(path: Path) -> None:
    root, mz_precision, int_precision, byte_offsets, coordinates, spectrum_mode, mz_min, mz_max = init_metadata(
        path, parse_lib="ElementTree"
    )

    assert root is not None
    assert mz_precision in {"f", "d", "i", "l"}
    assert int_precision in {"f", "d", "i", "l"}
    assert byte_offsets.shape[1] == 4
    assert coordinates.shape[1] == 3
    assert byte_offsets.shape[0] == coordinates.shape[0]
    assert spectrum_mode in {SPECTRUM_MODE_CENTROID, SPECTRUM_MODE_PROFILE}
    if spectrum_mode == SPECTRUM_MODE_CENTROID:
        assert mz_min is not None
        assert mz_max is not None
        assert mz_min < mz_max


@pytest.mark.skipif(not is_installed("lxml"), reason="lxml not installed")
@pytest.mark.parametrize("path", get_imzml_data())
def test_init_metadata_lxml_matches_elementtree(path: Path) -> None:
    (
        _,
        et_mz_precision,
        et_int_precision,
        et_byte_offsets,
        et_coordinates,
        et_spectrum_mode,
        et_mz_min,
        et_mz_max,
    ) = init_metadata(
        path, parse_lib="ElementTree"
    )
    (
        _,
        lxml_mz_precision,
        lxml_int_precision,
        lxml_byte_offsets,
        lxml_coordinates,
        lxml_spectrum_mode,
        lxml_mz_min,
        lxml_mz_max,
    ) = init_metadata(path, parse_lib="lxml")

    assert lxml_mz_precision == et_mz_precision
    assert lxml_int_precision == et_int_precision
    np.testing.assert_array_equal(lxml_byte_offsets, et_byte_offsets)
    np.testing.assert_array_equal(lxml_coordinates, et_coordinates)
    assert lxml_spectrum_mode == et_spectrum_mode
    assert lxml_mz_min == et_mz_min
    assert lxml_mz_max == et_mz_max


@pytest.mark.parametrize(
    ("filename", "expected_mode", "expected_centroid"),
    [
        ("simple_imzml.imzML", SPECTRUM_MODE_CENTROID, True),
        ("Example_Processed.imzML", SPECTRUM_MODE_PROFILE, False),
        ("Example_Continuous.imzML", SPECTRUM_MODE_PROFILE, False),
    ],
)
def test_imzml_spectrum_mode_uses_cvparams(
    filename: str,
    expected_mode: str,
    expected_centroid: bool,
    tmp_path: Path,
) -> None:
    """Test that centroid/profile mode is read from imzML cvParams."""
    path = _copy_imzml_dataset(Path(__file__).parent / "_test_data" / filename, tmp_path)

    *_, spectrum_mode, _mz_min, _mz_max = init_metadata(path, parse_lib="ElementTree")
    reader = IMZMLReader(path, parse_lib="ElementTree")

    assert spectrum_mode == expected_mode
    assert reader.is_centroid is expected_centroid


def test_centroid_imzml_uses_xml_bounds_without_binary_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that complete XML observed bounds are enough to build a stable m/z axis."""
    path = _copy_imzml_dataset(
        Path(__file__).parent / "_test_data" / "simple_imzml.imzML",
        tmp_path,
        copy_cache=False,
    )

    def _fail_binary_scan(self: IMZMLReader) -> tuple[float, float]:
        raise AssertionError("XML observed bounds should avoid binary m/z range scanning.")

    monkeypatch.setattr(IMZMLReader, "_estimate_centroid_mass_range", _fail_binary_scan)

    first = IMZMLReader(path, parse_lib="ElementTree")
    first_mz_min = first.mz_min
    first_mz_max = first.mz_max
    first_mz_x = first.mz_x.copy()

    second = IMZMLReader(path, parse_lib="ElementTree")

    assert second.mz_min == first_mz_min
    assert second.mz_max == first_mz_max
    np.testing.assert_array_equal(second.mz_x, first_mz_x)
    with np.load(path.with_suffix(".icache")) as f_ptr:
        assert float(np.asarray(f_ptr["mz_min"]).item()) == first_mz_min
        assert float(np.asarray(f_ptr["mz_max"]).item()) == first_mz_max
        assert str(np.asarray(f_ptr["spectrum_mode"]).item()) == SPECTRUM_MODE_CENTROID


# def test_centroid_imzml_old_cache_uses_xml_bounds_and_upgrades(
#     monkeypatch: pytest.MonkeyPatch,
#     tmp_path: Path,
# ) -> None:
#     """Test that old caches are upgraded from XML bounds without binary scanning."""
#     path = _copy_imzml_dataset(
#         Path(__file__).parent / "_test_data" / "simple_imzml.imzML",
#         tmp_path,
#         copy_cache=True,
#     )
#     with np.load(path.with_suffix(".icache")) as f_ptr:
#         assert "mz_min" not in f_ptr.files
#         assert "mz_max" not in f_ptr.files
#         assert "spectrum_mode" not in f_ptr.files

#     def _fail_binary_scan(self: IMZMLReader) -> tuple[float, float]:
#         raise AssertionError("XML observed bounds should avoid binary m/z range scanning.")

#     monkeypatch.setattr(IMZMLReader, "_estimate_centroid_mass_range", _fail_binary_scan)

#     reader = IMZMLReader(path, parse_lib="ElementTree")

#     assert reader.mz_min == 1.0
#     assert reader.mz_max == 4.0
#     with np.load(path.with_suffix(".icache")) as f_ptr:
#         assert float(np.asarray(f_ptr["mz_min"]).item()) == reader.mz_min
#         assert float(np.asarray(f_ptr["mz_max"]).item()) == reader.mz_max
#         assert str(np.asarray(f_ptr["spectrum_mode"]).item()) == SPECTRUM_MODE_CENTROID


def test_centroid_imzml_missing_xml_bounds_falls_back_to_exact_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that missing XML observed bounds fall back to exact binary scanning."""
    path = _copy_imzml_dataset(
        Path(__file__).parent / "_test_data" / "simple_imzml.imzML",
        tmp_path,
        copy_cache=False,
    )
    text = path.read_text()
    lines = [
        line
        for line in text.splitlines()
        if "MS:1000528" not in line and "MS:1000527" not in line
    ]
    path.write_text("\n".join(lines))
    original_scan = IMZMLReader._estimate_centroid_mass_range
    calls = 0

    def _spy_binary_scan(self: IMZMLReader) -> tuple[float, float]:
        nonlocal calls
        calls += 1
        return original_scan(self)

    monkeypatch.setattr(IMZMLReader, "_estimate_centroid_mass_range", _spy_binary_scan)

    reader = IMZMLReader(path, parse_lib="ElementTree")

    assert reader.mz_min == 1.0
    assert reader.mz_max == 4.0
    assert calls == 1
    with np.load(path.with_suffix(".icache")) as f_ptr:
        assert float(np.asarray(f_ptr["mz_min"]).item()) == reader.mz_min
        assert float(np.asarray(f_ptr["mz_max"]).item()) == reader.mz_max
        assert str(np.asarray(f_ptr["spectrum_mode"]).item()) == SPECTRUM_MODE_CENTROID


def test_choose_iterparse_defaults_to_elementtree_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    assert choose_iterparse().__module__ == "xml.etree.ElementTree"


@pytest.mark.skipif(
    not (is_installed("zarr") and is_installed("rechunker") and is_installed("dask")), reason="zarr not installed"
)
@pytest.mark.parametrize("path", get_imzml_data())
def test_to_zarr(path, tmp_path):
    path = _copy_imzml_dataset(path, tmp_path)
    reader = IMZMLReader(path)

    mzs = [500, 550, 600, 601, 603]
    zarr_temp = tmp_path / "output.zarr"
    zarr_path = reader.to_zarr(zarr_temp, mzs, tol=0.5)
    assert zarr_path.exists()


@pytest.mark.skipif(
    not (is_installed("yoki5") and is_installed("h5py") and is_installed("hdf5plugin")), reason="zarr not installed"
)
@pytest.mark.parametrize("path", get_imzml_data())
def test_to_h5(path, tmp_path):
    path = _copy_imzml_dataset(path, tmp_path)
    reader = IMZMLReader(path)

    mzs = [500, 550, 600, 601, 603]
    h5_temp = tmp_path / "output"  # forgot to include .h5 extension
    h5_path = reader.to_hdf5(h5_temp, mzs, tol=0.5)
    assert h5_path.exists()
    assert h5_temp != h5_path


@pytest.mark.parametrize("path", get_imzml_data())
def test_norms(path, tmp_path):
    path = _copy_imzml_dataset(path, tmp_path)
    reader = IMZMLReader(path)

    h5_temp = tmp_path / "output"  # forgot to include .h5 extension
    h5_path = reader.extract_normalizations_hdf5(h5_temp)
    assert h5_path.exists()
    assert h5_temp != h5_path
