"""Tests for imzml files."""

import sys
from pathlib import Path

import numpy as np
import pytest
from koyo.system import is_installed

from imzy import IMZMLReader, get_reader
from imzy._readers.imzml._imzml import choose_iterparse, init_metadata

from .utilities import get_imzml_data


@pytest.mark.parametrize("path", get_imzml_data())
def test_init(path):
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
    root, mz_precision, int_precision, byte_offsets, coordinates = init_metadata(path, parse_lib="ElementTree")

    assert root is not None
    assert mz_precision in {"f", "d", "i", "l"}
    assert int_precision in {"f", "d", "i", "l"}
    assert byte_offsets.shape[1] == 4
    assert coordinates.shape[1] == 3
    assert byte_offsets.shape[0] == coordinates.shape[0]


@pytest.mark.skipif(not is_installed("lxml"), reason="lxml not installed")
@pytest.mark.parametrize("path", get_imzml_data())
def test_init_metadata_lxml_matches_elementtree(path: Path) -> None:
    _, et_mz_precision, et_int_precision, et_byte_offsets, et_coordinates = init_metadata(
        path, parse_lib="ElementTree"
    )
    _, lxml_mz_precision, lxml_int_precision, lxml_byte_offsets, lxml_coordinates = init_metadata(
        path, parse_lib="lxml"
    )

    assert lxml_mz_precision == et_mz_precision
    assert lxml_int_precision == et_int_precision
    np.testing.assert_array_equal(lxml_byte_offsets, et_byte_offsets)
    np.testing.assert_array_equal(lxml_coordinates, et_coordinates)


def test_choose_iterparse_defaults_to_elementtree_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    assert choose_iterparse().__module__ == "xml.etree.ElementTree"


@pytest.mark.skipif(
    not (is_installed("zarr") and is_installed("rechunker") and is_installed("dask")), reason="zarr not installed"
)
@pytest.mark.parametrize("path", get_imzml_data())
def test_to_zarr(path, tmp_path):
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
    reader = IMZMLReader(path)

    mzs = [500, 550, 600, 601, 603]
    h5_temp = tmp_path / "output"  # forgot to include .h5 extension
    h5_path = reader.to_hdf5(h5_temp, mzs, tol=0.5)
    assert h5_path.exists()
    assert h5_temp != h5_path


@pytest.mark.parametrize("path", get_imzml_data())
def test_norms(path, tmp_path):
    reader = IMZMLReader(path)

    h5_temp = tmp_path / "output"  # forgot to include .h5 extension
    h5_path = reader.extract_normalizations_hdf5(h5_temp)
    assert h5_path.exists()
    assert h5_temp != h5_path
