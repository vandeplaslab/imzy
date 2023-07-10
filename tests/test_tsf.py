"""Tests for imzml files."""
import pytest
from imzy import TSFReader, get_reader
from koyo.system import IS_MAC

from .utilities import get_tsf_data


@pytest.mark.skipif(IS_MAC, reason="Bruker reader is not supported on macOS.")
@pytest.mark.parametrize("path", get_tsf_data())
def test_init(path):
    reader: TSFReader = get_reader(path)
    assert type(reader) == TSFReader
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
    mz_min, mz_max = reader.get_acquisition_mass_range()
    assert mz_min != mz_max
    mz_x, mz_y = reader.get_summed_spectrum(reader.pixels)
    assert mz_x.shape == mz_y.shape


@pytest.mark.skipif(IS_MAC, reason="Bruker reader is not supported on macOS.")
@pytest.mark.parametrize("path", get_tsf_data())
def test_to_zarr(path, tmp_path):
    reader = TSFReader(path)

    mzs = [500, 550, 600, 601, 603]
    zarr_temp = tmp_path / "output.zarr"
    zarr_path = reader.to_zarr(zarr_temp, mzs, tol=0.5)
    assert zarr_path.exists()


@pytest.mark.skipif(IS_MAC, reason="Bruker reader is not supported on macOS.")
@pytest.mark.parametrize("path", get_tsf_data())
def test_to_h5(path, tmp_path):
    reader = TSFReader(path)

    mzs = [500, 550, 600, 601, 603]
    h5_temp = tmp_path / "output"  # forgot to include .h5 extension
    h5_path = reader.to_hdf5(h5_temp, mzs, tol=0.5)
    assert h5_path.exists()
    assert h5_temp != h5_path
