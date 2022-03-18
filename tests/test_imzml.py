"""Tests for imzml files."""
import pytest

from imzy import IMZMLReader, get_reader

from .utilities import get_imzml_data


@pytest.mark.parametrize("path", get_imzml_data())
def test_init(path):
    reader = get_reader(path)
    assert type(reader) == IMZMLReader
    assert reader.n_pixels > 0
    assert reader.xyz_coordinates.shape[0] == reader.n_pixels
    assert reader.xyz_coordinates.shape[1] == 3  # x, y and z

    # get spectrum
    x, y = reader.get_spectrum(0)
    assert x.shape == y.shape
    # also possible by getter
    x, y = reader[1]
    assert x.shape == y.shape

    # get tic
    tic = reader.get_tic()
    assert len(tic) == reader.n_pixels

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

    for x, y in reader.iter_spectra():
        assert x.shape == y.shape

    # get summed spectrum
    mz_min, mz_max = reader._estimate_mass_range()
    assert mz_min != mz_max
    mz_x, mz_y = reader.get_summed_spectrum(reader.pixels)
    assert mz_x.shape == mz_y.shape


@pytest.mark.parametrize("path", get_imzml_data())
def test_to_zarr(path, tmp_path):
    reader = IMZMLReader(path)

    mzs = [500, 550, 600, 601, 603]
    zarr_temp = tmp_path / "output.zarr"
    zarr_path = reader.to_zarr(zarr_temp, mzs, tol=0.5)
    assert zarr_path.exists()
