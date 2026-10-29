# Quickstart

This example creates a two-by-two imzML dataset, so no vendor data or download is needed.
Run it in a writable directory after `pip install imzy`. Use a new output name when repeating it;
the writer refuses to overwrite existing files by default.

## Create sample data

```python
from imzy import IMZMLWriter

with IMZMLWriter("sample", spectrum_type="centroid", ibd_mode="processed") as writer:
    writer.add_spectrum([100.0, 200.0, 300.0], [1.0, 2.0, 3.0], (1, 1))
    writer.add_spectrum([100.0, 200.0, 300.0], [2.0, 4.0, 6.0], (2, 1))
    writer.add_spectrum([100.0, 200.0, 300.0], [3.0, 6.0, 9.0], (1, 2))
    writer.add_spectrum([100.0, 200.0, 300.0], [4.0, 8.0, 12.0], (2, 2))
```

This writes `sample.imzML` and `sample.ibd`. Writer coordinates are `(x, y)` and default to a one-based origin.

## Read spectra and images

```python
from imzy import IMZMLReader

with IMZMLReader("sample.imzML") as reader:
    mzs, intensities = reader.get_spectrum(0)
    print(mzs)          # [100. 200. 300.]
    print(intensities)  # [1. 2. 3.]

    tic = reader.get_tic(silent=True)  # shape (4,)
    tic_image = reader.reshape(tic)  # shape (2, 2)
    image = reader.get_ion_image(200.0, tol=0.1, silent=True)
    print(image)  # [[2. 4.], [6. 8.]]

    images = reader.get_ion_images([100.0, 200.0], ppm=10, silent=True)
    print(images.shape)  # (2, 2, 2): masses, rows, columns
```

Pixel indices start at zero. Images use NumPy `(y, x)` order. Supply exactly one of `tol` (Da) and `ppm`.
Opening the reader can create a metadata cache next to the input; the XML remains the metadata source.

## Normalize and convert

```python
from imzy import IMZMLReader, write_imzml

with IMZMLReader("sample.imzML") as reader:
    scales = reader.get_normalizations(silent=True)["TIC"]
    image = reader.get_ion_image(200.0, tol=0.1, silent=True)
    normalized_image = image * reader.reshape(scales)
    converted = write_imzml(reader, "normalized", normalization="TIC", silent=True)
```

Normalization values are multipliers. See [normalization and storage](storage.md) for batch operations,
or run the [complete example](examples.md) to generate plots and validate a conversion.
