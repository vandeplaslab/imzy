# Spectra and images

## Select a reader

`get_reader()` detects built-in formats and installed reader plugins:

```python
from imzy import get_reader

reader = get_reader("path/to/dataset")
print(reader.n_pixels, reader.image_shape, reader.is_centroid)
mzs, intensities = reader.get_spectrum(0)
```

Additional keyword arguments are forwarded to the selected reader. Consult the [reader reference](api.md#readers)
before passing format-specific options. For imzML, prefer a context manager to release file mappings promptly:

```python
from imzy import IMZMLReader

with IMZMLReader("sample.imzML", backend="auto") as reader:
    for mzs, intensities in reader.spectra_iter(indices=[0, 1], silent=True):
        print(float(intensities.sum()))
```

An individual spectrum is a pair of one-dimensional arrays with matching lengths. Spectrum lengths may vary
in processed centroid datasets. Copy read-only arrays before modifying them.

## Sum selected pixels

```python
with IMZMLReader("sample.imzML") as reader:
    selected = reader.pixels[:min(100, reader.n_pixels)]
    mzs, summed = reader.get_summed_spectrum(selected, silent=True)
    average = summed / len(selected)
```

Select at least one pixel. For centroid imzML, summation resamples onto a profile grid controlled by
`resolution` and `mz_ppm`; it does not simply concatenate the peaks. Vendor readers also have format-specific
centroid-to-profile behavior. In particular, NeoFlex and Waters readers expose `auto_profile=False` for raw centroid access.

## Extract ion images

```python
with IMZMLReader("sample.imzML") as reader:
    image = reader.get_ion_image(200.0, ppm=10, silent=True)
    images = reader.get_ion_images([100.0, 200.0, 300.0], tol=0.05, silent=True)
    table = reader.to_table([100.0, 200.0, 300.0], tol=0.05, silent=True)
```

| Operation | Shape |
| --- | --- |
| `get_tic()` | `(n_pixels,)` |
| `get_ion_image(...)` | `(height, width)` |
| `get_ion_images(mzs, ...)` | `(n_masses, height, width)` |
| `to_table(mzs, ...)` | `(n_pixels, n_masses)` |

The intensity at each pixel is the sum within the target mass window. `tol=0.05` selects a window of
`mz +/- 0.05` Da; `ppm=10` scales its half-width with the target m/z. Use one positive tolerance,
and keep input m/z arrays sorted ascending.

Use batch extraction to obtain several masses in one pass. TIC values are cached within a reader;
repeated `get_tic()` calls reuse the calculated vector.

## Coordinates and missing pixels

`image_shape` is `(height, width)`. Coordinates describe acquired pixels, which may not fill a rectangular grid.
Use `reshape()` and `reshape_batch()` instead of reshaping with NumPy: they place data at the recorded coordinates.
`flatten(image)` retrieves acquired pixels in reader order.

```python
import numpy as np

with IMZMLReader("sample.imzML") as reader:
    tic_image = reader.reshape(reader.get_tic(silent=True), fill_value=np.nan)
    tic_vector = reader.flatten(tic_image)
```

`reshape()` defaults to zero for unacquired pixels; ion-image methods default to NaN for those locations.
An acquired pixel with no peak in the window contributes zero. imzML coordinates are one-based, while
Bruker reader coordinates use a zero-based image origin. Let the reader handle placement.

## Bruker regions of interest

```python
from imzy import get_reader

reader = get_reader("sample.d")
print(reader.rois)
roi_reader = get_reader("sample.d", roi=0)  # choose an ID reported by reader.rois
```

ROI selection applies to Bruker TSF/TDF/NeoFlex. The selected reader exposes the pixels for that region.
These examples require a supported vendor platform and real input data.
