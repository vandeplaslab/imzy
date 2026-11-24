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

Zero-based imzML inputs are normalized to one-based reader coordinates. `source_coordinate_base` records
the detected original bases, with `None` for an absent z axis. Cropped acquisitions keep their slide offsets.
For a multi-plane acquisition, select a normalized z plane before extracting 2-D images or centroid stores:

```python
with IMZMLReader("volume.imzML", z_plane=2, pixel_size=(50.0, 50.0)) as reader:
    image = reader.get_ion_image(200.0, ppm=10, silent=True)
```

The selected reader's spectra, pixel indices, tables, and exports all refer to that plane; the full acquisition
grid extent is preserved. The `pixel_size` override is in um and is needed only when physical sizing cannot be
determined. Unknown sizes warn but leave spectral access available. Declared length units are converted to um;
current unitless length terms are assumed to be um. Legacy pixel areas are not interpreted as lengths.

## Bruker regions of interest

```python
from imzy import get_reader

reader = get_reader("sample.d")
print(reader.rois)
roi_reader = get_reader("sample.d", roi=0)  # choose an ID reported by reader.rois
```

ROI selection applies to Bruker TSF/TDF/NeoFlex. The selected reader exposes the pixels for that region.
These examples require a supported vendor platform and real input data.

## TOFWERK LA-ICP-TOF

`TofwerkReader` reads TOFWERK TofDAQ `.h5` files from laser-ablation imaging (e.g. icpTOF). It needs `h5py`
(`pip install "imzy[hdf5]"`). `get_reader()` picks it when the file has the `ImageData` spot table and `PeakData`.

```python
from imzy import get_reader

reader = get_reader("LA_Image.h5")  # sample scans, integrated peaks
print(reader.image_shape, reader.pixel_size, reader.peak_labels[:3])  # e.g. ['[6Li]+', '[7Li]+', '[9Be]+']
cu = reader.get_ion_image(62.93, tol=0.1, silent=True)

standards = get_reader("LA_Image.h5", roi=1)  # reference standard scans
profile = get_reader("LA_Image.h5", mode="profile")  # full TOF spectra on the mass axis
```

- Each valid laser spot becomes one pixel. Stage coordinates (um) are snapped onto a grid with the spot spacing
  as pixel size; x and y follow the stage axes. `stage_coordinates` keeps the original positions.
- `roi=0` (default) loads the sample scans and `roi=1` the reference standards; `reader.rois` lists what is present.
- `mode="peaks"` (default) returns one intensity per entry of the peak table (`is_centroid` is `True`).
  `mode="profile"` returns the full spectra, which are much larger and slower to read.
- Fitted peak intensities are returned as stored and can be slightly negative.

## PHI ToF-SIMS

`PhiReader` reads PHI SmartSoft-TOF `.raw` files from nanoTOF ToF-SIMS imaging. Unlike Waters `.raw` (a directory),
these are single files starting with `SOFH`; `get_reader()` picks `PhiReader` for them. No vendor library is needed.
The parser follows [Thyra](https://github.com/M4i-Imaging-Mass-Spectrometry/thyra).

```python
from imzy import get_reader

reader = get_reader("sample.raw")
print(reader.image_shape, reader.pixel_size, reader.calibration_source)
tic = reader.reshape(reader.get_tic())
image = reader.get_ion_image(27.0, tol=0.05)
```

- The file stores individual ion events, so they are binned onto the detector time channels when the reader is
  created. This reads the whole file once and keeps the sparse spectra in memory.
- Every pixel of the square raster (times the mosaic tiles) is a pixel of the reader; pixels without ions return
  empty spectra. Spectra hold the m/z of the occupied channels and their ion counts summed over all frames
  (`is_centroid` is `True`). `mz_axis` and `tof_axis` give the full channel grid.
- A recalibration appended to the file is used by default; pass `use_appended_calibration=False` to use the
  acquisition header calibration instead.
- The pixel size is `Raster Size (um) / ImagePixels` from the header; pass `pixel_size=` to override it.
- `dropped_events` reports how many events were invalid, outside the raster or outside the mass range.
