# API reference

Import public objects from `imzy`. This reference covers the main workflows; backend SDK internals are outside
the public reader interface. `PathLike` below means a string or a filesystem path accepted by the reader.

## Readers

### Discovery

```python
get_reader(path: PathLike, **kwargs) -> BaseReader
```

Detects a built-in or installed plugin reader. Extra keyword arguments are forwarded to that reader.
Unsupported inputs raise an error. Vendor readers are available only on their [supported platforms](installation.md#formats-and-platforms).

### Constructors

```python
IMZMLReader(
    path, ibd_path=None, auto_profile=True, mz_ppm="auto", resolution=50_000,
    mz_min=None, mz_max=None, parse_lib=None, *, backend="auto",
)
TSFReader(path, use_recalibrated_state=False, auto_profile=True, roi=None)
TDFReader(
    path, use_recalibrated_state=False,
    pressure_compensation_strategy=PressureStrategy.NoCompensation, roi=None,
)
NeoFlexReader(path, use_recalibrated_state=False, auto_profile=True, resolution=30_000, roi=None)
WatersReader(path, auto_profile=True, mz_ppm="auto", resolution="auto", enable_ion_mobility=True)
```

Each constructor returns its corresponding `BaseReader` subclass.
`ibd_path` overrides the imzML binary-file location; otherwise the matching `.ibd` is used.
`mz_min` / `mz_max` override imzML mass-range estimates. `parse_lib` selects the XML parser; leave it as `None`
for automatic selection. `backend` accepts `"auto"`, `"rust"`, or `"python"`.
`resolution` and `mz_ppm` control centroid resampling; their defaults are estimates, not instrument calibration.
`auto_profile` controls automatic centroid-to-profile conversion where the reader implements it.
Bruker `roi=None` uses the reader's default region selection; an integer selects a reported ROI.
`use_recalibrated_state` requests the vendor recalibration state. TDF pressure compensation uses the
`PressureStrategy` enum in `imzy._readers.bruker._tdf` and defaults to no compensation.
Waters `enable_ion_mobility` defaults to enabled.

`IMZMLReader` supports `with ... as reader` and `close() -> None`. Its `backend` property reports
`"python"` or `"rust"`. Closing prevents further reads; returned native arrays can retain the mapping.
See [backend lifetime](backends.md#arrays-and-file-lifetime).

### Shared spectra and image operations

These signatures describe the common usage supported by the built-in readers. Output arrays are NumPy arrays.

| Method | Return value / behavior |
| --- | --- |
| `get_spectrum(index: int)` | `(mzs, intensities)`, two 1-D arrays; indices are zero-based |
| `get_intensities(index: int)` | 1-D intensity array for one pixel |
| `spectra_iter(indices=None, silent=False)` | Iterator of `(mzs, intensities)` pairs; `None` selects all pixels |
| `get_summed_spectrum(indices, scales=None, silent=False)` | `(mzs, summed_intensities)`; `indices` must be nonempty; `scales` has one multiplier per original pixel |
| `get_tic(silent=False)` | Cached total-ion-current vector, shape `(n_pixels,)` |
| `get_ion_image(mz, tol=None, ppm=None, fill_value=np.nan, silent=False)` | One image, shape `(height, width)` |
| `get_ion_images(mzs, tol=None, ppm=None, fill_value=np.nan, silent=False)` | Image stack, shape `(n_masses, height, width)` |
| `to_table(mzs, tol=None, ppm=None, fill_value=np.nan, silent=False)` | Flat intensity table, shape `(n_pixels, n_masses)` |
| `reshape(array, fill_value=0)` | Place `(n_pixels,)` values into a `(height, width)` image |
| `reshape_batch(array, fill_value=0)` | Convert `(n_pixels, n_masses)` to `(n_masses, height, width)` |
| `flatten(array)` | Retrieve acquired pixels from a 2-D image, shape `(n_pixels,)` |

Mass extraction takes exactly one positive `tol` (Da half-width) or `ppm` (relative half-width).
`fill_value` describes unacquired image positions. Centroid summation uses resampling and can produce
a different mass grid from individual spectra.

### Properties

| Property | Value |
| --- | --- |
| `path` | Input dataset path |
| `n_pixels`, `pixels` | Acquired-pixel count and original pixel indices |
| `image_shape` | `(height, width)` |
| `xyz_coordinates` | `(n_pixels, 3)` coordinates in reader convention |
| `x_coordinates`, `y_coordinates`, `z_coordinates` | Per-pixel coordinate vectors |
| `mz_min`, `mz_max` | Reader mass-range bounds |
| `is_centroid` | Whether the input is centroid data |
| `rois` | Available ROI identifiers |
| `x_pixel_size`, `y_pixel_size` | Physical pixel sizes where supplied by the input |

Use reader reshape methods to account for coordinate origin and missing pixels.

## Normalization and extraction

```python
get_normalizations() -> list[str]
reader.get_normalizations(silent=False) -> dict[str, np.ndarray]
reader.extract_normalizations_hdf5(hdf_path, silent=False) -> Path
reader.to_hdf5(
    hdf_path, mzs, tol=None, ppm=None, as_flat=True, max_mem=512, silent=False,
) -> Path
reader.to_zarr(
    zarr_path, mzs, tol=None, ppm=None, as_flat=True, chunk_size=None, silent=False,
) -> Path
```

The module function lists supported names; the reader method calculates `(n_pixels,)` multiplier arrays.
`extract_normalizations_hdf5()` persists factors, reusing an existing file. Storage exports require a nonempty
mass list and the corresponding optional extra. Only flat tables are supported.
HDF5 `max_mem` is in MB; Zarr `chunk_size` is `(pixels, masses)`.
See [normalization and storage](storage.md) for examples and output reuse behavior.

## Writers

```python
write_imzml(
    reader, output_path, *, ibd_mode="auto", spectrum_type="auto", coordinate_origin="auto",
    mz_dtype=None, intensity_dtype=None, on_error="error", overwrite=False, indices=None,
    source_path=None, pixel_size=None, image_shape=None, normalization=None,
    spectrum_transform=None, silent=False,
) -> Path

IMZMLWriter(
    output_path, *, mz_dtype=np.float64, intensity_dtype=np.float32, ibd_mode="auto",
    spectrum_type="centroid", polarity=None, coordinate_origin="one", on_error="error",
    overwrite=False, source_path=None, pixel_size=None, image_shape=None,
)
```

`write_imzml()` returns the output `.imzML` path. `IMZMLWriter.from_reader()` accepts the same parameters
and returns the same result. The manual constructor returns a writer; always close it or use a context manager.

- `ibd_mode`: `"auto"`, `"continuous"`, or `"processed"`. Auto selects continuous for profile and processed for centroid.
- `spectrum_type`: `"centroid"` or `"profile"`; reader conversion additionally accepts `"auto"`.
- `coordinate_origin`: `"zero"`, `"one"`, or `"auto"`. In reader conversion, auto uses the reader format.
  On the manual writer, explicit auto is treated as zero; the manual default is one.
- `mz_dtype`, `intensity_dtype`: float32, float64, int32, or int64; conversion with `None` uses reader precision when available.
- `on_error`: `"error"` or `"warn"`. Empty spectra always warn and skip.
- `indices`: original pixel indices to export; `None` means all.
- `source_path`, `pixel_size`, `image_shape`: source metadata, `(x_size, y_size)` physical pixel size,
  and `(height, width)` image size. Conversion inherits them from the reader when omitted.
- `normalization`: multiplier name, or `None`. `spectrum_transform` is an optional
  `(index, mzs, intensities) -> (mzs, intensities)` callback applied after normalization.
- `polarity`: `"positive"` or `"negative"`, or `None` for unspecified.

```python
writer.add_spectrum(mzs, intensities, coords, user_params=None, *, error_context=None) -> bool
writer.close() -> None
```

`add_spectrum()` accepts matching 1-D arrays and `(x, y)` / `(x, y, z)` coordinates. It returns whether the
spectrum was written. `user_params` accepts mappings with required `name` and `value` entries.
`error_context` adds context to skip warnings.
`close()` finalizes output. Existing files raise `FileExistsError` unless overwrite is enabled.

Warnings exported from `imzy`: `IMZMLWriterWarning`, `SkippedSpectrumWarning`, `EmptySpectrumWarning`.
The latter two specialize skipped-spectrum reporting. See [writing](writing.md) for mode constraints.

## Storage classes

```python
H5CentroidsStore(path, xyz_coordinates=None, pixel_index=None, image_shape=None, mode="a")
ZarrCentroidsStore(path, xyz_coordinates=None, pixel_index=None, image_shape=None)
H5NormalizationStore(path, mode="a")
InMemoryStore(reader, xyz_coordinates=None, pixel_index=None, image_shape=None)
```

HDF5 and Zarr centroid stores expose `xs` (stored mass axis) and `n_peaks` (mass count):

| Method | Result |
| --- | --- |
| `get_ion(value: int \| float)` | `(n_pixels,)`; integer column index or nearest float mass |
| `get_ions(mzs: np.ndarray)` | `(n_pixels, n_masses)` table for nearest stored masses |
| `get_ion_indices(xs: np.ndarray)` | `(matched_masses, column_indices)` |

Zarr `get_ions(mzs, filtered=False)` can select a pre-existing filtered table; requesting missing filtered
data raises `ValueError`. For images, use `reader.reshape()` / `reshape_batch()` with the stored vectors/tables.
HDF5 store methods manage their file access internally; `mode="r"` requests read access.
Zarr opens a group in append mode.

`InMemoryStore` delegates extraction to a reader: `get_ion(mz, fill_value=np.nan, ppm=3.0)` returns a 2-D
image and `get_ions(mzs, fill_value=np.nan, ppm=3.0)` returns a 3-D stack. Unlike persisted stores, it extracts
from the source for each call.

`H5NormalizationStore.get_normalization_list() -> list[str]` lists stored names.
`get_normalization(name, as_median=True, as_multiplier=True)` returns a stored `(n_pixels,)` factor array;
the two flags currently do not alter retrieval. `store[name]` is equivalent.
`normalize(array, name, **kwargs)` applies multipliers to a vector or a flat `(n_pixels, n_masses)` table.

## Reader plugins

Register a package module through the `imzy.plugins` entry point. Implement the `imzy_reader` hook,
decorated with `imzy.hookspec.hook_impl`, returning a `BaseReader` for supported paths and `None` otherwise:

```python
from pathlib import Path
from typing import Any

from imzy import BaseReader
from imzy.hookspec import hook_impl


@hook_impl
def imzy_reader(path: str | Path, **kwargs: Any) -> BaseReader | None:
    """Return a reader for this plugin's supported format."""
    if Path(path).suffix != ".yourformat":
        return None
    # Implement YourReader as a BaseReader subclass in your package.
    return YourReader(path, **kwargs)
```

```toml
[project.entry-points."imzy.plugins"]
your_project_name = "your_project_name.imzy"
```

This is an extension-point sketch: `YourReader` must implement spectrum access, metadata, coordinates,
and any advertised operations. `get_reader()` loads the registered plugins automatically.
