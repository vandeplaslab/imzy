# Writing imzML

## Convert a reader

```python
from imzy import IMZMLReader, write_imzml

with IMZMLReader("sample.imzML") as reader:
    output = write_imzml(reader, "converted", normalization="TIC", silent=True)
print(output)  # Path to converted.imzML
```

Use `get_reader()` for vendor input. The output is a pair of `.imzML` and `.ibd` files.
Keep both files together. The writer refuses existing outputs unless `overwrite=True`.

| Option | Default | Meaning |
| --- | --- | --- |
| `spectrum_type` | `"auto"` | Infer centroid/profile from the reader, or choose `"centroid"` / `"profile"` |
| `ibd_mode` | `"auto"` | Continuous for profile, processed for centroid; can be explicitly selected |
| `coordinate_origin` | `"auto"` | One-based for imzML readers, zero-based for other readers |
| `indices` | `None` | Export every pixel, or supply original zero-based pixel indices |
| `normalization` | `None` | Apply the named normalization multiplier before writing |
| `on_error` | `"error"` | Stop on errors, or use `"warn"` to skip unreadable/invalid spectra |
| `silent` | `False` | Show progress by default |

Continuous mode requires identical m/z arrays for every spectrum. Processed mode permits independent axes.
Bruker TDF spectra are collapsed onto m/z when exported; ion mobility is not retained.
For readers with centroid-to-profile conversion support, explicitly requesting profile output can resample
centroid data; it does not recover a measured profile spectrum.

## Add spectra manually

```python
from imzy import IMZMLWriter

with IMZMLWriter("manual", spectrum_type="centroid", ibd_mode="processed") as writer:
    writer.add_spectrum([100.0, 101.0], [10.0, 50.0], (1, 1))
    writer.add_spectrum([100.0, 102.0], [20.0, 30.0], (2, 1))
```

Coordinates are `(x, y)` or `(x, y, z)`, in imzML axis order, with one-based coordinates by default.
Two-dimensional coordinates are written with `z=1`. Use `coordinate_origin="zero"` to shift zero-based
coordinates by one during export. Arrays must be finite, one-dimensional, and have matching lengths;
m/z values must be sorted ascending.

`IMZMLWriter` defaults to float64 m/z and float32 intensity arrays. `write_imzml()` preserves reader precision
when available, otherwise using those defaults. Supply `mz_dtype` and `intensity_dtype` to choose explicitly.
Metadata overrides include `polarity` on the manual writer and `pixel_size` / `image_shape` on both interfaces.
See the [writer reference](api.md#writers) for the full signatures.

## Finalization and skipped spectra

Always close the writer, preferably with a context manager. Closing finalizes XML and binary metadata and moves
temporary outputs into place. A context manager cleans up temporary files when its body raises an exception.
Empty spectra are skipped with `EmptySpectrumWarning`, including under `on_error="error"`.
Other skipped spectra emit `SkippedSpectrumWarning` when `on_error="warn"`.

`add_spectrum()` returns `True` when written and `False` when skipped. Review warnings when skipping is enabled;
the output can contain fewer pixels than the input.
