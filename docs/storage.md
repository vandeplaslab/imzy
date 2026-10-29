# Normalization and storage

The snippets here use `sample.imzML` from the [quickstart](quickstart.md).

## Normalization multipliers

```python
from imzy import IMZMLReader, get_normalizations

print(get_normalizations())  # supported names; this does not calculate factors

with IMZMLReader("sample.imzML") as reader:
    normalizations = reader.get_normalizations(silent=True)
    scales = normalizations["TIC"]  # one multiplier per acquired pixel

    image = reader.get_ion_image(200.0, tol=0.1, silent=True)
    normalized_image = image * reader.reshape(scales)

    table = reader.to_table([100.0, 200.0], tol=0.1, silent=True)
    normalized_table = table * scales[:, None]

    mzs, normalized_sum = reader.get_summed_spectrum(reader.pixels, scales=scales, silent=True)
```

Multiply by the factors; do not divide by them. Factors have shape `(n_pixels,)`, including when you sum a
subset of pixels: `scales` is indexed by the original pixel index. Strategies include TIC, RMS, Median,
percentile-filtered TIC, and 0/2/3-norm. Use `get_normalizations()` for the exact names.
For zero-signal or non-finite input, inspect the resulting factors before analysis.

## HDF5 centroid tables

Install `imzy[hdf5]`. Exporting extracts the requested mass windows and stores a flat table, not every original spectrum:

```python
from imzy import H5CentroidsStore, IMZMLReader

with IMZMLReader("sample.imzML") as reader:
    path = reader.to_hdf5("centroids.h5", [100.0, 200.0], tol=0.1, silent=True)
    store = H5CentroidsStore(path, mode="r")
    ion = store.get_ion(200.0)  # (n_pixels,), nearest stored mass
    image = reader.reshape(ion)
```

`max_mem=512` is the default extraction memory budget in MB. Use a fresh output path for each extraction setup:
existing HDF5 output may be reused when its stored mass list matches. A reused path is not a request to recompute
with new tolerances. Only `as_flat=True` is supported.

## Zarr centroid tables

Install `imzy[zarr]` on Python 3.11 or newer. Exports use Zarr 3:

```python
import numpy as np
from imzy import IMZMLReader, ZarrCentroidsStore

with IMZMLReader("sample.imzML") as reader:
    path = reader.to_zarr("centroids.zarr", [100.0, 200.0], tol=0.1, silent=True)
    store = ZarrCentroidsStore(path)
    table = store.get_ions(np.array([200.0, 100.0]))  # (n_pixels, 2)
    images = reader.reshape_batch(table)
```

The default `chunk_size=None` lets the exporter choose the chunk shape. A supplied tuple describes
`(pixels_per_chunk, masses_per_chunk)`; keep individual chunks within the exporter's 512 MiB limit.
Only flat export is supported. Store queries select the nearest mass already extracted; they do not re-integrate
the source spectra. Pass an integer to `get_ion()` to select a stored column by index, or a float to select a mass.

## Persist normalization factors

```python
from imzy import H5NormalizationStore, IMZMLReader

with IMZMLReader("sample.imzML") as reader:
    path = reader.extract_normalizations_hdf5("normalizations.h5", silent=True)
store = H5NormalizationStore(path, mode="r")
names = store.get_normalization_list()
scales = store.get_normalization("TIC")
```

This requires the HDF5 extra. An existing normalization file is reused. See the [storage example](examples.md#storage)
for a complete export and read-back check.
