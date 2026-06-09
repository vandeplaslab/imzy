# imzy

<div align="center">

[![Python Version](https://img.shields.io/pypi/pyversions/imzy.svg)](https://pypi.org/project/imzy/)
[![PyPI](https://img.shields.io/pypi/v/imzy.svg)](https://pypi.org/project/imzy)
[![Downloads](https://img.shields.io/pypi/dm/imzy.svg)](https://pypistats.org/packages/imzy)
[![Code coverage](https://codecov.io/gh/vandeplaslab/imzy/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/imzy)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/vandeplaslab/imzy/blob/main/.pre-commit-config.yaml)
[![License](https://img.shields.io/github/license/vandeplaslab/imzy)](https://github.com/vandeplaslab/imzy/blob/main/LICENSE)

Read, inspect, process, and export imaging mass spectrometry datasets from Python.

</div>

imzy provides a common reader interface for imzML, Bruker, and Waters imaging mass spectrometry data. It includes tools
for reading spectra, extracting ion images, computing common normalization factors, exporting centroid tables to HDF5 or
Zarr, and converting supported inputs to imzML.

## Installation

Install the core package with pip:

```bash
pip install imzy
```

Optional storage and plotting extras are available when you need them:

```bash
pip install "imzy[hdf5]"
pip install "imzy[zarr]"
pip install "imzy[plot]"
pip install "imzy[all]"
```

imzy requires Python 3.10 or newer.

## Supported Formats

| Format | Support |
| --- | --- |
| imzML (`.imzML`/`.ibd`) | Windows, macOS, Linux |
| Bruker TSF/TDF/NeoFlex (`.d`) | Windows, Linux |
| Waters MassLynx (`.raw`) | Windows |

Bruker TDF ion mobility data can be read, but imzML export currently writes spectra collapsed to the m/z axis.

## Python Usage

```python
import numpy as np
from imzy import get_reader, write_imzml

path = "path/to/dataset"
reader = get_reader(path)

# Read one spectrum by pixel index.
mzs, intensities = reader.get_spectrum(0)

# Sum spectra across selected pixels.
summed_mzs, summed_intensities = reader.get_summed_spectrum(np.arange(100))

# Iterate through spectra. Each item is an (m/z, intensity) pair.
for mzs, intensities in reader.spectra_iter():
    ...

# Create a 2-D total ion current image.
tic_image = reader.reshape(reader.get_tic())

# Extract one ion image, using either a tolerance in Da or ppm.
ion_image = reader.get_ion_image(885.549, ppm=10)

# Extract many ion images in one pass.
target_mzs = [885.549, 886.552, 887.555]
images = reader.get_ion_images(target_mzs, tol=0.05)

# Export any imzy reader to imzML.
output_path = write_imzml(reader, "converted_dataset", silent=True)
```

Useful reader properties include `n_pixels`, `image_shape`, `x_coordinates`, `y_coordinates`, `xyz_coordinates`,
`mz_min`, `mz_max`, `is_centroid`, `rois`, `x_pixel_size`, and `y_pixel_size`.

## Command Line Usage

The `imzy` command can convert any supported input to imzML:

```bash
imzy convert path/to/input path/to/output --silent
```

Common options:

```bash
imzy convert path/to/input path/to/output \
  --spectrum-type auto \
  --ibd-mode auto \
  --normalization TIC \
  --overwrite
```

Use `--roi` to export a specific Bruker TSF/TDF/NeoFlex region of interest:

```bash
imzy convert path/to/bruker.d path/to/output --roi 0
```

Run `imzy convert --help` for the full list of options.

## Normalization and Batch Exports

Readers can compute normalization multipliers for common strategies such as TIC, RMS, median, percentile-filtered TIC,
and vector norms:

```python
normalizations = reader.get_normalizations(silent=True)
tic_scales = normalizations["TIC"]
```

For larger extraction jobs, write flat centroid tables to optional HDF5 or Zarr stores:

```python
target_mzs = [734.569, 760.585, 782.567]

hdf5_path = reader.to_hdf5("centroids.h5", target_mzs, ppm=10, silent=True)
zarr_path = reader.to_zarr("centroids.zarr", target_mzs, ppm=10, silent=True)
```

Install `imzy[hdf5]` or `imzy[zarr]` before using those storage backends.

## imzML Writing

Use `write_imzml` for reader-to-reader conversion, or `IMZMLWriter` when you need to add spectra manually:

```python
from imzy import IMZMLWriter

with IMZMLWriter("manual_export", spectrum_type="centroid", ibd_mode="processed") as writer:
    writer.add_spectrum([100.0, 101.0], [10.0, 50.0], (1, 1))
    writer.add_spectrum([100.0, 101.0], [20.0, 30.0], (2, 1))
```

Writer options include `spectrum_type`, `ibd_mode`, `coordinate_origin`, `normalization`, `indices`, `overwrite`, and
`on_error`.

## Plugins

Custom readers can be registered through the `imzy.plugins` entry point. Implement the `imzy_reader` hook and return a
reader when the input path is supported:

```python
from __future__ import annotations

import typing as ty
from pathlib import Path

from koyo.typing import PathLike

from imzy import BaseReader
from imzy.hookspec import hook_impl


class YourReader(BaseReader):
    """Reader for your custom imaging format."""


@hook_impl
def imzy_reader(path: PathLike, **kwargs: ty.Any) -> BaseReader | None:
    """Return a reader when path points to a supported dataset."""
    if Path(path).suffix != ".yourformat":
        return None
    return YourReader(path, **kwargs)
```

Then expose the module from your package metadata:

```toml
[project.entry-points."imzy.plugins"]
your_project_name = "your_project_name.imzy"
```

When `get_reader()` is called, imzy registers built-in readers and then loads installed plugins from this entry point.

## Development

Install the project in development mode and set up the local checks:

```bash
make develop
make pre-commit-install
```

Run the main checks before opening a change:

```bash
make codestyle
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [SECURITY.md](SECURITY.md) for reporting security
issues.

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@misc{imzy,
  author = {Migas, Lukasz G. and contributors},
  title = {imzy: A reader and writer interface for imzML and other imaging mass spectrometry formats},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/vandeplaslab/imzy}}
}
```
