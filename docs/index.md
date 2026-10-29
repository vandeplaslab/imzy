# imzy

Read, inspect, process, and export imaging mass spectrometry datasets from Python.
imzy provides a common reader interface for imzML, Bruker TSF/TDF/NeoFlex, and Waters MassLynx data.

```bash
pip install imzy
```

```python
from imzy import get_reader

reader = get_reader("sample.imzML")
mzs, intensities = reader.get_spectrum(0)
image = reader.get_ion_image(734.569, ppm=10, silent=True)
```

Start with [installation](installation.md) and the [quickstart](quickstart.md), which creates its own sample dataset.
The [user guide](usage.md) explains spectra and images, and the [examples](examples.md) provide runnable scripts.
Use the [API reference](api.md) for parameters, defaults, and return shapes.

## Capabilities

- Read individual spectra, iterate pixels, and sum spectra over selected pixels.
- Extract single or multiple ion images with Da or ppm windows.
- Calculate normalization multipliers and export mass-targeted tables to HDF5 or Zarr.
- Convert supported inputs to imzML through Python or the [command line](cli.md).
- Choose the [Rust or Python imzML backend](backends.md).

See [performance methodology](performance/index.md), the [recorded backend comparison](performance/imzml.md),
and [interactive benchmark history](https://vandeplaslab.github.io/imzy/benchmarks/).

## Project

imzy is released under the [BSD 3-Clause License](https://github.com/vandeplaslab/imzy/blob/main/LICENSE).
Report bugs on [GitHub](https://github.com/vandeplaslab/imzy/issues).
See the repository's [contribution guide](https://github.com/vandeplaslab/imzy/blob/main/CONTRIBUTING.md)
and [security policy](https://github.com/vandeplaslab/imzy/blob/main/SECURITY.md).

These docs track `main`. A released PyPI version can differ from the current development version.
