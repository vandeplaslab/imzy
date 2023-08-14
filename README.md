# imzy

<div align="center">

[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/imzy/imzy/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)
[![Python Version](https://img.shields.io/pypi/pyversions/imzy.svg)](https://pypi.org/project/imzy/)
[![Python package index](https://img.shields.io/pypi/v/imzy.svg)](https://pypi.org/project/imzy)
[![Python package index download statistics](https://img.shields.io/pypi/dm/imzy.svg)](https://pypistats.org/packages/imzy)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/imzy/imzy/blob/main/.pre-commit-config.yaml)
[![Code coverage](https://codecov.io/gh/vandeplaslab/imzy/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/imzy)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/imzy/imzy/releases)
[![License](https://img.shields.io/github/license/imzy/imzy)](https://github.com/imzy/imzy/blob/main/LICENSE)

imzy: A simple reader interface to imzML, Bruker (.tdf/.tsf) file formats

</div>

## Getting started
Install using pip
```bash
pip install imzy
```

Analyse your data
```python
import numpy as np
from imzy import get_reader

PATH_TO_FILE = "path/to/file"

# we currently support imzML, Bruker .d (.tsf/.tdf) formats
reader = get_reader(PATH_TO_FILE)
# will extract mass spectrum for pixel index '0'
mz_x, mz_y = reader.get_mass_spectrum(0)
# will extract summed mass spectrum for pixel indices 0-100
mz_x, mz_y = reader.get_summed_spectrum(np.arange(100))
# iterate over all mass spectra in the dataset
for mz_x, mz_y in reader.spectra_iter():
  ...
# get tic array
tic = reader.get_tic()
# this array is 1d so needs to be reshaped to 2d if you want to view it as an image
tic = reader.reshape(tic)
# will extract the 885.549 ion image with 10 ppm window around it
image = reader.get_ion_image(885.549, ppm=10)
# you can also extract multiple images at the same time (which is much more efficient since the spectra
# only need to be loaded into memory once)
mzs = [...] # list of m/zs to extract
images = reader.get_ion_images(mzs, tol=0.05)
```

## Supported formats
- imzML on Windows, macOS and Linux
- Bruker (.tdf/.tsf) on Windows and Linux


## Plugins

It is now possible to create your own readers by implementing the `imzy.hookspec` interface. This allows you to create
your own readers for any format you want. You can then register your reader with imzy by adding the following to your
`setup.py` or `pyproject.toml` or `setup.cfg` file:

If you have project named `your_project_name`, you could add a file `imzy.py` to your project with the following code:

```python
from imzy import BaseReader
from imzy.hookspec import hook_impl

class YourReader(BaseReader):
  """Your reader class."""
  

@hook_impl
def imzy_reader(path: str, **kwargs) -> ty.Optional[YourReader]:
    """Return YourReader if path is valid."""
    ...
```

In the `pyproject.toml` file, please define the interface:
```toml
[options.entry_points."imzy.plugins"]
your_project_name = "your_project_name.imzy"
```

Your reader will be automatically detected when the `ImzyPluginManager` is initialized, which happens when the
`get_reader` function is called. You can then use your reader as follows:


## Planned features
- add functionality to readers
- improve performance
- improve tests
- add better caching support
- add support for Waters (.raw) files
- add support for Thermo (.raw) files

## Contributing

### Initialize your code

1. Initialize `git` inside your repo:

```bash
cd imzy && git init
```

2. Create conda environment. We are using `imzy` as its name.

```bash
conda create -n imzy python=3.9
```

3. Initialize and install `pre-commit` hooks:

```bash
make develop
make pre-commit-install
```

4. Run the codestyle:

```bash
make codestyle
```

5. Upload initial code to GitHub:

```bash
git add .
git commit -m ":tada: Initial commit"
git branch -M main
git remote add origin https://github.com/imzy/imzy.git
git push -u origin main
```


## 🛡 License

[![License](https://img.shields.io/github/license/vandeplaslab/imzy)](https://github.com/vandeplaslab/imzy/blob/main/LICENSE)

This project is licensed under the terms of the `BSD-3` license. See [LICENSE](https://github.com/imzy/imzy/blob/main/LICENSE) for more details.

## 📃 Citation

```bibtex
@misc{imzy,
  author = {imzy},
  title = {imzy: A new reader/writer interface to imzML and other imaging mass spectrometry formats.},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/imzy/imzy}}
}
```
