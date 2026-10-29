# Runnable examples

These scripts generate their own two-by-two imzML data. No external datasets or notebook runtime is required.
Run commands from a repository checkout, or download both scripts into the same directory and omit `examples/`
from the commands. Use a fresh output directory for each run to avoid replacing previous data.

## Spectra, images, and conversion

```bash
pip install "imzy[plot]"
python examples/quickstart.py /tmp/imzy-quickstart --plot
```

On Windows, use an output path such as `example-output/quickstart`.
Omit `--plot` to run without Matplotlib. The script writes `sample.imzML` / `sample.ibd`,
`normalized.imzML` / `normalized.ibd`, and optionally `quickstart.png`.

The input spectra contain `[1, 2, 3]` multiplied by pixel numbers 1 through 4.
The TIC vector is `[6, 12, 18, 24]`; the m/z 200 image is `[[2, 4], [6, 8]]`.
TIC factors scale each spectrum to a total intensity of 15, the median input TIC.
The script reads the normalized output back to check this result.

[Download quickstart.py](examples/quickstart.py).

```python
--8<-- "examples/quickstart.py"
```

## Storage

Run each backend in a different output directory. HDF5 requires `imzy[hdf5]`;
Zarr requires Python 3.11+ and `imzy[zarr]`.

```bash
pip install "imzy[hdf5,zarr]"
python examples/storage.py /tmp/imzy-hdf5 --backend hdf5
python examples/storage.py /tmp/imzy-zarr --backend zarr
```

Keep `quickstart.py` alongside `storage.py`: it supplies the shared sample-data function.
The script checks the exported `(4, 2)` table against known intensities and reshapes it into `(2, 2, 2)` images.

[Download storage.py](examples/storage.py).

```python
--8<-- "examples/storage.py"
```

## Notebooks

The repository also contains exploratory notebooks for
[basic reading and plotting](https://github.com/vandeplaslab/imzy/blob/main/examples/basic_reader.ipynb)
and [imzML conversion](https://github.com/vandeplaslab/imzy/blob/main/examples/convert_to_imzml.ipynb).
Those require your own input data and platform-specific readers. The scripts above are the automated smoke-tested examples.
