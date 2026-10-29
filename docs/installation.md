# Installation

imzy requires Python 3.10 or newer. Install it in a virtual environment:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install imzy
```

## Optional features

| Extra | Install | Purpose |
| --- | --- | --- |
| HDF5 | `pip install "imzy[hdf5]"` | Export and read HDF5 centroid/normalization stores |
| Zarr | `pip install "imzy[zarr]"` | Export and read Zarr centroid stores |
| Plotting | `pip install "imzy[plot]"` | Plot examples with Matplotlib |
| All | `pip install "imzy[all]"` | All features supported on your Python version |

Zarr support requires Python 3.11 or newer and Zarr 3 (`zarr>3,<4`). On Python 3.10, the `all` and `test`
extras omit Zarr. You can combine extras, for example `pip install "imzy[hdf5,plot]"`.

## Formats and platforms

| Input | Windows | macOS | Linux |
| --- | --- | --- | --- |
| imzML (`.imzML` with its `.ibd`) | Yes | Yes | Yes |
| Bruker TSF/TDF/NeoFlex (`.d`) | Yes | No | Yes |
| Waters MassLynx (`.raw`) | Yes | No | No |

Vendor formats depend on their native SDKs. Use imzML for the portable examples.
Keep the imzML XML and binary `.ibd` file together. Bruker TDF export collapses ion mobility onto the m/z axis.

## Installing from source

Release wheels include the Rust imzML extension and need no Rust compiler. Source builds compile the extension
when Cargo is available. To require compilation, install the [Rust toolchain](https://rustup.rs/) and run:

```bash
IMZY_BUILD_RUST=1 pip install .
```

To explicitly install without compilation:

```bash
IMZY_BUILD_RUST=0 pip install .
```

On PowerShell, set `$env:IMZY_BUILD_RUST = "1"` (or `"0"`) before `pip install .`.
Compilation failures are reported. See [backend selection](backends.md) for fallback behavior and thread settings.
