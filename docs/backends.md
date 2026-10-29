# Rust imzML backend

The imzML reader supports memory-mapped Rust access and a Python fallback behind the same reader interface.
The XML parser and metadata cache remain in Python.

```python
from imzy import IMZMLReader

with IMZMLReader("sample.imzML", backend="auto") as reader:
    print(reader.backend)
    mzs, intensities = reader.get_spectrum(0)
    writable_intensities = intensities.copy()
```

| Selection | Behavior |
| --- | --- |
| `backend="auto"` | Use Rust when the extension loads, otherwise Python |
| `backend="rust"` | Require Rust; raise `ImportError` when the extension cannot load |
| `backend="python"` | Explicitly use Python |

Auto selection is a fallback for extension availability; malformed data still raises errors.
Release wheels include the extension. See [source installation](installation.md#installing-from-source)
when compiling locally. After editing Rust sources, reinstall the package to rebuild the extension.

## Arrays and file lifetime

Returned spectrum arrays are read-only. Aligned native arrays can be views into the file; unaligned arrays
are copied safely. Use `.copy()` when you need writable data.

Keep input files unchanged while a reader, array, or iterator exists. `close()` releases the reader's mapping,
but native arrays and iterators retain their own mappings. They may keep file locks alive on Windows after
the reader closes. Release those references before replacing the dataset.

## Supported arrays and parallelism

Supported imzML arrays are uncompressed, little-endian float32, float64, int32, and int64.
Ion extraction requires ascending m/z arrays. Rust accelerates spectrum access/iteration, TIC calculation,
and ion-image extraction. Rayon parallelizes bulk extraction. Set the worker count before starting Python:

```bash
RAYON_NUM_THREADS=2 python analysis.py
```

On PowerShell, set `$env:RAYON_NUM_THREADS = "2"` before starting Python.
See [benchmark methodology](performance/index.md) for the controlled thread settings used in CI,
and the [recorded comparison](performance/imzml.md) for measured workloads and limitations.
