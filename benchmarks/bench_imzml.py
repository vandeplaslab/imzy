"""Warm-page-cache timings; fixture generation, JIT warmup, and cleanup are untimed."""

from __future__ import annotations

import importlib
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from imzy import IMZMLReader, write_imzml
from imzy._normalizations._extract import calculate_normalizations_optimized

from ._data import LAYOUTS, PIXELS, TARGETS, TOLERANCE, consume, dataset_path, generate

params = (["python", "rust"], list(LAYOUTS), list(PIXELS))
param_names = ["backend", "layout", "pixels"]
number = 1
repeat = 3
rounds = 2
# Warmup is explicit: automatic timed warmup would populate TIC/export caches.
warmup_time = 0


def setup_cache() -> None:
    """Generate all fixtures once per environment and revision, including metadata caches."""
    for layout in LAYOUTS:
        for pixels in PIXELS:
            path = generate(Path.cwd(), layout, pixels)
            with IMZMLReader(path, backend="python", parse_lib="ElementTree") as reader:
                # Profile bounds are resolved lazily; persist them for a complete cache.
                _ = reader.mz_min, reader.mz_max
            shutil.copyfile(path, path.with_name(f"{path.stem}-uncached.imzML"))


def setup(backend: str, layout: str, pixels: int) -> None:
    """Reject a disabled Python JIT rather than publish an invalid backend comparison."""
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        raise RuntimeError("Benchmarks require Numba JIT to be enabled.")


def _open(backend: str, layout: str, pixels: int) -> IMZMLReader:
    return IMZMLReader(dataset_path(Path.cwd(), layout, pixels), backend=backend, parse_lib="ElementTree")


def _warm(reader: IMZMLReader) -> None:
    # Warm every extraction signature and the file pages before measurement.
    consume(*reader.get_spectrum(0))
    reader.get_ion_image(float(TARGETS[0]), tol=TOLERANCE, silent=True)
    reader.get_ion_images(TARGETS, tol=TOLERANCE, silent=True)
    calculate_normalizations_optimized(reader.get_intensities(0).astype(np.float32))


class ReaderInitialization:
    """Open and close readers with cached metadata or with XML parsing and cache creation."""

    def setup(self, backend: str, layout: str, pixels: int) -> None:
        """Prepare independent paths so each uncached measurement must parse XML."""
        self.path = dataset_path(Path.cwd(), layout, pixels)
        self.uncached = self.path.with_name(f"{self.path.stem}-uncached.imzML")
        self.uncached.with_suffix(".icache").unlink(missing_ok=True)
        # Import/backend loading and OS page warmup are outside initialization timings.
        with _open(backend, layout, pixels):
            pass

    def time_open_cached(self, backend: str, layout: str, pixels: int) -> None:
        """Measure opening and closing a reader using an existing metadata cache."""
        with IMZMLReader(self.path, backend=backend, parse_lib="ElementTree"):
            pass

    def time_open_uncached(self, backend: str, layout: str, pixels: int) -> None:
        """Measure XML parsing, metadata cache creation, and reader open/close."""
        with IMZMLReader(
            self.uncached, ibd_path=self.path.with_suffix(".ibd"), backend=backend, parse_lib="ElementTree"
        ):
            pass


class Read:
    """Read and consume spectra, extracted images, TICs, and normalization factors."""

    def setup(self, backend: str, layout: str, pixels: int) -> None:
        """Open a reader and warm Numba kernels and input pages outside the timer."""
        self.reader = _open(backend, layout, pixels)
        self.indices = np.random.default_rng(17).integers(0, pixels, size=min(1024, pixels))
        _warm(self.reader)

    def teardown(self, backend: str, layout: str, pixels: int) -> None:
        """Release mappings and handles after each sample."""
        self.reader.close()

    def time_random_spectra(self, backend: str, layout: str, pixels: int) -> float:
        """Consume both complete arrays for a fixed random selection of pixels."""
        return sum(consume(*self.reader.get_spectrum(int(i))) for i in self.indices)

    def time_iteration(self, backend: str, layout: str, pixels: int) -> float:
        """Consume both complete arrays for every spectrum."""
        return sum(consume(x, y) for x, y in self.reader.spectra_iter(silent=True))

    def time_tic(self, backend: str, layout: str, pixels: int) -> float:
        """Compute a fresh TIC vector on every call and consume it."""
        self.reader._tic = None
        return float(self.reader.get_tic(silent=True).sum(dtype=np.float64))

    def time_ion_image(self, backend: str, layout: str, pixels: int) -> float:
        """Extract and consume one ion image."""
        return float(self.reader.get_ion_image(float(TARGETS[0]), tol=TOLERANCE, silent=True).sum(dtype=np.float64))

    def time_ion_images(self, backend: str, layout: str, pixels: int) -> float:
        """Extract and consume 32 ion images in one pass."""
        return float(self.reader.get_ion_images(TARGETS, tol=TOLERANCE, silent=True).sum(dtype=np.float64))

    def time_normalizations(self, backend: str, layout: str, pixels: int) -> float:
        """Calculate and consume every normalization factor for the dataset."""
        factors = self.reader.get_normalizations(silent=True)
        return sum(float(values.sum(dtype=np.float64)) for values in factors.values())


class Export:
    """Measure complete synchronous exports to fresh destinations for every sample."""

    def setup(self, backend: str, layout: str, pixels: int) -> None:
        """Warm input reads and allocate a new output directory outside the timer."""
        for module in ("dask.array", "h5py", "hdf5plugin", "zarr", "imzy._centroids._extract"):
            importlib.import_module(module)
        self.reader = _open(backend, layout, pixels)
        _warm(self.reader)
        self.directory = tempfile.TemporaryDirectory(prefix="imzy-asv-export-")
        self.output = Path(self.directory.name)

    def teardown(self, backend: str, layout: str, pixels: int) -> None:
        """Close readers and remove files outside the timer."""
        self.reader.close()
        self.directory.cleanup()

    def time_imzml(self, backend: str, layout: str, pixels: int) -> None:
        """Convert every spectrum to imzML, including finalization and file close."""
        write_imzml(self.reader, self.output / "converted.imzML", silent=True)

    def time_hdf5(self, backend: str, layout: str, pixels: int) -> None:
        """Extract 32 masses and write a complete HDF5 centroid store."""
        self.reader.to_hdf5(self.output / "ions.h5", TARGETS, tol=TOLERANCE, silent=True)

    def time_zarr(self, backend: str, layout: str, pixels: int) -> None:
        """Extract, write, and rechunk a complete Zarr 3 centroid store."""
        self.reader.to_zarr(self.output / "ions.zarr", TARGETS, tol=TOLERANCE, silent=True)
