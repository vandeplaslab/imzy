"""Benchmark both backends with equivalent, consumed results and warm OS caches.

Data generation, metadata parsing, correctness checks, and JIT warmup are outside
the timings. Every timed spectrum read consumes both complete arrays. TIC caching
is reset on every call. Backend order alternates to reduce ordering bias.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from imzy import IMZMLReader, IMZMLWriter


def generate(directory: Path, pixels: int, points: int, continuous: bool) -> Path:
    """Generate seeded spectra with varying intensities and, when processed, axes."""
    path = directory / ("continuous.imzML" if continuous else "processed.imzML")
    rng = np.random.default_rng(20261026)
    axis = np.linspace(100.0, 1000.0, points)
    with IMZMLWriter(
        path,
        spectrum_type="profile" if continuous else "centroid",
        ibd_mode="continuous" if continuous else "processed",
    ) as writer:
        for i in range(pixels):
            mzs = (
                axis if continuous else np.sort(rng.uniform(100.0, 1000.0, int(rng.integers(points // 2, points + 1))))
            )
            intensities = rng.lognormal(3.0, 1.2, size=len(mzs)).astype(np.float32)
            writer.add_spectrum(mzs, intensities, (i % 128 + 1, i // 128 + 1))
    return path


def consume(x: np.ndarray, y: np.ndarray) -> float:
    """Touch every input value with the same reduction for both backends."""
    return float(x.sum(dtype=np.float64) + y.sum(dtype=np.float64))


def operations(reader: IMZMLReader, indices: np.ndarray, targets: np.ndarray) -> dict[str, Callable[[], float]]:
    """Construct matched workloads, including uncached TIC computation."""

    def random_spectra() -> float:
        return sum(consume(*reader.get_spectrum(int(i))) for i in indices)

    def iterate() -> float:
        return sum(consume(x, y) for x, y in reader.spectra_iter(silent=True))

    def image() -> float:
        return float(reader.get_ion_image(float(targets[0]), tol=0.3, silent=True).sum(dtype=np.float64))

    def images() -> float:
        return float(reader.get_ion_images(targets, tol=0.3, silent=True).sum(dtype=np.float64))

    def tic() -> float:
        reader._tic = None
        return float(reader.get_tic(silent=True).sum(dtype=np.float64))

    return {"random_spectra": random_spectra, "iterate": iterate, "image": image, "images": images, "tic": tic}


def benchmark(path: Path, targets: np.ndarray, repeats: int) -> dict:
    """Validate numerical agreement and report median timings for warm reads."""
    with IMZMLReader(path, backend="python") as python, IMZMLReader(path, backend="rust") as rust:
        indices = np.random.default_rng(17).integers(0, python.n_pixels, size=min(1024, python.n_pixels))
        readers = {"python": python, "rust": rust}
        work = {backend: operations(reader, indices, targets) for backend, reader in readers.items()}
        # Compare complete outputs as well as checksums before timing anything.
        np.testing.assert_allclose(
            rust.to_table(targets, tol=0.3, silent=True),
            python.to_table(targets, tol=0.3, silent=True),
            rtol=1e-5,
        )
        np.testing.assert_allclose(rust.get_tic(silent=True), python.get_tic(silent=True), rtol=1e-5)
        for index in indices:
            for actual, expected in zip(rust[int(index)], python[int(index)], strict=True):
                np.testing.assert_array_equal(actual, expected)
        times = {name: {backend: [] for backend in readers} for name in work["python"]}
        for name in times:
            reference = work["python"][name]()  # Warm JIT and the file pages for both implementations.
            np.testing.assert_allclose(work["rust"][name](), reference, rtol=1e-5)
            for repeat in range(repeats):
                for backend in ("python", "rust") if repeat % 2 == 0 else ("rust", "python"):
                    start = time.perf_counter()
                    checksum = work[backend][name]()
                    elapsed = time.perf_counter() - start
                    times[name][backend].append(elapsed)
                    np.testing.assert_allclose(checksum, reference, rtol=1e-5)
        result = {}
        for name, values in times.items():
            medians = {backend: statistics.median(samples) for backend, samples in values.items()}
            count = len(indices) if name == "random_spectra" else python.n_pixels
            result[name] = {
                "seconds": medians,
                "samples_seconds": values,
                "speedup": medians["python"] / medians["rust"],
                "pixels_per_second": {backend: count / value for backend, value in medians.items()},
            }
        return {"ibd_bytes": path.with_suffix(".ibd").stat().st_size, "operations": result}


def main() -> None:
    """Run portable synthetic benchmarks and optionally save their JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=8192)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--targets", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.pixels < 1 or args.points < 2 or args.targets < 1 or args.repeats < 1:
        parser.error("pixels, targets, and repeats must be positive; points must be at least two")
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        parser.error("Enable Numba JIT for a fair Python baseline")
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "parameters": {key: value for key, value in vars(args).items() if key != "output"},
        "cache": "warm OS page cache; data generation, parsing, and JIT excluded",
        "cases": {},
    }
    with tempfile.TemporaryDirectory(prefix="imzy-benchmark-") as directory:
        for continuous in (True, False):
            path = generate(Path(directory), args.pixels, args.points, continuous)
            case = "continuous_profile" if continuous else "processed_centroid"
            result["cases"][case] = benchmark(path, np.linspace(150.0, 950.0, args.targets), args.repeats)
            for name, measurements in result["cases"][case]["operations"].items():
                print(f"{case:20s} {name:15s} {measurements['speedup']:6.2f}x {measurements['seconds']}")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
