"""Benchmark the imzML writer: spectrum appends, finalization, and reader export.

Spectra are generated before timing from a fixed pool that is cycled, so the
timings cover only writer work. Peak RSS is reported for the whole process, so
run one case per process when comparing memory.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from imzy import IMZMLReader, IMZMLWriter, write_imzml

POOL_SIZE = 256


def make_pool(points: int, seed: int = 20261105) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate a pool of sorted centroid spectra with varying lengths."""
    rng = np.random.default_rng(seed)
    pool = []
    for _ in range(POOL_SIZE):
        n = int(rng.integers(points // 2, points + 1))
        mzs = np.sort(rng.uniform(100.0, 1000.0, n))
        intensities = rng.lognormal(3.0, 1.2, size=n).astype(np.float32)
        pool.append((mzs, intensities))
    return pool


def peak_rss_mb() -> float:
    """Return the peak resident set size of this process in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024


def bench_manual(directory: Path, pixels: int, points: int) -> dict[str, float]:
    """Time ``add_spectrum`` calls and ``close`` separately."""
    pool = make_pool(points)
    width = int(np.ceil(np.sqrt(pixels)))
    writer = IMZMLWriter(directory / "manual", ibd_mode="processed", spectrum_type="centroid")
    start = time.perf_counter()
    for i in range(pixels):
        mzs, intensities = pool[i % POOL_SIZE]
        writer.add_spectrum(mzs, intensities, (i % width + 1, i // width + 1))
    add_seconds = time.perf_counter() - start
    start = time.perf_counter()
    writer.close()
    close_seconds = time.perf_counter() - start
    return {
        "add_seconds": add_seconds,
        "close_seconds": close_seconds,
        "total_seconds": add_seconds + close_seconds,
        "us_per_spectrum": 1e6 * (add_seconds + close_seconds) / pixels,
        "imzml_mb": (directory / "manual.imzML").stat().st_size / 1024**2,
        "ibd_mb": (directory / "manual.ibd").stat().st_size / 1024**2,
    }


def bench_export(directory: Path, pixels: int, points: int) -> dict[str, float]:
    """Time ``write_imzml`` from an imzML reader (source generation untimed)."""
    pool = make_pool(points)
    width = int(np.ceil(np.sqrt(pixels)))
    with IMZMLWriter(directory / "source", ibd_mode="processed", spectrum_type="centroid") as writer:
        for i in range(pixels):
            mzs, intensities = pool[i % POOL_SIZE]
            writer.add_spectrum(mzs, intensities, (i % width + 1, i // width + 1))
    with IMZMLReader(directory / "source.imzML") as reader:
        start = time.perf_counter()
        write_imzml(reader, directory / "export", silent=True)
        export_seconds = time.perf_counter() - start
    return {"export_seconds": export_seconds, "us_per_spectrum": 1e6 * export_seconds / pixels}


def main() -> None:
    """Run one benchmark case and print JSON results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=["manual", "export"])
    parser.add_argument("--pixels", type=int, default=100_000)
    parser.add_argument("--points", type=int, default=1_000)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="imzy-writer-bench-") as name:
        func = bench_manual if args.case == "manual" else bench_export
        result = func(Path(name), args.pixels, args.points)
    result = {"case": args.case, "pixels": args.pixels, "points": args.points, **result}
    result["peak_rss_mb"] = peak_rss_mb()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
