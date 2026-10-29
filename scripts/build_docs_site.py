"""Assemble documentation and ASV charts into a single GitHub Pages artifact."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def assemble_site(docs: Path, benchmarks: Path, output: Path) -> None:
    """Copy both sites into a fresh output directory, retaining the root history URL."""
    archive = benchmarks / "asv-results.tar.gz"
    for source in (docs / "index.html", benchmarks / "index.html", archive):
        if not source.is_file():
            raise FileNotFoundError(f"Missing site input: {source}")
    for reserved in ("benchmarks", "asv-results.tar.gz"):
        if (docs / reserved).exists():
            raise ValueError(f"Documentation output uses a reserved path: {reserved}")
    for source in (docs, benchmarks):
        if output.resolve().is_relative_to(source.resolve()):
            raise ValueError("The combined output must be outside both source directories.")
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    shutil.copytree(docs, output)
    shutil.copytree(benchmarks, output / "benchmarks")
    shutil.copy2(archive, output / archive.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble_site(args.docs, args.benchmarks, args.output)
