"""Publish repository examples and benchmark reports without duplicating their sources."""

from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import File, Files


def on_files(files: Files, *, config: MkDocsConfig) -> Files:
    """Include existing reports, raw measurements, and downloadable example scripts."""
    root = Path(config.config_file_path).resolve().parent
    sources = {
        "performance/index.md": "benchmarks/README.md",
        "performance/imzml.md": "benchmarks/imzml.md",
        "performance/imzml-macos-arm64.json": "benchmarks/imzml-macos-arm64.json",
        "examples/quickstart.py": "examples/quickstart.py",
        "examples/storage.py": "examples/storage.py",
    }
    for destination, source in sources.items():
        files.append(File.generated(config, destination, abs_src_path=str(root / source)))
    return files
