"""Runnable documentation examples and combined Pages layout."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scripts.build_docs_site import assemble_site

from imzy import IMZMLReader

ROOT = Path(__file__).resolve().parents[1]


def _run_example(name: str, output: Path, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), str(output), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(output.parent / "matplotlib"),
            "XDG_CACHE_HOME": str(output.parent / "cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_quickstart_example(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output = tmp_path / "quickstart"
    _run_example("quickstart.py", output, "--plot")
    with IMZMLReader(output / "sample.imzML") as reader:
        np.testing.assert_array_equal(reader.get_tic(silent=True), [6, 12, 18, 24])
        images = reader.get_ion_images([100.0, 200.0], tol=0.1, silent=True)
        np.testing.assert_array_equal(images, [[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        np.testing.assert_allclose(reader.get_normalizations(silent=True)["TIC"], [2.5, 1.25, 5 / 6, 0.625])
    with IMZMLReader(output / "normalized.imzML") as reader:
        assert reader.n_pixels == 4
        for mzs, intensities in reader.spectra_iter(silent=True):
            np.testing.assert_array_equal(mzs, [100, 200, 300])
            np.testing.assert_allclose(intensities, [2.5, 5, 7.5], rtol=1e-6)
    assert (output / "quickstart.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("backend", ["hdf5", "zarr"])
def test_storage_example(tmp_path: Path, backend: str) -> None:
    if backend == "hdf5":
        pytest.importorskip("h5py")
        pytest.importorskip("hdf5plugin")
    else:
        if sys.version_info < (3, 11):
            pytest.skip("The Zarr extra requires Python 3.11+.")
        pytest.importorskip("zarr", minversion="3.0")
        pytest.importorskip("dask.array")
    _run_example("storage.py", tmp_path / backend, "--backend", backend)


@pytest.mark.parametrize("missing", ["docs/index.html", "charts/index.html", "charts/asv-results.tar.gz"])
def test_assembly_requires_complete_inputs(tmp_path: Path, missing: str) -> None:
    for name in ("docs/index.html", "charts/index.html", "charts/asv-results.tar.gz"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        if name != missing:
            path.write_bytes(b"input")
    output = tmp_path / "combined"
    with pytest.raises(FileNotFoundError):
        assemble_site(tmp_path / "docs", tmp_path / "charts", output)
    assert not output.exists()


@pytest.mark.parametrize("reserved", ["benchmarks", "asv-results.tar.gz"])
def test_assembly_rejects_documentation_collisions(tmp_path: Path, reserved: str) -> None:
    docs, charts = tmp_path / "docs", tmp_path / "charts"
    docs.mkdir()
    charts.mkdir()
    (docs / "index.html").write_text("Docs")
    (docs / reserved).write_text("Reserved")
    (charts / "index.html").write_text("Charts")
    (charts / "asv-results.tar.gz").write_bytes(b"history")
    with pytest.raises(ValueError, match="reserved"):
        assemble_site(docs, charts, tmp_path / "combined")


def test_assembly_preserves_existing_output(tmp_path: Path) -> None:
    docs, charts, output = tmp_path / "docs", tmp_path / "charts", tmp_path / "combined"
    for directory in (docs, charts, output):
        directory.mkdir()
        (directory / "index.html").write_text(directory.name)
    (charts / "asv-results.tar.gz").write_bytes(b"history")
    with pytest.raises(FileExistsError):
        assemble_site(docs, charts, output)
    assert (output / "index.html").read_text() == "combined"
