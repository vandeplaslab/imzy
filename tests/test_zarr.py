"""Zarr 3 export and centroid-store checks with independent NumPy expectations."""

from pathlib import Path

import numpy as np
import pytest

from imzy import IMZMLReader
from imzy._centroids import ZarrCentroidsStore
from imzy._centroids._extract import rechunk_zarr_array

from .test_imzml_native import _dataset

zarr = pytest.importorskip("zarr")
pytest.importorskip("dask.array")


@pytest.mark.usefixtures("imzml_backend")
@pytest.mark.parametrize("mode", ["centroid", "profile"])
@pytest.mark.parametrize("chunk_size", [None, (7, 2)])
def test_zarr_export(tmp_path: Path, mode: str, chunk_size: tuple[int, int] | None) -> None:
    axis = np.array([100.0, 100.5, 101.0, 103.0])
    spectra = []
    for i in range(253):  # Cross the 250-row extraction boundary and leave a partial block.
        x = axis + (0.125 * (i % 3) if mode == "centroid" else 0)
        y = np.array([1, 2, 4, 8], dtype=np.float32) + (i % 11)
        if mode == "centroid" and i % 31 == 30:
            x, y = x[:0], y[:0]
        spectra.append((x, y))
    path = _dataset(tmp_path, spectra, shared=mode == "profile", mode=mode)
    mzs = np.array([100.25, 101.0, 106.0])
    expected = np.asarray(
        [[y[(x >= mz - 0.25) & (x <= mz + 0.25)].sum() for mz in mzs] for x, y in spectra], dtype=np.float32
    )

    with IMZMLReader(path) as reader:
        result = reader.to_zarr(tmp_path / "peaks.zarr", mzs, tol=0.25, chunk_size=chunk_size, silent=True)

    group = zarr.open_group(str(result), mode="r")
    assert group.metadata.zarr_format == 3
    assert set(group.array_keys()) == {"array", "mzs", "mzs_min", "mzs_max", "ys"}
    assert group["array"].dtype == np.dtype(np.float32)
    assert group["array"].chunks == (chunk_size or (253, 3))
    assert group.attrs["tol"] == 0.25
    assert group.attrs["ppm"] is None
    np.testing.assert_array_equal(group["array"][:], expected)
    np.testing.assert_array_equal(group["mzs"][:], mzs)
    np.testing.assert_array_equal(group["mzs_min"][:], mzs - 0.25)
    np.testing.assert_array_equal(group["mzs_max"][:], mzs + 0.25)
    np.testing.assert_allclose(group["ys"][:], expected.sum(axis=0))

    store = ZarrCentroidsStore(result)
    np.testing.assert_array_equal(store.get_ion(1), expected[:, 1])
    np.testing.assert_array_equal(store.get_ion(100.25), expected[:, 0])
    np.testing.assert_array_equal(store.get_ions(mzs[[2, 0, 0]]), expected[:, [2, 0, 0]])
    with pytest.raises(ValueError, match="cache filtered"):
        store.get_ions(mzs, filtered=True)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("axis_key", ["xs", "mzs"])
def test_store_reads_existing_arrays(tmp_path: Path, zarr_format: int, axis_key: str) -> None:
    path = tmp_path / "existing.zarr"
    group = zarr.open_group(str(path), mode="w", zarr_format=zarr_format)
    mzs = np.array([100.0, 101.0, 102.0])
    values = np.arange(15, dtype=np.float32).reshape(5, 3)
    group.create_array(axis_key, data=mzs)
    group.create_array("array", data=values)

    store = ZarrCentroidsStore(path)
    store.z_store.create_array("filtered", data=values * 2)
    store = ZarrCentroidsStore(path)
    np.testing.assert_array_equal(store.xs, mzs)
    np.testing.assert_array_equal(store.get_ions(mzs[[2, 0]]), values[:, [2, 0]])
    np.testing.assert_array_equal(store.get_ions(mzs[[2, 0]], filtered=True), values[:, [2, 0]] * 2)


def test_rechunk_rejects_oversized_chunks_without_removing_source(tmp_path: Path) -> None:
    source = tmp_path / "source.zarr"
    target = tmp_path / "target.zarr"
    expected = np.arange(12, dtype=np.float32).reshape(4, 3)
    zarr.create_array(str(source), data=expected)

    with pytest.raises(ValueError, match="512 MiB"):
        rechunk_zarr_array(tmp_path, source, target, chunk_size=(1_000_000, 1_000_000), silent=True)

    np.testing.assert_array_equal(zarr.open_array(str(source), mode="r")[:], expected)
    assert not target.exists()
