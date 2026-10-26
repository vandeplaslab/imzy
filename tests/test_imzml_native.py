"""Independent binary reader checks for both imzML backends."""

from __future__ import annotations

import gc
import importlib
import importlib.util
import os
import typing as ty
import weakref
from pathlib import Path

import numpy as np
import pytest
from ims_utils.spectrum import get_mzs_for_tol

from imzy import IMZMLReader
from imzy._readers.imzml._imzml import DTYPE_DICT

_HAS_RUST = importlib.util.find_spec("imzy._imzml_rust") is not None
if os.environ.get("IMZY_REQUIRE_RUST") == "1":
    importlib.import_module("imzy._imzml_rust")


@pytest.fixture(
    params=["python", pytest.param("rust", marks=pytest.mark.skipif(not _HAS_RUST, reason="Rust unavailable"))]
)
def backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _dataset(
    directory: Path,
    spectra: list[tuple[np.ndarray, np.ndarray]],
    mz_code: str = "d",
    int_code: str = "f",
    *,
    shared: bool = False,
    unaligned: bool = False,
    mode: str = "centroid",
) -> Path:
    """Write independent little-endian arrays and an existing-format metadata cache."""
    path = directory / "data.imzML"
    path.write_text("<mzML/>")  # Complete cached metadata avoids XML parsing in these binary tests.
    offsets = []
    with path.with_suffix(".ibd").open("wb") as stream:
        stream.write(bytes(16) + (b"x" if unaligned else b""))
        mz_offset = stream.tell()
        for index, (mz, intensity) in enumerate(spectra):
            if not shared or index == 0:
                mz_offset = stream.tell()
                stream.write(mz.astype(DTYPE_DICT[mz_code]).tobytes())
            int_offset = stream.tell()
            stream.write(intensity.astype(DTYPE_DICT[int_code]).tobytes())
            offsets.append((mz_offset, len(mz), int_offset, len(intensity)))
    with path.with_suffix(".icache").open("wb") as stream:
        np.savez(
            stream,
            px_max_x=len(spectra) + 1,
            px_max_y=1,
            px_max_z=1,
            px_size_x=1.0,
            px_size_y=1.0,
            mz_precision=mz_code,
            int_precision=int_code,
            byte_offsets=np.asarray(offsets, dtype=np.int64).reshape(-1, 4),
            xyz_coordinates=np.asarray([(i + 1, 1, 1) for i in range(len(spectra))], dtype=np.int64).reshape(-1, 3),
            mz_min=100.0,
            mz_max=104.0,
            spectrum_mode=mode,
        )
    return path


def _edit_cache(path: Path, **updates: ty.Any) -> None:
    with np.load(path.with_suffix(".icache")) as cache:
        values = {name: cache[name] for name in cache.files}
    values.update(updates)
    with path.with_suffix(".icache").open("wb") as stream:
        np.savez(stream, **values)


@pytest.mark.parametrize("mz_code", list(DTYPE_DICT))
@pytest.mark.parametrize("int_code", list(DTYPE_DICT))
@pytest.mark.parametrize("unaligned", [False, True])
def test_fixed_width_spectra_and_iteration(
    tmp_path: Path, backend: str, mz_code: str, int_code: str, unaligned: bool
) -> None:
    mz = np.array([100, 101, 103])
    intensity = np.array([1, 7, 23]) + (2**40 if int_code == "l" else 0)
    spectra = [(mz, intensity), (mz + 1, intensity + 2), (mz[:0], intensity[:0])]
    path = _dataset(tmp_path, spectra, mz_code, int_code, unaligned=unaligned)
    with IMZMLReader(path, backend=backend) as reader:
        assert reader.backend == backend
        for i, (x, y) in enumerate(spectra):
            actual_x, actual_y = reader[i]
            np.testing.assert_array_equal(actual_x, x.astype(DTYPE_DICT[mz_code]))
            np.testing.assert_array_equal(actual_y, y.astype(DTYPE_DICT[int_code]))
            assert actual_x.dtype == DTYPE_DICT[mz_code]
            assert actual_y.dtype == DTYPE_DICT[int_code]
            assert not actual_x.flags.writeable
            assert not actual_y.flags.writeable
            np.testing.assert_array_equal(reader.get_intensities(i), actual_y)
        np.testing.assert_array_equal(reader[-2][0], spectra[1][0])
        selection = [1, 0, -1, 1]
        for (x, y), index in zip(reader.spectra_iter(iter(selection), silent=True), selection, strict=True):
            np.testing.assert_array_equal(x, spectra[index][0])
            np.testing.assert_array_equal(y, spectra[index][1])
        assert list(reader.spectra_iter([], silent=True)) == []
        assert len(list(reader)) == len(spectra)
        assert len(list(reader)) == len(spectra)
        low, high = reader._estimate_centroid_mass_range()
        assert low == 100.0 and high == 104.0
        expected_tic = np.asarray([y.astype(DTYPE_DICT[int_code]).sum() for _, y in spectra], dtype=np.float64)
        np.testing.assert_allclose(reader.get_tic(silent=True), expected_tic, rtol=2e-6)


@pytest.mark.parametrize("mode", ["centroid", "profile"])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("ppm", [None, 10000.0])
@pytest.mark.parametrize("code", list(DTYPE_DICT))
def test_images_match_independent_masks(
    tmp_path: Path, backend: str, mode: str, shared: bool, ppm: float | None, code: str
) -> None:
    mz = np.array([100.0, 101.0, 101.0, 102.0])
    spectra = [(mz, np.array([1, 2, 7, 4])), (mz if shared else mz + 0.5, np.array([3, 5, 9, 8]))]
    spectra = [(x.astype(DTYPE_DICT[code]), y.astype(DTYPE_DICT[code])) for x, y in spectra]
    path = _dataset(tmp_path, spectra, code, code, shared=shared, mode=mode)
    kwargs = {"tol": 0.5} if ppm is None else {"ppm": ppm}
    targets = np.array([101.0, 103.0, 101.0, 500.0])
    lower, upper = get_mzs_for_tol(targets, **kwargs)
    expected = np.asarray(
        [[y[(x >= lo) & (x <= hi)].sum() for lo, hi in zip(lower, upper, strict=True)] for x, y in spectra]
    )
    with IMZMLReader(path, backend=backend) as reader:
        table = reader.to_table(iter(targets), silent=True, **kwargs)
        np.testing.assert_array_equal(table, expected)
        assert table.dtype == np.float32
        images = reader.get_ion_images(targets, silent=True, **kwargs)
        assert images.shape == (len(targets), 1, 3)
        np.testing.assert_array_equal(images[:, 0, :2], expected.T)
        np.testing.assert_array_equal(images[:, 0, 2], 0)  # Existing missing-pixel fill behavior.
        np.testing.assert_array_equal(reader.get_ion_image(101.0, silent=True, **kwargs), images[0])
        events = list(reader._get_ions_yield(targets, silent=True, **kwargs))
        assert events[-1][:2] == (None, None)
        np.testing.assert_array_equal(events[-1][2], table)
        assert reader.to_table([], tol=0.1, silent=True).shape == (2, 0)
        assert list(reader._get_ions_yield([], tol=0.1, silent=True))[-1][2].shape == (2, 0)


def test_external_binary_path_and_reopen_settings(tmp_path: Path, backend: str) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([7]))])
    binary = tmp_path / "external.bin"
    path.with_suffix(".ibd").rename(binary)
    with IMZMLReader(path, ibd_path=str(binary), backend=backend) as reader:
        with IMZMLReader(path, **reader._get_reader_kwargs()) as reopened:
            assert reopened.backend == backend
            np.testing.assert_array_equal(reopened[0][1], [7])


def test_block_boundaries_and_progress(tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from imzy._readers.imzml import _imzml

    monkeypatch.setattr(_imzml, "_BLOCK_SIZE", 2)
    spectra = [(np.array([100, 101, 102]), np.array([i, i + 1, i + 2])) for i in range(5)]
    path = _dataset(tmp_path, spectra)
    with IMZMLReader(path, backend=backend) as reader:
        events = list(reader._get_ions_yield([101], tol=0.1, silent=True))
        assert [event[0] for event in events] == [1, 3, 4, None]
        np.testing.assert_array_equal(events[-1][2][:, 0], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(reader.get_tic(silent=True), [3, 6, 9, 12, 15])


def test_mapping_ownership_close_and_context(tmp_path: Path, backend: str) -> None:
    spectra = [(np.array([100.0, 101.0]), np.array([2, 3])), (np.array([103.0]), np.array([7]))]
    path = _dataset(tmp_path, spectra)
    reader = IMZMLReader(path, backend=backend)
    x, y = reader[0]
    iterator = reader.spectra_iter(silent=True)
    next(iterator)
    if backend == "rust":
        assert np.shares_memory(x, reader[0][0])
        with pytest.raises(ValueError):
            x.setflags(write=True)
    reader.close()
    reader.close()
    np.testing.assert_array_equal(next(iterator)[1], [7])
    with pytest.raises(ValueError, match="closed"):
        reader[0]
    with pytest.raises(ValueError, match="closed"):
        reader.get_tic(silent=True)
    with pytest.raises(ValueError, match="closed"):
        reader.get_ion_image(100, tol=1, silent=True)
    del iterator
    reference = weakref.ref(reader)
    del reader
    gc.collect()
    assert reference() is None
    np.testing.assert_array_equal(x, [100.0, 101.0])
    np.testing.assert_array_equal(y, [2, 3])
    with IMZMLReader(path, backend=backend) as scoped:
        retained = scoped[1][0]
    np.testing.assert_array_equal(retained, [103.0])
    with pytest.raises(ValueError, match="closed"):
        scoped.get_intensities(0)


@pytest.mark.parametrize("index", [-3, 2, 10**30])
def test_out_of_range_indices(tmp_path: Path, backend: str, index: int) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([1]))] * 2)
    with IMZMLReader(path, backend=backend) as reader, pytest.raises(IndexError):
        reader[index]


@pytest.mark.parametrize("indices", [[0.5], [[0]], ["0"]])
def test_invalid_selections(tmp_path: Path, backend: str, indices: list) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([1]))])
    with IMZMLReader(path, backend=backend) as reader, pytest.raises((TypeError, ValueError)):
        list(reader.spectra_iter(indices, silent=True))


@pytest.mark.parametrize("column,value", [(0, -1), (1, -1), (1, 2**62), (2, 2**62), (3, 2)])
def test_bad_offsets_fail_before_binary_access(tmp_path: Path, backend: str, column: int, value: int) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([1]))])
    with np.load(path.with_suffix(".icache")) as cache:
        offsets = cache["byte_offsets"]
    offsets[0, column] = value
    _edit_cache(path, byte_offsets=offsets)
    with pytest.raises(ValueError):
        IMZMLReader(path, backend=backend)


def test_truncated_binary_and_all_empty_spectra(tmp_path: Path, backend: str) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([1]))])
    with path.with_suffix(".ibd").open("r+b") as stream:
        stream.truncate(17)
    with pytest.raises(ValueError, match="beyond"):
        IMZMLReader(path, backend=backend)
    path = _dataset(tmp_path, [(np.array([]), np.array([]))])
    with IMZMLReader(path, backend=backend) as reader:
        assert reader[0][0].size == 0
        np.testing.assert_array_equal(reader.get_tic(silent=True), [0])
        with pytest.raises(ValueError, match="empty"):
            reader._estimate_centroid_mass_range()


def test_automatic_fallback_and_required_native_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _dataset(tmp_path, [(np.array([100]), np.array([1]))])
    original = importlib.import_module

    def unavailable(name: str, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        if name == "imzy._imzml_rust":
            raise ImportError("simulated missing extension")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", unavailable)
    with IMZMLReader(path) as reader:
        assert reader.backend == "python"
    with pytest.raises(ImportError, match="IMZY_BUILD_RUST=1"):
        IMZMLReader(path, backend="rust")
    with pytest.raises(ValueError, match="backend"):
        IMZMLReader(path, backend="invalid")
