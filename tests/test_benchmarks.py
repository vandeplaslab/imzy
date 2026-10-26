"""Independent checks that ASV workloads perform equivalent, uncached work."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from benchmarks import bench_imzml
from benchmarks._data import LAYOUTS, TARGETS, TOLERANCE, consume, generate, spectra

from imzy import IMZMLReader


def _ion_table(inputs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.asarray(
        [[y[(x >= mz - TOLERANCE) & (x <= mz + TOLERANCE)].sum() for mz in TARGETS] for x, y in inputs],
        dtype=np.float32,
    )


def _normalizations(inputs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    raw = []
    for _, y in inputs:
        q95, q90, q10, q5 = np.quantile(y, [0.95, 0.9, 0.1, 0.05])
        raw.append(
            [
                y.sum(),
                np.sqrt(np.mean(y**2)),
                np.median(y),
                y[y < q95].sum(),
                y[y < q90].sum(),
                y[y > q5].sum(),
                y[y > q10].sum(),
                y[(y > q5) & (y < q95)].sum(),
                y[(y > q10) & (y < q90)].sum(),
                np.count_nonzero(y),
                np.linalg.norm(y, 2),
                np.linalg.norm(y, 3),
            ]
        )
    values = np.asarray(raw, dtype=np.float32)
    return np.median(values, axis=0) / values


@pytest.mark.parametrize("layout", LAYOUTS)
def test_fixtures_are_seeded_and_varied(tmp_path: Path, layout: str) -> None:
    inputs = list(spectra(layout, 16, 64))
    repeated = list(spectra(layout, 16, 64))
    for (x, y), (other_x, other_y) in zip(inputs, repeated, strict=True):
        np.testing.assert_array_equal(x, other_x)
        np.testing.assert_array_equal(y, other_y)
        assert np.all(np.diff(x) > 0)
        assert y.dtype == np.float32
        assert np.unique(y).size > 1
    assert not np.array_equal(inputs[0][1], inputs[1][1])
    if layout == "continuous_profile":
        assert all(len(x) == 64 for x, _ in inputs)
        assert all(np.array_equal(x, inputs[0][0]) for x, _ in inputs)
    else:
        assert len({len(x) for x, _ in inputs}) > 1
        assert all(32 <= len(x) <= 64 for x, _ in inputs)
    path = generate(tmp_path, layout, 16, 64)
    with IMZMLReader(path, backend="python") as reader:
        for index, expected in enumerate(inputs):
            for actual, values in zip(reader.get_spectrum(index), expected, strict=True):
                np.testing.assert_array_equal(actual, values)


@pytest.mark.usefixtures("imzml_backend")
@pytest.mark.parametrize("layout", LAYOUTS)
def test_read_workloads_match_numpy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    inputs = list(spectra(layout, 8, 64))
    path = generate(tmp_path, layout, 8, 64)
    monkeypatch.chdir(tmp_path)
    # The existing backend fixture selects Python/Rust when no explicit backend is supplied.
    with IMZMLReader(path) as reader:
        backend = reader.backend
        np.testing.assert_allclose(reader.to_table(TARGETS, tol=TOLERANCE, silent=True), _ion_table(inputs), rtol=1e-5)
        actual = np.column_stack(list(reader.get_normalizations(silent=True).values()))
        np.testing.assert_allclose(actual, _normalizations(inputs), rtol=1e-5)

    workload = bench_imzml.Read()
    workload.setup(backend, layout, 8)
    try:
        expected = sum(consume(x, y) for x, y in inputs)
        assert workload.time_iteration(backend, layout, 8) == pytest.approx(expected)
        indices = np.random.default_rng(17).integers(0, 8, size=8)
        assert workload.time_random_spectra(backend, layout, 8) == pytest.approx(
            sum(consume(*inputs[i]) for i in indices)
        )
        assert workload.time_ion_images(backend, layout, 8) == pytest.approx(_ion_table(inputs).sum(dtype=np.float64))
        assert workload.time_ion_image(backend, layout, 8) == pytest.approx(
            _ion_table(inputs)[:, 0].sum(dtype=np.float64)
        )
        assert workload.time_normalizations(backend, layout, 8) == pytest.approx(
            _normalizations(inputs).sum(dtype=np.float64), rel=1e-5
        )
        for _ in range(2):
            assert workload.time_tic(backend, layout, 8) == pytest.approx(sum(float(y.sum()) for _, y in inputs))
    finally:
        workload.teardown(backend, layout, 8)


def test_tic_recomputes_on_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = "processed_centroid"
    generate(tmp_path, layout, 8, 64)
    monkeypatch.chdir(tmp_path)
    workload = bench_imzml.Read()
    workload.setup("python", layout, 8)
    calls = 0
    original = workload.reader._read_spectra

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(workload.reader, "_read_spectra", counted)
    try:
        workload.time_tic("python", layout, 8)
        workload.time_tic("python", layout, 8)
        assert calls == 2
    finally:
        workload.teardown("python", layout, 8)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_initialization_reparses_each_uncached_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    from imzy._readers.imzml import _imzml

    path = generate(tmp_path, layout, 8, 64)
    shutil.copyfile(path, path.with_name(f"{path.stem}-uncached.imzML"))
    with IMZMLReader(path, backend="python") as reader:
        _ = reader.mz_min, reader.mz_max
    monkeypatch.chdir(tmp_path)
    calls = 0
    original = _imzml.init_metadata

    def counted(*args: object, **kwargs: object) -> tuple:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_imzml, "init_metadata", counted)

    def unexpected_upgrade(*args: object, **kwargs: object) -> None:
        raise AssertionError("A complete metadata cache must not reparse XML.")

    monkeypatch.setattr(_imzml, "init_spectrum_metadata", unexpected_upgrade)
    workload = bench_imzml.ReaderInitialization()
    for expected in (1, 2):
        workload.setup("python", layout, 8)
        workload.time_open_cached("python", layout, 8)
        assert calls == expected - 1
        workload.time_open_uncached("python", layout, 8)
        assert calls == expected
    assert bench_imzml.number == 1
    assert bench_imzml.warmup_time == 0


@pytest.mark.usefixtures("imzml_backend")
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("format_name", ["imzml", "hdf5", "zarr"])
def test_exports_write_fresh_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str, format_name: str
) -> None:
    pytest.importorskip("dask.array")
    pytest.importorskip("zarr", minversion="3.0")
    from imzy._centroids import H5CentroidsStore

    inputs = list(spectra(layout, 8, 64))
    path = generate(tmp_path, layout, 8, 64)
    with IMZMLReader(path) as reader:
        backend = reader.backend
    monkeypatch.chdir(tmp_path)
    workload = bench_imzml.Export()
    previous = None
    for _ in range(2):
        workload.setup(backend, layout, 8)
        output = workload.output
        try:
            assert output != previous
            assert list(output.iterdir()) == []
            getattr(workload, f"time_{format_name}")(backend, layout, 8)
            if format_name == "imzml":
                with IMZMLReader(output / "converted.imzML", backend=backend) as reader:
                    for i, (x, y) in enumerate(inputs):
                        actual_x, actual_y = reader.get_spectrum(i)
                        np.testing.assert_array_equal(actual_x, x)
                        np.testing.assert_array_equal(actual_y, y)
            elif format_name == "hdf5":
                np.testing.assert_allclose(H5CentroidsStore(output / "ions.h5").peaks, _ion_table(inputs), rtol=1e-5)
            else:
                import zarr

                group = zarr.open_group(str(output / "ions.zarr"), mode="r")
                assert group.metadata.zarr_format == 3
                np.testing.assert_allclose(group["array"][:], _ion_table(inputs), rtol=1e-5)
        finally:
            workload.teardown(backend, layout, 8)
        assert not output.exists()
        previous = output


def test_disabled_jit_fails_instead_of_comparing_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NUMBA_DISABLE_JIT", "1")
    with pytest.raises(RuntimeError, match="JIT"):
        bench_imzml.setup("python", "processed_centroid", 8)
