"""Tests for the Bruker frame decoding kernels in `imzy._readers.bruker.utilities`."""

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from imzy._readers.bruker.utilities import (
    get_sparse_data_from_buffer,
    peaks_to_csc,
    peaks_to_profile,
    scan_buffer_to_csc,
    scan_buffer_to_profile,
)

BUFFER_DIR = Path(__file__).parent / "_test_data" / "bruker_buffers"
BUFFER_FILES = sorted(BUFFER_DIR.glob("test_sparse_from_buffer_*.npz"))


def _load(filename: Path) -> dict[str, np.ndarray]:
    with np.load(filename) as f_ptr:
        return {key: f_ptr[key] for key in f_ptr.files}


def _make_frame(rng: np.random.Generator, n_scans: int, n_mz_bins: int) -> tuple[np.ndarray, ...]:
    """Make random frame as (scan_offsets, tof_indices, intensities) with sorted TOF indices within each scan."""
    counts = rng.integers(0, 20, size=n_scans)
    counts[rng.random(n_scans) < 0.3] = 0  # plenty of empty scans
    tofs, ints = [], []
    for count in counts:
        tofs.append(np.sort(rng.choice(n_mz_bins, size=count, replace=False)).astype(np.uint32))
        ints.append(rng.integers(1, 1000, size=count).astype(np.uint32))
    scan_offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return scan_offsets, np.concatenate(tofs), np.concatenate(ints)


def _to_buffer(scan_offsets: np.ndarray, tof_indices: np.ndarray, intensities: np.ndarray) -> np.ndarray:
    """Encode frame in the vendor `tims_read_scans_v2` layout."""
    counts = np.diff(scan_offsets)
    parts = [counts.astype(np.uint32)]
    for start, end in pairwise(scan_offsets):
        parts.extend([tof_indices[start:end], intensities[start:end]])
    return np.concatenate(parts).astype(np.uint32)


def _reference_dense(scan_offsets, tof_indices, intensities, n_scans, mz_min_idx, mz_max_idx) -> np.ndarray:
    dense = np.zeros((mz_max_idx - mz_min_idx, n_scans), dtype=np.int64)
    for scan in range(n_scans):
        for i in range(scan_offsets[scan], scan_offsets[scan + 1]):
            if mz_min_idx <= tof_indices[i] < mz_max_idx:
                dense[tof_indices[i] - mz_min_idx, scan] += intensities[i]
    return dense


@pytest.mark.parametrize("filename", BUFFER_FILES, ids=lambda p: p.stem)
def test_get_sparse_data_from_buffer_golden(filename):
    """The COO wrapper reproduces the output of the original implementation."""
    d = _load(filename)
    data, rows, cols = get_sparse_data_from_buffer(
        d["buffer"].astype(np.uint32), d["mz_range"][0], d["mz_range"][1], d["dt_range"][0], d["dt_range"][1]
    )
    assert_array_equal(d["result_data"], data)
    assert_array_equal(d["result_row"], rows)
    assert_array_equal(d["result_col"], cols)


@pytest.mark.parametrize("filename", BUFFER_FILES, ids=lambda p: p.stem)
def test_scan_buffer_to_csc_golden(filename):
    """The CSC kernel matches the COO reference (including m/z trimming)."""
    d = _load(filename)
    (mz_min, mz_max), (scan_begin, scan_end) = d["mz_range"], d["dt_range"]
    n_scans = scan_end - scan_begin
    data, indices, indptr = scan_buffer_to_csc(d["buffer"].astype(np.uint32), n_scans, mz_min, mz_max)
    result = sparse.csc_matrix((data, indices, indptr), shape=(mz_max - mz_min, n_scans))
    expected = sparse.coo_matrix(
        (d["result_data"].astype(np.int64), (d["result_row"] - mz_min, d["result_col"] - scan_begin)),
        shape=(mz_max - mz_min, n_scans),
    )
    assert data.dtype == np.int32 and indices.dtype == np.int32 and indptr.dtype == np.int32
    assert (result != expected.tocsc()).nnz == 0


@pytest.mark.parametrize("mz_range", [(0, 500), (100, 400), (450, 500), (0, 1)])
def test_kernels_match_dense_reference(mz_range):
    """Buffer and peak kernels agree with a dense reference on random frames."""
    rng = np.random.default_rng(42)
    n_scans, n_mz_bins = 50, 500
    scan_offsets, tofs, ints = _make_frame(rng, n_scans, n_mz_bins)
    buffer = _to_buffer(scan_offsets, tofs, ints)
    expected = _reference_dense(scan_offsets, tofs, ints, n_scans, *mz_range)
    shape = (mz_range[1] - mz_range[0], n_scans)

    from_buffer = sparse.csc_matrix(scan_buffer_to_csc(buffer, n_scans, *mz_range), shape=shape)
    from_peaks = sparse.csc_matrix(peaks_to_csc(scan_offsets, tofs, ints, n_scans, *mz_range), shape=shape)
    assert_array_equal(from_buffer.toarray(), expected)
    assert_array_equal(from_peaks.toarray(), expected)
    assert from_buffer.has_sorted_indices


def test_peaks_to_csc_extra_scans():
    """Scans beyond `n_scans` are ignored and missing scans are empty."""
    rng = np.random.default_rng(0)
    scan_offsets, tofs, ints = _make_frame(rng, 10, 100)
    full = _reference_dense(scan_offsets, tofs, ints, 10, 0, 100)
    fewer = sparse.csc_matrix(peaks_to_csc(scan_offsets, tofs, ints, 6, 0, 100), shape=(100, 6))
    assert_array_equal(fewer.toarray(), full[:, :6])
    more = sparse.csc_matrix(peaks_to_csc(scan_offsets, tofs, ints, 15, 0, 100), shape=(100, 15))
    assert_array_equal(more.toarray()[:, :10], full)
    assert more[:, 10:].nnz == 0


def test_profile_kernels():
    """Profile kernels equal the sum over scans."""
    rng = np.random.default_rng(1)
    n_scans, n_mz_bins = 40, 300
    scan_offsets, tofs, ints = _make_frame(rng, n_scans, n_mz_bins)
    expected = _reference_dense(scan_offsets, tofs, ints, n_scans, 0, n_mz_bins).sum(axis=1)
    assert_array_equal(scan_buffer_to_profile(_to_buffer(scan_offsets, tofs, ints), n_scans, n_mz_bins), expected)
    assert_array_equal(peaks_to_profile(tofs, ints, n_mz_bins), expected)
    # bins beyond `n_mz_bins` are dropped
    assert_array_equal(peaks_to_profile(tofs, ints, 100), expected[:100])


def test_truncated_buffer_raises():
    """A buffer shorter than its peak counts is rejected instead of reading out of bounds."""
    buffer = np.array([3, 1, 2], dtype=np.uint32)
    with pytest.raises(ValueError):
        scan_buffer_to_csc(buffer, 1, 0, 10)
    with pytest.raises(ValueError):
        scan_buffer_to_profile(buffer, 1, 10)
