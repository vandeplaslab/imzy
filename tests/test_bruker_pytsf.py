"""Tests for the macOS `pytsf` backend of the Bruker readers."""

import numpy as np
import pytest
from koyo.system import IS_MAC
from scipy import sparse

from imzy._readers.bruker import HAS_BRUKER_BACKEND

from .utilities import get_tdf_data, get_tsf_data

pytestmark = pytest.mark.skipif(not (IS_MAC and HAS_BRUKER_BACKEND), reason="pytsf backend is only used on macOS.")


@pytest.mark.parametrize("path", get_tsf_data())
def test_tsf_backend(path):
    from imzy._readers.bruker._tsf_pytsf import TSFReader

    with TSFReader(path) as reader:
        assert not reader.is_centroid
        profile = reader.read_profile_spectrum(0)
        assert profile.dtype == np.uint32
        assert profile.shape == reader.mz_x.shape
        mzs, intensities = reader.read_centroid_spectrum(0)
        assert mzs.dtype == np.float64
        assert intensities.dtype == np.float32
        assert np.all(np.diff(mzs) > 0)
        if not reader._tsf.meta.has_line_spectra_peak_width:
            # vendor library reports missing data as RuntimeError
            with pytest.raises(RuntimeError):
                reader.read_centroid_spectrum_with_width(0)
        # index <-> m/z round-trip
        indices = np.asarray([10.0, 1000.0, 100000.0])
        np.testing.assert_allclose(reader.mz_to_index(1, reader.index_to_mz(1, indices)), indices, atol=1e-6)


@pytest.mark.parametrize("path", get_tdf_data())
def test_tdf_backend(path):
    from imzy._readers.bruker._tdf_pytsf import TDFReader

    with TDFReader(path) as reader:
        frame = reader.read_frame(1)
        assert frame.shape == reader.frame_shape
        np.testing.assert_array_equal(reader.read_profile_spectrum(0), np.asarray(frame.sum(axis=1)).ravel())
        assert reader.read_profile_spectrum(0).shape == reader.mz_x.shape
        ook0 = reader.scan_num_to_ook0(1, np.arange(reader.n_dt_bins))
        assert np.all(np.diff(ook0) < 0)  # 1/K0 falls with scan number
        np.testing.assert_allclose(reader.ook0_to_scan_num(1, ook0[[0, -1]]), [0, reader.n_dt_bins - 1], atol=1e-3)
        with pytest.raises(NotImplementedError):
            reader.scan_num_to_voltage(1, np.arange(3))


@pytest.mark.parametrize("path", get_tdf_data())
def test_tdf_read_frame_csc(path):
    from imzy._readers.bruker._tdf_pytsf import TDFReader

    with TDFReader(path) as reader:
        for frame_id in (1, int(reader.frame_indices[-1])):
            raw = reader._tdf.read_frame(frame_id)
            rows, cols, data = raw.tof_indices.astype(np.int64), raw.scan, raw.intensities.astype(np.int64)
            for mz_min_idx, mz_max_idx in ((0, reader.n_mz_bins), (50_000, 120_000)):
                mask = (rows >= mz_min_idx) & (rows < mz_max_idx) & (cols < reader.n_dt_bins)
                shape = (mz_max_idx - mz_min_idx, reader.n_dt_bins)
                expected = sparse.coo_matrix((data[mask], (rows[mask] - mz_min_idx, cols[mask])), shape=shape)
                result = reader.read_frame_csc(frame_id, mz_min_idx, mz_max_idx)
                assert result.format == "csc"
                assert result.shape == shape
                assert (result != expected.tocsc()).nnz == 0
            # numpy integers are accepted as frame ids
            assert reader.read_frame_csc(np.int64(frame_id)).nnz == reader.read_frame_csc(frame_id).nnz
            np.testing.assert_array_equal(
                reader.read_frame_profile(frame_id), np.asarray(reader.read_frame(frame_id).sum(axis=1)).ravel()
            )


@pytest.mark.parametrize("path", get_tsf_data())
def test_tsf_frame_id_methods(path):
    from imzy._readers.bruker._tsf_pytsf import TSFReader

    with TSFReader(path) as reader:
        frame_id = int(reader.frame_indices[3])
        np.testing.assert_array_equal(reader.read_profile_frame(np.int64(frame_id)), reader.read_profile_spectrum(3))
        for a, b in zip(reader.read_centroid_frame(frame_id), reader.read_centroid_spectrum(3), strict=True):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("path", get_tsf_data() + get_tdf_data())
def test_summed_spectrum_subset(path):
    """Summing a subset of pixels with default scales must not index scales by position."""
    from imzy import get_reader

    reader = get_reader(path)
    indices = np.asarray([reader.n_pixels - 1, 0])
    _, summed = reader.get_summed_spectrum(indices, silent=True)
    expected = sum(np.asarray(reader._read_spectrum(i)[1], dtype=np.float64) for i in indices)
    np.testing.assert_allclose(summed, expected)
    assert reader.is_centroid is reader.is_centroid  # cached
    reader.close()


@pytest.mark.parametrize("path", get_tdf_data())
def test_tdf_read_scans_layout(path):
    """`read_scans` produces the vendor buffer layout, which decodes back to the same frame."""
    from imzy._readers.bruker._tdf_pytsf import TDFReader
    from imzy._readers.bruker.utilities import scan_buffer_to_csc

    with TDFReader(path) as reader:
        for scan_begin, scan_end in ((0, reader.n_dt_bins), (100, 200)):
            buffer = reader.read_scans(1, scan_begin, scan_end)
            assert buffer.dtype == np.uint32
            n_scans = scan_end - scan_begin
            data, indices, indptr = scan_buffer_to_csc(buffer, n_scans, 0, reader.n_mz_bins)
            decoded = sparse.csc_matrix((data, indices, indptr), shape=(reader.n_mz_bins, n_scans))
            expected = reader.read_frame_csc(1)[:, scan_begin:scan_end]
            assert (decoded != expected).nnz == 0
