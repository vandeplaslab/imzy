"""Tests for the macOS `pytsf` backend of the Bruker readers."""

import numpy as np
import pytest
from koyo.system import IS_MAC

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
