import typing as ty

import numba
import numpy as np


@numba.njit(fastmath=True, cache=True)
def get_sparse_data_from_buffer(
    buffer: np.ndarray, mz_min_idx: int, mz_max_idx: int, scan_begin: int, scan_end: int
) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the instantiation values for COO array by iterating over the data extracted from the Bruker frame.

    Parameters
    ----------
    buffer : np.ndarray[np.uint32]
        array containing data extracted for particular frame
    mz_min_idx : int
        m/z start index of what data should be included in the COO array
    mz_max_idx : int
        m/z end index of what data should be included in the COO array
    scan_begin : int
        start mobility bin
    scan_end : int
        end mobility bin

    Returns
    -------
    data : np.ndarray
        standard list containing `data` information to be used by the COO sparse matrix
    rows : np.ndarray
        standard list containing `rows` information to be used by the COO sparse matrix
    cols : np.ndarray
        standard list containing `cols` information to be used by the COO sparse matrix
    """
    data = []
    rows = []
    cols = []

    d = scan_end - scan_begin
    for col in range(scan_begin, scan_end):
        n_peaks = buffer[col - scan_begin]
        indices = buffer[d : d + n_peaks]
        d += n_peaks
        intensities = buffer[d : d + n_peaks]
        d += n_peaks
        for i in range(indices.shape[0]):
            mz_val = indices[i]
            mz_int = intensities[i]
            if mz_min_idx <= mz_val < mz_max_idx:
                data.append(mz_int)
                rows.append(mz_val)
                cols.append(col)
    return np.asarray(data), np.asarray(rows), np.asarray(cols)
