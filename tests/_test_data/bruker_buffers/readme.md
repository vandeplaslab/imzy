# Bruker scan buffers

The `tims_read_scans_v2` buffers (vendor layout: number of peaks per scan, then the TOF indices and intensities of
each scan in turn) together with the COO arrays produced by the original reference implementation.

Each `.npz` contains:

- `buffer` - uint32 scan buffer of a single frame (trimmed after the last peak)
- `result_data`, `result_row`, `result_col` - expected COO arrays (absolute TOF index / scan number)
- `mz_range` - `[mz_min_idx, mz_max_idx]` used for the extraction
- `dt_range` - `[scan_begin, scan_end]` contained in the buffer