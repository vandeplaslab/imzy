"""Export sample ion tables to HDF5 and Zarr and verify their stored intensities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from quickstart import create_sample

from imzy import IMZMLReader


def run(directory: Path, *, backend: str) -> Path:
    """Create sample data and verify one optional storage backend against known sums."""
    path = create_sample(directory)
    expected = np.arange(1, 5)[:, None] * np.array([1.0, 2.0])
    with IMZMLReader(path) as reader:
        if backend == "hdf5":
            from imzy import H5CentroidsStore

            output = reader.to_hdf5(directory / "centroids.h5", [100.0, 200.0], tol=0.1, silent=True)
            store = H5CentroidsStore(output, mode="r")
            table = np.column_stack([store.get_ion(100.0), store.get_ion(200.0)])
        elif backend == "zarr":
            from imzy import ZarrCentroidsStore

            output = reader.to_zarr(directory / "centroids.zarr", [100.0, 200.0], tol=0.1, silent=True)
            store = ZarrCentroidsStore(output)
            table = store.get_ions(np.array([100.0, 200.0]))
        else:
            raise ValueError("Choose hdf5 or zarr.")
        np.testing.assert_allclose(table, expected)
        np.testing.assert_allclose(reader.reshape_batch(table), expected.T.reshape(2, 2, 2))
    print(f"Verified {output}: {table.shape}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for new datasets; use a fresh directory for each run.")
    parser.add_argument("--backend", choices=["hdf5", "zarr"], required=True)
    args = parser.parse_args()
    run(args.output, backend=args.backend)
