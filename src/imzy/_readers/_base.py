"""Base reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.spectrum import find_between_batch, find_between_ppm, find_between_tol, get_mzs_for_tol
from koyo.typing import PathLike
from tqdm import tqdm

from imzy.utilities import accumulate_peaks_centroid, accumulate_peaks_profile


class BaseReader:
    """Base reader class."""

    # private attributes
    _xyz_coordinates: ty.Optional[np.ndarray] = None
    _tic: ty.Optional[np.ndarray] = None
    _current = -1

    def __init__(self, path: PathLike):
        self.path = Path(path)

    def _init(self):
        """Method which is called to initialize the reader."""
        raise NotImplementedError("Must implement method")

    @property
    def mz_min(self) -> float:
        """Minimum m/z value."""
        raise NotImplementedError("Must implement method")

    @property
    def mz_max(self) -> float:
        """Maximum m/z value."""
        raise NotImplementedError("Must implement method")

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether the data is in centroid or profile mode."""
        raise NotImplementedError("Must implement method")

    def get_spectrum(self, index: int):
        """Return mass spectrum."""
        return self._read_spectrum(index)

    def get_summed_spectrum(self, indices: ty.Iterable[int], silent: bool = False):
        """Sum pixel data to produce summed mass spectrum."""
        raise NotImplementedError("Must implement method")

    def _read_spectrum(self, index: int) -> ty.Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Must implement method")

    def _read_spectra(self, indices: ty.Optional[np.ndarray] = None) -> ty.Iterator[ty.Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("Must implement method")

    def reshape(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Reshape vector into an image."""
        raise NotImplementedError("Must implement method")

    def reshape_batch(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Reshape multiple vectors into a stack of images."""
        raise NotImplementedError("Must implement method")

    def __iter__(self):
        return self

    def __next__(self):
        """Get next spectrum."""
        if self._current < self.n_pixels - 1:
            self._current += 1
            return self[self._current]
        else:
            self._current = -1
            raise StopIteration

    def __getitem__(self, item: int):
        """Retrieve spectrum."""
        return self.get_spectrum(item)

    @property
    def xyz_coordinates(self) -> np.ndarray:
        """Return xyz coordinates."""
        return self._xyz_coordinates

    @property
    def x_coordinates(self) -> np.ndarray:
        """Return x-axis coordinates/."""
        return self._xyz_coordinates[:, 0]

    @property
    def y_coordinates(self) -> np.ndarray:
        """Return y-axis coordinates."""
        return self._xyz_coordinates[:, 1]

    @property
    def z_coordinates(self) -> np.ndarray:
        """Return z-axis coordinates."""
        return self._xyz_coordinates[:, 2]

    @property
    def pixels(self) -> np.ndarray:
        """Iterable of pixels in the dataset."""
        return np.arange(self.n_pixels)

    @property
    def n_pixels(self):
        """Return the total number of pixels in the dataset."""
        return len(self.x_coordinates)

    def get_chromatogram(self, indices: ty.Iterable[int]):
        """Return chromatogram."""
        indices = np.asarray(indices)
        array = np.zeros(len(indices), dtype=np.float32)
        for y in self.spectra_iter(indices):
            array += np.sum(y)
        return array

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return TIC image."""
        if self._tic is None:
            res = np.zeros(self.n_pixels)
            for i, (_, y) in enumerate(self.spectra_iter(silent=silent)):
                res[i] = y.sum()
            self._tic = res
        return self._tic

    def get_ion_image(
        self, mz: float, tol: float = None, ppm: float = None, fill_value: float = np.nan, silent: bool = False
    ) -> np.ndarray:
        """Return ion image for specified m/z with tolerance or ppm."""
        if tol is None and ppm is None or tol == 0 and ppm == 0:
            raise ValueError("Please specify `tol` or `ppm`.")
        elif tol is not None and ppm is not None:
            raise ValueError("Please only specify `tol` or `ppm`.")
        func = find_between_tol if tol else find_between_ppm
        val = tol if tol else ppm
        res = np.full(self.n_pixels, dtype=np.float32, fill_value=fill_value)
        if self.is_centroid:
            for i, (x, y) in enumerate(self.spectra_iter(silent=silent)):
                mask = func(x, mz, val)
                res[i] = y[mask].sum()
        else:
            x, _ = self[0]
            mask = func(x, mz, val)
            for i, (_, y) in enumerate(self.spectra_iter(silent=silent)):
                res[i] = y[mask].sum()
        return self.reshape(res)

    def _get_ions(
        self,
        mzs: ty.Iterable[float],
        tol: float = None,
        ppm: float = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ):
        mzs = np.asarray(mzs)
        mzs_min, mzs_max = get_mzs_for_tol(mzs, tol, ppm)
        res = np.full((self.n_pixels, len(mzs)), dtype=np.float32, fill_value=fill_value)

        if self.is_centroid:
            for i, (x, y) in enumerate(self.spectra_iter(silent=silent)):
                res[i] = accumulate_peaks_centroid(mzs_min, mzs_max, x, y)
        else:
            x, _ = self.get_spectrum(0)
            indices = find_between_batch(x, mzs_min, mzs_max)
            for i, (_, y) in enumerate(self.spectra_iter(silent=silent)):
                res[i] = accumulate_peaks_profile(indices, y)
        return res

    def get_ion_images(
        self,
        mzs: ty.Iterable[float],
        tol: float = None,
        ppm: float = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> np.ndarray:
        """Return many ion images for specified m/z values."""
        res = self._get_ions(mzs, tol, ppm, fill_value, silent)
        return self.reshape_batch(res)

    def to_table(
        self,
        mzs: ty.Iterable[float],
        tol: float = None,
        ppm: float = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ):
        """Return many ion images for specified m/z values without reshaping."""
        return self._get_ions(mzs, tol, ppm, fill_value, silent)

    def to_zarr(
        self,
        zarr_path: PathLike,
        mzs: ty.Iterable[float],
        tol: float = None,
        ppm: float = None,
        as_flat: bool = True,
        chunk_size: ty.Optional[ty.Tuple[int, int]] = None,
        silent: bool = False,
    ) -> Path:
        """Export many ion images for specified m/z values (+ tolerance) to Zarr array."""
        from imzy._extract import check_zarr, create_centroids_zarr, extract_centroids_zarr, rechunk_zarr_array

        if not as_flat:
            raise ValueError("Only flat images are supported at the moment.")
        check_zarr()

        mzs = np.asarray(mzs)
        if mzs.size == 0:
            raise ValueError("Expect at least 1 mass to extract.")
        mzs_min, mzs_max = get_mzs_for_tol(mzs, tol, ppm)

        zarr_path = Path(zarr_path)
        # prepare output directory
        ds = create_centroids_zarr(
            self.path,
            zarr_path,
            len(mzs),
            mzs=mzs,
            mzs_min=mzs_min,
            mzs_max=mzs_max,
            ppm=ppm,
            tol=tol,
        )
        zarr_array_path = str(zarr_path / ds.path)
        extract_centroids_zarr(
            input_dir=self.path,
            zarr_path=zarr_array_path,
            mzs_min=mzs_min,
            mzs_max=mzs_max,
            indices=self.pixels,
            silent=silent,
        )

        import dask.array as dsa

        ds = dsa.from_zarr(zarr_array_path)
        ys = ds.sum(axis=0).compute()
        create_centroids_zarr(self.path, zarr_path, len(mzs_min), ys=np.asarray(ys))

        target_path = str(zarr_path / "array")
        rechunk_zarr_array(self.path, zarr_array_path, target_path, chunk_size=chunk_size)
        return zarr_path

    def to_hdf5(
        self,
        hdf_path: PathLike,
        mzs: ty.Iterable[float],
        tol: float = None,
        ppm: float = None,
        as_flat: bool = True,
        max_mem: float = 512,  # mb
        silent: bool = False,
    ) -> Path:
        """Export many ion images for specified m/z values (+ tolerance) to a HDF5 store."""
        from imzy._extract import check_hdf5, create_centroids_hdf5, extract_centroids_hdf5, get_chunk_info

        if not as_flat:
            raise ValueError("Only flat images are supported at the moment.")
        check_hdf5()

        mzs = np.asarray(mzs)
        if mzs.size == 0:
            raise ValueError("Expect at least 1 mass to extract.")
        mzs_min, mzs_max = get_mzs_for_tol(mzs, tol, ppm)

        chunk_info = get_chunk_info(self.n_pixels, len(mzs), max_mem)
        hdf_path = Path(hdf_path)
        if not hdf_path.suffix == ".h5":
            hdf_path = hdf_path.with_suffix(".h5")

        if hdf_path.exists():
            from imzy._centroids import H5CentroidsStore

            store = H5CentroidsStore(hdf_path)
            if store.n_peaks == len(mzs):
                if np.allclose(store.xs, mzs, rtol=1e-3):
                    return hdf_path

        # prepare output directory
        hdf_path = create_centroids_hdf5(
            self.path,
            hdf_path,
            len(mzs),
            mzs=mzs,
            mzs_min=mzs_min,
            mzs_max=mzs_max,
            ppm=ppm,
            tol=tol,
            chunk_info=chunk_info,
        )
        extract_centroids_hdf5(
            input_dir=self.path,
            hdf_path=hdf_path,
            mzs_min=mzs_min,
            mzs_max=mzs_max,
            indices=self.pixels,
            silent=silent,
        )
        return hdf_path

    def spectra_iter(self, indices: ty.Optional[ty.Iterable[int]] = None, silent: bool = False):
        """Yield spectra."""
        indices = self.pixels if indices is None else np.asarray(indices)
        yield from tqdm(
            self._read_spectra(indices), total=len(indices), disable=silent, miniters=500, desc="Iterating spectra..."
        )