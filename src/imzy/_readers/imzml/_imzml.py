"""imzML reader."""

from __future__ import annotations

import importlib
import operator
import typing as ty
from pathlib import Path
from warnings import warn

import numpy as np
from ims_utils.spectrum import find_between_batch, get_mzs_for_tol, get_ppm_axis
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy._readers.imzml._cache import fingerprint, load_cache, save_cache
from imzy._readers.imzml._metadata import (
    check_uuid,
    choose_iterparse,
    parse_metadata,
    read_values,
    validate_metadata,
)
from imzy.hookspec import hook_impl
from imzy.processing import accumulate_peaks_centroid, accumulate_peaks_profile
from imzy.utilities import _auto_guess_ppm

PRECISION_DICT = {"32-bit float": "f", "64-bit float": "d", "32-bit integer": "i", "64-bit integer": "l"}
SIZE_DICT = {"f": 4, "d": 8, "i": 4, "l": 8}
DTYPE_DICT: dict[str, np.dtype[ty.Any]] = {
    "f": np.dtype("<f4"),
    "d": np.dtype("<f8"),
    "i": np.dtype("<i4"),
    "l": np.dtype("<i8"),
}
SPECTRUM_MODE_CENTROID = "centroid"
SPECTRUM_MODE_PROFILE = "profile"
LOWEST_OBSERVED_MZ_ACCESSION = "MS:1000528"
HIGHEST_OBSERVED_MZ_ACCESSION = "MS:1000527"
SpectrumMode = ty.Literal["centroid", "profile"]
MzRange: ty.TypeAlias = tuple[float, float]
Backend = ty.Literal["auto", "rust", "python"]
_BLOCK_SIZE = 1024


class IMZMLCache:
    """Cache for imzML keys/values."""

    def __init__(self, metadata_dict: dict[str, ty.Any]) -> None:
        self.metadata_dict = metadata_dict
        self.PX_MAX_X: int = metadata_dict["max count of pixels x"]
        self.PX_MAX_Y: int = metadata_dict["max count of pixels y"]
        self.PX_MAX_Z: int = metadata_dict.get("max count of pixels z", 1)
        self.PX_SIZE_X: float | None = metadata_dict.get("pixel size x")
        self.PX_SIZE_Y: float | None = metadata_dict.get("pixel size y")


class IMZMLReader(BaseReader):
    """Read imzML spectra using an optional memory-mapped Rust backend.

    The ``backend`` argument selects ``"auto"`` (Rust when installed), ``"rust"``
    (required), or ``"python"``. Spectrum arrays are read-only; use ``.copy()``
    for writable data. Keep the dataset unchanged while readers, arrays, or
    iterators exist. Closing a reader releases its mapping, but existing arrays
    and iterators retain their backing storage until they are released.

    ``pixel_size=(x_um, y_um)`` supplies positive physical sizes when metadata is
    missing or ambiguous. ``z_plane`` selects a normalized one-based plane for all
    spectrum and export operations. Without selection, multi-plane acquisitions
    support spectra and tables but require a plane for two-dimensional images.
    """

    _ibd_path: Path | None = None
    _icache_path: Path | None = None
    _is_centroid: bool | None = None
    _mz_grid: np.ndarray | None = None
    _spectrum_mode: SpectrumMode | None = None
    byte_offsets: np.ndarray

    def __init__(
        self,
        path: PathLike,
        ibd_path: PathLike | None = None,
        auto_profile: bool = True,
        mz_ppm: float | str = "auto",
        resolution: int = 50_000,  # this is a pure guess and should match the instrument
        mz_min: float | None = None,
        mz_max: float | None = None,
        parse_lib: str | None = None,
        *,
        backend: Backend = "auto",
        pixel_size: tuple[float, float] | None = None,
        z_plane: int | None = None,
    ) -> None:
        if backend not in {"auto", "rust", "python"}:
            raise ValueError("backend must be 'auto', 'rust', or 'python'.")
        choose_iterparse(parse_lib)
        if pixel_size is not None:
            sizes = np.asarray(pixel_size, dtype=float)
            if sizes.shape != (2,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
                raise ValueError("pixel_size must contain two finite, positive sizes in um.")
            pixel_size = (float(sizes[0]), float(sizes[1]))
        if z_plane is not None:
            try:
                z_plane = operator.index(z_plane)
            except TypeError as error:
                raise ValueError("z_plane must be a normalized one-based integer.") from error
            if z_plane < 1:
                raise ValueError("z_plane must be a normalized one-based integer.")
        super().__init__(path, auto_profile=auto_profile)
        self._closed = False
        self._native: ty.Any = None
        self._iterator: ty.Iterator[tuple[np.ndarray, np.ndarray]] | None = None
        self._requested_backend = backend
        self._parse_lib = parse_lib
        self._pixel_size_override = pixel_size
        self._z_plane = z_plane
        self._explicit_mz_bounds = (mz_min, mz_max)
        self._is_centroid = None
        self._tic = None
        self._mz_min = mz_min
        self._mz_max = mz_max
        self._init(ibd_path, parse_lib=parse_lib)
        self._init_backend(backend)
        self.resolution = resolution
        self.mz_ppm = _auto_guess_ppm(self.resolution, mz_ppm)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.path}; centroid={self.is_centroid}>"

    def _init_backend(self, backend: Backend) -> None:
        """Validate the index and load the native extension when available."""
        self._mz_dtype = DTYPE_DICT[self.mz_precision]
        self._int_dtype = DTYPE_DICT[self.int_precision]
        offsets = np.array(self.byte_offsets, dtype=np.int64, order="C", copy=True)
        if offsets.shape != (self.n_pixels, 4):
            raise ValueError("Expected an (n_pixels, 4) offset array.")
        file_size = self.ibd_path.stat().st_size
        for column, width in ((0, self._mz_size), (2, self._int_size)):
            starts, lengths = offsets[:, column], offsets[:, column + 1]
            if np.any(starts < 0) or np.any(lengths < 0) or np.any(lengths > file_size // width):
                raise ValueError("Invalid external array offset or length.")
            if np.any(starts > file_size - lengths * width):
                raise ValueError("External array extends beyond the ibd file.")
        if np.any(offsets[:, 1] != offsets[:, 3]):
            raise ValueError("m/z and intensity array lengths differ.")
        offsets.setflags(write=False)
        self.byte_offsets = offsets
        self._shared_mz = bool(len(offsets) and np.all(offsets[:, :2] == offsets[0, :2]))
        if backend != "python":
            try:
                module = importlib.import_module("imzy._imzml_rust")
            except (ImportError, OSError) as error:
                if backend == "rust":
                    raise ImportError(
                        "The Rust backend is unavailable. Install a native imzy wheel or build with IMZY_BUILD_RUST=1."
                    ) from error
            else:
                self._native = module.IbdReader(self.ibd_path, offsets, self.mz_precision, self.int_precision)

    @property
    def backend(self) -> ty.Literal["rust", "python"]:
        """Return the backend selected for this reader."""
        return "rust" if self._native is not None else "python"

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("Reader is closed.")

    def close(self) -> None:
        """Release this reader; existing spectrum arrays and iterators remain valid."""
        if self._native is not None:
            self._native.close()
        self._iterator = None
        self._mz_grid = None
        self._closed = True

    def __enter__(self) -> IMZMLReader:
        self._check_open()
        return self

    def __exit__(self, *args: ty.Any) -> None:
        self.close()

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        self._check_open()
        if self._iterator is None:
            self._iterator = self._read_spectra()
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = None
            raise

    def _get_reader_kwargs(self) -> dict[str, ty.Any]:
        """Return settings needed to reopen this dataset for batch exports."""
        return {
            "ibd_path": str(self.ibd_path),
            "auto_profile": self.auto_profile,
            "mz_ppm": self.mz_ppm,
            "resolution": self.resolution,
            "mz_min": self._mz_min,
            "mz_max": self._mz_max,
            "parse_lib": self._parse_lib,
            "backend": self._requested_backend,
            "pixel_size": self._pixel_size_override,
            "z_plane": self._z_plane,
        }

    def _init(self, ibd_path: PathLike | None = None, parse_lib: str | None = None) -> None:
        """Initialize metadata."""
        _, self._ibd_path, self._icache_path = infer_path(self.path, ibd_path)
        assert self._icache_path is not None
        self._source_fingerprint = fingerprint(self.path, self.ibd_path)
        metadata = load_cache(self._icache_path, self._source_fingerprint)
        if metadata is not None:
            try:
                validate_metadata(metadata, self.path, self.ibd_path)
            except (ValueError, KeyError, IndexError):
                metadata = None
        rebuild = metadata is None
        if metadata is None:
            metadata = parse_metadata(self.path, parse_lib)
            validate_metadata(metadata, self.path, self.ibd_path)
        self._source_metadata = metadata
        check_uuid(metadata.uuid, self.path, self.ibd_path)
        if rebuild:
            self._write_icache_safely()
        self.mz_precision, self.int_precision = metadata.mz_precision, metadata.int_precision
        self._mz_size, self._int_size = SIZE_DICT[self.mz_precision], SIZE_DICT[self.int_precision]
        self.byte_offsets = metadata.offsets
        self._xyz_coordinates = metadata.coordinates
        self._spectrum_mode = ty.cast(SpectrumMode | None, metadata.mode)
        self._apply_geometry()

    def _apply_geometry(self) -> None:
        """Apply per-reader overrides and selections to the complete source metadata."""
        metadata = self._source_metadata
        values = metadata.values.copy()
        values["max count of pixels z"] = int(metadata.coordinates[:, 2].max()) if len(metadata.coordinates) else 1
        if self._pixel_size_override is not None:
            values["pixel size x"], values["pixel size y"] = self._pixel_size_override
        else:
            for axis in ("x", "y"):
                if values[f"pixel size {axis}"] is None:
                    warn(
                        f"{self.path}: {values[f'pixel size {axis} reason']} for {axis}; "
                        "provide pixel_size=(x_um, y_um) for physical sizing.",
                        stacklevel=3,
                    )
        self._imzml_cache = IMZMLCache(values)
        if self._z_plane is not None:
            selection = metadata.coordinates[:, 2] == self._z_plane
            if not np.any(selection):
                raise ValueError(f"{self.path}: z_plane={self._z_plane} is absent from normalized z coordinates.")
            self.byte_offsets = metadata.offsets[selection]
            self._xyz_coordinates = metadata.coordinates[selection]
        elif self._spectrum_mode == SPECTRUM_MODE_CENTROID:
            self._apply_mass_range(metadata.mz_min, metadata.mz_max)
        assert self._xyz_coordinates is not None
        self._xyz_coordinates.setflags(write=False)
        self._image_geometry_error: str | None = None
        self._image_geometry_checked = False

    def _write_icache_safely(self) -> None:
        """Write the imzML cache when the cache path is available."""
        if self._icache_path is None:
            return
        try:
            save_cache(self._icache_path, self._source_metadata, self._source_fingerprint)
        except OSError:
            # Caching is optional on read-only or full filesystems.
            pass

    def _apply_mass_range(self, mz_min: float | None, mz_max: float | None) -> None:
        """Apply cached mass bounds without overriding explicit constructor values."""
        if self._mz_min is None:
            self._mz_min = mz_min
        if self._mz_max is None:
            self._mz_max = mz_max

    def _apply_centroid_xml_mass_range(self, mz_min: float | None, mz_max: float | None) -> None:
        """Apply complete XML observed bounds for centroid data."""
        if self._spectrum_mode != SPECTRUM_MODE_CENTROID or mz_min is None or mz_max is None:
            return
        self._apply_mass_range(mz_min, mz_max)

    @property
    def mz_ppm(self) -> float:
        """Return m/z ppm spacing."""
        return self._mz_ppm

    @mz_ppm.setter
    def mz_ppm(self, value: float | str) -> None:
        """Set m/z ppm spacing."""
        self._mz_ppm = float(value)
        if self._is_centroid is not False:
            self._mz_grid = None  # reset mz grid

    @property
    def ibd_path(self) -> Path:
        """Return path to ibd file."""
        if not self._ibd_path:
            raise ValueError("ibd path is not set.")
        return self._ibd_path

    @property
    def mz_min(self) -> float:
        """Minimum m/z value."""
        mz_min, _ = self._estimate_mass_range()
        return mz_min

    @property
    def mz_max(self) -> float:
        """Maximum m/z value."""
        _, mz_max = self._estimate_mass_range()
        return mz_max

    @property
    def metadata(self) -> IMZMLCache:
        """Cache."""
        return self._imzml_cache

    @property
    def rois(self) -> list[int]:
        """Return a list of ROI indices."""
        return [0]  # imzML files always have single ROI

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether data is in centroid or profile mode."""
        if self._is_centroid is None:
            if self._spectrum_mode == SPECTRUM_MODE_CENTROID:
                self._is_centroid = True
            elif self._spectrum_mode == SPECTRUM_MODE_PROFILE:
                self._is_centroid = False
            else:
                x, _ = self.get_spectrum(0)
                for _x, _ in self._read_spectra(range(1, self.n_pixels)):
                    if _x.shape != x.shape:
                        self._is_centroid = True
                        break
                if self._is_centroid is None:
                    self._is_centroid = False
        return self._is_centroid

    @property
    def x_pixel_size(self) -> float:
        """Return x pixel size in micrometers."""
        return self._physical_size("x")

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size in micrometers."""
        return self._physical_size("y")

    def _physical_size(self, axis: str) -> float:
        value = self.metadata.PX_SIZE_X if axis == "x" else self.metadata.PX_SIZE_Y
        if value is None:
            reason = self.metadata.metadata_dict[f"pixel size {axis} reason"]
            raise ValueError(f"{self.path}: {reason} for {axis}; provide pixel_size=(x_um, y_um).")
        return value

    @property
    def source_coordinate_base(self) -> tuple[int, int, int | None]:
        """Return detected input axis bases; None means z was absent throughout."""
        return self._source_metadata.coordinate_base

    @property
    def x_size(self) -> int:
        """Return the full acquisition grid width, including unoccupied columns."""
        return self.metadata.PX_MAX_X

    @property
    def y_size(self) -> int:
        """Return the full acquisition grid height, including unoccupied rows."""
        return self.metadata.PX_MAX_Y

    def _check_image_geometry(self) -> None:
        if not self._image_geometry_checked:
            if len(np.unique(self.z_coordinates)) > 1:
                self._image_geometry_error = "select z_plane= before using a 2-D image or centroid store"
            elif len(np.unique(self.xyz_coordinates[:, :2], axis=0)) != self.n_pixels:
                self._image_geometry_error = "duplicate x/y coordinates cannot form an unambiguous 2-D image"
            self._image_geometry_checked = True
        if self._image_geometry_error:
            raise ValueError(f"{self.path}: {self._image_geometry_error}.")

    def get_index_for_location(self, x: float, y: float) -> int | None:
        """Find a pixel by its normalized one-based x/y location."""
        self._check_image_geometry()
        if not float(x).is_integer() or not float(y).is_integer():
            raise ValueError("Image coordinates must be whole numbers.")
        if not (1 <= x <= self.x_size and 1 <= y <= self.y_size):
            raise ValueError(f"Coordinate ({x}, {y}) is outside the acquisition grid.")
        indices = np.flatnonzero((self.x_coordinates == x) & (self.y_coordinates == y))
        return int(indices[0]) if len(indices) else None

    @property
    def mz_x(self) -> np.ndarray:
        """Return m/z axis."""
        if self._mz_grid is None:
            if self.is_centroid:
                mz_min, mz_max = self._estimate_mass_range()
                self._mz_grid = get_ppm_axis(mz_min, mz_max, self.mz_ppm)
            else:
                self._mz_grid = self.get_spectrum(0)[0]
        return self._mz_grid

    def get_physical_coordinates(self, index: int) -> tuple[float, float]:
        """For a pixel index i, return real-world coordinates in micrometers.

        This is equivalent to multiplying the image coordinates of the given pixel with the pixel size.
        """
        x, y, _ = self.xyz_coordinates[index]
        return float(x * self.x_pixel_size), float(y * self.y_pixel_size)

    def reshape(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Reshape vector of intensities."""
        self._check_image_geometry()
        if len(array) != self.n_pixels:
            raise ValueError("Wrong size and shape of the array.")
        dtype = np.float32 if np.isnan(fill_value) else array.dtype
        im = np.full((self.metadata.PX_MAX_Y, self.metadata.PX_MAX_X), fill_value=fill_value, dtype=dtype)
        im[self.y_coordinates - 1, self.x_coordinates - 1] = array
        return im

    def reshape_batch(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Batch reshaping of images."""
        self._check_image_geometry()
        if array.ndim != 2:
            raise ValueError("Expected 2-D array.")
        if len(array) != self.n_pixels:
            raise ValueError("Wrong size and shape of the array.")
        n = array.shape[1]
        dtype = np.float32 if np.isnan(fill_value) else array.dtype
        im = np.full((n, self.metadata.PX_MAX_Y, self.metadata.PX_MAX_X), fill_value=fill_value, dtype=dtype)
        for i in range(n):
            im[i, self.y_coordinates - 1, self.x_coordinates - 1] = array[:, i]
        return im

    def flatten(self, image: np.ndarray) -> np.ndarray:
        """Retrieve the original vector of intensities from an image."""
        self._check_image_geometry()
        if image.shape != self.image_shape:
            raise ValueError(f"Expected an image with shape {self.image_shape}.")
        return ty.cast(np.ndarray, image[self.y_coordinates - 1, self.x_coordinates - 1])

    def to_hdf5(
        self,
        hdf_path: PathLike,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        as_flat: bool = True,
        max_mem: float = 512,
        silent: bool = False,
    ) -> Path:
        """Export a single z plane to a two-dimensional HDF5 centroid store."""
        self._check_image_geometry()
        return super().to_hdf5(hdf_path, mzs, tol, ppm, as_flat, max_mem, silent)

    extract_centroids_hdf5 = to_hdf5

    def to_zarr(
        self,
        zarr_path: PathLike,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        as_flat: bool = True,
        chunk_size: tuple[int, int] | None = None,
        silent: bool = False,
    ) -> Path:
        """Export a single z plane to a two-dimensional Zarr centroid store."""
        self._check_image_geometry()
        return super().to_zarr(zarr_path, mzs, tol, ppm, as_flat, chunk_size, silent)

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum."""
        indices = np.asarray(indices)
        if np.any(indices >= self.n_pixels):
            raise ValueError("You cannot specify indices that are greater than the total number of pixels.")
        if scales is None:
            scales = np.ones(self.n_pixels, dtype=np.float32)
        if self.is_centroid:
            return self._get_summed_spectrum_centroid(indices, scales=scales, silent=silent)
        return self._get_summed_spectrum_profile(indices, scales=scales, silent=silent)

    def _get_summed_spectrum_profile(
        self, indices: ty.Iterable[int], scales: np.ndarray, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")
        mz_x, mz_y = self[indices[0]]
        mz_y = mz_y.copy().astype(np.float64) * scales[indices[0]]
        for index, (_, y) in enumerate(
            tqdm(self._read_spectra(indices[1::]), total=len(indices) - 1, disable=silent, desc="Summing spectra..."),
            start=1,
        ):
            mz_y += y * scales[indices[index]]
        return mz_x, mz_y

    def _get_summed_spectrum_centroid(
        self, indices: ty.Iterable[int], scales: np.ndarray, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        # Creating summed spectrum from centroided data is a lot harder because there is no consensus axis in which case
        # we must create our own.
        # We have decided to create a resampled spectrum with a pre-defined ppm limit. This is not ideal but it's better
        # than not doing it at all.

        indices = np.asarray(indices)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")

        mz_x = self.mz_x
        mz_y = np.zeros_like(mz_x, dtype=np.float64)
        for index, (x, y) in enumerate(
            tqdm(self._read_spectra(indices), total=len(indices), disable=silent, desc="Summing spectra...")
        ):
            y = y * scales[indices[index]]
            x, y, _ = self._centroid_to_profile(x, y, resolution=self.resolution, mz_grid=mz_x)
            mz_y += y
        return mz_x, mz_y

    def _estimate_mass_range(self) -> tuple[float, float]:
        """Return the exact acquisition m/z range for the imzML file."""
        if self._mz_min is None or self._mz_max is None:
            if self.is_centroid:
                mz_min, mz_max = self._estimate_centroid_mass_range()
            else:
                mz_x, _ = self.get_spectrum(0)
                if mz_x.size == 0:
                    raise ValueError("Cannot determine m/z range from an empty profile spectrum.")
                mz_min, mz_max = float(np.min(mz_x)), float(np.max(mz_x))
            if self._mz_min is None:
                self._mz_min = float(mz_min)
            if self._mz_max is None:
                self._mz_max = float(mz_max)
            if self._z_plane is None and self._explicit_mz_bounds == (None, None):
                self._source_metadata.mz_min, self._source_metadata.mz_max = self._mz_min, self._mz_max
                self._write_icache_safely()
        return self._mz_min, self._mz_max

    def _estimate_centroid_mass_range(self) -> tuple[float, float]:
        """Scan all centroid spectra and return exact global m/z bounds."""
        self._check_open()
        mz_min, mz_max = np.inf, -np.inf
        if self._native is not None:
            for start in range(0, self.n_pixels, _BLOCK_SIZE):
                low, high = self._native.mass_range(start, min(start + _BLOCK_SIZE, self.n_pixels))
                mz_min, mz_max = min(mz_min, low), max(mz_max, high)
            if not np.isfinite(mz_min) or not np.isfinite(mz_max):
                raise ValueError("Cannot determine m/z range from empty centroid spectra.")
            return float(mz_min), float(mz_max)
        with open(self.ibd_path, "rb") as f_ptr:
            for index in self.pixels:
                mz_o, mz_l, _, _ = self.byte_offsets[index]
                f_ptr.seek(mz_o)
                x = np.frombuffer(f_ptr.read(mz_l * self._mz_size), dtype=self._mz_dtype)
                if x.size == 0:
                    continue
                mz_min = min(mz_min, float(np.min(x)))
                mz_max = max(mz_max, float(np.max(x)))
        if not np.isfinite(mz_min) or not np.isfinite(mz_max):
            raise ValueError("Cannot determine m/z range from empty centroid spectra.")
        return float(mz_min), float(mz_max)

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        self._check_open()
        index = self._normalize_index(index)
        if self._native is not None:
            return ty.cast(tuple[np.ndarray, np.ndarray], self._native.spectrum(index))
        with open(self.ibd_path, "rb") as f_ptr:
            return self._python_spectrum(f_ptr, index)

    def get_intensities(self, index: int) -> np.ndarray:
        """Return pixel intensities without loading its m/z array."""
        self._check_open()
        index = self._normalize_index(index)
        if self._native is not None:
            return ty.cast(np.ndarray, self._native.intensities(index))
        offset, length = self.byte_offsets[index, 2:]
        with open(self.ibd_path, "rb") as f_ptr:
            f_ptr.seek(offset)
            return self._python_array(f_ptr, length, self._int_dtype)

    def _normalize_index(self, index: int) -> int:
        index = operator.index(index)
        n_pixels = self.n_pixels
        if index < 0:
            index += n_pixels
        if not 0 <= index < n_pixels:
            raise IndexError("Pixel index out of range.")
        return index

    @staticmethod
    def _selection(indices: ty.Iterable[int]) -> np.ndarray:
        values = np.asarray(indices if isinstance(indices, (np.ndarray, list, tuple, range)) else list(indices))
        if values.ndim != 1:
            raise ValueError("Expected one-dimensional pixel indices.")
        if values.size and values.dtype.kind not in "iu":
            raise TypeError("Pixel indices must be integers.")
        if values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
            raise IndexError("Pixel index out of range.")
        return np.ascontiguousarray(values, dtype=np.int64)

    @staticmethod
    def _python_array(f_ptr: ty.BinaryIO, length: int, dtype: np.dtype) -> np.ndarray:
        expected = int(length) * dtype.itemsize
        data = f_ptr.read(expected)
        if len(data) != expected:
            raise ValueError("External array extends beyond the ibd file.")
        return np.frombuffer(data, dtype=dtype)

    def _python_spectrum(self, f_ptr: ty.BinaryIO, index: int) -> tuple[np.ndarray, np.ndarray]:
        mz_offset, mz_length, int_offset, int_length = self.byte_offsets[index]
        f_ptr.seek(mz_offset)
        mzs = self._python_array(f_ptr, mz_length, self._mz_dtype)
        f_ptr.seek(int_offset)
        return mzs, self._python_array(f_ptr, int_length, self._int_dtype)

    def _read_spectra(self, indices: ty.Iterable[int] | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Read spectra without constantly opening and closing the file handle."""
        self._check_open()
        selection = None if indices is None else self._selection(indices)
        if self._native is not None:
            return ty.cast(ty.Iterator[tuple[np.ndarray, np.ndarray]], self._native.spectra(selection))
        return self._python_spectra(range(self.n_pixels) if selection is None else selection)

    def _python_spectra(self, indices: ty.Iterable[int]) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        with open(self.ibd_path, "rb") as f_ptr:
            for index in indices:
                yield self._python_spectrum(f_ptr, self._normalize_index(index))

    def spectra_iter(
        self, indices: ty.Iterable[int] | None = None, silent: bool = False
    ) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate spectra in selection order, without allocating a default pixel index."""
        selection = None if indices is None else self._selection(indices)
        spectra = self._read_spectra(selection)
        if silent:
            return spectra
        total = self.n_pixels if selection is None else len(selection)
        return iter(tqdm(spectra, total=total, miniters=500, desc="Iterating spectra..."))

    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return cached total ion currents using intensity-only native reduction."""
        self._check_open()
        if self._native is None:
            return super().get_tic(silent=silent)
        if self._tic is None:
            result = np.empty(self.n_pixels, dtype=np.float64)
            for start in tqdm(range(0, self.n_pixels, _BLOCK_SIZE), disable=silent, desc="Extracting TIC..."):
                stop = min(start + _BLOCK_SIZE, self.n_pixels)
                result[start:stop] = self._native.tic(start, stop)
            self._tic = result
        return self._tic

    def get_ion_image(
        self,
        mz: float,
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> np.ndarray:
        """Return an ion image using the existing inclusive Da or ppm window."""
        self._check_open()
        if self._native is None and (self.is_centroid or self._shared_mz):
            return super().get_ion_image(mz, tol, ppm, fill_value, silent)
        # Single-image extraction historically accepts either sign of ppm.
        ppm = abs(ppm) if ppm is not None else None
        return self.reshape(self._get_ions([mz], tol, ppm, fill_value, silent)[:, 0])

    def _ion_blocks(
        self, lower: np.ndarray, upper: np.ndarray, silent: bool
    ) -> ty.Iterator[tuple[int, int, np.ndarray]]:
        self._check_open()
        if lower.size == 0:
            return
        shared_indices = None
        if self._native is None and self._shared_mz:
            shared_indices = find_between_batch(self.get_spectrum(0)[0], lower, upper)
        for start in tqdm(range(0, self.n_pixels, _BLOCK_SIZE), disable=silent, desc="Extracting ions..."):
            stop = min(start + _BLOCK_SIZE, self.n_pixels)
            if self._native is not None:
                block = self._native.ion_table(start, stop, lower, upper)
            else:
                block = np.empty((stop - start, len(lower)), dtype=np.float32)
                for row, (x, y) in enumerate(self._read_spectra(range(start, stop))):
                    block[row] = (
                        accumulate_peaks_centroid(lower, upper, x, y)
                        if shared_indices is None
                        else accumulate_peaks_profile(shared_indices, y)
                    )
            yield start, stop, block

    def _get_ions(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> np.ndarray:
        self._check_open()
        values = np.asarray(mzs if isinstance(mzs, (np.ndarray, list, tuple)) else list(mzs), dtype=np.float64)
        lower, upper = (np.ascontiguousarray(v, dtype=np.float64) for v in get_mzs_for_tol(values, tol, ppm))
        result = np.full((self.n_pixels, len(values)), fill_value, dtype=np.float32)
        for start, stop, block in self._ion_blocks(lower, upper, silent):
            result[start:stop] = block
        return result

    def _get_ions_yield(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> ty.Iterator[tuple[int | None, int | None, np.ndarray | None]]:
        self._check_open()
        values = np.asarray(list(mzs), dtype=np.float64)
        lower, upper = (np.ascontiguousarray(v, dtype=np.float64) for v in get_mzs_for_tol(values, tol, ppm))
        result = np.full((self.n_pixels, len(values)), fill_value, dtype=np.float32)
        for start, stop, block in self._ion_blocks(lower, upper, silent):
            result[start:stop] = block
            yield stop - 1, self.n_pixels, None
        yield None, None, result


def infer_path(path: Path, ibd_path: PathLike | None = None) -> tuple[Path, Path, Path]:
    """Resolve an exact binary sibling or a unique case-insensitive filename match."""
    if ibd_path is None:
        exact = path.with_suffix(".ibd")
        if exact.is_file():
            ibd_path = exact
        else:
            matches = [
                item
                for item in path.parent.iterdir()
                if item.is_file() and item.name.casefold() == exact.name.casefold()
            ]
            if len(matches) > 1:
                raise ValueError(f"{path}: ambiguous binary siblings: {matches}.")
            ibd_path = matches[0] if matches else exact
    return path, Path(ibd_path), path.with_suffix(".icache")


def read_imzml_metadata(root: ty.Any, sl: str = "{http://psi.hupo.org/ms/mzml}") -> dict[str, ty.Any]:
    """Read resolved acquisition dimensions and pixel sizes in um."""
    return read_values(root)


def init_metadata(
    path: Path,
    parse_lib: str | None = None,
    sl: str = "{http://psi.hupo.org/ms/mzml}",
) -> tuple[ty.Any, str, str, np.ndarray, np.ndarray, SpectrumMode | None, float | None, float | None]:
    """Stream resolved metadata, pruning spectrum XML after each record."""
    metadata = parse_metadata(path, parse_lib)
    return (
        metadata.root,
        metadata.mz_precision,
        metadata.int_precision,
        metadata.offsets,
        metadata.coordinates,
        ty.cast(SpectrumMode | None, metadata.mode),
        metadata.mz_min,
        metadata.mz_max,
    )


def init_spectrum_metadata(
    path: Path,
    parse_lib: str | None = None,
    sl: str = "{http://psi.hupo.org/ms/mzml}",
    fallback_spectrum_mode: SpectrumMode | None = None,
) -> tuple[SpectrumMode | None, float | None, float | None]:
    """Read resolved mode and complete observed mass bounds from spectrum XML."""
    metadata = parse_metadata(path, parse_lib)
    return ty.cast(SpectrumMode | None, metadata.mode) or fallback_spectrum_mode, metadata.mz_min, metadata.mz_max


def init_spectrum_mode(
    path: Path, parse_lib: str | None = None, sl: str = "{http://psi.hupo.org/ms/mzml}"
) -> SpectrumMode | None:
    """Read resolved centroid/profile metadata without reading binary arrays."""
    return init_spectrum_metadata(path, parse_lib, sl)[0]


def is_imzml(path: PathLike) -> bool:
    """Check if path is imzml."""
    path = Path(path)
    return path.suffix.lower() == ".imzml"


@hook_impl
def imzy_reader(path: PathLike, reader_kwargs: dict[str, ty.Any], **kwargs: ty.Any) -> IMZMLReader | None:
    """Return an imzML reader with the options explicitly forwarded by Pluggy."""
    if is_imzml(path):
        return IMZMLReader(path, **(reader_kwargs | kwargs))
    return None
