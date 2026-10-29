"""Reader for SCiLS Lab datasets (`.sbd` + `.slx`).

File format
-----------
A SCiLS dataset is a pair of files that share a stem:

- `.sbd` - HDF5 file with the spectral data, coordinates, regions, normalizations, optical images, etc.
- `.slx` - SQLite file with acquisition metadata (optional for reading ion images, see `_slx.py`).

Everything needed for ion images lives in the SBD file, mostly inside `SpectraGroup/`. The (sparse, centroided) peak
data is stored twice:

- **image-major** (per m/z bin; used here): `imageSpotOffsets`, `imageIntensityOffsets`, `compressedImageSpots` and
  `imageIntensities`. The m/z axis is `SamplePositions/GlobalMassAxis/SamplePositions` and bin `j` corresponds to
  axis value `j`. Bin `j` owns bytes `imageSpotOffsets[j]:imageSpotOffsets[j+1]` of `compressedImageSpots` and floats
  `imageIntensityOffsets[j]:imageIntensityOffsets[j+1]` of `imageIntensities` (uncompressed).
- spectrum-major (per spot): `mzOffsets`, `intensityOffsets`, `compressedMz`, `intensities`. The m/z encoding of
  `compressedMz` has not been decoded, so per-pixel spectra are not supported yet.

Each bin's byte range is an independent zlib stream. Once decompressed it holds `n` unsigned integers (1, 2, 4 or
8 bytes wide; `n` is the number of spots in the bin), stored as byte-planes (all least-significant bytes first, i.e.
the array is transposed) and delta-encoded. Decoding is therefore: `zlib` -> reshape to `(width, n)` -> transpose -> view as
`uint{8 * width}` -> `cumsum` -> spot indices (sorted). The intensities pair one-to-one with these spot indices.

Other relevant datasets: `Registrations/<i>/OriginalCoordinates` (2, n_spots) integer grid coordinates,
`Registrations/<i>/Rasters/<r>/PixelToWorld` (4, 4; pixel size in micrometers), `MeasurementRasterIndices` (raster index
of each spot) and `SpectraGroup/Normalizations/<id>/{Values,TargetValue}` where `Values = TIC / TargetValue`.

Note
----
The ion images reflect the processed peak list that SCiLS stored on its global m/z axis (quantized to the axis spacing,
roughly 1.5 - 10 mDa), not the raw vendor data.

SCiLS Lab (or HDFView) sometimes leave the HDF5 "open for write" flag set in the SBD file, which makes a plain read-only
open fail. The file is then re-opened in SWMR read mode, which never modifies the file.
"""

from __future__ import annotations

import typing as ty
import warnings
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from ims_utils.spectrum import get_mzs_for_tol
from koyo.typing import PathLike

from imzy._readers._base import BaseReader
from imzy._readers.scils._slx import is_sqlite, read_slx_metadata
from imzy.hookspec import hook_impl

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
SBD_SUFFIX = ".sbd"
SLX_SUFFIX = ".slx"
_NORMALIZATION_ALIASES = {
    "tic": "total ion count",
    "total ion count": "total ion count",
    "rms": "root mean square",
    "root mean square": "root mean square",
}
_SHORT_NORMALIZATION_NAMES = {"total ion count": "TIC", "root mean square": "RMS"}


def _find_sibling(path: Path, suffix: str) -> Path | None:
    """Find file with the same stem and a different suffix (case-insensitive)."""
    for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
        if candidate.is_file():
            return candidate
    for candidate in path.parent.glob(f"{path.stem}.*"):
        if candidate.suffix.lower() == suffix and candidate.is_file():
            return candidate
    return None


def resolve_scils_paths(path: PathLike) -> tuple[Path | None, Path | None]:
    """Return `(sbd_path, slx_path)` for either of the two files. Missing files are returned as `None`."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == SBD_SUFFIX:
        return (path if path.is_file() else None), _find_sibling(path, SLX_SUFFIX)
    if suffix == SLX_SUFFIX:
        return _find_sibling(path, SBD_SUFFIX), (path if path.is_file() else None)
    return None, None


def is_scils(path: PathLike) -> bool:
    """Check whether the path points to a SCiLS `.sbd` (HDF5) or `.slx` (SQLite) file with an existing SBD file."""
    sbd_path, slx_path = resolve_scils_paths(path)
    if sbd_path is None:
        return False
    if Path(path).suffix.lower() == SLX_SUFFIX and not (slx_path and is_sqlite(slx_path)):
        return False
    try:
        with open(sbd_path, "rb") as f:
            return f.read(len(HDF5_MAGIC)) == HDF5_MAGIC
    except OSError:
        return False


def decode_spot_indices(raw: bytes, n: int) -> np.ndarray:
    """Decode the spot indices of a single m/z bin.

    Parameters
    ----------
    raw : bytes
        Compressed (zlib) bytes of the bin, taken from `SpectraGroup/compressedImageSpots`.
    n : int
        Number of spots in the bin (`imageIntensityOffsets[j + 1] - imageIntensityOffsets[j]`).
    """
    data = zlib.decompress(raw)
    width, remainder = divmod(len(data), n)
    if remainder or width not in (1, 2, 4, 8):
        raise ValueError(f"Unexpected size of decompressed SCiLS spot block ({len(data)} bytes for {n} spots).")
    planes = np.frombuffer(data, dtype=np.uint8).reshape(width, n)
    deltas = np.ascontiguousarray(planes.T).view(f"<u{width}").reshape(n)
    return np.cumsum(deltas, dtype=np.int64)


def _attr(attrs: ty.Mapping[str, ty.Any], key: str, default: ty.Any = None) -> ty.Any:
    """Read scalar attribute (SCiLS stores them as 1-element byte-string arrays)."""
    value = attrs.get(key)
    if value is None:
        return default
    value = np.asarray(value).ravel()
    if not value.size:
        return default
    item = value[0]
    return item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item


class SCiLSReader(BaseReader):
    """Reader for SCiLS Lab `.sbd`/`.slx` datasets.

    Ion images are extracted from the image-major layout of the SBD file, so the cost of an ion image scales with the
    number of m/z bins inside the window rather than with the size of the dataset.

    Parameters
    ----------
    path : PathLike
        Path to the `.sbd` or `.slx` file. The sibling file is found automatically; the SLX file is optional.
    auto_profile : bool
        Unused, kept for API compatibility with other readers.
    roi : int, optional
        Raster (measurement region) index to restrict the reader to. By default, all spots are used. Datasets that
        contain multiple rasters can have overlapping coordinates, in which case selecting each raster is advised.
    """

    def __init__(self, path: PathLike, auto_profile: bool = True, roi: int | None = None):
        sbd_path, slx_path = resolve_scils_paths(path)
        if sbd_path is None:
            raise FileNotFoundError(f"Could not find the SCiLS .sbd file for '{path}'.")
        super().__init__(sbd_path, auto_profile=auto_profile)
        self.slx_path = slx_path
        self.roi = None if roi in (None, "None") else int(roi)  # type: ignore[call-overload]
        self._init()

    # ------------------------------------------------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------------------------------------------------
    @contextmanager
    def _open(self) -> ty.Iterator[ty.Any]:
        """Open the SBD file for reading, also when SCiLS/HDFView left the "open for write" flag set."""
        import h5py

        error: OSError | None = None
        for kwargs in ({}, {"swmr": True}, {"swmr": True, "locking": False}):
            try:
                f = h5py.File(self.path, "r", **kwargs)
            except OSError as exc:
                error = error or exc
                continue
            try:
                yield f
            finally:
                f.close()
            return
        raise error  # type: ignore[misc]

    def _init(self) -> None:
        with self._open() as f:
            if "SpectraGroup" not in f:
                raise ValueError(f"'{self.path}' does not look like a SCiLS .sbd file (missing 'SpectraGroup').")
            attrs = f.attrs
            self.spectra_type = str(_attr(attrs, "SpectraType", ""))
            if self.spectra_type and self.spectra_type != "Peaks":
                warnings.warn(
                    f"SCiLS spectra type '{self.spectra_type}' has not been tested - only 'Peaks' is supported.",
                    stacklevel=3,
                )
            self.sbd_attributes: dict[str, ty.Any] = {key: _attr(attrs, key) for key in attrs}
            self._init_index(f)
            coordinates = self._init_geometry(f)
            self._init_normalizations(f)
            self._source_attributes = self._read_source_attributes(f)

        x, y = coordinates[0, self._spots], coordinates[1, self._spots]
        self.x_min_raw, self.x_max_raw = int(x.min()), int(x.max())
        self.y_min_raw, self.y_max_raw = int(y.min()), int(y.max())
        self._xyz_coordinates = np.column_stack((x - self.x_min_raw, y - self.y_min_raw, np.zeros_like(x)))

        self.slx_metadata: dict[str, ty.Any] = read_slx_metadata(self.slx_path) if self.slx_path else {}

    def _init_index(self, f: ty.Any) -> None:
        """Load the m/z axis and the image-major index (small arrays; the peak data itself is read on demand)."""
        positions = f["SamplePositions"]
        axis_key = "GlobalMassAxis" if "GlobalMassAxis" in positions else next(iter(positions))
        self._mz_axis = positions[axis_key]["SamplePositions"][:].astype(np.float64)
        spectra = f["SpectraGroup"]
        self._spot_offsets = self._extend_offsets(
            spectra["imageSpotOffsets"][:], spectra["compressedImageSpots"].shape[0]
        )
        self._intensity_offsets = self._extend_offsets(
            spectra["imageIntensityOffsets"][:], spectra["imageIntensities"].shape[0]
        )

    def _init_geometry(self, f: ty.Any) -> np.ndarray:
        """Load rasters, pixel size and return the raw integer coordinates of all spots, shape (2, n_spots)."""
        registrations = f["Registrations"]
        registration = registrations["0" if "0" in registrations else min(registrations.keys())]
        coordinates = registration["OriginalCoordinates"][:].astype(np.int64)
        n_spots = coordinates.shape[1]
        self.n_spots_all = n_spots
        if "MeasurementRasterIndices" in f:
            self._raster_indices = f["MeasurementRasterIndices"][:].ravel().astype(int)
        else:
            self._raster_indices = np.zeros(n_spots, dtype=int)
        self._rois = np.unique(self._raster_indices).tolist()
        if self.roi is None:
            self._spots = np.arange(n_spots)
        elif self.roi in self._rois:
            self._spots = np.flatnonzero(self._raster_indices == self.roi)
        else:
            raise ValueError(f"ROI {self.roi} was not found. Available ROIs: {self._rois}")

        # pixel size is stored in the pixel -> world (micrometers) matrix of the raster
        self._pixel_size = self._pixel_size_y = 1.0
        rasters = registration.get("Rasters", {})
        raster_key = str(self._rois[0] if self.roi is None else self.roi)
        if raster_key not in rasters and len(rasters):
            raster_key = min(rasters.keys())
        if raster_key in rasters and "PixelToWorld" in rasters[raster_key]:
            matrix = rasters[raster_key]["PixelToWorld"][:]
            self._pixel_size, self._pixel_size_y = float(abs(matrix[0, 0])), float(abs(matrix[1, 1]))
        return coordinates

    def _init_normalizations(self, f: ty.Any) -> None:
        """Load the normalizations (`Values` are divisors, such that `TIC / Values == TargetValue`)."""
        self._normalizations: dict[str, tuple[np.ndarray, float]] = {}
        group = f["SpectraGroup"].get("Normalizations", {})
        for key in group:
            norm = group[key]
            if "Values" not in norm:
                continue
            name = str(_attr(norm.attrs, "name", key))
            target = float(norm["TargetValue"][0]) if "TargetValue" in norm else 1.0
            self._normalizations[name] = (norm["Values"][:].astype(np.float64), target)

    @staticmethod
    def _extend_offsets(offsets: np.ndarray, total: int) -> np.ndarray:
        """Offsets are bin *starts*; append the total length so that every bin has an end."""
        return np.append(offsets.astype(np.int64), np.int64(total))

    @staticmethod
    def _read_source_attributes(f: ty.Any) -> dict[str, ty.Any]:
        """Read `Source/...` attributes that SCiLS stores on the top-level region."""
        regions = f.get("Regions", {})
        for key in regions:
            group = regions[key]
            if hasattr(group, "attrs") and any(str(k).startswith("Source/") for k in group.attrs):
                return {str(k): _attr(group.attrs, k) for k in group.attrs}
        return {}

    # ------------------------------------------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------------------------------------------
    @property
    def mz_min(self) -> float:
        """Minimum m/z value of the global mass axis."""
        return float(self._mz_axis[0])

    @property
    def mz_max(self) -> float:
        """Maximum m/z value of the global mass axis."""
        return float(self._mz_axis[-1])

    @property
    def mz_axis(self) -> np.ndarray:
        """Global mass axis on which SCiLS stores the peaks."""
        return self._mz_axis

    @property
    def is_centroid(self) -> bool:
        """SCiLS stores peak lists, which are centroided."""
        return True

    @property
    def rois(self) -> list[int]:
        """List of rasters (measurement regions)."""
        return list(self._rois)

    def get_n_pixels_for_roi(self, roi: int) -> int:
        """Return the number of pixels in the raster."""
        return int(np.count_nonzero(self._raster_indices == roi))

    def get_pixels_for_roi(self, roi: int) -> np.ndarray:
        """Return indices of the pixels that belong to the raster."""
        if self.roi is not None:
            return self.pixels if roi == self.roi else np.asarray([], dtype=int)
        return np.flatnonzero(self._raster_indices == roi)

    def _get_reader_kwargs(self) -> dict[str, ty.Any]:
        return {} if self.roi is None else {"roi": self.roi}

    @property
    def x_pixel_size(self) -> float:
        """X pixel size in micrometers."""
        return self._pixel_size

    @property
    def y_pixel_size(self) -> float:
        """Y pixel size in micrometers."""
        return self._pixel_size_y

    @property
    def metadata(self) -> dict[str, ty.Any]:
        """Metadata from the SBD attributes and (if available) SLX file."""
        return {"sbd": self.sbd_attributes, "source": self._source_attributes, "slx": self.slx_metadata}

    @property
    def polarity(self) -> str | None:
        """Ion polarity ('positive'/'negative') if known."""
        for source in self.slx_metadata.get("data_sources", []):
            if source.get("polarity"):
                return str(source["polarity"])
        polarity = self._source_attributes.get("Source/Polarity")
        return str(polarity) if polarity else None

    # ------------------------------------------------------------------------------------------------------------
    # normalizations
    # ------------------------------------------------------------------------------------------------------------
    @property
    def normalization_names(self) -> list[str]:
        """Names of the normalizations stored in the file, e.g. 'Total Ion Count'."""
        return list(self._normalizations)

    def _resolve_normalization(self, name: str) -> str:
        lookup = {key.lower(): key for key in self._normalizations}
        key = _NORMALIZATION_ALIASES.get(name.lower(), name.lower())
        if key not in lookup:
            raise ValueError(f"Normalization '{name}' not found. Available normalizations: {self.normalization_names}")
        return lookup[key]

    def get_normalization_factors(self, name: str) -> np.ndarray:
        """Return SCiLS' per-pixel normalization divisors (`Values`).

        An image normalized the SCiLS way is `image / factors`, which makes the per-pixel TIC (or RMS) equal to the
        normalization's target value.
        """
        values, _ = self._normalizations[self._resolve_normalization(name)]
        return values[self._spots]

    def get_normalizations(self, silent: bool = False) -> dict[str, np.ndarray]:
        """Get normalizations stored in the file as imzy-style multipliers.

        Keys are 'TIC' and 'RMS' where available (other normalizations keep their SCiLS name). Like in other imzy
        readers, the multiplier is `1 / (value / median(value))`.
        """
        data = {}
        for name, (values, _) in self._normalizations.items():
            values = values[self._spots]
            positive = values > 0
            median = np.median(values[positive]) if positive.any() else 1.0
            multiplier = np.ones(len(values), dtype=np.float32)
            multiplier[positive] = median / values[positive]
            data[_SHORT_NORMALIZATION_NAMES.get(name.lower(), name)] = multiplier
        return data

    # ------------------------------------------------------------------------------------------------------------
    # spectra (not supported yet)
    # ------------------------------------------------------------------------------------------------------------
    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Reading per-pixel spectra is not supported yet for SCiLS datasets.")

    def _read_spectra(self, indices: np.ndarray | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError("Reading per-pixel spectra is not supported yet for SCiLS datasets.")

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Summed spectrum is not supported yet for SCiLS datasets."""
        raise NotImplementedError("Summed spectra are not supported yet for SCiLS datasets.")

    # ------------------------------------------------------------------------------------------------------------
    # TIC
    # ------------------------------------------------------------------------------------------------------------
    def get_tic(self, silent: bool = False) -> np.ndarray:
        """Return TIC per pixel.

        When SCiLS stored a 'Total Ion Count' normalization, the TIC is derived from it (`Values * TargetValue`), which
        avoids reading all intensities; otherwise intensities are summed per spot.
        """
        if self._tic is None:
            for name, (values, target) in self._normalizations.items():
                if name.lower() == "total ion count":
                    tic = values * target
                    break
            else:
                with self._open() as f:
                    spectra = f["SpectraGroup"]
                    offsets = spectra["intensityOffsets"][:].astype(np.int64)
                    intensities = spectra["intensities"][:].astype(np.float64)
                tic = np.zeros(len(offsets))
                ends = np.append(offsets[1:], len(intensities))
                nonempty = ends > offsets
                tic[nonempty] = np.add.reduceat(intensities, offsets[nonempty])
            self._tic = tic[self._spots].astype(np.float32)
        return self._tic

    # ------------------------------------------------------------------------------------------------------------
    # ion images
    # ------------------------------------------------------------------------------------------------------------
    def _window_values(self, f: ty.Any, mz_min: float, mz_max: float) -> np.ndarray:
        """Sum intensities of all bins with `mz_min <= m/z <= mz_max` for each spot of the file."""
        n_spots = self.n_spots_all
        start = int(np.searchsorted(self._mz_axis, mz_min, side="left"))
        end = int(np.searchsorted(self._mz_axis, mz_max, side="right"))
        if end <= start:
            return np.zeros(n_spots)
        spot_offsets, intensity_offsets = self._spot_offsets, self._intensity_offsets
        if intensity_offsets[end] == intensity_offsets[start]:
            return np.zeros(n_spots)

        spectra = f["SpectraGroup"]
        spot_start = spot_offsets[start]
        compressed = spectra["compressedImageSpots"][spot_start : spot_offsets[end]].tobytes()
        intensities = spectra["imageIntensities"][intensity_offsets[start] : intensity_offsets[end]]

        blocks = []
        for j in range(start, end):
            n = int(intensity_offsets[j + 1] - intensity_offsets[j])
            if n:
                blocks.append(
                    decode_spot_indices(compressed[spot_offsets[j] - spot_start : spot_offsets[j + 1] - spot_start], n)
                )
        spots = np.concatenate(blocks)
        return np.bincount(spots, weights=intensities, minlength=n_spots)[:n_spots]

    def _windows_to_pixels(
        self,
        mzs_min: np.ndarray,
        mzs_max: np.ndarray,
        normalization: str | None = None,
    ) -> np.ndarray:
        """Return array of shape (n_pixels, n_windows)."""
        factors = None
        if normalization is not None:
            factors = self.get_normalization_factors(normalization)
            factors = np.where(factors > 0, factors, np.inf)  # pixels without signal stay at 0
        res = np.zeros((self.n_pixels, len(mzs_min)), dtype=np.float32)
        with self._open() as f:
            for i in range(len(mzs_min)):
                values = self._window_values(f, float(mzs_min[i]), float(mzs_max[i]))[self._spots]
                res[:, i] = values if factors is None else values / factors
        return res

    def get_ion_image_range(self, mz_min: float, mz_max: float, normalization: str | None = None) -> np.ndarray:
        """Return ion image for the explicit window `mz_min <= m/z <= mz_max`.

        Parameters
        ----------
        mz_min, mz_max : float
            Boundaries of the m/z window (inclusive).
        normalization : str, optional
            Name of one of the stored normalizations (e.g. 'TIC', 'RMS'); see `normalization_names`.
        """
        if mz_max < mz_min:
            raise ValueError("`mz_max` must not be smaller than `mz_min`.")
        res = self._windows_to_pixels(np.asarray([mz_min]), np.asarray([mz_max]), normalization)
        return self.reshape(res[:, 0])

    def get_ion_image(
        self,
        mz: float,
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
        normalization: str | None = None,
    ) -> np.ndarray:
        """Return ion image for specified m/z with tolerance or ppm.

        `fill_value` is only accepted for API compatibility; every pixel has a value.
        """
        mzs_min, mzs_max = get_mzs_for_tol(np.asarray([mz]), tol, ppm)
        res = self._windows_to_pixels(mzs_min, mzs_max, normalization)
        return self.reshape(res[:, 0])

    def _get_ions(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
        normalization: str | None = None,
    ) -> np.ndarray:
        mzs_min, mzs_max = get_mzs_for_tol(np.asarray(list(mzs), dtype=np.float64), tol, ppm)
        return self._windows_to_pixels(mzs_min, mzs_max, normalization)

    def _get_ions_yield(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
    ) -> ty.Iterator[tuple[int | None, int | None, np.ndarray | None]]:
        mzs_min, mzs_max = get_mzs_for_tol(np.asarray(list(mzs), dtype=np.float64), tol, ppm)
        res = np.zeros((self.n_pixels, len(mzs_min)), dtype=np.float32)
        with self._open() as f:
            for i in range(len(mzs_min)):
                res[:, i] = self._window_values(f, float(mzs_min[i]), float(mzs_max[i]))[self._spots]
                yield i, len(mzs_min), None
        yield None, None, res

    def get_ion_images(
        self,
        mzs: ty.Iterable[float],
        tol: float | None = None,
        ppm: float | None = None,
        fill_value: float = np.nan,
        silent: bool = False,
        normalization: str | None = None,
    ) -> np.ndarray:
        """Return many ion images for specified m/z values. Output shape is (n_mzs, height, width)."""
        res = self._get_ions(mzs, tol, ppm, fill_value, silent, normalization)
        return self.reshape_batch(res)


@hook_impl
def imzy_reader(path: PathLike, **kwargs: ty.Any) -> SCiLSReader | None:
    """Return SCiLSReader if path is a SCiLS .sbd/.slx file."""
    if is_scils(path):
        return SCiLSReader(path, **kwargs)
    return None
