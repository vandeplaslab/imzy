"""Pure Python reader for Bruker rapifleX MALDI-TOF imaging data.

A rapifleX export is a directory containing:

- ``<name>.dat`` - binary file with a 48-byte header, a uint32 offset table (one entry per raster cell) and
  float32 profile spectra
- ``<name>_info.txt`` - acquisition metadata (mass range, raster step, polarity, ...)
- ``<name>_poslog.txt`` - position log which maps raster positions to regions
- ``<name>.mis`` - optional FlexImaging XML with teaching points and areas
"""

from __future__ import annotations

import re
import struct
import typing as ty
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from koyo.utilities import get_min_max
from loguru import logger
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy.hookspec import hook_impl

HEADER_SIZE = 48
POSITION_PATTERN = re.compile(r"R(\d+)X(\d+)Y(\d+)")


class RapiflexReader(BaseReader):
    """Reader for Bruker rapifleX MALDI-TOF profile data.

    All spectra share a single linear m/z axis spanning ``Mass Start`` - ``Mass End`` from the ``_info.txt`` file.

    Parameters
    ----------
    path : PathLike
        Path to the rapifleX export directory.
    roi : int | None, optional
        Region of interest to load. If None, all regions are loaded.
    """

    _mz_x: np.ndarray | None = None
    _mmap: np.memmap | None = None

    def __init__(self, path: PathLike, roi: int | None = None, **kwargs: ty.Any):
        super().__init__(path, auto_profile=False)
        self.roi = roi
        self.dat_path: Path
        self.info_path: Path
        self.poslog_path: Path
        self.mis_path: Path | None = None
        self.info_metadata: dict[str, ty.Any] = {}
        self.mis_metadata: dict[str, ty.Any] = {}
        self.positions: list[dict[str, ty.Any]] = []
        self.header: dict[str, int] = {}
        self._init()

    def _init(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """Parse metadata, position log and binary header, then select the requested region."""
        self._find_data_files()
        self.info_metadata = parse_info_file(self.info_path)
        if self.mis_path is not None:
            self.mis_metadata = parse_mis_file(self.mis_path)
        self.positions = parse_poslog_file(self.poslog_path)
        self._parse_dat_header()
        self._prepare_raster_metadata()
        self._mmap = np.memmap(self.dat_path, dtype=np.uint8, mode="r")
        self._validate_offsets()
        self._mz_min, self._mz_max = self.get_acquisition_mass_range()
        self.set_region_information(self.roi)

    def __enter__(self) -> RapiflexReader:
        return self

    def __exit__(self, *args: ty.Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Release the memory-mapped data file."""
        self._mmap = None

    def _find_data_files(self) -> None:
        """Find the required rapifleX data files."""
        folder = self.path
        if not folder.is_dir():
            raise ValueError(f"rapifleX path must be a directory: {folder}")

        dat_files = sorted(folder.glob("*.dat"))
        if not dat_files:
            raise ValueError(f"No .dat file found in {folder}")
        if len(dat_files) > 1:
            logger.warning(f"Multiple .dat files found, using first: {dat_files[0]}")
        self.dat_path = dat_files[0]

        info_files = sorted(folder.glob("*_info.txt"))
        if not info_files:
            raise ValueError(f"No *_info.txt file found in {folder}")
        self.info_path = info_files[0]

        poslog_files = sorted(folder.glob("*_poslog.txt"))
        if not poslog_files:
            raise ValueError(f"No *_poslog.txt file found in {folder}")
        self.poslog_path = poslog_files[0]

        mis_files = sorted(folder.glob("*.mis"))
        self.mis_path = mis_files[0] if mis_files else None

    def _parse_dat_header(self) -> None:
        """Parse the binary header and the per-raster offset table."""
        with self.dat_path.open("rb") as handle:
            header_data = handle.read(HEADER_SIZE)
            if len(header_data) != HEADER_SIZE:
                raise ValueError(f"Invalid rapifleX .dat file: header is shorter than {HEADER_SIZE} bytes.")
            values = struct.unpack("<12I", header_data)
            self.header = {
                "header_size": int(values[0]),
                "unknown1": int(values[1]),
                "first_raster_x": int(values[2]),
                "first_raster_y": int(values[3]),
                "raster_width": int(values[4]),
                "raster_height": int(values[5]),
                "n_datapoints": int(values[6]),
            }
            n_raster = self.header["raster_width"] * self.header["raster_height"]
            if n_raster <= 0:
                raise ValueError("Invalid rapifleX raster dimensions in .dat header.")
            self.offsets = np.frombuffer(handle.read(n_raster * 4), dtype="<u4").copy()
            if self.offsets.size != n_raster:
                raise ValueError("Invalid rapifleX .dat file: offset table is incomplete.")
        self.data_start = HEADER_SIZE + n_raster * 4
        # cells without a spectrum have an offset that points inside the header/offset table (usually 0)
        self.valid_raster_indices = np.where(self.offsets >= self.data_start)[0].astype(np.int64)

    def _prepare_raster_metadata(self) -> None:
        """Compute raster x/y and region number for every raster cell."""
        width, height = self.header["raster_width"], self.header["raster_height"]
        n_raster = width * height
        cells = np.arange(n_raster, dtype=np.int32)
        self.raster_x = cells % width
        self.raster_y = cells // width
        self.raster_region = np.zeros(n_raster, dtype=np.int32)

        first_x, first_y = self.header["first_raster_x"], self.header["first_raster_y"]
        for position in self.positions:
            raster_idx = (position["raster_y"] - first_y) * width + (position["raster_x"] - first_x)
            if 0 <= raster_idx < n_raster:
                self.raster_region[raster_idx] = position["region"]

    def _validate_offsets(self) -> None:
        """Make sure that every spectrum fits inside the data file."""
        assert self._mmap is not None
        ends = self.offsets[self.valid_raster_indices].astype(np.int64) + self.n_datapoints * 4
        truncated = np.where(ends > self._mmap.size)[0]
        if truncated.size:
            raster_idx = int(self.valid_raster_indices[truncated[0]])
            raise ValueError(
                f"rapifleX .dat file is truncated: spectrum at raster position x={self.raster_x[raster_idx]},"
                f" y={self.raster_y[raster_idx]} ends at byte {ends[truncated[0]]} but the file has only"
                f" {self._mmap.size} bytes ({truncated.size} spectra affected)."
            )

    # noinspection PyAttributeOutsideInit
    def set_region_information(self, roi: int | None = None) -> None:
        """Select the pixels that belong to the requested region of interest."""
        self.region_number_all = self.raster_region[self.valid_raster_indices]
        self._rois: list[int] = [int(value) for value in np.unique(self.region_number_all)]
        self.region_frames = np.arange(self.valid_raster_indices.size)
        if roi is not None:
            if roi not in self._rois:
                raise ValueError(f"ROI {roi} was not found. Available ROIs: {self._rois}")
            self.region_frames = np.where(self.region_number_all == roi)[0]
        if self.region_frames.size == 0:
            raise ValueError(f"No spectra found for region of interest {roi}.")

        self.region_number = self.region_number_all[self.region_frames]
        # raster index of every pixel
        self.pixel_raster_indices = self.valid_raster_indices[self.region_frames]
        x = self.raster_x[self.pixel_raster_indices]
        y = self.raster_y[self.pixel_raster_indices]
        x_min, _ = get_min_max(x)
        y_min, _ = get_min_max(y)
        self._xyz_coordinates = np.column_stack((x - x_min, y - y_min, np.zeros_like(x)))

    @property
    def n_datapoints(self) -> int:
        """Return the number of m/z points per spectrum."""
        return int(self.header.get("n_datapoints") or self.info_metadata.get("DataPoints", 0))

    @property
    def mz_min(self) -> float:
        """Return minimum m/z value."""
        return self._mz_min

    @property
    def mz_max(self) -> float:
        """Return maximum m/z value."""
        return self._mz_max

    @property
    def mz_x(self) -> np.ndarray:
        """Return the m/z axis shared by all spectra."""
        if self._mz_x is None:
            if self.n_datapoints <= 0:
                raise ValueError("Cannot create m/z axis because the number of data points is zero.")
            if self.mz_min >= self.mz_max:
                raise ValueError(f"Invalid rapifleX m/z range: {self.mz_min} to {self.mz_max}.")
            self._mz_x = np.linspace(self.mz_min, self.mz_max, self.n_datapoints)
        return self._mz_x

    @property
    def is_centroid(self) -> bool:
        """Return False, since rapifleX exports always contain profile spectra."""
        return False

    @property
    def rois(self) -> list[int]:
        """Return list of all ROIs in the dataset."""
        return self._rois

    @property
    def polarity(self) -> str:
        """Return acquisition polarity ('Positive', 'Negative' or 'Unknown')."""
        value = str(self.info_metadata.get("Polarity", self.info_metadata.get("Ion Polarity", ""))).strip().lower()
        if value.startswith("+") or "positive" in value:
            return "Positive"
        if value.startswith("-") or "negative" in value:
            return "Negative"
        return "Unknown"

    def _get_raster_step(self, axis: int) -> float | None:
        """Return the raster step (in micrometers) for the x (0) or y (1) axis."""
        key = "raster_x" if axis == 0 else "raster_y"
        if key in self.info_metadata:
            return float(self.info_metadata[key])
        raster = self.mis_metadata.get("raster")
        if isinstance(raster, list) and len(raster) > axis:
            return float(raster[axis])
        return None

    @property
    def x_pixel_size(self) -> float:
        """Return x pixel size in micrometers."""
        value = self._get_raster_step(0)
        if value is None:
            logger.warning("Could not determine rapifleX pixel size - defaulting to 1.0.")
            return 1.0
        return value

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size in micrometers."""
        value = self._get_raster_step(1)
        return self.x_pixel_size if value is None else value

    def get_acquisition_mass_range(self) -> tuple[float, float]:
        """Return the acquired m/z range from the ``_info.txt`` file."""
        mz_min = self.info_metadata.get("Mass Start", self.info_metadata.get("MassStart"))
        mz_max = self.info_metadata.get("Mass End", self.info_metadata.get("MassEnd"))
        if mz_min is None or mz_max is None:
            raise ValueError("Failed to extract rapifleX mass range from *_info.txt.")
        return float(mz_min), float(mz_max)

    def get_n_pixels_for_roi(self, roi: int) -> int:
        """Return the number of pixels in a ROI."""
        return int(np.count_nonzero(self.region_number_all == roi))

    def get_pixels_for_roi(self, roi: int) -> np.ndarray:
        """Return the pixel indices of a ROI."""
        if self.roi is not None:
            return self.pixels if roi == self.roi else np.asarray([], dtype=int)
        return np.where(self.region_number_all == roi)[0]

    def _get_reader_kwargs(self) -> dict[str, ty.Any]:
        """Return kwargs needed to re-open this reader with the same ROI selection."""
        if self.roi is None:
            return {}
        return {"roi": self.roi}

    def read_raster_spectrum(self, raster_idx: int) -> np.ndarray:
        """Read the profile spectrum (intensities) for a zero-based raster cell index.

        Cells without a spectrum return an array of zeros.
        """
        if self._mmap is None:
            raise RuntimeError("Reader has been closed.")
        if raster_idx < 0 or raster_idx >= self.offsets.size:
            raise IndexError(f"Raster index out of range: {raster_idx}")
        offset = int(self.offsets[raster_idx])
        if offset < self.data_start:
            return np.zeros(self.n_datapoints, dtype=np.float32)
        return np.frombuffer(self._mmap, dtype="<f4", count=self.n_datapoints, offset=offset).astype(np.float32)

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index < 0 or index >= self.n_pixels:
            raise IndexError(f"Pixel index {index} is out of bounds for {self.n_pixels} pixels.")
        return self.mz_x, self.read_raster_spectrum(int(self.pixel_raster_indices[index]))

    def _read_spectra(self, indices: np.ndarray | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        if indices is None:
            indices = self.pixels
        for index in indices:
            yield self._read_spectrum(index)

    def get_summed_spectrum(
        self, indices: ty.Iterable[int], scales: np.ndarray | None = None, silent: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum."""
        indices = np.asarray(indices)
        if np.any(indices >= self.n_pixels):
            raise ValueError("You cannot specify indices that are greater than the total number of pixels.")
        if scales is None:
            scales = np.ones(self.n_pixels, dtype=np.float32)
        mz_y = np.zeros_like(self.mz_x, dtype=np.float64)
        for index in tqdm(indices, disable=silent, total=len(indices), desc="Summing spectra..."):
            mz_y += self._read_spectrum(index)[1] * scales[index]
        return self.mz_x, mz_y


def coerce_metadata_value(value: str) -> int | float | str:
    """Coerce a metadata value to a numeric type when possible."""
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_info_file(path: Path) -> dict[str, ty.Any]:
    """Parse the rapifleX ``_info.txt`` file into a dictionary.

    The ``Raster`` entry (e.g. ``20,20``) is additionally split into ``raster_x`` and ``raster_y``.
    """
    metadata: dict[str, ty.Any] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            separator = ":" if ":" in line else "=" if "=" in line else None
            if separator is None:
                continue
            key, value = line.split(separator, 1)
            metadata[key.strip()] = coerce_metadata_value(value.strip())

    raster = metadata.get("Raster")
    if isinstance(raster, str) and "," in raster:
        parts = raster.split(",", 1)
        metadata["raster_x"] = float(parts[0])
        metadata["raster_y"] = float(parts[1])
    elif isinstance(raster, (int, float)):
        metadata["raster_x"] = metadata["raster_y"] = float(raster)
    return metadata


def parse_mis_file(path: Path) -> dict[str, ty.Any]:
    """Parse the optional FlexImaging ``.mis`` XML file (method, raster, teaching points and areas)."""
    metadata: dict[str, ty.Any] = {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse MIS file: {exc}")
        return metadata

    for elem_name in ("Method", "ImageFile", "OriginalImage", "BaseGeometry"):
        elem = root.find(f".//{elem_name}")
        if elem is not None and elem.text:
            metadata[elem_name] = elem.text

    raster_elem = root.find(".//Raster")
    if raster_elem is not None and raster_elem.text:
        parts = raster_elem.text.split(",")
        if len(parts) == 2:
            metadata["raster"] = [float(parts[0]), float(parts[1])]

    teaching_points = _parse_mis_teaching_points(root)
    if teaching_points:
        metadata["teaching_points"] = teaching_points
    areas = _parse_mis_areas(root)
    if areas:
        metadata["areas"] = areas
    return metadata


def _parse_mis_teaching_points(root: ET.Element) -> list[dict[str, list[int]]]:
    """Parse ``TeachPoint`` elements (``img_x,img_y;stage_x,stage_y``)."""
    teaching_points = []
    for teaching_point in root.findall(".//TeachPoint"):
        if not teaching_point.text or ";" not in teaching_point.text:
            continue
        image_coords, stage_coords = teaching_point.text.split(";", 1)
        img_x, img_y = map(int, image_coords.split(","))
        stage_x, stage_y = map(int, stage_coords.split(","))
        teaching_points.append({"image": [img_x, img_y], "stage": [stage_x, stage_y]})
    return teaching_points


def _parse_mis_areas(root: ET.Element) -> list[dict[str, ty.Any]]:
    """Parse ``Area`` elements into their name and bounding box (``p1`` = min corner, ``p2`` = max corner)."""
    areas = []
    for area in root.findall(".//Area"):
        points = [[int(value) for value in point.text.split(",")[:2]] for point in area.findall("Point") if point.text]
        if points:
            xy = np.asarray(points, dtype=np.int32)
            areas.append(
                {
                    "name": area.get("Name", ""),
                    "p1": [int(xy[:, 0].min()), int(xy[:, 1].min())],
                    "p2": [int(xy[:, 0].max()), int(xy[:, 1].max())],
                }
            )
    return areas


def parse_poslog_file(path: Path) -> list[dict[str, ty.Any]]:
    """Parse region and raster/physical coordinates from the ``_poslog.txt`` file."""
    positions = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5 or parts[0].startswith("#") or parts[2] == "__":
                continue
            match = POSITION_PATTERN.match(parts[2])
            if match is None:
                continue
            positions.append(
                {
                    "region": int(match.group(1)),
                    "raster_x": int(match.group(2)),
                    "raster_y": int(match.group(3)),
                    "phys_x": float(parts[3]),
                    "phys_y": float(parts[4]),
                    "phys_z": float(parts[5]) if len(parts) > 5 else 0.0,
                }
            )
    return positions


def is_rapiflex(path: PathLike) -> bool:
    """Check whether the path is a Bruker rapifleX export directory."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(path.glob("*.dat")) and any(path.glob("*_info.txt")) and any(path.glob("*_poslog.txt"))


@hook_impl
def imzy_reader(path: PathLike, **kwargs: ty.Any) -> RapiflexReader | None:
    """Return RapiflexReader if path is a Bruker rapifleX export."""
    if is_rapiflex(path):
        return RapiflexReader(path, **kwargs)
    return None
