"""imzML reader."""
import typing as ty
from pathlib import Path
from warnings import warn

import numpy as np
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy._readers.imzml._ontology import get_cv_param
from imzy.hookspec import hook_impl

PRECISION_DICT = {"32-bit float": "f", "64-bit float": "d", "32-bit integer": "i", "64-bit integer": "l"}
SIZE_DICT = {"f": 4, "d": 8, "i": 4, "l": 8}


class IMZMLCache:
    """Cache for imzML keys/values."""

    def __init__(self, metadata_dict: ty.Dict):
        self.metadata_dict = metadata_dict
        self.PX_MAX_X: int = metadata_dict["max count of pixels x"]
        self.PX_MAX_Y: int = metadata_dict["max count of pixels y"]
        self.PX_MAX_Z: int = metadata_dict.get("max count of pixels z", 1)
        self.PX_SIZE_X: float = metadata_dict.get("pixel size x", 1)
        self.PX_SIZE_Y: float = metadata_dict.get("pixel size y", 1)

    def to_cache(self) -> ty.Dict:
        """Serialize metadata to cache."""
        return {
            "px_max_x": self.PX_MAX_X,
            "px_max_y": self.PX_MAX_Y,
            "px_max_z": self.PX_MAX_Z,
            "px_size_x": self.PX_SIZE_X,
            "px_size_y": self.PX_SIZE_Y,
        }

    @classmethod
    def from_cache(cls, path: Path) -> "IMZMLCache":
        """Read data from cache."""
        data = {}
        with np.load(path) as f_ptr:
            data["max count of pixels x"] = f_ptr["px_max_x"]
            data["max count of pixels y"] = f_ptr["px_max_y"]
            data["max count of pixels z"] = f_ptr["px_max_z"]
            data["pixel size x"] = f_ptr["px_size_x"]
            data["pixel size y"] = f_ptr["px_size_y"]
        return cls(data)


class IMZMLReader(BaseReader):
    """ImzML file reader."""

    _ibd_path: ty.Optional[Path] = None
    _icache_path: ty.Optional[Path] = None
    _is_centroid: ty.Optional[bool] = None

    def __init__(
        self,
        path: PathLike,
        ibd_path: ty.Optional[PathLike] = None,
        mz_ppm: float = 1.0,
        mz_min: ty.Optional[float] = None,
        mz_max: ty.Optional[float] = None,
    ):
        super().__init__(path)
        self._init(ibd_path)
        self.mz_ppm = mz_ppm
        self._mz_min = mz_min
        self._mz_max = mz_max

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.path}; centroid={self.is_centroid}>"

    @property
    def ibd_path(self) -> Path:
        """Return path to ibd file."""
        if not self._ibd_path:
            raise ValueError("ibd path is not set.")
        return self._ibd_path

    def _init(self, ibd_path: ty.Optional[PathLike] = None) -> None:
        """Initialize metadata."""
        _, self._ibd_path, self._icache_path = infer_path(self.path, ibd_path)
        if self._icache_path and self._icache_path.exists():
            self.mz_precision, self.int_precision, self.byte_offsets, self._xyz_coordinates = read_icache(
                self._icache_path
            )
            self._imzml_cache = IMZMLCache.from_cache(self._icache_path)
        else:
            root, self.mz_precision, self.int_precision, self.byte_offsets, self._xyz_coordinates = init_metadata(
                self.path
            )
            self._icache_path = self.path.with_suffix(".icache")
            metadata = read_imzml_metadata(root)
            self._imzml_cache = IMZMLCache(metadata)
        self._mz_size, self._int_size = SIZE_DICT[self.mz_precision], SIZE_DICT[self.int_precision]

        # if cache file does not exist, write it immediately
        if not self._icache_path.exists():
            try:
                write_icache(self, self._icache_path)
            except OSError as error:  # in case there is no space or can't write?
                print(error)

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
    def rois(self) -> ty.List[int]:
        """Return list of ROI indices."""
        return [0]  # imzML files always have single ROI

    @property
    def is_centroid(self) -> bool:
        """Flag to indicate whether data is in centroid or profile mode."""
        if self._is_centroid is None:
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
        return self.metadata.PX_SIZE_X

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size in micrometers."""
        return self.metadata.PX_SIZE_Y

    def get_physical_coordinates(self, index: int) -> ty.Tuple[float, float]:
        """For a pixel index i, return real-world coordinates in micrometers.

        This is equivalent to multiplying the image coordinates of the given pixel with the pixel size.
        """
        x, y, _ = self.xyz_coordinates[index]
        return x * self.metadata.PX_SIZE_X, y * self.metadata.PX_SIZE_Y

    def reshape(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Reshape vector of intensities."""
        if len(array) != self.n_pixels:
            raise ValueError("Wrong size and shape of the array.")
        dtype = np.float32 if np.isnan(fill_value) else array.dtype
        im = np.full((self.metadata.PX_MAX_Y, self.metadata.PX_MAX_X), fill_value=fill_value, dtype=dtype)
        im[self.y_coordinates - 1, self.x_coordinates - 1] = array
        return im

    def reshape_batch(self, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
        """Batch reshaping of images."""
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

    def get_summed_spectrum(self, indices: ty.Iterable[int], silent: bool = False) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Sum pixel data to produce summed mass spectrum."""
        indices = np.asarray(indices)
        if np.any(indices >= self.n_pixels):
            raise ValueError("You cannot specify indices that are greater than the total number of pixels.")
        if self.is_centroid:
            return self._get_summed_spectrum_centroid(indices)
        return self._get_summed_spectrum_profile(indices)

    def _get_summed_spectrum_profile(
        self, indices: ty.Iterable[int], silent: bool = False
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(indices)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")
        mz_x, mz_y = self[indices[0]]
        mz_y = mz_y.copy().astype(np.float64)
        for _, y in tqdm(
            self._read_spectra(indices[1::]), total=len(indices) - 1, disable=silent, desc="Summing spectra..."
        ):
            mz_y += y
        return mz_x, mz_y

    def _get_summed_spectrum_centroid(
        self, indices: ty.Iterable[int], silent: bool = False
    ) -> ty.Tuple[np.ndarray, np.ndarray]:
        # creating summed spectrum from centroided data is a lot harder because there is no consensus axis in which case
        # we must create our own.
        # We have decided to create resampled spectrum with pre-defined ppm limit. This is not ideal but its better than
        # not doing it at all.
        from koyo.spectrum import get_ppm_axis, set_ppm_axis, trim_axis

        indices = np.asarray(indices)
        if indices.size == 0:
            raise ValueError("You must specify at least one index.")

        mz_min, mz_max = self._estimate_mass_range()
        mz_x = get_ppm_axis(mz_min, mz_max, self.mz_ppm)
        mz_y = np.zeros_like(mz_x, dtype=np.float64)
        for x, y in tqdm(self._read_spectra(indices), total=len(indices), disable=silent, desc="Summing spectra..."):
            x, y = trim_axis(x, y, self._mz_min, self._mz_max)
            mz_y = set_ppm_axis(mz_x, mz_y, x, y)
        return mz_x, mz_y

    def _estimate_mass_range(self) -> ty.Tuple[float, float]:
        """Function will iterate over portion of the spectra and try to determine what is min/max m/z value."""
        if self._mz_min is None or self._mz_max is None:
            if self.n_pixels < 5000:
                indices = self.pixels
            else:
                indices = np.unique(np.random.choice(self.pixels, 5000, replace=False))
            mz_min, mz_max = 1e6, 0
            with open(self.ibd_path, "rb") as f_ptr:
                for index in indices:
                    mz_o, mz_l, _, _ = self.byte_offsets[index]
                    f_ptr.seek(mz_o)
                    x = np.frombuffer(f_ptr.read(mz_l * self._mz_size), dtype=self.mz_precision)
                    x_min, x_max = x.min(), x.max()
                    if x_min < mz_min:
                        mz_min = x_min
                    if x_max > mz_max:
                        mz_max = x_max
            if self._mz_min is None:
                self._mz_min = mz_min
            if self._mz_max is None:
                self._mz_max = mz_max
        return self._mz_min, self._mz_max

    def _read_spectrum(self, index: int) -> ty.Tuple[np.ndarray, np.ndarray]:
        with open(self.ibd_path, "rb") as f_ptr:
            mz_o, mz_l, int_o, int_l = self.byte_offsets[index]
            f_ptr.seek(mz_o)
            mz_bytes = f_ptr.read(mz_l * self._mz_size)
            f_ptr.seek(int_o)
            int_bytes = f_ptr.read(int_l * self._int_size)
        return np.frombuffer(mz_bytes, dtype=self.mz_precision), np.frombuffer(int_bytes, dtype=self.int_precision)

    def _read_spectra(
        self, indices: ty.Optional[ty.Iterable[int]] = None
    ) -> ty.Iterator[ty.Tuple[np.ndarray, np.ndarray]]:
        """Read spectra without constantly opening and closing the file handle."""
        if indices is None:
            indices = self.pixels
        with open(self.ibd_path, "rb") as f_ptr:
            for index in indices:
                mz_o, mz_l, int_o, int_l = self.byte_offsets[index]
                f_ptr.seek(mz_o)
                mz_bytes = f_ptr.read(mz_l * self._mz_size)
                f_ptr.seek(int_o)
                int_bytes = f_ptr.read(int_l * self._int_size)
                yield np.frombuffer(mz_bytes, dtype=self.mz_precision), np.frombuffer(
                    int_bytes, dtype=self.int_precision
                )


def infer_path(path: Path, ibd_path: ty.Optional[PathLike] = None) -> ty.Tuple[Path, Path, ty.Optional[Path]]:
    """Infer imzml/ibd path."""
    import re

    if ibd_path is None:
        ibd_path = path.name

        for _path in path.parent.glob("*.ibd"):
            if re.match(r".+\.ibd", str(_path), re.IGNORECASE) and _path.stem == path.stem:
                ibd_path = _path
                break
    icache_path = None
    for _path in path.parent.glob("*.icache"):
        if re.match(r".+\.icache", str(_path), re.IGNORECASE) and _path.stem == path.stem:
            icache_path = _path
            break
    return path, ibd_path, icache_path


# noinspection HttpUrlsUsage
def read_imzml_metadata(root, sl: str = "{http://psi.hupo.org/ms/mzml}"):
    """Initializes the imzml dict with frequently used metadata from the .imzML file.

    This method reads only a subset of the available meta information and may be extended in the future. The keys
    are named similarly to the imzML names. Currently supported keys: "max dimension x", "max dimension y",
    "pixel size x", "pixel size y", "matrix solution concentration", "wavelength", "focus diameter x",
    "focus diameter y", "pulse energy", "pulse duration", "attenuation".

    If a key is not found in the XML tree, it will not be in the dict either.

    :return d:
        dict containing above mentioned meta data
    :rtype:
        dict
    :raises Warning:
        if an xml attribute has a number format different from the imzML specification
    """

    def _check_meta(param, accession, elem_list):
        for idx, _ in enumerate(param):
            acc, attr = accession[idx]
            elem = elem_list.find(f'.//{sl}cvParam[@accession="{acc}"]')
            if elem is None:
                break
            name, value = param[idx]
            try:
                metadata_dict[name] = value(elem.attrib[attr])
            except ValueError:
                warn(Warning(f"Wrong data type in XML file. Skipped attribute '{name}'"), stacklevel=3)

    metadata_dict = {}
    scan_settings_list_elem = root.find("%sscanSettingsList" % sl)
    instrument_config_list_elem = root.find("%sinstrumentConfigurationList" % sl)
    supported_params_1 = [
        ("max count of pixels x", int),
        ("max count of pixels y", int),
        ("max dimension x", int),
        ("max dimension y", int),
        ("pixel size x", float),
        ("pixel size y", float),
        ("matrix solution concentration", float),
    ]
    supported_params_2 = [
        ("wavelength", float),
        ("focus diameter x", float),
        ("focus diameter y", float),
        ("pulse energy", float),
        ("pulse duration", float),
        ("attenuation", float),
    ]
    supported_accession_1 = [
        ("IMS:1000042", "value"),
        ("IMS:1000043", "value"),
        ("IMS:1000044", "value"),
        ("IMS:1000045", "value"),
        ("IMS:1000046", "value"),
        ("IMS:1000047", "value"),
        ("MS:1000835", "value"),
    ]
    supported_accession_2 = [
        ("MS:1000843", "value"),
        ("MS:1000844", "value"),
        ("MS:1000845", "value"),
        ("MS:1000846", "value"),
        ("MS:1000847", "value"),
        ("MS:1000848", "value"),
    ]
    _check_meta(supported_params_1, supported_accession_1, scan_settings_list_elem)
    _check_meta(supported_params_2, supported_accession_2, instrument_config_list_elem)
    return metadata_dict


def init_metadata(path: Path, parse_lib: ty.Optional[str] = None, sl: str = "{http://psi.hupo.org/ms/mzml}"):
    """Method to initialize formats, coordinates and offsets from the imzML file format.

    This method should only be called by __init__. Reads the data formats, coordinates and offsets from
    the .imzML file and initializes the respective attributes. While traversing the XML tree, the per-spectrum
    metadata is pruned, i.e. the <spectrumList> element(s) are left behind empty.

    Supported accession values for the number formats: "MS:1000521", "MS:1000523", "IMS:1000141" or
    "IMS:1000142". The string values are "32-bit float", "64-bit float", "32-bit integer", "64-bit integer".
    """
    mz_group = int_group = None

    # get iterator
    iterparse = choose_iterparse(parse_lib)
    elem_iterator = iterparse(str(path), events=("start", "end"))

    temp, mz_group_id, int_group_id = None, None, None
    _, root = next(elem_iterator)

    offsets = []
    for event, elem in elem_iterator:
        if elem.tag == sl + "spectrumList" and event == "start":
            temp = elem
        elif elem.tag == sl + "spectrum" and event == "end":
            offsets.append(process_spectrum(elem, mz_group_id, int_group_id))
            temp.remove(elem)
        elif elem.tag == sl + "referenceableParamGroup" and event == "end":
            for param in elem:
                if param.attrib["name"] == "m/z array":
                    mz_group_id = elem.attrib["id"]
                    mz_group = elem
                elif param.attrib["name"] == "intensity array":
                    int_group_id = elem.attrib["id"]
                    int_group = elem

    # cleanup
    mz_precision, int_precision = assign_precision(int_group, mz_group)
    fix_offsets(offsets)
    offsets = np.array(offsets, dtype=np.int64)
    byte_offsets = offsets[:, 0:4]
    coordinates = offsets[:, 4::]
    return root, mz_precision, int_precision, byte_offsets, coordinates


def fix_offsets(offsets):
    """Fix errors introduced by incorrect signed 32bit integers when unsigned 64bit was appropriate."""

    def _fix(offsets, index: int):
        delta = 0
        prev_value = float("nan")
        for values in offsets:
            value = values[index]
            if value < 0 <= prev_value:
                delta += 2**32
            values[index] = value + delta
            prev_value = value

    # correct offsets
    _fix(offsets, OffsetIndices.MZ_OFFSET)
    _fix(offsets, OffsetIndices.INT_OFFSET)


def assign_precision(int_group, mz_group, sl: str = "{http://psi.hupo.org/ms/mzml}"):
    """Determine precision."""
    valid_accession_strings = (
        "MS:1000521",
        "MS:1000523",
        "IMS:1000141",
        "IMS:1000142",
        "MS:1000519",
        "MS:1000522",
    )
    mz_precision = int_precision = None
    for s in valid_accession_strings:
        param = mz_group.find(f'{sl}cvParam[@accession="{s}"]')
        if param is not None:
            mz_precision = PRECISION_DICT[param.attrib["name"]]
            break
    for s in valid_accession_strings:
        param = int_group.find(f'{sl}cvParam[@accession="{s}"]')
        if param is not None:
            int_precision = PRECISION_DICT[param.attrib["name"]]
            break
    if (mz_precision is None) or (int_precision is None):
        raise RuntimeError(f"Unsupported number format: mz = {mz_precision}, int = {int_precision}")
    return mz_precision, int_precision


def process_spectrum(elem, mz_group_id, int_group_id, sl: str = "{http://psi.hupo.org/ms/mzml}"):
    """Process spectrum."""
    array_list_item = elem.find("%sbinaryDataArrayList" % sl)
    element_list = list(array_list_item)
    mz_group, int_group = None, None
    for element in element_list:
        ref = element.find("%sreferenceableParamGroupRef" % sl).attrib["ref"]
        if ref == mz_group_id:
            mz_group = element
        elif ref == int_group_id:
            int_group = element

    mz_offset = int(get_cv_param(mz_group, "IMS:1000102"))
    mz_length = int(get_cv_param(mz_group, "IMS:1000103"))
    intensity_offset = int(get_cv_param(int_group, "IMS:1000102"))
    intensity_length = int(get_cv_param(int_group, "IMS:1000103"))

    scan_elem = elem.find(f"{sl}scanList/{sl}scan")
    x = int(get_cv_param(scan_elem, "IMS:1000050"))
    y = int(get_cv_param(scan_elem, "IMS:1000051"))
    z = get_cv_param(scan_elem, "IMS:1000052")
    z = int(z) if z is not None else 1
    return [mz_offset, mz_length, intensity_offset, intensity_length, x, y, z]


class OffsetIndices:
    """Indices."""

    MZ_OFFSET = 0
    MZ_LENGTH = 1
    INT_OFFSET = 2
    INT_LENGTH = 3


class CoordinateIndices:
    """Coordinate indices."""

    X = 0
    Y = 1
    Z = 2


def choose_iterparse(parse_lib: ty.Optional[str] = None) -> ty.Callable:
    """Choose iterparse."""
    if parse_lib == "ElementTree":
        from xml.etree.ElementTree import iterparse
    elif parse_lib == "lxml":
        try:
            from lxml.etree import iterparse
        except ImportError:
            from xml.etree.ElementTree import iterparse
    else:
        try:
            from lxml.etree import iterparse
        except ImportError:
            from xml.etree.ElementTree import iterparse
    return iterparse


def read_icache(path: Path) -> ty.Tuple[str, str, np.ndarray, np.ndarray]:
    """Read icache file into memory."""
    with np.load(path) as f_ptr:
        mz_precision = str(f_ptr["mz_precision"])
        int_precision = str(f_ptr["int_precision"])
        byte_offsets = f_ptr["byte_offsets"]
        xyz_coordinates = f_ptr["xyz_coordinates"]
    return mz_precision, int_precision, byte_offsets, xyz_coordinates


def write_icache(obj: IMZMLReader, path: Path) -> None:
    """Write icache file to disk so next time the imzML file is being opened, it will be much, much faster."""
    np.savez(
        path,
        **obj._imzml_cache.to_cache(),
        mz_precision=obj.mz_precision,
        int_precision=obj.int_precision,
        byte_offsets=obj.byte_offsets,
        xyz_coordinates=obj.xyz_coordinates,
    )
    npz_path = path.with_suffix(".icache.npz")  # need to include both extensions
    # unfortunately, numpy automatically adds the .npz extension which might not be desirable, so we might as well
    # rename it to the .icache
    npz_path.rename(path)


def is_imzml(path: PathLike) -> bool:
    """Check if path is imzml."""
    path = Path(path)
    return path.suffix.lower() == ".imzml"


@hook_impl
def imzy_reader(path: PathLike, **kwargs) -> ty.Optional[IMZMLReader]:
    """Return TDFReader if path is Bruker .d/tdf."""
    if is_imzml(path):
        return IMZMLReader(path, **kwargs)
    return None
