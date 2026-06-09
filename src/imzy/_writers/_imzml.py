"""imzML writer."""

from __future__ import annotations

import hashlib
import typing as ty
import uuid
import warnings
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from koyo.typing import PathLike
from tqdm import tqdm

from imzy._readers._base import BaseReader
from imzy._readers.imzml._imzml import IMZMLReader

MZML_NAMESPACE = "http://psi.hupo.org/ms/mzml"
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

IbdMode = ty.Literal["auto", "continuous", "processed"]
SpectrumType = ty.Literal["auto", "centroid", "profile"]
ResolvedSpectrumType = ty.Literal["centroid", "profile"]
CoordinateOrigin = ty.Literal["auto", "zero", "one"]
OnError = ty.Literal["error", "warn"]
UserParamValue = str | int | float
UserParam = ty.Mapping[str, UserParamValue]

IBD_MODE_CONTINUOUS: ty.Literal["continuous"] = "continuous"
IBD_MODE_PROCESSED: ty.Literal["processed"] = "processed"
IBD_MODE_AUTO: ty.Literal["auto"] = "auto"
SPECTRUM_TYPE_CENTROID: ty.Literal["centroid"] = "centroid"
SPECTRUM_TYPE_PROFILE: ty.Literal["profile"] = "profile"
COORDINATE_ORIGIN_AUTO: ty.Literal["auto"] = "auto"
COORDINATE_ORIGIN_ZERO: ty.Literal["zero"] = "zero"
COORDINATE_ORIGIN_ONE: ty.Literal["one"] = "one"
ON_ERROR_ERROR: ty.Literal["error"] = "error"
ON_ERROR_WARN: ty.Literal["warn"] = "warn"

_VALID_IBD_MODES = {IBD_MODE_AUTO, IBD_MODE_CONTINUOUS, IBD_MODE_PROCESSED}
_VALID_SPECTRUM_TYPES = {SPECTRUM_TYPE_CENTROID, SPECTRUM_TYPE_PROFILE}
_VALID_COORDINATE_ORIGINS = {COORDINATE_ORIGIN_AUTO, COORDINATE_ORIGIN_ZERO, COORDINATE_ORIGIN_ONE}
_VALID_ON_ERROR = {ON_ERROR_ERROR, ON_ERROR_WARN}
_DTYPE_TO_CV: dict[np.dtype[ty.Any], tuple[str, str]] = {
    np.dtype(np.float32): ("32-bit float", "MS:1000521"),
    np.dtype(np.float64): ("64-bit float", "MS:1000523"),
    np.dtype(np.int32): ("32-bit integer", "MS:1000519"),
    np.dtype(np.int64): ("64-bit integer", "MS:1000522"),
}
_PRECISION_TO_DTYPE: dict[str, np.dtype[ty.Any]] = {
    "f": np.dtype(np.float32),
    "d": np.dtype(np.float64),
    "i": np.dtype(np.int32),
    "l": np.dtype(np.int64),
}


@dataclass(frozen=True)
class _ExternalArray:
    """External binary array location."""

    offset: int
    length: int
    encoded_length: int


@dataclass(frozen=True)
class _Spectrum:
    """Metadata for one exported spectrum."""

    coords: tuple[int, int, int]
    mz: _ExternalArray
    intensity: _ExternalArray
    mz_min: float
    mz_max: float
    mz_base: float
    intensity_base: float
    intensity_tic: float
    user_params: tuple[UserParam, ...]


@dataclass(frozen=True)
class _WriterState:
    """Mutable writer state before a spectrum write."""

    ibd_position: int
    sha1: ty.Any
    first_mz: _ExternalArray | None
    mz_cache: dict[tuple[tuple[int, ...], bytes], _ExternalArray]
    n_spectra: int


class IMZMLWriter:
    """Write imzML and ibd files."""

    def __init__(
        self,
        output_path: PathLike,
        *,
        mz_dtype: ty.Any = np.float64,
        intensity_dtype: ty.Any = np.float32,
        ibd_mode: IbdMode = IBD_MODE_AUTO,
        spectrum_type: ResolvedSpectrumType = SPECTRUM_TYPE_CENTROID,
        polarity: str | None = None,
        coordinate_origin: CoordinateOrigin = COORDINATE_ORIGIN_ONE,
        on_error: OnError = ON_ERROR_ERROR,
    ) -> None:
        self.mz_dtype = _validate_dtype(mz_dtype, name="mz_dtype")
        self.intensity_dtype = _validate_dtype(intensity_dtype, name="intensity_dtype")
        self.ibd_mode = _validate_ibd_mode(ibd_mode)
        self.spectrum_type = _validate_resolved_spectrum_type(spectrum_type)
        self.polarity = _validate_polarity(polarity)
        self.coordinate_origin = _validate_coordinate_origin(coordinate_origin)
        self.on_error = _validate_on_error(on_error)
        if self.coordinate_origin == COORDINATE_ORIGIN_AUTO:
            self.coordinate_origin = COORDINATE_ORIGIN_ZERO

        self.base_path, self.imzml_path, self.ibd_path = _resolve_output_paths(output_path)
        self.run_id = self.base_path.name
        self.uuid = uuid.uuid4()
        self.sha1 = hashlib.sha1()
        self._closed = False
        self._spectra: list[_Spectrum] = []
        self._first_mz: _ExternalArray | None = None
        self._mz_cache: dict[tuple[tuple[int, ...], bytes], _ExternalArray] = {}

        self._ibd = self.ibd_path.open("wb+")
        self._write_ibd(self.uuid.bytes)

    @property
    def spectra(self) -> tuple[_Spectrum, ...]:
        """Return metadata for spectra added to the writer."""
        return tuple(self._spectra)

    @classmethod
    def from_reader(
        cls,
        reader: BaseReader,
        output_path: PathLike,
        *,
        ibd_mode: IbdMode = IBD_MODE_AUTO,
        spectrum_type: SpectrumType = COORDINATE_ORIGIN_AUTO,
        coordinate_origin: CoordinateOrigin = COORDINATE_ORIGIN_AUTO,
        mz_dtype: ty.Any | None = None,
        intensity_dtype: ty.Any | None = None,
        on_error: OnError = ON_ERROR_ERROR,
        silent: bool = False,
    ) -> Path:
        """Write an imzy reader to imzML."""
        resolved_spectrum_type = _resolve_reader_spectrum_type(reader, spectrum_type)
        resolved_coordinate_origin = _resolve_reader_coordinate_origin(reader, coordinate_origin)
        if mz_dtype is None:
            mz_dtype = _reader_dtype(reader, "mz_precision", np.float64)
        if intensity_dtype is None:
            intensity_dtype = _reader_dtype(reader, "int_precision", np.float32)

        with cls(
            output_path,
            mz_dtype=mz_dtype,
            intensity_dtype=intensity_dtype,
            ibd_mode=ibd_mode,
            spectrum_type=resolved_spectrum_type,
            coordinate_origin=resolved_coordinate_origin,
            on_error=on_error,
        ) as writer:
            context = reader._disable_auto_profile() if hasattr(reader, "_disable_auto_profile") else nullcontext()
            with context:
                coordinates = np.asarray(reader.xyz_coordinates)
                iterator = tqdm(
                    enumerate(coordinates),
                    total=reader.n_pixels,
                    disable=silent,
                    miniters=500,
                    desc="Writing imzML...",
                )
                for index, coords in iterator:
                    try:
                        mzs, intensities = reader.get_spectrum(index)
                    except Exception as error:
                        if writer.on_error == ON_ERROR_WARN:
                            writer._warn_skipped_spectrum(error, context=f"pixel {index} at {tuple(coords)}")
                            continue
                        raise
                    writer.add_spectrum(mzs, intensities, coords, error_context=f"pixel {index} at {tuple(coords)}")
        return writer.imzml_path

    def add_spectrum(
        self,
        mzs: ty.Iterable[float],
        intensities: ty.Iterable[float],
        coords: ty.Sequence[int | float],
        user_params: ty.Iterable[UserParam] | None = None,
        *,
        error_context: str | None = None,
    ) -> bool:
        """Add one spectrum to the output files."""
        if self._closed:
            raise ValueError("Cannot add spectra after the writer has been closed.")

        state = self._capture_state()
        try:
            self._add_spectrum_checked(mzs, intensities, coords, user_params=user_params)
        except Exception as error:
            self._restore_state(state)
            if self.on_error == ON_ERROR_WARN:
                self._warn_skipped_spectrum(error, context=error_context)
                return False
            self._cleanup_outputs()
            raise
        return True

    def _add_spectrum_checked(
        self,
        mzs: ty.Iterable[float],
        intensities: ty.Iterable[float],
        coords: ty.Sequence[int | float],
        *,
        user_params: ty.Iterable[UserParam] | None = None,
    ) -> None:
        """Add one validated spectrum to the output files."""
        mz_array = np.asarray(mzs, dtype=self.mz_dtype)
        intensity_array = np.asarray(intensities, dtype=self.intensity_dtype)
        if mz_array.ndim != 1 or intensity_array.ndim != 1:
            raise ValueError("m/z and intensity arrays must be one-dimensional.")
        if mz_array.shape != intensity_array.shape:
            raise ValueError("m/z and intensity arrays must have the same shape.")
        if mz_array.size == 0:
            raise ValueError("Cannot write an empty spectrum.")

        mz_location = self._get_mz_location(mz_array)
        intensity_location = self._encode_and_write(intensity_array, self.intensity_dtype)
        base_index = int(np.argmax(intensity_array))
        self._spectra.append(
            _Spectrum(
                coords=self._normalize_coordinates(coords),
                mz=mz_location,
                intensity=intensity_location,
                mz_min=float(np.min(mz_array)),
                mz_max=float(np.max(mz_array)),
                mz_base=float(mz_array[base_index]),
                intensity_base=float(intensity_array[base_index]),
                intensity_tic=float(np.sum(intensity_array)),
                user_params=tuple(user_params or ()),
            )
        )

    def close(self) -> None:
        """Write the imzML file and close the writer."""
        if self._closed:
            return
        try:
            if not self._spectra:
                raise ValueError("Cannot write imzML output without any spectra.")
            self._ibd.close()
            self._write_xml()
            self._closed = True
        except Exception:
            self._cleanup_outputs()
            raise

    finish = close

    def __enter__(self) -> IMZMLWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: ty.Any,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._cleanup_outputs()

    def _capture_state(self) -> _WriterState:
        self._ibd.flush()
        return _WriterState(
            ibd_position=self._ibd.tell(),
            sha1=self.sha1.copy(),
            first_mz=self._first_mz,
            mz_cache=self._mz_cache.copy(),
            n_spectra=len(self._spectra),
        )

    def _restore_state(self, state: _WriterState) -> None:
        self._ibd.seek(state.ibd_position)
        self._ibd.truncate(state.ibd_position)
        self.sha1 = state.sha1.copy()
        self._first_mz = state.first_mz
        self._mz_cache = state.mz_cache.copy()
        del self._spectra[state.n_spectra :]

    def _cleanup_outputs(self) -> None:
        with suppress(ValueError, OSError):
            self._ibd.close()
        self._closed = True
        with suppress(FileNotFoundError):
            self.imzml_path.unlink()
        with suppress(FileNotFoundError):
            self.ibd_path.unlink()

    def _warn_skipped_spectrum(self, error: Exception, *, context: str | None = None) -> None:
        message = "Skipping spectrum"
        if context:
            message = f"{message} for {context}"
        warnings.warn(f"{message}: {error}", stacklevel=3)

    def _get_mz_location(self, mz_array: np.ndarray) -> _ExternalArray:
        if self.ibd_mode == IBD_MODE_CONTINUOUS:
            if self._first_mz is None:
                self._first_mz = self._encode_and_write(mz_array, self.mz_dtype)
            elif not self._same_external_mz(self._first_mz, mz_array):
                raise ValueError("Continuous imzML output requires every m/z array to be identical.")
            return self._first_mz
        if self.ibd_mode == IBD_MODE_PROCESSED:
            return self._encode_and_write(mz_array, self.mz_dtype)

        key = (mz_array.shape, mz_array.tobytes())
        mz_location = self._mz_cache.get(key)
        if mz_location is None:
            mz_location = self._encode_and_write(mz_array, self.mz_dtype)
            self._mz_cache[key] = mz_location
        return mz_location

    def _same_external_mz(self, location: _ExternalArray, mz_array: np.ndarray) -> bool:
        self._ibd.flush()
        self._ibd.seek(location.offset)
        existing = np.frombuffer(self._ibd.read(location.encoded_length), dtype=self.mz_dtype)
        self._ibd.seek(0, 2)
        return np.array_equal(existing, mz_array)

    def _normalize_coordinates(self, coords: ty.Sequence[int | float]) -> tuple[int, int, int]:
        if len(coords) not in {2, 3}:
            raise ValueError("Coordinates must contain x/y or x/y/z values.")
        normalized = [int(value) for value in coords]
        if self.coordinate_origin == COORDINATE_ORIGIN_ZERO:
            normalized = [value + 1 for value in normalized]
        if len(normalized) == 2:
            normalized.append(1)
        return normalized[0], normalized[1], normalized[2]

    def _encode_and_write(self, data: np.ndarray, dtype: np.dtype) -> _ExternalArray:
        array = np.asarray(data, dtype=dtype)
        offset = self._ibd.tell()
        data_bytes = array.tobytes()
        encoded_length = self._write_ibd(data_bytes)
        return _ExternalArray(offset=offset, length=int(array.shape[0]), encoded_length=encoded_length)

    def _write_ibd(self, data: bytes) -> int:
        self._ibd.write(data)
        self.sha1.update(data)
        return len(data)

    def _write_xml(self) -> None:
        ET.register_namespace("", MZML_NAMESPACE)
        ET.register_namespace("xsi", XSI_NAMESPACE)
        root = ET.Element(
            _tag("mzML"),
            {
                f"{{{XSI_NAMESPACE}}}schemaLocation": (
                    "http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0_idx.xsd"
                ),
                "version": "1.1",
            },
        )
        self._add_cv_list(root)
        self._add_file_description(root)
        self._add_referenceable_param_groups(root)
        self._add_software_list(root)
        self._add_scan_settings(root)
        self._add_instrument_configuration(root)
        self._add_data_processing(root)
        self._add_run(root)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(self.imzml_path, encoding="ISO-8859-1", xml_declaration=True)

    def _add_cv_list(self, root: ET.Element) -> None:
        cv_list = _sub(root, "cvList", count="3")
        _sub(
            cv_list,
            "cv",
            uri="http://psidev.cvs.sourceforge.net/*checkout*/psidev/psi/psi-ms/mzML/controlledVocabulary/psi-ms.obo",
            fullName="Proteomics Standards Initiative Mass Spectrometry Ontology",
            id="MS",
            version="3.65.0",
        )
        _sub(
            cv_list,
            "cv",
            uri="https://raw.githubusercontent.com/imzML/imzML/master/imagingMS.obo",
            fullName="Mass Spectrometry Imaging Ontology",
            id="IMS",
            version="1.1.0",
        )
        _sub(
            cv_list,
            "cv",
            uri="http://obo.cvs.sourceforge.net/*checkout*/obo/obo/ontology/phenotype/unit.obo",
            fullName="Unit Ontology",
            id="UO",
            version="12:10:2011",
        )

    def _add_file_description(self, root: ET.Element) -> None:
        file_content = _sub(_sub(root, "fileDescription"), "fileContent")
        _cv(file_content, "MS", "MS:1000579", "MS1 spectrum")
        self._add_spectrum_type_param(file_content)
        _cv(file_content, "IMS", _ibd_mode_accession(self.resolved_ibd_mode), self.resolved_ibd_mode)
        _cv(file_content, "IMS", "IMS:1000080", "universally unique identifier", value=f"{{{self.uuid}}}".upper())
        _cv(file_content, "IMS", "IMS:1000091", "ibd SHA-1", value=self.sha1.hexdigest().upper())

    def _add_referenceable_param_groups(self, root: ET.Element) -> None:
        groups = _sub(root, "referenceableParamGroupList", count="4")
        mz_name, mz_accession = _DTYPE_TO_CV[self.mz_dtype]
        int_name, int_accession = _DTYPE_TO_CV[self.intensity_dtype]

        mz_group = _sub(groups, "referenceableParamGroup", id="mzArray")
        _cv(mz_group, "MS", "MS:1000576", "no compression")
        _cv(
            mz_group,
            "MS",
            "MS:1000514",
            "m/z array",
            unitCvRef="MS",
            unitAccession="MS:1000040",
            unitName="m/z",
        )
        _cv(mz_group, "MS", mz_accession, mz_name)
        _cv(mz_group, "IMS", "IMS:1000101", "external data", value="true")

        intensity_group = _sub(groups, "referenceableParamGroup", id="intensityArray")
        _cv(intensity_group, "MS", int_accession, int_name)
        _cv(
            intensity_group,
            "MS",
            "MS:1000515",
            "intensity array",
            unitCvRef="MS",
            unitAccession="MS:1000131",
            unitName="number of detector counts",
        )
        _cv(intensity_group, "MS", "MS:1000576", "no compression")
        _cv(intensity_group, "IMS", "IMS:1000101", "external data", value="true")

        scan_group = _sub(groups, "referenceableParamGroup", id="scan1")
        _cv(scan_group, "MS", "MS:1000093", "increasing m/z scan")
        _cv(scan_group, "MS", "MS:1000512", "filter string")

        spectrum_group = _sub(groups, "referenceableParamGroup", id="spectrum1")
        _cv(spectrum_group, "MS", "MS:1000579", "MS1 spectrum")
        _cv(spectrum_group, "MS", "MS:1000511", "ms level", value="0")
        self._add_spectrum_type_param(spectrum_group)
        if self.polarity == "positive":
            _cv(spectrum_group, "MS", "MS:1000130", "positive scan")
        elif self.polarity == "negative":
            _cv(spectrum_group, "MS", "MS:1000129", "negative scan")

    def _add_spectrum_type_param(self, element: ET.Element) -> None:
        if self.spectrum_type == SPECTRUM_TYPE_CENTROID:
            _cv(element, "MS", "MS:1000127", "centroid spectrum")
        else:
            _cv(element, "MS", "MS:1000128", "profile spectrum")

    def _add_software_list(self, root: ET.Element) -> None:
        software = _sub(_sub(root, "softwareList", count="1"), "software", id="imzy", version="unknown")
        _cv(software, "MS", "MS:1000799", "custom unreleased software tool", value="imzy imzML writer")

    def _add_scan_settings(self, root: ET.Element) -> None:
        scan_settings = _sub(_sub(root, "scanSettingsList", count="1"), "scanSettings", id="scanSettings1")
        _cv(scan_settings, "IMS", "IMS:1000401", "top down")
        _cv(scan_settings, "IMS", "IMS:1000411", "one way")
        _cv(scan_settings, "IMS", "IMS:1000480", "horizontal line scan")
        _cv(scan_settings, "IMS", "IMS:1000491", "linescan left right")
        _cv(scan_settings, "IMS", "IMS:1000042", "max count of pixels x", value=str(self._max_coordinate(0)))
        _cv(scan_settings, "IMS", "IMS:1000043", "max count of pixels y", value=str(self._max_coordinate(1)))
        _cv(scan_settings, "IMS", "IMS:1000044", "max dimension x", value=str(self._max_coordinate(0)))
        _cv(scan_settings, "IMS", "IMS:1000045", "max dimension y", value=str(self._max_coordinate(1)))

    def _add_instrument_configuration(self, root: ET.Element) -> None:
        _sub(_sub(root, "instrumentConfigurationList", count="1"), "instrumentConfiguration", id="IC1")

    def _add_data_processing(self, root: ET.Element) -> None:
        data_processing = _sub(_sub(root, "dataProcessingList", count="1"), "dataProcessing", id="export_from_imzy")
        method = _sub(data_processing, "processingMethod", order="0", softwareRef="imzy")
        _cv(method, "MS", "MS:1000530", "file format conversion", value="Output to imzML")

    def _add_run(self, root: ET.Element) -> None:
        run = _sub(root, "run", defaultInstrumentConfigurationRef="IC1", id=self.run_id)
        spectrum_list = _sub(
            run,
            "spectrumList",
            count=str(len(self._spectra)),
            defaultDataProcessingRef="export_from_imzy",
        )
        for index, spectrum in enumerate(self._spectra, start=1):
            self._add_spectrum(spectrum_list, index, spectrum)

    def _add_spectrum(self, spectrum_list: ET.Element, index: int, spectrum: _Spectrum) -> None:
        element = _sub(spectrum_list, "spectrum", defaultArrayLength="0", id=f"spectrum={index}", index=str(index))
        _sub(element, "referenceableParamGroupRef", ref="spectrum1")
        _cv(
            element,
            "MS",
            "MS:1000528",
            "lowest observed m/z",
            value=_format_float(spectrum.mz_min),
            unitCvRef="MS",
            unitAccession="MS:1000040",
            unitName="m/z",
        )
        _cv(
            element,
            "MS",
            "MS:1000527",
            "highest observed m/z",
            value=_format_float(spectrum.mz_max),
            unitCvRef="MS",
            unitAccession="MS:1000040",
            unitName="m/z",
        )
        _cv(
            element,
            "MS",
            "MS:1000504",
            "base peak m/z",
            value=_format_float(spectrum.mz_base),
            unitCvRef="MS",
            unitAccession="MS:1000040",
            unitName="m/z",
        )
        _cv(
            element,
            "MS",
            "MS:1000505",
            "base peak intensity",
            value=_format_float(spectrum.intensity_base),
            unitCvRef="MS",
            unitAccession="MS:1000131",
            unitName="number of detector counts",
        )
        _cv(element, "MS", "MS:1000285", "total ion current", value=_format_float(spectrum.intensity_tic))
        self._add_scan_list(element, spectrum)
        self._add_binary_arrays(element, spectrum)

    def _add_scan_list(self, spectrum_element: ET.Element, spectrum: _Spectrum) -> None:
        scan_list = _sub(spectrum_element, "scanList", count="1")
        _cv(scan_list, "MS", "MS:1000795", "no combination")
        scan = _sub(scan_list, "scan", instrumentConfigurationRef="IC1")
        _sub(scan, "referenceableParamGroupRef", ref="scan1")
        x, y, z = spectrum.coords
        _cv(scan, "IMS", "IMS:1000050", "position x", value=str(x))
        _cv(scan, "IMS", "IMS:1000051", "position y", value=str(y))
        _cv(scan, "IMS", "IMS:1000052", "position z", value=str(z))
        for user_param in spectrum.user_params:
            _sub(
                scan,
                "userParam",
                name=str(user_param["name"]),
                value=str(user_param["value"]),
            )

    def _add_binary_arrays(self, spectrum_element: ET.Element, spectrum: _Spectrum) -> None:
        arrays = _sub(spectrum_element, "binaryDataArrayList", count="2")
        self._add_binary_array(arrays, "mzArray", spectrum.mz)
        self._add_binary_array(arrays, "intensityArray", spectrum.intensity)

    def _add_binary_array(self, arrays: ET.Element, reference: str, location: _ExternalArray) -> None:
        binary_array = _sub(arrays, "binaryDataArray", encodedLength="0")
        _sub(binary_array, "referenceableParamGroupRef", ref=reference)
        _cv(binary_array, "IMS", "IMS:1000103", "external array length", value=str(location.length))
        _cv(binary_array, "IMS", "IMS:1000104", "external encoded length", value=str(location.encoded_length))
        _cv(binary_array, "IMS", "IMS:1000102", "external offset", value=str(location.offset))
        _sub(binary_array, "binary")

    @property
    def resolved_ibd_mode(self) -> str:
        """Return the resolved imzML ibd mode."""
        if self.ibd_mode != IBD_MODE_AUTO:
            return self.ibd_mode
        return IBD_MODE_CONTINUOUS if len(self._unique_mz_locations()) <= 1 else IBD_MODE_PROCESSED

    def _unique_mz_locations(self) -> set[tuple[int, int, int]]:
        return {(spectrum.mz.offset, spectrum.mz.length, spectrum.mz.encoded_length) for spectrum in self._spectra}

    def _max_coordinate(self, index: int) -> int:
        if not self._spectra:
            return 0
        return max(spectrum.coords[index] for spectrum in self._spectra)


def write_imzml(reader: BaseReader, output_path: PathLike, **kwargs: ty.Any) -> Path:
    """Write an imzy reader to imzML."""
    return IMZMLWriter.from_reader(reader, output_path, **kwargs)


def _resolve_output_paths(output_path: PathLike) -> tuple[Path, Path, Path]:
    path = Path(output_path)
    if path.suffix.lower() in {".imzml", ".ibd"}:
        base_path = path.with_suffix("")
    else:
        base_path = path
    return base_path, base_path.with_suffix(".imzML"), base_path.with_suffix(".ibd")


def _validate_dtype(dtype: ty.Any, *, name: str) -> np.dtype:
    np_dtype = np.dtype(dtype)
    if np_dtype not in _DTYPE_TO_CV:
        valid = ", ".join(str(dtype_) for dtype_ in _DTYPE_TO_CV)
        raise ValueError(f"Unsupported {name}: {np_dtype}. Expected one of: {valid}.")
    return ty.cast(np.dtype[ty.Any], np_dtype)


def _validate_ibd_mode(ibd_mode: str) -> IbdMode:
    if ibd_mode not in _VALID_IBD_MODES:
        raise ValueError(f"Invalid ibd_mode: {ibd_mode!r}.")
    return ty.cast(IbdMode, ibd_mode)


def _validate_resolved_spectrum_type(spectrum_type: str) -> ResolvedSpectrumType:
    if spectrum_type not in _VALID_SPECTRUM_TYPES:
        raise ValueError(f"Invalid spectrum_type: {spectrum_type!r}.")
    return ty.cast(ResolvedSpectrumType, spectrum_type)


def _validate_coordinate_origin(coordinate_origin: str) -> CoordinateOrigin:
    if coordinate_origin not in _VALID_COORDINATE_ORIGINS:
        raise ValueError(f"Invalid coordinate_origin: {coordinate_origin!r}.")
    return ty.cast(CoordinateOrigin, coordinate_origin)


def _validate_on_error(on_error: str) -> OnError:
    if on_error not in _VALID_ON_ERROR:
        raise ValueError(f"Invalid on_error: {on_error!r}.")
    return ty.cast(OnError, on_error)


def _validate_polarity(polarity: str | None) -> str:
    if polarity is None:
        return ""
    polarity = polarity.lower()
    if polarity not in {"positive", "negative"}:
        raise ValueError("polarity must be one of 'positive' or 'negative'.")
    return polarity


def _resolve_reader_spectrum_type(reader: BaseReader, spectrum_type: str) -> ResolvedSpectrumType:
    if spectrum_type == COORDINATE_ORIGIN_AUTO:
        return SPECTRUM_TYPE_CENTROID if reader.is_centroid else SPECTRUM_TYPE_PROFILE
    return _validate_resolved_spectrum_type(spectrum_type)


def _resolve_reader_coordinate_origin(reader: BaseReader, coordinate_origin: str) -> CoordinateOrigin:
    coordinate_origin = _validate_coordinate_origin(coordinate_origin)
    if coordinate_origin != COORDINATE_ORIGIN_AUTO:
        return coordinate_origin
    return COORDINATE_ORIGIN_ONE if isinstance(reader, IMZMLReader) else COORDINATE_ORIGIN_ZERO


def _reader_dtype(reader: BaseReader, attribute: str, default: ty.Any) -> np.dtype:
    precision = getattr(reader, attribute, None)
    if isinstance(precision, str):
        dtype = _PRECISION_TO_DTYPE.get(precision)
        if dtype is not None:
            return dtype
    return ty.cast(np.dtype[ty.Any], np.dtype(default))


def _ibd_mode_accession(ibd_mode: str) -> str:
    if ibd_mode == IBD_MODE_CONTINUOUS:
        return "IMS:1000030"
    if ibd_mode == IBD_MODE_PROCESSED:
        return "IMS:1000031"
    raise ValueError(f"Invalid resolved ibd mode: {ibd_mode!r}.")


def _tag(name: str) -> str:
    return f"{{{MZML_NAMESPACE}}}{name}"


def _sub(parent: ET.Element, tag_name: str, **attrs: str) -> ET.Element:
    return ET.SubElement(parent, _tag(tag_name), attrs)


def _cv(parent: ET.Element, cv_ref: str, accession: str, name: str, **attrs: str) -> ET.Element:
    return _sub(parent, "cvParam", cvRef=cv_ref, accession=accession, name=name, **attrs)


def _format_float(value: float) -> str:
    return f"{value:.12g}"
