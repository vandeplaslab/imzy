"""Resolve and validate imzML metadata without retaining spectrum XML trees."""

from __future__ import annotations

import sys
import typing as ty
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID
from warnings import warn

import numpy as np

from imzy._readers.imzml._ontology import ACCESSION_FIX_MAPPING, all_terms
from imzy._readers.imzml._uo import terms as unit_terms

NS = "{http://psi.hupo.org/ms/mzml}"
PRECISIONS = {
    "MS:1000521": ("f", 4),
    "MS:1000523": ("d", 8),
    "MS:1000519": ("i", 4),
    "MS:1000522": ("l", 8),
    "IMS:1000141": ("i", 4),
    "IMS:1000142": ("l", 8),
    "MS:1000520": ("half", 2),
    "MS:1001479": ("ascii", 0),
}
MODES = {"MS:1000127": "centroid", "MS:1000128": "profile"}
COMPRESSIONS = {
    accession
    for accession, (name, _) in all_terms.items()
    if "compression" in name.lower() and (name != "binary data compression type")
}
ARRAY_KINDS = {
    accession
    for accession, (name, _) in all_terms.items()
    if name.endswith(" array") and accession.startswith("MS:") and accession != "MS:1000513"
}
FAMILIES = (set(PRECISIONS), set(MODES), COMPRESSIONS, ARRAY_KINDS)
UNIT_FACTORS = {
    "UO:0000008": 1e6,
    "UO:0000015": 1e4,
    "UO:0000016": 1e3,
    "UO:0000017": 1.0,
    "UO:0000018": 1e-3,
    "UO:0000019": 1e-4,
    "UO:0000020": 1e-6,
}


@dataclass(frozen=True)
class Param:
    """Original value, name, and units of a resolved cvParam."""

    name: str
    value: str | None
    unit: str | None
    unit_name: str | None


class ParamResolver:
    """Resolve shared groups once and apply direct cvParam overrides."""

    def __init__(self, root: ty.Any) -> None:
        self.groups: dict[str, ty.Any] = {}
        self.resolved: dict[str, dict[str, Param]] = {}
        for group in root.findall(f"{NS}referenceableParamGroupList/{NS}referenceableParamGroup"):
            group_id = group.attrib["id"]
            if group_id in self.groups:
                raise ValueError(f"Duplicate referenceable parameter group {group_id!r}.")
            self.groups[group_id] = group
        for group_id in self.groups:
            self.group(group_id)

    def group(self, group_id: str, visiting: tuple[str, ...] = ()) -> dict[str, Param]:
        """Return a group, refusing missing or cyclic references."""
        if group_id in visiting:
            raise ValueError(f"Reference cycle in parameter group {group_id!r}.")
        if group_id not in self.resolved:
            if group_id not in self.groups:
                raise ValueError(f"Missing referenceable parameter group {group_id!r}.")
            self.resolved[group_id] = self.params(self.groups[group_id], (*visiting, group_id))
        return self.resolved[group_id]

    def params(self, elem: ty.Any, visiting: tuple[str, ...] = ()) -> dict[str, Param]:
        """Merge inherited terms, with direct terms taking precedence by family."""
        if elem is None:
            return {}
        direct: dict[str, Param] = {}
        for node in elem.findall(f"{NS}cvParam"):
            accession = node.attrib["accession"]
            name = node.get("name", "")
            accession = ACCESSION_FIX_MAPPING.get((accession, name), accession)
            param = Param(name, node.get("value"), node.get("unitAccession"), node.get("unitName"))
            if accession in direct and direct[accession] != param:
                raise ValueError(f"Conflicting direct cvParam {accession}.")
            direct[accession] = param
        overridden = set(direct)
        for family in FAMILIES:
            if family.intersection(direct):
                overridden.update(family)
        inherited: dict[str, Param] = {}
        for ref in elem.findall(f"{NS}referenceableParamGroupRef"):
            for accession, param in self.group(ref.attrib["ref"], visiting).items():
                if accession in overridden:
                    continue
                if accession in inherited and inherited[accession] != param:
                    raise ValueError(f"Conflicting inherited cvParam {accession}.")
                inherited[accession] = param
        return inherited | direct


@dataclass
class Metadata:
    """Complete acquisition index and metadata, independent of reader selections."""

    root: ty.Any
    mz_precision: str
    int_precision: str
    offsets: np.ndarray
    coordinates: np.ndarray
    spans: np.ndarray
    kinds: list[str]
    values: dict[str, ty.Any]
    mode: str | None
    mz_min: float | None
    mz_max: float | None
    coordinate_base: tuple[int, int, int | None]
    uuid: str | None


def choose_iterparse(parse_lib: str | None = None) -> ty.Callable:
    """Select a supported XML parser, honoring an explicit lxml request."""
    if parse_lib not in (None, "ElementTree", "lxml"):
        raise ValueError("parse_lib must be None, 'ElementTree', or 'lxml'.")
    if parse_lib == "ElementTree" or (parse_lib is None and sys.platform == "win32"):
        from xml.etree.ElementTree import iterparse

        return iterparse
    try:
        from lxml.etree import iterparse as lxml_iterparse

        return ty.cast(ty.Callable, lxml_iterparse)
    except ImportError:
        if parse_lib == "lxml":
            raise
        from xml.etree.ElementTree import iterparse as et_iterparse

        return et_iterparse


def _spectrum_arrays(
    elem: ty.Any,
    resolver: ParamResolver,
    precisions: dict[str, str],
    kinds: list[str],
    spans: list[tuple[int, int, int, int, int, int]],
    index: int,
) -> tuple[int, int, int, int]:
    arrays = {}
    for array in elem.findall(f"{NS}binaryDataArrayList/{NS}binaryDataArray"):
        kind, code, offset, length, width = _array(resolver.params(array))
        if kind in arrays:
            raise ValueError(f"Duplicate binary array {kind}.")
        arrays[kind] = (offset, length)
        if kind in {"MS:1000514", "MS:1000515"}:
            if kind in precisions and precisions[kind] != code:
                raise ValueError(f"Mixed precision across spectra for {kind}.")
            precisions[kind] = code
        span_kind = f"{kind}/{code}"
        if span_kind not in kinds:
            kinds.append(span_kind)
        spans.append((offset, length * width, kinds.index(span_kind), width, length, index))
    if not {"MS:1000514", "MS:1000515"}.issubset(arrays):
        raise ValueError("Missing m/z or intensity binary array.")
    return (*arrays["MS:1000514"], *arrays["MS:1000515"])


def _observed_range(params: dict[str, Param]) -> tuple[float, float] | None:
    try:
        low = float(ty.cast(str, params["MS:1000528"].value))
        high = float(ty.cast(str, params["MS:1000527"].value))
    except (KeyError, TypeError, ValueError):
        return None
    return (low, high) if np.isfinite([low, high]).all() and low <= high else None


def _normalize_coordinates(
    coordinates: list[tuple[int, int, int]],
    z_presence: set[bool],
    values: dict[str, ty.Any],
) -> tuple[np.ndarray, tuple[int, int, int | None]]:
    if len(z_presence) > 1:
        raise ValueError("Mixed explicit and missing z coordinates (IMS:1000052).")
    coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
    base = tuple(0 if len(coords) and coords[:, i].min() == 0 else 1 for i in range(3))
    for i in range(3):
        if base[i] == 0:
            if np.any(coords[:, i] == np.iinfo(np.int64).max):
                raise ValueError("Coordinate normalization exceeds the supported integer range.")
            coords[:, i] += 1
    for i, axis in enumerate(("x", "y")):
        observed = int(coords[:, i].max()) if len(coords) else 1
        count = values.setdefault(f"max count of pixels {axis}", observed)
        if count < observed or count < 1:
            raise ValueError(f"Declared {axis} dimension {count} cannot contain coordinates through {observed}.")
    return coords, (base[0], base[1], base[2] if True in z_presence else None)


def _integer(params: dict[str, Param], accession: str) -> int:
    try:
        value = int(params[accession].value)  # type: ignore[arg-type]
        if value < 0 or value > np.iinfo(np.int64).max:
            raise ValueError("outside the supported nonnegative integer range")
        return value
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid or missing {accession}: {params.get(accession)}.") from error


def _mode(params: dict[str, Param]) -> str | None:
    modes = {MODES[key] for key in MODES.keys() & params.keys()}
    if len(modes) > 1:
        raise ValueError("Conflicting centroid/profile spectrum declarations.")
    return next(iter(modes), None)


def _coordinates(spectrum: ty.Any, resolver: ParamResolver) -> tuple[int, int, int | None]:
    candidates = []
    for scan in spectrum.findall(f"{NS}scanList/{NS}scan"):
        params = resolver.params(scan)
        if not {"IMS:1000050", "IMS:1000051", "IMS:1000052"}.intersection(params):
            continue
        candidates.append(
            (
                _integer(params, "IMS:1000050"),
                _integer(params, "IMS:1000051"),
                _integer(params, "IMS:1000052") if "IMS:1000052" in params else None,
            )
        )
    if not candidates or len(set(candidates)) != 1:
        raise ValueError("Expected one unambiguous coordinate-bearing scan (IMS:1000050/51/52).")
    return candidates[0]


def _array(params: dict[str, Param]) -> tuple[str, str, int, int, int]:
    kinds = ARRAY_KINDS.intersection(params)
    if len(kinds) != 1:
        raise ValueError(f"Expected one binary array kind, found {sorted(kinds)}.")
    kind = next(iter(kinds))
    compression = COMPRESSIONS.intersection(params)
    if compression - {"MS:1000576"}:
        raise ValueError(f"Unsupported compression: {sorted(compression)}.")
    precisions = {PRECISIONS[key] for key in PRECISIONS.keys() & params.keys()}
    if len(precisions) != 1:
        raise ValueError(f"Expected one supported precision for {kind}.")
    code, width = next(iter(precisions))
    if not width:
        raise ValueError(f"Unsupported null-terminated ASCII precision for {kind}.")
    if kind in {"MS:1000514", "MS:1000515"} and code == "half":
        raise ValueError(f"Unsupported 16-bit float precision for {kind}.")
    offset = _integer(params, "IMS:1000102")
    length = _integer(params, "IMS:1000103")
    encoded = length * width
    if "IMS:1000104" in params and _integer(params, "IMS:1000104") != encoded:
        raise ValueError(f"External encoded length IMS:1000104 disagrees with {kind} length and precision.")
    return kind, code, offset, length, width


def _pixel_size(param: Param | None) -> tuple[float | None, str]:
    if param is None:
        return None, "missing pixel size"
    if param.name.strip().lower() == "pixel size":
        return None, "legacy pixel size is an area"
    unit = param.unit
    # Old unitless length declarations retain the historical um interpretation.
    if unit is None and param.unit_name:
        matches = [key for key in UNIT_FACTORS if unit_terms[key][0] == param.unit_name.replace("metre", "meter")]
        if len(matches) != 1:
            return None, f"unsupported pixel-size unit {param.unit_name!r}"
        unit = matches[0]
    if unit is not None:
        if unit not in UNIT_FACTORS:
            return None, f"unsupported pixel-size unit {unit!r}"
        if param.unit_name and param.unit_name.replace("metre", "meter") != unit_terms[unit][0]:
            return None, f"conflicting pixel-size unit {unit!r} and name {param.unit_name!r}"
    try:
        value = float(param.value) * UNIT_FACTORS.get(unit, 1.0)  # type: ignore[arg-type]
        if not np.isfinite(value) or value <= 0:
            raise ValueError("nonpositive or nonfinite size")
    except (TypeError, ValueError, OverflowError):
        return None, f"invalid pixel size {param.value!r}"
    return value, ""


def read_values(root: ty.Any, resolver: ParamResolver | None = None) -> dict[str, ty.Any]:
    """Read grid dimensions and physical sizes through referenced scan settings."""
    resolver = resolver or ParamResolver(root)
    combined: dict[str, Param] = {}
    for settings in root.findall(f"{NS}scanSettingsList/{NS}scanSettings"):
        for accession, param in resolver.params(settings).items():
            if accession in combined and combined[accession] != param:
                raise ValueError(f"Conflicting scan settings for {accession}.")
            combined[accession] = param
    values: dict[str, ty.Any] = {}
    for axis, accession in (("x", "IMS:1000042"), ("y", "IMS:1000043")):
        if accession in combined:
            values[f"max count of pixels {axis}"] = _integer(combined, accession)
    for axis, accession in (("x", "IMS:1000046"), ("y", "IMS:1000047")):
        size, reason = _pixel_size(combined.get(accession))
        values[f"pixel size {axis}"] = size
        values[f"pixel size {axis} reason"] = reason
    _read_optional_values(root, resolver, combined, values)
    return values


def _read_optional_values(
    root: ty.Any,
    resolver: ParamResolver,
    scan_params: dict[str, Param],
    values: dict[str, ty.Any],
) -> None:
    """Preserve the reader's existing optional acquisition/instrument metadata."""
    fields = {
        "IMS:1000044": ("max dimension x", int),
        "IMS:1000045": ("max dimension y", int),
        "MS:1000835": ("matrix solution concentration", float),
        "MS:1000843": ("wavelength", float),
        "MS:1000844": ("focus diameter x", float),
        "MS:1000845": ("focus diameter y", float),
        "MS:1000846": ("pulse energy", float),
        "MS:1000847": ("pulse duration", float),
        "MS:1000848": ("attenuation", float),
    }
    params = scan_params.copy()
    for config in root.findall(f"{NS}instrumentConfigurationList/{NS}instrumentConfiguration"):
        for elem in config.iter():
            params.update(resolver.params(elem))
    for accession in fields.keys() & params.keys():
        name, convert = fields[accession]
        try:
            values[name] = convert(ty.cast(str, params[accession].value))
        except (TypeError, ValueError):
            warn(f"Wrong data type in XML file. Skipped attribute {name!r}.", stacklevel=3)


def parse_metadata(path: Path, parse_lib: str | None = None) -> Metadata:
    """Stream all spectra and resolve their array layouts, modes, and coordinates."""
    iterparse = choose_iterparse(parse_lib)
    context = "header"
    try:
        with path.open("rb") as stream:
            iterator = iterparse(stream, events=("start", "end"))
            _, root = next(iterator)
            resolver = None
            parent = None
            offsets: list[tuple[int, int, int, int]] = []
            coordinates: list[tuple[int, int, int]] = []
            spans: list[tuple[int, int, int, int, int, int]] = []
            kinds: list[str] = []
            modes: set[str | None] = set()
            ranges: list[tuple[float, float] | None] = []
            z_presence: set[bool] = set()
            precisions: dict[str, str] = {}
            default_mode = None
            for event, elem in iterator:
                if event == "start" and elem.tag == NS + "spectrumList":
                    parent = elem
                    resolver = ParamResolver(root)
                    default_mode = _mode(resolver.params(root.find(f"{NS}fileDescription/{NS}fileContent")))
                if event != "end" or elem.tag != NS + "spectrum":
                    continue
                context = f"spectrum {elem.get('id', elem.get('index', str(len(offsets))))}"
                if resolver is None or parent is None:
                    raise ValueError("Spectrum outside spectrumList.")
                params = resolver.params(elem)
                modes.add(_mode(params) or default_mode)
                coord = _coordinates(elem, resolver)
                z_presence.add(coord[2] is not None)
                coordinates.append((coord[0], coord[1], 1 if coord[2] is None else coord[2]))
                offsets.append(_spectrum_arrays(elem, resolver, precisions, kinds, spans, len(offsets)))
                ranges.append(_observed_range(params))
                parent.remove(elem)
            context = "acquisition metadata"
            if resolver is None:
                raise ValueError("Missing spectrumList.")
            modes.discard(None)
            if len(modes) > 1:
                raise ValueError("Mixed centroid/profile spectrum modes.")
            values = read_values(root, resolver)
            coords, base = _normalize_coordinates(coordinates, z_presence, values)
            bounds = [item for item in ranges if item is not None]
            if len(bounds) != len(ranges):
                bounds = []
            file_params = resolver.params(root.find(f"{NS}fileDescription/{NS}fileContent"))
            uuid = file_params.get("IMS:1000080")
            return Metadata(
                root,
                precisions.get("MS:1000514", "d"),
                precisions.get("MS:1000515", "f"),
                np.asarray(offsets, dtype=np.int64).reshape(-1, 4),
                coords,
                np.asarray(spans, dtype=np.int64).reshape(-1, 6),
                kinds,
                values,
                next(iter(modes), None),
                min(item[0] for item in bounds) if bounds else None,
                max(item[1] for item in bounds) if bounds else None,
                base,
                uuid.value if uuid else None,
            )
    except (ValueError, TypeError, KeyError, AttributeError, StopIteration, OverflowError, SyntaxError) as error:
        raise ValueError(f"{path}: {context}: {error}") from error


def validate_metadata(metadata: Metadata, path: Path, ibd_path: Path) -> None:
    """Check all external ranges and paired lengths before either backend opens."""
    size = ibd_path.stat().st_size
    if size < 16:
        raise ValueError(f"{path}: truncated ibd UUID header in {ibd_path}.")
    offsets = metadata.offsets
    if offsets.shape != (len(metadata.coordinates), 4):
        raise ValueError(f"{path}: invalid spectrum offset index.")
    if np.any(offsets[:, 1] != offsets[:, 3]):
        raise ValueError(f"{path}: m/z and intensity array lengths differ.")
    for column, code, kind in ((0, metadata.mz_precision, "MS:1000514"), (2, metadata.int_precision, "MS:1000515")):
        width = dict(zip(("f", "d", "i", "l"), (4, 8, 4, 8), strict=True))[code]
        rows = (
            metadata.spans[metadata.spans[:, 2] == metadata.kinds.index(f"{kind}/{code}")]
            if len(offsets)
            else metadata.spans
        )
        if len(rows) != len(offsets) or (
            len(rows)
            and (
                np.any(rows[:, 0] != offsets[:, column])
                or np.any(rows[:, 4] != offsets[:, column + 1])
                or np.any(rows[:, 3] != width)
                or np.any(rows[:, 5] != np.arange(len(offsets)))
            )
        ):
            raise ValueError(f"{path}: cached array span disagrees with offset index.")
    _validate_spans(metadata, path, size)


def _validate_spans(metadata: Metadata, path: Path, size: int) -> None:
    previous: tuple[int, ...] | None = None
    for row in sorted(map(tuple, metadata.spans.tolist())):
        offset, encoded, _kind, width, length, index = row
        if index < 0 or index >= len(metadata.offsets):
            raise ValueError(f"{path}: invalid spectrum index in external array span.")
        if offset < 0 or length < 0 or width <= 0 or encoded != length * width:
            raise ValueError(f"{path}: spectrum {index}: invalid external array offset or length.")
        if offset > size or encoded > size - offset:
            raise ValueError(f"{path}: spectrum {index}: external array extends beyond the ibd file.")
        if not encoded:
            continue
        if offset < 16:
            raise ValueError(f"{path}: spectrum {index}: external array overlaps the UUID header.")
        if previous is not None and offset < previous[0] + previous[1] and row[:5] != previous[:5]:
            raise ValueError(f"{path}: spectrum {index}: overlapping external arrays.")
        previous = row


def check_uuid(declared: str | None, path: Path, ibd_path: Path) -> None:
    """Warn about missing, malformed, or mismatched UUIDs without rejecting vendor data."""
    with ibd_path.open("rb") as stream:
        binary = UUID(bytes=stream.read(16))
    try:
        xml = UUID(declared) if declared else None
    except (ValueError, AttributeError):
        xml = None
    if xml is None:
        warn(f"{path}: missing or malformed UUID {declared!r}; {ibd_path} header UUID is {binary}.", stacklevel=3)
    elif xml != binary:
        warn(f"UUID mismatch: {path} declares {xml}; {ibd_path} header contains {binary}.", stacklevel=3)
