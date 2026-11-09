"""Small independently encoded imzML acquisitions for reader regression tests."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

NS = "{http://psi.hupo.org/ms/mzml}"
ET.register_namespace("", NS[1:-1])
TYPES = {
    "f": ("MS:1000521", "32-bit float", "<f4"),
    "d": ("MS:1000523", "64-bit float", "<f8"),
    "i": ("MS:1000519", "32-bit integer", "<i4"),
    "l": ("MS:1000522", "64-bit integer", "<i8"),
}


def cv(parent: ET.Element, accession: str, name: str, value: object = None, **attrs: str) -> ET.Element:
    """Append a cvParam, preserving an intentionally absent value."""
    attributes = {"accession": accession, "name": name} | attrs
    if value is not None:
        attributes["value"] = str(value)
    return ET.SubElement(parent, NS + "cvParam", attributes)


def save(path: Path, root: ET.Element) -> None:
    """Write an edited fixture as XML."""
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def dataset(
    directory: Path,
    spectra: list[tuple[np.ndarray, np.ndarray]] | None = None,
    mz_code: str = "d",
    int_code: str = "f",
    *,
    shared: bool = False,
    unaligned: bool = False,
    mode: str = "centroid",
    coordinates: list[tuple[int, int, int]] | None = None,
) -> Path:
    """Encode little-endian binary data and matching XML without imzy's writer."""
    directory.mkdir(parents=True, exist_ok=True)
    if spectra is None:
        spectra = [
            (np.array([100.0, 101.0, 102.0]), np.array([1.0, 7.0, 23.0])),
            (np.array([200.0, 201.0, 202.0]), np.array([3.0, 5.0, 11.0])),
        ]
    coordinates = coordinates if coordinates is not None else [(i + 1, 1, 1) for i in range(len(spectra))]
    path = directory / "data.imzML"
    root = ET.Element(NS + "mzML")
    content = ET.SubElement(ET.SubElement(root, NS + "fileDescription"), NS + "fileContent")
    cv(content, "IMS:1000080", "universally unique identifier", "00000000-0000-0000-0000-000000000000")
    cv(content, "MS:1000127" if mode == "centroid" else "MS:1000128", f"{mode} spectrum")
    groups = ET.SubElement(root, NS + "referenceableParamGroupList")
    for gid, accession, name, code in (
        ("mz", "MS:1000514", "m/z array", mz_code),
        ("intensity", "MS:1000515", "intensity array", int_code),
    ):
        group = ET.SubElement(groups, NS + "referenceableParamGroup", id=gid)
        cv(group, accession, name)
        cv(group, TYPES[code][0], TYPES[code][1])
        cv(group, "MS:1000576", "no compression")
    settings = ET.SubElement(ET.SubElement(root, NS + "scanSettingsList"), NS + "scanSettings")
    for accession, name, value in (
        ("IMS:1000042", "max count of pixels x", max((c[0] for c in coordinates), default=0) + 1),
        (
            "IMS:1000043",
            "max count of pixels y",
            max((c[1] for c in coordinates), default=1) + int(any(c[1] == 0 for c in coordinates)),
        ),
        ("IMS:1000046", "pixel size (x)", 50),
        ("IMS:1000047", "pixel size y", 50),
    ):
        cv(settings, accession, name, value)
    ET.SubElement(root, NS + "instrumentConfigurationList")
    spectrum_list = ET.SubElement(ET.SubElement(root, NS + "run"), NS + "spectrumList", count=str(len(spectra)))
    with path.with_suffix(".ibd").open("wb") as stream:
        stream.write(bytes(16) + (b"x" if unaligned else b""))
        mz_offset = stream.tell()
        for i, ((mz, intensity), coords) in enumerate(zip(spectra, coordinates, strict=True)):
            spectrum = ET.SubElement(spectrum_list, NS + "spectrum", id=f"spectrum={i}")
            if len(mz):
                cv(spectrum, "MS:1000528", "lowest observed m/z", float(np.min(mz)))
                cv(spectrum, "MS:1000527", "highest observed m/z", float(np.max(mz)))
            scan = ET.SubElement(ET.SubElement(spectrum, NS + "scanList"), NS + "scan")
            for accession, name, value in zip(
                ("IMS:1000050", "IMS:1000051", "IMS:1000052"),
                ("position x", "position y", "position z"),
                coords,
                strict=True,
            ):
                cv(scan, accession, name, value)
            arrays = ET.SubElement(spectrum, NS + "binaryDataArrayList")
            for gid, values, code in (("mz", mz, mz_code), ("intensity", intensity, int_code)):
                if gid == "mz" and shared and i > 0:
                    offset = mz_offset
                else:
                    offset = stream.tell()
                    stream.write(values.astype(TYPES[code][2]).tobytes())
                    if gid == "mz":
                        mz_offset = offset
                array = ET.SubElement(arrays, NS + "binaryDataArray")
                ET.SubElement(array, NS + "referenceableParamGroupRef", ref=gid)
                cv(array, "IMS:1000102", "external offset", offset)
                cv(array, "IMS:1000103", "external array length", len(values))
                cv(array, "IMS:1000104", "external encoded length", len(values) * np.dtype(TYPES[code][2]).itemsize)
    save(path, root)
    return path
