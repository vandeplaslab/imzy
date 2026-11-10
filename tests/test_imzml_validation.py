"""Independent regression cases for imzML metadata and acquisition caches."""

from __future__ import annotations

import builtins
import json
import typing as ty
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import UUID
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from imzy import IMZMLReader, write_imzml
from imzy._readers.imzml import _cache, _imzml
from imzy._readers.imzml._ontology import convert_xml_value

from .imzml_fixtures import NS, cv, dataset, save

pytestmark = pytest.mark.usefixtures("imzml_backend")


def _term(root: ET.Element, accession: str) -> ET.Element:
    term = root.find(f'.//{NS}cvParam[@accession="{accession}"]')
    assert term is not None
    return term


def _arrays(root: ET.Element, spectrum: int = 0) -> list[ET.Element]:
    return root.findall(f".//{NS}spectrum")[spectrum].findall(f"{NS}binaryDataArrayList/{NS}binaryDataArray")


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "unit,value,expected",
    [
        (None, "50", 50.0),
        ("UO:0000016", "0.05", 50.0),
        ("UO:0000018", "4406.25", 4.40625),
        ("UO:0000008", "0.00005", 50.0),
        ("UO:0000015", "0.005", 50.0),
    ],
)
def test_pixel_units(tmp_path: Path, cached: bool, unit: str | None, value: str, expected: float) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    for accession in ("IMS:1000046", "IMS:1000047"):
        term = _term(root, accession)
        term.set("value", value)
        if unit:
            term.set("unitAccession", unit)
    save(path, root)
    if cached:
        IMZMLReader(path).close()
    with IMZMLReader(path) as reader:
        assert reader.pixel_size == pytest.approx(expected)
        assert reader.get_physical_coordinates(0) == pytest.approx((expected, expected))


@pytest.mark.parametrize("case", ["area", "unit", "unit_conflict", "missing", "negative", "nan", "no_value"])
def test_unknown_pixel_sizes_allow_spectra(tmp_path: Path, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    term = _term(root, "IMS:1000046")
    if case == "area":
        term.set("name", "pixel size")
        term.set("value", "10000")
    elif case == "unit":
        term.set("unitAccession", "UO:0000010")
    elif case == "unit_conflict":
        term.set("unitAccession", "UO:0000015")
        term.set("unitName", "micrometer")
    elif case == "missing":
        root.find(f"{NS}scanSettingsList/{NS}scanSettings").remove(term)
    elif case == "no_value":
        term.attrib.pop("value")
    else:
        term.set("value", "-1" if case == "negative" else "nan")
    save(path, root)
    for _ in range(2):
        with pytest.warns(UserWarning, match="pixel.*size"), IMZMLReader(path) as reader:
            np.testing.assert_array_equal(reader[0][0], [100, 101, 102])
            with pytest.raises(ValueError, match="pixel_size="):
                _ = reader.x_pixel_size
            with pytest.raises(ValueError, match="pixel_size="):
                reader.get_physical_coordinates(0)
    before = path.with_suffix(".icache").read_bytes()
    with IMZMLReader(path, pixel_size=(25.0, 30.0)) as reader:
        assert (reader.x_pixel_size, reader.y_pixel_size) == (25.0, 30.0)
        assert reader._get_reader_kwargs()["pixel_size"] == (25.0, 30.0)
    assert path.with_suffix(".icache").read_bytes() == before


@pytest.mark.parametrize("value", [(0.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0), (1.0, float("inf")), (1.0,)])
def test_pixel_override_validation(tmp_path: Path, value: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="pixel_size"):
        IMZMLReader(dataset(tmp_path), pixel_size=value)  # type: ignore[arg-type]


def test_referenced_metadata_and_direct_override(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    groups = root.find(NS + "referenceableParamGroupList")
    shared = ET.SubElement(groups, NS + "referenceableParamGroup", id="pixels")
    for acc in ("IMS:1000046", "IMS:1000047"):
        term = _term(root, acc)
        root.find(f"{NS}scanSettingsList/{NS}scanSettings").remove(term)
        term.set("value", "10000")
        term.set("unitAccession", "UO:0000018")
        shared.append(term)
    settings = root.find(f"{NS}scanSettingsList/{NS}scanSettings")
    ET.SubElement(settings, NS + "referenceableParamGroupRef", ref="pixels")
    cv(settings, "IMS:1000046", "pixel size (x)", 20)
    save(path, root)
    with IMZMLReader(path) as reader:
        assert (reader.x_pixel_size, reader.y_pixel_size) == (20, 10)


@pytest.mark.parametrize("case", ["missing_ref", "cycle", "inherited_conflict"])
def test_invalid_references_have_context(tmp_path: Path, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    groups = root.find(NS + "referenceableParamGroupList")
    first = ET.SubElement(groups, NS + "referenceableParamGroup", id="first")
    second = ET.SubElement(groups, NS + "referenceableParamGroup", id="second")
    if case == "missing_ref":
        ET.SubElement(first, NS + "referenceableParamGroupRef", ref="absent")
    elif case == "cycle":
        ET.SubElement(first, NS + "referenceableParamGroupRef", ref="second")
        ET.SubElement(second, NS + "referenceableParamGroupRef", ref="first")
    else:
        cv(first, "MS:1000511", "ms level", 1)
        cv(second, "MS:1000511", "ms level", 2)
        spectrum = root.find(f".//{NS}spectrum")
        for ref in ("first", "second"):
            ET.SubElement(spectrum, NS + "referenceableParamGroupRef", ref=ref)
    save(path, root)
    with pytest.raises(ValueError, match=r"data\.imzML"):
        IMZMLReader(path)
    assert not path.with_suffix(".icache").exists()


@pytest.mark.parametrize("origin", [0, 1, 5])
def test_coordinate_and_image_roundtrip(tmp_path: Path, origin: int) -> None:
    path = dataset(tmp_path, coordinates=[(origin, origin, 1), (origin + 1, origin, 1)])
    with IMZMLReader(path) as reader:
        normalized = max(origin, 1)
        assert reader.source_coordinate_base == (min(origin, 1), min(origin, 1), 1)
        assert reader.x_coordinates.tolist() == [normalized, normalized + 1]
        values = np.array([2, 3])
        image = reader.reshape(values)
        assert image.shape == reader.image_shape == (reader.y_size, reader.x_size)
        np.testing.assert_array_equal(reader.flatten(image), values)
        np.testing.assert_array_equal(reader.reshape_batch(values[:, None])[0], image)
        assert reader.get_index_for_location(normalized, normalized) == 0
        assert reader.get_index_for_location(normalized + 1, normalized) == 1
        assert reader.get_location_for_index(0) == (normalized, normalized)


@pytest.mark.parametrize("second_scan", [False, True])
def test_coordinates_from_groups_and_second_scan(tmp_path: Path, second_scan: bool) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    groups = root.find(NS + "referenceableParamGroupList")
    for index, scanlist in enumerate(root.findall(f".//{NS}scanList")):
        scan = scanlist[0]
        shared = ET.SubElement(groups, NS + "referenceableParamGroup", id=f"coords{index}")
        for term in list(scan):
            scan.remove(term)
            shared.append(term)
        if second_scan:
            scan = ET.SubElement(scanlist, NS + "scan")
        ET.SubElement(scan, NS + "referenceableParamGroupRef", ref=f"coords{index}")
    save(path, root)
    with IMZMLReader(path) as reader:
        assert reader.xyz_coordinates.tolist() == [[1, 1, 1], [2, 1, 1]]


@pytest.mark.parametrize("case", ["negative", "fractional", "mixed_z", "ambiguous_scan", "small_grid"])
def test_invalid_geometry(tmp_path: Path, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    if case in ("negative", "fractional"):
        _term(root, "IMS:1000050").set("value", "-1" if case == "negative" else "1.5")
    elif case == "mixed_z":
        scan = root.find(f".//{NS}scan")
        scan.remove(_term(scan, "IMS:1000052"))
    elif case == "ambiguous_scan":
        scanlist = root.find(f".//{NS}scanList")
        scan = ET.fromstring(ET.tostring(scanlist[0]))
        _term(scan, "IMS:1000050").set("value", "2")
        scanlist.append(scan)
    else:
        _term(root, "IMS:1000042").set("value", "1")
    save(path, root)
    with pytest.raises(ValueError, match=r"data\.imzML"):
        IMZMLReader(path)


def test_missing_z_and_dimensions(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    for scan in root.findall(f".//{NS}scan"):
        scan.remove(_term(scan, "IMS:1000052"))
    settings = root.find(f"{NS}scanSettingsList/{NS}scanSettings")
    for accession in ("IMS:1000042", "IMS:1000043"):
        settings.remove(_term(settings, accession))
    save(path, root)
    with IMZMLReader(path) as reader:
        assert reader.source_coordinate_base == (1, 1, None)
        assert reader.z_coordinates.tolist() == [1, 1]
        assert reader.image_shape == (1, 2)


def test_plane_views_and_cache_isolation(tmp_path: Path) -> None:
    path = dataset(tmp_path, coordinates=[(0, 0, 0), (0, 0, 1)])
    with IMZMLReader(path) as full:
        assert full.n_pixels == 2
        assert full.source_coordinate_base == (0, 0, 0)
        assert full.to_table([100], tol=0.1, silent=True).shape == (2, 1)
        for operation in (
            lambda: full.reshape(np.array([1, 2])),
            lambda: full.get_ion_image(100, tol=0.1, silent=True),
            lambda: full.to_hdf5(tmp_path / "bad.h5", [100], tol=0.1),
            lambda: full.to_zarr(tmp_path / "bad.zarr", [100], tol=0.1),
        ):
            with pytest.raises(ValueError, match="z_plane"):
                operation()
        write_imzml(full, tmp_path / "all_planes", silent=True)
    cache = path.with_suffix(".icache").read_bytes()
    for plane, low in ((1, 100.0), (2, 200.0)):
        with IMZMLReader(path, z_plane=plane, pixel_size=(10.0, 20.0)) as reader:
            assert reader.n_pixels == 1
            assert reader.mz_min == low
            assert reader.mz_max == low + 2
            assert reader.image_shape == (1, 1)
            assert reader.reshape(np.array([3])).tolist() == [[3]]
            with IMZMLReader(path, **reader._get_reader_kwargs()) as reopened:
                np.testing.assert_array_equal(reopened[0][0], reader[0][0])
            output = write_imzml(reader, tmp_path / f"plane{plane}", silent=True)
            with IMZMLReader(output) as exported:
                assert exported.n_pixels == 1
                assert exported.z_coordinates.tolist() == [plane]
                assert exported.x_pixel_size == 10
    assert path.with_suffix(".icache").read_bytes() == cache
    with IMZMLReader(path) as full:
        assert full.n_pixels == 2
        assert full.mz_min == 100 and full.mz_max == 202
    with pytest.raises(ValueError, match="absent"):
        IMZMLReader(path, z_plane=3)


@pytest.mark.parametrize("plane", [0, -1, 1.5, "1"])
def test_plane_argument_validation(tmp_path: Path, plane: ty.Any) -> None:
    with pytest.raises(ValueError, match="z_plane"):
        IMZMLReader(dataset(tmp_path), z_plane=plane)


@pytest.mark.parametrize("spectrum", [0, 1])
def test_uniform_array_precision_overrides(tmp_path: Path, spectrum: int) -> None:
    path = dataset(tmp_path, mz_code="f")
    root = ET.parse(path).getroot()
    group = root.find(f'{NS}referenceableParamGroupList/{NS}referenceableParamGroup[@id="mz"]')
    precision = _term(group, "MS:1000521")
    precision.set("accession", "MS:1000523")
    precision.set("name", "64-bit float")
    for index in (0, 1):
        cv(_arrays(root, index)[0], "MS:1000521", "32-bit float")
    save(path, root)
    with IMZMLReader(path) as reader:
        assert reader.mz_precision == "f"
        np.testing.assert_array_equal(
            reader[spectrum][0], [100 + spectrum * 100, 101 + spectrum * 100, 102 + spectrum * 100]
        )


@pytest.mark.parametrize("inline", [False, True])
def test_array_roles_and_alternative_groups(tmp_path: Path, inline: bool) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    groups = root.find(NS + "referenceableParamGroupList")
    original = groups.find(f'{NS}referenceableParamGroup[@id="mz"]')
    alternate = ET.fromstring(ET.tostring(original))
    alternate.set("id", "alternate")
    _term(alternate, "MS:1000514").set("name", "vendor label")
    groups.append(alternate)
    array = _arrays(root, 1)[0]
    ref = array.find(NS + "referenceableParamGroupRef")
    if inline:
        array.remove(ref)
        for term in alternate:
            array.append(ET.fromstring(ET.tostring(term)))
    else:
        ref.set("ref", "alternate")
    save(path, root)
    with IMZMLReader(path) as reader:
        np.testing.assert_array_equal(reader[0][0], [100, 101, 102])
        np.testing.assert_array_equal(reader[1][0], [200, 201, 202])


def test_exact_cross_kind_alias_is_rejected(tmp_path: Path) -> None:
    path = dataset(tmp_path, int_code="d")
    root = ET.parse(path).getroot()
    mz, intensity = _arrays(root)
    _term(intensity, "IMS:1000102").set("value", _term(mz, "IMS:1000102").get("value"))
    save(path, root)
    with pytest.raises(ValueError, match="overlapping"):
        IMZMLReader(path)


@pytest.mark.parametrize(
    "case", ["mixed", "encoded", "overlap", "header", "bounds", "short", "compression", "half", "ascii"]
)
def test_bad_array_layouts_fail_before_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    array = _arrays(root, 1)[0]
    if case == "mixed":
        cv(array, "MS:1000521", "32-bit float")
        _term(array, "IMS:1000104").set("value", "12")
    elif case in ("encoded", "short"):
        _term(array, "IMS:1000104" if case == "encoded" else "IMS:1000103").set("value", "2")
    elif case in ("overlap", "header", "bounds"):
        _term(array, "IMS:1000102").set("value", {"overlap": "20", "header": "0", "bounds": "100000"}[case])
    elif case == "compression":
        cv(array, "MS:1000574", "zlib compression")
    elif case == "ascii":
        cv(array, "MS:1001479", "null-terminated ASCII string")
    else:
        cv(array, "MS:1000520", "16-bit float")
    save(path, root)

    def unexpected_backend(*args: ty.Any) -> None:
        raise AssertionError("Malformed layout reached a binary backend")

    monkeypatch.setattr(IMZMLReader, "_init_backend", unexpected_backend)
    with pytest.raises(ValueError, match=r"data\.imzML"):
        IMZMLReader(path)
    assert not path.with_suffix(".icache").exists()


def test_auxiliary_array_and_missing_typed_value(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    group = ET.SubElement(root.find(NS + "referenceableParamGroupList"), NS + "referenceableParamGroup", id="mobility")
    cv(group, "MS:1003006", "mean inverse reduced ion mobility array")
    cv(group, "MS:1000520", "16-bit float")
    cv(group, "MS:1000576", "no compression")
    with path.with_suffix(".ibd").open("ab") as stream:
        offset = stream.tell()
        stream.write(np.array([1.0, 2.0, 3.0], dtype="<f2").tobytes())
    array = ET.SubElement(root.find(f".//{NS}binaryDataArrayList"), NS + "binaryDataArray")
    ET.SubElement(array, NS + "referenceableParamGroupRef", ref="mobility")
    cv(array, "IMS:1000102", "external offset", offset)
    cv(array, "IMS:1000103", "external array length", 3)
    cv(array, "IMS:1000104", "external encoded length", 6)
    save(path, root)
    with IMZMLReader(path) as reader:
        np.testing.assert_array_equal(reader[0][0], [100, 101, 102])
    _term(array, "IMS:1000102").set("value", "16")
    save(path, root)
    with pytest.raises(ValueError, match="overlapping"):
        IMZMLReader(path)
    assert convert_xml_value("xsd:float", None) is None
    assert convert_xml_value(None, None) is True


@pytest.mark.parametrize("case", ["unused", "polarity", "override", "mixed"])
def test_effective_spectrum_modes(tmp_path: Path, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    group = ET.SubElement(root.find(NS + "referenceableParamGroupList"), NS + "referenceableParamGroup", id="other")
    cv(group, "MS:1000128", "positive scan" if case == "polarity" else "profile spectrum")
    if case in ("override", "mixed"):
        cv(root.findall(f".//{NS}spectrum")[0], "MS:1000128", "profile spectrum")
        if case == "override":
            cv(root.findall(f".//{NS}spectrum")[1], "MS:1000128", "profile spectrum")
    if case == "polarity":
        for spectrum in root.findall(f".//{NS}spectrum"):
            ET.SubElement(spectrum, NS + "referenceableParamGroupRef", ref="other")
    save(path, root)
    if case == "mixed":
        with pytest.raises(ValueError, match="Mixed centroid/profile"):
            IMZMLReader(path)
    else:
        with IMZMLReader(path) as reader:
            assert reader.is_centroid is (case != "override")


@pytest.mark.parametrize("case", ["spelling", "mismatch", "missing", "malformed", "truncated"])
def test_uuid_checks_on_every_open(tmp_path: Path, case: str) -> None:
    path = dataset(tmp_path)
    root = ET.parse(path).getroot()
    uuid = _term(root, "IMS:1000080")
    if case == "spelling":
        uuid.set("value", "{ABCDEF01-2345-6789-ABCD-EF0123456789}")
        data = path.with_suffix(".ibd").read_bytes()
        path.with_suffix(".ibd").write_bytes(UUID(uuid.get("value")).bytes + data[16:])
    elif case == "mismatch":
        uuid.set("value", "12345678-1234-1234-1234-123456789012")
    elif case == "malformed":
        uuid.set("value", "broken")
    elif case == "missing":
        root.find(f"{NS}fileDescription/{NS}fileContent").remove(uuid)
    else:
        path.with_suffix(".ibd").write_bytes(bytes(8))
    save(path, root)
    for _ in range(2):
        if case == "truncated":
            with pytest.raises(ValueError, match="UUID header"):
                IMZMLReader(path)
        elif case == "spelling":
            IMZMLReader(path).close()
        else:
            with pytest.warns(UserWarning, match="UUID") as caught:
                IMZMLReader(path).close()
            assert str(path) in str(caught[0].message)
            assert str(path.with_suffix(".ibd")) in str(caught[0].message)


@pytest.mark.parametrize("content", ["", "<mzML", "<mzML/>", '<mzML xmlns="http://psi.hupo.org/ms/mzml"><run>'])
def test_xml_errors_are_contextual(tmp_path: Path, content: str) -> None:
    path = dataset(tmp_path)
    path.write_text(content)
    with pytest.raises(ValueError, match=r"data\.imzML") as caught:
        IMZMLReader(path)
    assert caught.value.__cause__ is not None


def test_stale_corrupt_and_legacy_caches_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = dataset(tmp_path)
    IMZMLReader(path).close()
    cache_path = path.with_suffix(".icache")
    root = ET.parse(path).getroot()
    _term(root, "IMS:1000046").set("value", "99")
    save(path, root)
    with IMZMLReader(path) as reader:
        assert reader.x_pixel_size == 99
    for content in (b"broken", b"", None):
        if content is None:
            with cache_path.open("wb") as stream:
                np.savez(stream, mz_precision="d")
        else:
            cache_path.write_bytes(content)
        with IMZMLReader(path) as reader:
            assert reader.n_pixels == 2
    metadata = _cache.load_cache(cache_path, _cache.fingerprint(path, path.with_suffix(".ibd")))
    assert metadata is not None

    def unexpected_parse(*args: ty.Any) -> None:
        raise AssertionError("A valid cache reparsed XML")

    monkeypatch.setattr(_imzml, "parse_metadata", unexpected_parse)
    with IMZMLReader(path) as reader:
        assert reader.x_pixel_size == 99


def test_corrupt_index_rebuilds_from_xml(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    IMZMLReader(path).close()
    cache_path = path.with_suffix(".icache")
    with np.load(cache_path) as cache:
        values = {key: cache[key] for key in cache.files}
    values["byte_offsets"][0, 0] = 18
    with cache_path.open("wb") as stream:
        np.savez(stream, **values)
    with IMZMLReader(path) as reader:
        np.testing.assert_array_equal(reader[0][0], [100, 101, 102])


def test_binary_change_invalidates_cache(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    IMZMLReader(path).close()
    with path.with_suffix(".ibd").open("r+b") as stream:
        stream.truncate(17)
    with pytest.raises(ValueError, match="beyond"):
        IMZMLReader(path)


def test_atomic_cache_failure_and_read_only_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = dataset(tmp_path)
    IMZMLReader(path).close()
    cache_path = path.with_suffix(".icache")
    previous = cache_path.read_bytes()

    def cannot_replace(*args: ty.Any) -> None:
        raise PermissionError("read-only cache directory")

    monkeypatch.setattr(_cache.os, "replace", cannot_replace)
    with IMZMLReader(path) as reader:
        reader._write_icache_safely()
        assert reader[0][0].size == 3
    assert cache_path.read_bytes() == previous
    assert not list(tmp_path.glob(".data.icache.*"))
    cache_path.unlink()
    with IMZMLReader(path) as reader:
        assert reader.n_pixels == 2
    assert not cache_path.exists()
    assert not list(tmp_path.glob(".data.icache.*"))


def test_concurrent_atomic_cache_replacements(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    with IMZMLReader(path) as reader:
        metadata = reader._source_metadata
        source_fingerprint = reader._source_fingerprint
    cache_path = path.with_suffix(".icache")

    def replace_cache(index: int) -> None:
        _cache.save_cache(cache_path, metadata, source_fingerprint)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(replace_cache, range(20)))
    cached = _cache.load_cache(cache_path, source_fingerprint)
    assert cached is not None
    np.testing.assert_array_equal(cached.offsets, metadata.offsets)
    assert not list(tmp_path.glob(".data.icache.*"))


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_windows_cache_replacement_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int) -> None:
    path = dataset(tmp_path)
    with IMZMLReader(path) as reader:
        metadata = reader._source_metadata
        source_fingerprint = reader._source_fingerprint
    cache_path = path.with_suffix(".icache")
    replace = _cache.os.replace
    attempts = 0
    delays: list[float] = []

    def contended_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            error = PermissionError("Windows cache replacement contention")
            error.winerror = winerror
            raise error
        replace(source, destination)

    monkeypatch.setattr(_cache.os, "replace", contended_replace)
    monkeypatch.setattr(_cache, "sleep", delays.append)
    _cache.save_cache(cache_path, metadata, source_fingerprint)
    assert attempts == 3
    assert len(delays) == 2 and all(delay > 0 for delay in delays)
    cached = _cache.load_cache(cache_path, source_fingerprint)
    assert cached is not None
    np.testing.assert_array_equal(cached.offsets, metadata.offsets)
    assert not list(tmp_path.glob(".data.icache.*"))


@pytest.mark.parametrize("winerror", [None, 5, 32, 33, 19])
def test_cache_replacement_failure_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winerror: int | None,
) -> None:
    path = dataset(tmp_path)
    with IMZMLReader(path) as reader:
        metadata = reader._source_metadata
        source_fingerprint = reader._source_fingerprint
    cache_path = path.with_suffix(".icache")
    previous = cache_path.read_bytes()
    attempts = 0
    delays: list[float] = []

    def denied_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        error = PermissionError("Cache replacement denied")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(_cache.os, "replace", denied_replace)
    monkeypatch.setattr(_cache, "sleep", delays.append)
    with pytest.raises(PermissionError, match="Cache replacement denied"):
        _cache.save_cache(cache_path, metadata, source_fingerprint)
    if winerror in (5, 32, 33):
        assert 1 < attempts <= 10
        assert len(delays) == attempts - 1
        assert 0 < sum(delays) < 1
    else:
        assert attempts == 1
        assert delays == []
    assert cache_path.read_bytes() == previous
    assert not list(tmp_path.glob(".data.icache.*"))


@pytest.mark.parametrize("directory", ["second", r"second\acquisition"])
def test_source_path_is_part_of_cache_fingerprint(tmp_path: Path, directory: str) -> None:
    first, second = dataset(tmp_path / "first"), dataset(tmp_path / directory)
    IMZMLReader(first).close()
    second.with_suffix(".icache").write_bytes(first.with_suffix(".icache").read_bytes())
    with IMZMLReader(second) as reader:
        assert reader.n_pixels == 2
    with np.load(second.with_suffix(".icache")) as cache:
        sources = json.loads(str(cache["source_fingerprint"].item()))
        assert sources[0][0] == str(second.resolve())
        assert sources[1][0] == str(second.with_suffix(".ibd").resolve())


def test_binary_filename_resolution(tmp_path: Path) -> None:
    path = dataset(tmp_path / "proj.ibd")
    path.with_suffix(".ibdtmp").touch()
    with IMZMLReader(path) as reader:
        assert reader.ibd_path == path.with_suffix(".ibd")
    binary = path.with_name("DATA.IBD")
    path.with_suffix(".ibd").rename(binary)
    with IMZMLReader(path) as reader:
        assert reader.ibd_path.name.casefold() == "data.ibd"
    binary.unlink()
    with pytest.raises(FileNotFoundError):
        IMZMLReader(path)


def test_ambiguous_binary_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "data.imzML"

    # Mock the directory listing so case-insensitive filesystems can exercise ambiguity.
    class Candidate:
        def __init__(self, name: str) -> None:
            self.name = name

        def is_file(self) -> bool:
            return True

    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Candidate("DATA.ibd"), Candidate("Data.IBD")]))
    with pytest.raises(ValueError, match="ambiguous"):
        _imzml.infer_path(path)


def test_parser_selection_is_validated_before_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = dataset(tmp_path)
    IMZMLReader(path).close()
    with pytest.raises(ValueError, match="parse_lib"):
        IMZMLReader(path, parse_lib="LXML")
    original = builtins.__import__

    def no_lxml(name: str, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        if name.startswith("lxml"):
            raise ImportError("lxml unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_lxml)
    with pytest.raises(ImportError, match="lxml"):
        IMZMLReader(path, parse_lib="lxml")
    assert _imzml.choose_iterparse().__module__ == "xml.etree.ElementTree"


def test_parallel_reads_and_below_range_images(tmp_path: Path) -> None:
    path = dataset(tmp_path)
    with IMZMLReader(path) as reader:
        expected = [tuple(array.copy() for array in reader[index]) for index in range(2)]

        def check(index: int) -> bool:
            return all(np.array_equal(a, b) for a, b in zip(reader[index % 2], expected[index % 2], strict=True))

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert all(pool.map(check, range(2000)))
        assert not reader.get_ion_image(0.5, tol=0.05, silent=True).any()
        assert not reader.to_table([0.5], tol=0.05, silent=True).any()


@pytest.mark.parametrize("format_name", ["hdf5", "zarr", "normalizations"])
def test_selected_plane_store_exports(tmp_path: Path, format_name: str) -> None:
    path = dataset(tmp_path, coordinates=[(1, 1, 1), (1, 1, 2)])
    with IMZMLReader(path, z_plane=2, pixel_size=(25.0, 30.0)) as reader:
        if format_name == "hdf5":
            h5py = pytest.importorskip("h5py")
            output = reader.to_hdf5(tmp_path / "selected", [201], tol=0.1, silent=True)
            from imzy import H5CentroidsStore

            store = H5CentroidsStore(output)
            np.testing.assert_array_equal(store.get_ion(201.0), [5.0])
            with h5py.File(output) as stream:
                assert len(stream[f"{store.SPATIAL_KEY}/x_coordinates"]) == 1
                np.testing.assert_array_equal(stream[store.SPATIAL_KEY].attrs["pixel_size"], [25.0, 30.0])
        elif format_name == "zarr":
            zarr = pytest.importorskip("zarr", minversion="3")
            pytest.importorskip("dask.array")
            output = reader.to_zarr(tmp_path / "selected.zarr", [201], tol=0.1, silent=True)
            np.testing.assert_array_equal(zarr.open_group(str(output))["array"][:], [[5.0]])
        else:
            h5py = pytest.importorskip("h5py")
            output = reader.extract_normalizations_hdf5(tmp_path / "norms", silent=True)
            with h5py.File(output) as stream:
                arrays = []
                stream.visititems(lambda name, obj: arrays.append(obj.shape) if isinstance(obj, h5py.Dataset) else None)
                assert arrays and all(shape == (1,) for shape in arrays)
