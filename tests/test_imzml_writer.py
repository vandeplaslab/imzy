"""Tests for imzML writer."""

from __future__ import annotations

import shutil
import typing as ty
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

import imzy
from imzy import (
    EmptySpectrumWarning,
    IMZMLReader,
    IMZMLWriter,
    SkippedSpectrumWarning,
    write_imzml,
)
from imzy._readers._base import BaseReader
from imzy._readers.imzml._imzml import SPECTRUM_MODE_CENTROID, SPECTRUM_MODE_PROFILE, init_metadata

_MZML_NS = {"mzml": "http://psi.hupo.org/ms/mzml"}


def _copy_imzml_dataset(path: Path, tmp_path: Path) -> Path:
    """Copy an imzML dataset to a temporary path."""
    imzml_path = tmp_path / path.name
    shutil.copyfile(path, imzml_path)
    shutil.copyfile(path.with_suffix(".ibd"), imzml_path.with_suffix(".ibd"))
    cache_path = path.with_suffix(".icache")
    if cache_path.exists():
        shutil.copyfile(cache_path, imzml_path.with_suffix(".icache"))
    return imzml_path


def _has_cv_param(path: Path, accession: str) -> bool:
    """Return whether an imzML file contains a cvParam accession."""
    root = ET.parse(path).getroot()
    return root.find(f'.//mzml:cvParam[@accession="{accession}"]', _MZML_NS) is not None


def _cv_value(path: Path, accession: str) -> str:
    """Return a cvParam value from an imzML file."""
    root = ET.parse(path).getroot()
    element = root.find(f'.//mzml:cvParam[@accession="{accession}"]', _MZML_NS)
    assert element is not None
    return element.attrib["value"]


class DummyReader(BaseReader):
    """Small reader used to test writer export behavior."""

    def __init__(
        self,
        spectra: list[tuple[np.ndarray, np.ndarray]],
        coordinates: np.ndarray,
        *,
        is_centroid: bool = True,
        failing_indices: set[int] | None = None,
    ) -> None:
        self._spectra_data = spectra
        self._xyz_coordinates = coordinates
        self._is_centroid = is_centroid
        self._failing_indices = failing_indices or set()
        super().__init__("dummy")

    @property
    def mz_min(self) -> float:
        """Return the minimum m/z value."""
        return float(min(np.min(mzs) for mzs, _ in self._spectra_data))

    @property
    def mz_max(self) -> float:
        """Return the maximum m/z value."""
        return float(max(np.max(mzs) for mzs, _ in self._spectra_data))

    @property
    def is_centroid(self) -> bool:
        """Return whether spectra are centroided."""
        return self._is_centroid

    @property
    def rois(self) -> list[int]:
        """Return available ROIs."""
        return [0]

    @property
    def x_pixel_size(self) -> float:
        """Return x pixel size."""
        return 1.0

    @property
    def y_pixel_size(self) -> float:
        """Return y pixel size."""
        return 1.0

    def get_summed_spectrum(
        self,
        indices: ty.Iterable[int],
        scales: np.ndarray | None = None,
        silent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a summed spectrum."""
        indices = np.asarray(list(indices))
        mzs = self._spectra_data[int(indices[0])][0]
        intensities = np.zeros_like(mzs, dtype=np.float64)
        for index in indices:
            intensities += self._spectra_data[int(index)][1]
        return mzs, intensities

    def _read_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if index in self._failing_indices:
            raise RuntimeError(f"Could not read pixel {index}.")
        return self._spectra_data[index]

    def _read_spectra(self, indices: ty.Iterable[int] | None = None) -> ty.Iterator[tuple[np.ndarray, np.ndarray]]:
        if indices is None:
            indices = self.pixels
        for index in indices:
            yield self._read_spectrum(int(index))


def test_manual_centroid_processed_roundtrip(tmp_path: Path) -> None:
    """Write centroid spectra manually and read them back."""
    mzs_1 = np.asarray([100.0, 101.0, 102.0], dtype=np.float64)
    ints_1 = np.asarray([1.0, 5.0, 2.0], dtype=np.float32)
    mzs_2 = np.asarray([200.0, 201.0], dtype=np.float64)
    ints_2 = np.asarray([3.0, 7.0], dtype=np.float32)

    with IMZMLWriter(tmp_path / "manual", ibd_mode="processed", spectrum_type="centroid") as writer:
        writer.add_spectrum(mzs_1, ints_1, (1, 1, 1))
        writer.add_spectrum(mzs_2, ints_2, (2, 1, 1))

    reader = IMZMLReader(tmp_path / "manual.imzML", parse_lib="ElementTree")
    assert reader.n_pixels == 2
    assert reader.is_centroid is True
    np.testing.assert_array_equal(reader.xyz_coordinates, np.asarray([[1, 1, 1], [2, 1, 1]]))
    np.testing.assert_array_equal(reader.get_tic(silent=True), np.asarray([8.0, 10.0]))
    np.testing.assert_array_equal(reader.get_spectrum(0)[0], mzs_1)
    np.testing.assert_array_equal(reader.get_spectrum(0)[1], ints_1)
    assert reader.mz_min == 100.0
    assert reader.mz_max == 201.0


def test_profile_continuous_roundtrip(tmp_path: Path) -> None:
    """Write profile spectra with one shared m/z axis."""
    mzs = np.asarray([100.0, 101.0, 102.0], dtype=np.float64)
    with IMZMLWriter(tmp_path / "profile", ibd_mode="continuous", spectrum_type="profile") as writer:
        writer.add_spectrum(mzs, np.asarray([1.0, 2.0, 3.0], dtype=np.float32), (1, 1))
        writer.add_spectrum(mzs, np.asarray([4.0, 5.0, 6.0], dtype=np.float32), (2, 1))

    *_, spectrum_mode, mz_min, mz_max = init_metadata(tmp_path / "profile.imzML", parse_lib="ElementTree")
    reader = IMZMLReader(tmp_path / "profile.imzML", parse_lib="ElementTree")

    assert spectrum_mode == SPECTRUM_MODE_PROFILE
    assert reader.is_centroid is False
    assert mz_min == 100.0
    assert mz_max == 102.0
    np.testing.assert_array_equal(reader.get_spectrum(1)[0], mzs)
    np.testing.assert_array_equal(reader.get_spectrum(1)[1], np.asarray([4.0, 5.0, 6.0], dtype=np.float32))


def test_auto_ibd_mode_uses_spectrum_type(tmp_path: Path) -> None:
    """Resolve automatic ibd mode from centroid/profile spectrum type."""
    mzs = np.asarray([100.0, 101.0], dtype=np.float64)
    intensities = np.asarray([1.0, 2.0], dtype=np.float32)

    with IMZMLWriter(tmp_path / "auto_centroid", spectrum_type="centroid") as writer:
        writer.add_spectrum(mzs, intensities, (1, 1))
        writer.add_spectrum(mzs, intensities, (2, 1))
    *_, centroid_offsets, _coords, centroid_spectrum_mode, _mz_min, _mz_max = init_metadata(
        tmp_path / "auto_centroid.imzML",
        parse_lib="ElementTree",
    )

    with IMZMLWriter(tmp_path / "auto_profile", spectrum_type="profile") as writer:
        writer.add_spectrum(mzs, intensities, (1, 1))
        writer.add_spectrum(mzs, intensities, (2, 1))
    *_, profile_offsets, _coords, profile_spectrum_mode, _mz_min, _mz_max = init_metadata(
        tmp_path / "auto_profile.imzML",
        parse_lib="ElementTree",
    )

    assert centroid_spectrum_mode == SPECTRUM_MODE_CENTROID
    assert profile_spectrum_mode == SPECTRUM_MODE_PROFILE
    assert centroid_offsets[0, 0] != centroid_offsets[1, 0]
    assert profile_offsets[0, 0] == profile_offsets[1, 0]
    assert _has_cv_param(tmp_path / "auto_centroid.imzML", "IMS:1000031")
    assert _has_cv_param(tmp_path / "auto_profile.imzML", "IMS:1000030")


@pytest.mark.parametrize(
    ("filename", "expected_mode"),
    [
        ("simple_imzml.imzML", SPECTRUM_MODE_CENTROID),
        ("Example_Processed.imzML", SPECTRUM_MODE_PROFILE),
    ],
)
def test_write_imzml_reader_roundtrip(tmp_path: Path, filename: str, expected_mode: str) -> None:
    """Export existing imzML readers and read the exported data back."""
    input_path = Path(__file__).parent / "_test_data" / filename
    input_path = _copy_imzml_dataset(input_path, tmp_path)
    source = IMZMLReader(input_path, parse_lib="ElementTree")
    output_path = write_imzml(source, tmp_path / f"exported_{filename}", silent=True)
    exported = IMZMLReader(output_path, parse_lib="ElementTree")
    *_, spectrum_mode, _mz_min, _mz_max = init_metadata(output_path, parse_lib="ElementTree")

    assert spectrum_mode == expected_mode
    np.testing.assert_array_equal(exported.xyz_coordinates, source.xyz_coordinates)
    assert exported.n_pixels == source.n_pixels
    np.testing.assert_array_equal(exported.get_spectrum(0)[0], source.get_spectrum(0)[0])
    np.testing.assert_array_equal(exported.get_spectrum(0)[1], source.get_spectrum(0)[1])


def test_reader_export_shifts_zero_based_coordinates(tmp_path: Path) -> None:
    """Export non-imzML reader coordinates as one-based imzML coordinates."""
    spectra = [(np.asarray([100.0]), np.asarray([1.0], dtype=np.float32))]
    coordinates = np.asarray([[0, 1, 0]])
    reader = DummyReader(spectra, coordinates)

    output_path = write_imzml(reader, tmp_path / "zero_based", silent=True)
    exported = IMZMLReader(output_path, parse_lib="ElementTree")

    np.testing.assert_array_equal(exported.xyz_coordinates, np.asarray([[1, 2, 1]]))


def test_manual_empty_spectrum_warns_and_skips(tmp_path: Path) -> None:
    """Skip manual spectra without m/z peaks."""
    with IMZMLWriter(tmp_path / "manual_empty") as writer:
        assert writer.add_spectrum([100.0], [1.0], (1, 1)) is True
        with pytest.warns(EmptySpectrumWarning, match="no m/z peaks"):
            assert writer.add_spectrum([], [], (2, 1)) is False
        assert writer.add_spectrum([102.0], [3.0], (3, 1)) is True

    reader = IMZMLReader(tmp_path / "manual_empty.imzML", parse_lib="ElementTree")
    assert reader.n_pixels == 2
    np.testing.assert_array_equal(reader.xyz_coordinates, np.asarray([[1, 1, 1], [3, 1, 1]]))
    np.testing.assert_array_equal(reader.get_tic(silent=True), np.asarray([1.0, 3.0]))


def test_reader_empty_spectrum_warns_and_skips(tmp_path: Path) -> None:
    """Skip reader pixels without m/z peaks."""
    spectra = [
        (np.asarray([100.0]), np.asarray([1.0], dtype=np.float32)),
        (np.asarray([]), np.asarray([], dtype=np.float32)),
        (np.asarray([102.0]), np.asarray([3.0], dtype=np.float32)),
    ]
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    reader = DummyReader(spectra, coordinates)

    with pytest.warns(EmptySpectrumWarning, match="no m/z peaks"):
        output_path = write_imzml(reader, tmp_path / "reader_empty", silent=True)
    exported = IMZMLReader(output_path, parse_lib="ElementTree")

    assert exported.n_pixels == 2
    np.testing.assert_array_equal(exported.xyz_coordinates, np.asarray([[1, 1, 1], [3, 1, 1]]))
    np.testing.assert_array_equal(exported.get_tic(silent=True), np.asarray([1.0, 3.0]))


def test_writer_validation(tmp_path: Path) -> None:
    """Validate writer error handling."""
    with pytest.raises(ValueError, match="Unsupported mz_dtype"):
        IMZMLWriter(tmp_path / "bad_dtype", mz_dtype=np.uint32)
    with pytest.raises(ValueError, match="Invalid spectrum_type"):
        IMZMLWriter(tmp_path / "bad_spectrum", spectrum_type="raw")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid ibd_mode"):
        IMZMLWriter(tmp_path / "bad_mode", ibd_mode="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid on_error"):
        IMZMLWriter(tmp_path / "bad_error", on_error="skip")  # type: ignore[arg-type]

    with IMZMLWriter(tmp_path / "continuous", ibd_mode="continuous") as writer:
        writer.add_spectrum([1.0, 2.0], [1.0, 2.0], (1, 1))
        with pytest.raises(ValueError, match="identical"):
            writer.add_spectrum([1.0, 3.0], [1.0, 2.0], (2, 1))

    writer = IMZMLWriter(tmp_path / "closed")
    writer.add_spectrum([1.0], [1.0], (1, 1))
    writer.close()
    with pytest.raises(ValueError, match="closed"):
        writer.add_spectrum([1.0], [1.0], (1, 1))


def test_manual_warn_skips_invalid_spectrum(tmp_path: Path) -> None:
    """Skip invalid manual spectra when warnings are requested."""
    with IMZMLWriter(tmp_path / "manual_warn", on_error="warn") as writer:
        assert writer.add_spectrum([100.0], [1.0], (1, 1)) is True
        with pytest.warns(SkippedSpectrumWarning, match="Skipping spectrum"):
            assert writer.add_spectrum([101.0], [2.0], (2,)) is False
        assert writer.add_spectrum([102.0], [3.0], (2, 1)) is True

    reader = IMZMLReader(tmp_path / "manual_warn.imzML", parse_lib="ElementTree")
    assert reader.n_pixels == 2
    np.testing.assert_array_equal(reader.xyz_coordinates, np.asarray([[1, 1, 1], [2, 1, 1]]))
    np.testing.assert_array_equal(reader.get_tic(silent=True), np.asarray([1.0, 3.0]))


def test_manual_error_deletes_partial_files(tmp_path: Path) -> None:
    """Delete partial files when manual writing fails in error mode."""
    with pytest.raises(ValueError, match="Coordinates"):
        with IMZMLWriter(tmp_path / "manual_error") as writer:
            writer.add_spectrum([100.0], [1.0], (1, 1))
            writer.add_spectrum([101.0], [2.0], (2,))

    assert not (tmp_path / "manual_error.imzML").exists()
    assert not (tmp_path / "manual_error.ibd").exists()


def test_reader_warn_skips_bad_pixel(tmp_path: Path) -> None:
    """Skip unreadable reader pixels when warnings are requested."""
    spectra = [
        (np.asarray([100.0]), np.asarray([1.0], dtype=np.float32)),
        (np.asarray([101.0]), np.asarray([2.0], dtype=np.float32)),
        (np.asarray([102.0]), np.asarray([3.0], dtype=np.float32)),
    ]
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    reader = DummyReader(spectra, coordinates, failing_indices={1})

    with pytest.warns(SkippedSpectrumWarning, match="pixel 1"):
        output_path = write_imzml(reader, tmp_path / "reader_warn", on_error="warn", silent=True)
    exported = IMZMLReader(output_path, parse_lib="ElementTree")

    assert exported.n_pixels == 2
    np.testing.assert_array_equal(exported.xyz_coordinates, np.asarray([[1, 1, 1], [3, 1, 1]]))
    np.testing.assert_array_equal(exported.get_tic(silent=True), np.asarray([1.0, 3.0]))


def test_reader_error_deletes_partial_files(tmp_path: Path) -> None:
    """Delete partial files when reader export fails in error mode."""
    spectra = [
        (np.asarray([100.0]), np.asarray([1.0], dtype=np.float32)),
        (np.asarray([101.0]), np.asarray([2.0], dtype=np.float32)),
    ]
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0]])
    reader = DummyReader(spectra, coordinates, failing_indices={1})

    with pytest.raises(RuntimeError, match="pixel 1"):
        write_imzml(reader, tmp_path / "reader_error", silent=True)

    assert not (tmp_path / "reader_error.imzML").exists()
    assert not (tmp_path / "reader_error.ibd").exists()


def test_warn_all_skipped_deletes_partial_files(tmp_path: Path) -> None:
    """Reject zero-spectrum files after all spectra are skipped."""
    with pytest.raises(ValueError, match="without any spectra"):
        with IMZMLWriter(tmp_path / "empty_warn", on_error="warn") as writer:
            with pytest.warns(SkippedSpectrumWarning, match="Skipping spectrum"):
                writer.add_spectrum([100.0], [1.0], (1,))

    assert not (tmp_path / "empty_warn.imzML").exists()
    assert not (tmp_path / "empty_warn.ibd").exists()


def test_overwrite_false_refuses_existing_outputs(tmp_path: Path) -> None:
    """Refuse to overwrite existing output files by default."""
    imzml_path = tmp_path / "existing.imzML"
    ibd_path = tmp_path / "existing.ibd"
    imzml_path.write_text("old xml", encoding="utf-8")
    ibd_path.write_bytes(b"old ibd")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        IMZMLWriter(tmp_path / "existing")

    assert imzml_path.read_text(encoding="utf-8") == "old xml"
    assert ibd_path.read_bytes() == b"old ibd"
    assert not list(tmp_path.glob(".*existing*.tmp"))


def test_overwrite_true_replaces_existing_outputs(tmp_path: Path) -> None:
    """Replace existing output files only after a successful write."""
    imzml_path = tmp_path / "replace.imzML"
    ibd_path = tmp_path / "replace.ibd"
    imzml_path.write_text("old xml", encoding="utf-8")
    ibd_path.write_bytes(b"old ibd")

    with IMZMLWriter(tmp_path / "replace", overwrite=True) as writer:
        writer.add_spectrum([100.0], [1.0], (1, 1))

    exported = IMZMLReader(imzml_path, parse_lib="ElementTree")
    assert exported.n_pixels == 1
    assert imzml_path.read_text(encoding="ISO-8859-1") != "old xml"
    assert ibd_path.read_bytes() != b"old ibd"


def test_atomic_failure_preserves_existing_outputs(tmp_path: Path) -> None:
    """Leave existing final files untouched when overwrite export fails."""
    output_path = tmp_path / "atomic"
    with IMZMLWriter(output_path) as writer:
        writer.add_spectrum([100.0], [1.0], (1, 1))
    old_xml = output_path.with_suffix(".imzML").read_bytes()
    old_ibd = output_path.with_suffix(".ibd").read_bytes()

    with pytest.raises(RuntimeError, match="planned failure"):
        with IMZMLWriter(output_path, overwrite=True) as writer:
            writer.add_spectrum([101.0], [2.0], (2, 1))
            raise RuntimeError("planned failure")

    assert output_path.with_suffix(".imzML").read_bytes() == old_xml
    assert output_path.with_suffix(".ibd").read_bytes() == old_ibd
    assert not list(tmp_path.glob(".*atomic*.tmp"))


def test_writer_metadata_is_exported(tmp_path: Path) -> None:
    """Write core imzy metadata to the XML file."""
    source_path = tmp_path / "source.raw"
    with IMZMLWriter(
        tmp_path / "metadata",
        source_path=source_path,
        pixel_size=(20.5, 21.5),
        image_shape=(4, 5),
    ) as writer:
        writer.add_spectrum([100.0], [1.0], (1, 1))

    imzml_path = tmp_path / "metadata.imzML"
    root = ET.parse(imzml_path).getroot()
    software = root.find('.//mzml:software[@id="imzy"]', _MZML_NS)
    assert software is not None
    assert software.attrib["version"] == imzy.get_version()
    source_file = root.find(".//mzml:sourceFile", _MZML_NS)
    assert source_file is not None
    assert source_file.attrib["name"] == source_path.name
    assert source_file.attrib["location"] == str(source_path.parent)
    assert root.find('.//mzml:userParam[@name="imzy export timestamp"]', _MZML_NS) is not None
    assert _cv_value(imzml_path, "IMS:1000042") == "5"
    assert _cv_value(imzml_path, "IMS:1000043") == "4"
    assert _cv_value(imzml_path, "IMS:1000046") == "20.5"
    assert _cv_value(imzml_path, "IMS:1000047") == "21.5"


def test_write_imzml_indices_preserve_selection_order(tmp_path: Path) -> None:
    """Export only selected reader indices in the requested order."""
    spectra = [
        (np.asarray([100.0]), np.asarray([1.0], dtype=np.float32)),
        (np.asarray([101.0]), np.asarray([2.0], dtype=np.float32)),
        (np.asarray([102.0]), np.asarray([3.0], dtype=np.float32)),
    ]
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [2, 1, 0]])
    reader = DummyReader(spectra, coordinates)

    output_path = write_imzml(reader, tmp_path / "indices", indices=[2, 0], silent=True)
    exported = IMZMLReader(output_path, parse_lib="ElementTree")

    assert exported.n_pixels == 2
    np.testing.assert_array_equal(exported.xyz_coordinates, np.asarray([[3, 2, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(exported.get_spectrum(0)[0], spectra[2][0])
    np.testing.assert_array_equal(exported.get_spectrum(0)[1], spectra[2][1])
    np.testing.assert_array_equal(exported.get_spectrum(1)[0], spectra[0][0])
    np.testing.assert_array_equal(exported.get_spectrum(1)[1], spectra[0][1])


def test_write_imzml_indices_validation(tmp_path: Path) -> None:
    """Validate reader export indices."""
    spectra = [(np.asarray([100.0]), np.asarray([1.0], dtype=np.float32))]
    coordinates = np.asarray([[0, 0, 0]])
    reader = DummyReader(spectra, coordinates)

    with pytest.raises(ValueError, match="within"):
        write_imzml(reader, tmp_path / "bad_index", indices=[1], silent=True)
    with pytest.raises(ValueError, match="integers"):
        write_imzml(reader, tmp_path / "bad_float_index", indices=[0.5], silent=True)
