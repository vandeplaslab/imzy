"""Tests for the imzy command line interface."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from imzy.cli import main


def _copy_imzml_dataset(path: Path, tmp_path: Path) -> Path:
    """Copy an imzML dataset to a temporary path."""
    imzml_path = tmp_path / path.name
    shutil.copyfile(path, imzml_path)
    shutil.copyfile(path.with_suffix(".ibd"), imzml_path.with_suffix(".ibd"))
    cache_path = path.with_suffix(".icache")
    if cache_path.exists():
        shutil.copyfile(cache_path, imzml_path.with_suffix(".icache"))
    return imzml_path


def test_cli_help_includes_convert_command() -> None:
    """Show the convert command in top-level help."""
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "convert" in result.output


def test_cli_rejects_invalid_normalization(tmp_path: Path) -> None:
    """Reject normalization names outside the available registry."""
    input_path = tmp_path / "input.imzML"
    input_path.write_text("", encoding="utf-8")

    result = CliRunner().invoke(main, ["convert", str(input_path), str(tmp_path / "out"), "--normalization", "bad"])

    assert result.exit_code != 0
    assert "Invalid normalization" in result.output


def test_cli_rejects_roi_for_non_bruker_input(tmp_path: Path) -> None:
    """Reject ROI selection for inputs that do not support Bruker ROI metadata."""
    input_path = _copy_imzml_dataset(Path(__file__).parent / "_test_data" / "simple_imzml.imzML", tmp_path)

    result = CliRunner().invoke(main, ["convert", str(input_path), str(tmp_path / "out"), "--roi", "1"])

    assert result.exit_code != 0
    assert "--roi is only supported" in result.output


def test_cli_converts_imzml_to_imzml(tmp_path: Path) -> None:
    """Convert a supported input to imzML through the CLI."""
    input_path = _copy_imzml_dataset(Path(__file__).parent / "_test_data" / "simple_imzml.imzML", tmp_path)
    output_path = tmp_path / "converted"

    result = CliRunner().invoke(main, ["convert", str(input_path), str(output_path), "--silent"])

    assert result.exit_code == 0
    assert (tmp_path / "converted.imzML").exists()
    assert (tmp_path / "converted.ibd").exists()
    assert str(tmp_path / "converted.imzML") in result.output


def test_cli_passes_roi_to_bruker_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reopen Bruker TSF/TDF readers with the requested ROI."""
    input_path = tmp_path / "input.d"
    input_path.mkdir()
    calls: list[dict[str, int]] = []

    class TDFReader:
        """Fake Bruker reader for CLI dispatch tests."""

        def close(self) -> None:
            """Close the fake reader."""

    def fake_get_reader(path: Path, **kwargs: int) -> TDFReader:
        calls.append(kwargs)
        return TDFReader()

    def fake_write_imzml(reader: TDFReader, output_path: Path, **kwargs: object) -> Path:
        return output_path.with_suffix(".imzML")

    monkeypatch.setattr("imzy.cli.imzy.get_reader", fake_get_reader)
    monkeypatch.setattr("imzy.cli.imzy.write_imzml", fake_write_imzml)

    result = CliRunner().invoke(main, ["convert", str(input_path), str(tmp_path / "out"), "--roi", "2", "--silent"])

    assert result.exit_code == 0
    assert calls == [{}, {"roi": 2}]
    assert "ion mobility is not exported" in result.output
