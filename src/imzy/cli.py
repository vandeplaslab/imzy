"""Command line interface for imzy."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import click

import imzy
from imzy._normalizations._extract import get_normalizations
from imzy._readers._base import BaseReader
from imzy._writers._imzml import IBD_MODE_AUTO, ON_ERROR_ERROR, SPECTRUM_TYPE_AUTO, IbdMode, OnError, SpectrumType

_BRUKER_ROI_READER_NAMES = {"NeoFlexReader", "TDFReader", "TSFReader"}
_BRUKER_TDF_READER_NAME = "TDFReader"
_NORMALIZATIONS = get_normalizations()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Run imzy commands."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--spectrum-type",
    type=click.Choice(["auto", "centroid", "profile"]),
    default=SPECTRUM_TYPE_AUTO,
    show_default=True,
    help="Spectrum mode to write.",
)
@click.option(
    "--normalization",
    type=click.Choice(_NORMALIZATIONS, case_sensitive=False),
    default=None,
    metavar="NAME",
    help="Normalization multiplier to apply by name. If not specified, no normalization is applied.",
)
@click.option("--roi", type=int, default=None, help="Bruker TSF/TDF region of interest to export.")
@click.option(
    "--ibd-mode",
    type=click.Choice(["auto", "continuous", "processed"], case_sensitive=False),
    default=IBD_MODE_AUTO,
    show_default=True,
    help="imzML binary layout mode.",
)
@click.option("--overwrite/--no-overwrite", default=False, show_default=True, help="Replace existing imzML outputs.")
@click.option(
    "--on-error",
    type=click.Choice(["error", "warn"], case_sensitive=False),
    default=ON_ERROR_ERROR,
    show_default=True,
    help="How to handle unreadable spectra.",
)
@click.option("--silent/--progress", default=False, show_default=True, help="Hide progress bars.")
def convert(
    input_path: Path,
    output_path: Path,
    spectrum_type: str,
    normalization: str | None,
    roi: int | None,
    ibd_mode: str,
    overwrite: bool,
    on_error: str,
    silent: bool,
) -> None:
    """Convert a supported imaging mass spectrometry file to imzML."""
    reader = _get_reader(input_path, roi=roi)
    reader_name = reader.__class__.__name__
    if roi is not None and reader_name not in _BRUKER_ROI_READER_NAMES:
        raise click.ClickException("--roi is only supported for Bruker TSF/TDF inputs.")
    if reader_name == _BRUKER_TDF_READER_NAME:
        click.echo(
            "Warning: Bruker TDF ion mobility is not exported; spectra will be squashed to the m/z axis only.",
            err=True,
        )

    try:
        output = imzy.write_imzml(
            reader,
            output_path,
            ibd_mode=ty.cast(IbdMode, ibd_mode),
            spectrum_type=ty.cast(SpectrumType, spectrum_type),
            normalization=normalization,
            overwrite=overwrite,
            on_error=ty.cast(OnError, on_error),
            silent=silent,
        )
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()
    click.echo(str(output))


def _get_reader(input_path: Path, *, roi: int | None) -> BaseReader:
    if roi is None:
        return imzy.get_reader(input_path)

    reader = imzy.get_reader(input_path)
    reader_name = reader.__class__.__name__
    close = getattr(reader, "close", None)
    if close is not None:
        close()
    if reader_name not in _BRUKER_ROI_READER_NAMES:
        raise click.ClickException("--roi is only supported for Bruker TSF/TDF inputs.")
    return imzy.get_reader(input_path, roi=roi)
