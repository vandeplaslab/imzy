"""Low-level parsing of PHI SmartSoft-TOF `.raw` files (ToF-SIMS imaging, nanoTOF instruments).

Ported from Thyra (https://github.com/M4i-Imaging-Mass-Spectrometry/thyra, `thyra/readers/phi`, MIT License,
Copyright (c) 2025 Theodoros Visvikis).

File format
-----------
The file opens with a CRLF-delimited ASCII header bracketed by `SOFH` and `EOFH`, zero-padded to the byte count the
header declares in `HeaderSize`. Top-level entries use `Key: value`; entries below a `[Section]` marker use
`Key=value`.

Everything past `HeaderSize` is a chain of blocks, each introduced by a 16-byte header: `uint16` block id at offset 0
and `uint32` record size at offset 12. Block ids:

- 1 - ion events (8 bytes each); 8 instead when MS/MS is active
- 2 - end of frame, no payload
- 6 - mosaic tile origin, payload is two `uint32` tile indices
- 14 - appended ASCII info block, e.g. a recalibration done after acquisition
- 0 - end of stream

Any other id is skipped using its record size. Each ion event is a little-endian `uint64`: bit 31 is set on a valid
event, bits 0-30 hold the flight time in picoseconds, bits 32-42 the x pixel, bits 43-53 the y pixel and bits 58-63
are clear on a valid event. Events failing the validity test are dropped.

Flight times are binned onto the detector channel grid (`SpecBinSize`) and converted to m/z with
`sqrt(m/z) = (Mass/Time) * t_us + MassOffset`.
"""

from __future__ import annotations

import logging
import math
import re
import struct
import typing as ty
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SOFH_MAGIC = b"SOFH"
EOFH_MARKER = b"EOFH"
MAX_HEADER_PROBE = 1 << 20

BLOCK_HEADER_SIZE = 16
BLOCK_END = 0
BLOCK_FRAME_END = 2
BLOCK_TILE = 6
BLOCK_ASCII_INFO = 14
ASCII_INFO_LIMIT = 1 << 16

EVENT_SIZE = 8
_BIT31 = np.uint64(1 << 31)
_TOF_MASK = np.uint64(0x7FFFFFFF)
_COORD_MASK = np.uint64(0x7FF)
_X_SHIFT = np.uint64(32)
_Y_SHIFT = np.uint64(43)
_HIGH_SHIFT = np.uint64(58)

_SECTION_RE = re.compile(r"^\[(.+)\]$")
_TOP_LEVEL_RE = re.compile(r"^([A-Za-z][^:]*): ?(.*)$")
_TILE_STRUCT = struct.Struct("<II")


# ----------------------------------------------------------------------------------------------------------------------
# header
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PhiHeader:
    """Parsed PHI acquisition header.

    Attributes
    ----------
    entries : dict
        Top-level `Key: value` pairs, verbatim.
    sections : dict
        `[Section]` blocks mapping key to value, verbatim.
    header_size : int
        Byte offset at which the binary block stream begins.
    image_pixels : int
        Raster edge length in pixels (the raster is square).
    mass_slope : float
        `Mass/Time`, in sqrt(amu) per microsecond.
    mass_offset : float
        `MassOffset`, in sqrt(amu).
    start_flight_time_us, stop_flight_time_us : float
        Flight time range retained by the acquisition.
    bin_size_ns : float
        Detector time-channel width (`SpecBinSize`).
    start_mz, stop_mz : float
        Declared mass range (0 when not declared).
    polarity : str
        'positive' or 'negative'.
    n_frames : int
        Number of acquisition frames declared in the header.
    msms_active : bool
        Whether MS/MS was enabled; selects the event block id.
    raster_size_um : float, optional
        Raster edge length in micrometers.
    """

    entries: dict[str, str]
    sections: dict[str, dict[str, str]]
    header_size: int
    image_pixels: int
    mass_slope: float
    mass_offset: float
    start_flight_time_us: float
    stop_flight_time_us: float
    bin_size_ns: float
    start_mz: float
    stop_mz: float
    polarity: str
    n_frames: int
    msms_active: bool
    raster_size_um: float | None = None

    @property
    def data_block_id(self) -> int:
        """Block id carrying ion events: 8 under MS/MS, otherwise 1."""
        return 8 if self.msms_active else 1

    @property
    def pixel_size_um(self) -> float | None:
        """Pixel edge length in micrometers (`Raster Size (um) / ImagePixels`), or None if unknown.

        The header also carries a `Raster Size Calibration` factor which is deliberately not applied: it trims the scan
        generator so the raster achieves the requested pitch rather than scaling the field of view.
        """
        if not self.raster_size_um or self.image_pixels <= 0:
            return None
        return self.raster_size_um / self.image_pixels


def _to_float(value: str | None, default: float = 0.0) -> float:
    """Read a header number; anything that is not a finite number gives `default`."""
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _to_int(value: str | None, default: int = 0) -> int:
    """Read a header count; anything that is not a finite number gives `default`."""
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return default


def _split_header_text(text: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Split the header text into top-level entries and sections."""
    entries: dict[str, str] = {}
    sections: dict[str, dict[str, str]] = {}
    current = ""  # empty means "above the first [Section]"
    for line in text.split("\r\n"):
        stripped = line.strip()
        if not stripped:
            continue
        section = _SECTION_RE.match(stripped)
        if section is not None:
            current = section.group(1)
            sections.setdefault(current, {})
            continue
        if not current:
            match = _TOP_LEVEL_RE.match(line)
            if match is not None:
                entries[match.group(1)] = match.group(2).strip()
        elif "=" in line:
            key, _, value = line.partition("=")
            sections[current][key.strip()] = value.strip()
    return entries, sections


def _read_header_text(path: Path) -> str:
    """Read the ASCII header region (between `SOFH` and `EOFH`)."""
    with path.open("rb") as handle:
        probe = handle.read(MAX_HEADER_PROBE)
    if not probe.startswith(SOFH_MAGIC):
        raise ValueError(f"Not a PHI SmartSoft-TOF raw file (missing SOFH magic): {path}")
    end = probe.find(EOFH_MARKER)
    if end == -1:
        raise ValueError(f"Malformed PHI header, no EOFH terminator: {path}")
    return probe[:end].decode("latin-1")


def _parse_polarity(entries: dict[str, str], sections: dict[str, dict[str, str]]) -> str:
    """Read the secondary ion polarity from either place it is recorded."""
    text = sections.get("Polarity", {}).get("Polarity", entries.get("SecIonPolarity", ""))
    return "negative" if "-" in text or "Negative" in text else "positive"


def parse_header(path: Path) -> PhiHeader:
    """Parse and validate the ASCII header of a PHI `.raw` file.

    Raises
    ------
    ValueError
        If the file is not a PHI raw file, the header is unterminated or a required field is missing or invalid.
    """
    path = Path(path)
    entries, sections = _split_header_text(_read_header_text(path))
    acq = sections.get("Acq Base", {})
    lmig = sections.get("PHI LMIG", {})

    for name in ("HeaderSize", "ImagePixels", "Mass/Time", "MassOffset"):
        if name not in entries:
            raise ValueError(f"PHI header is missing required field '{name}': {path}")

    raster = _to_float(lmig.get("Raster Size (um)") or entries.get("ScanWidthX"))
    header = PhiHeader(
        entries=entries,
        sections=sections,
        header_size=_to_int(entries["HeaderSize"]),
        image_pixels=_to_int(entries["ImagePixels"]),
        mass_slope=_to_float(entries["Mass/Time"]),
        mass_offset=_to_float(entries["MassOffset"]),
        start_flight_time_us=_to_float(entries.get("StartFlightTime"), 0.0),
        stop_flight_time_us=_to_float(entries.get("StopFlightTime"), 0.0),
        bin_size_ns=_to_float(entries.get("SpecBinSize"), 0.128),
        start_mz=_to_float(acq.get("Start Mass (amu)"), 0.0),
        stop_mz=_to_float(acq.get("End Mass (amu)"), 0.0),
        polarity=_parse_polarity(entries, sections),
        n_frames=_to_int(entries.get("NoFrames") or acq.get("Frames"), 1),
        msms_active="Inactive" not in acq.get("MSMS Active", "Inactive"),
        raster_size_um=raster or None,
    )
    if header.header_size <= 0:
        raise ValueError(f"PHI header declares a non-positive HeaderSize: {path}")
    if header.image_pixels <= 0:
        raise ValueError(f"PHI header declares a non-positive ImagePixels: {path}")
    if header.mass_slope <= 0:
        raise ValueError(f"PHI header declares a non-positive Mass/Time: {path}")
    if header.stop_flight_time_us <= header.start_flight_time_us:
        raise ValueError(
            f"PHI header flight-time range is empty: {header.start_flight_time_us} to {header.stop_flight_time_us} us"
            f" in {path}"
        )
    return header


# ----------------------------------------------------------------------------------------------------------------------
# block chain
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class DataSpan:
    """One run of ion events, with the mosaic tile origin in effect."""

    offset: int
    size: int
    tile: tuple[int, int] = (0, 0)

    @property
    def n_records(self) -> int:
        """Number of 8-byte event records in this span."""
        return self.size // EVENT_SIZE


@dataclass
class BlockIndex:
    """Result of walking the block chain without reading event payloads."""

    data_spans: list[DataSpan] = field(default_factory=list)
    n_frames: int = 0
    tiles: list[tuple[int, int]] = field(default_factory=list)
    appended: list[dict[str, str]] = field(default_factory=list)
    block_counts: dict[int, int] = field(default_factory=dict)
    clean: bool = True

    @property
    def n_records(self) -> int:
        """Total event records across every span, valid or not."""
        return sum(span.n_records for span in self.data_spans)

    @property
    def tile_grid(self) -> tuple[int, int]:
        """Mosaic tile counts `(nx, ny)` implied by the tile origins actually encountered."""
        if not self.tiles:
            return 1, 1
        return max(t[0] for t in self.tiles) + 1, max(t[1] for t in self.tiles) + 1

    def appended_calibration(self) -> dict[str, str] | None:
        """Return the newest appended info block with a usable `Mass/Time` and `MassOffset`, if any."""
        for info in reversed(self.appended):
            try:
                float(info["Mass/Time"])
                float(info["MassOffset"])
            except (KeyError, ValueError):
                continue
            return info
        return None


def _parse_ascii_info(payload: bytes) -> dict[str, str]:
    """Parse an appended ASCII info block into key/value pairs."""
    info: dict[str, str] = {}
    for line in payload.decode("latin-1", errors="replace").split("\r\n"):
        key, sep, value = line.partition(":")
        if sep and key and not key.startswith("\x00"):
            info[key.strip()] = value.strip()
    return info


def scan_blocks(path: Path, header: PhiHeader) -> BlockIndex:  # noqa: C901
    """Walk the block chain, recording its structure without reading the event payloads."""
    path = Path(path)
    file_size = path.stat().st_size
    data_id = header.data_block_id
    index = BlockIndex()
    tile = (0, 0)

    with path.open("rb") as handle:
        pos = header.header_size
        while pos + BLOCK_HEADER_SIZE <= file_size:
            handle.seek(pos)
            raw = handle.read(BLOCK_HEADER_SIZE)
            block_id = int.from_bytes(raw[0:2], "little")
            record_size = int.from_bytes(raw[12:16], "little")
            pos += BLOCK_HEADER_SIZE
            index.block_counts[block_id] = index.block_counts.get(block_id, 0) + 1

            if block_id == BLOCK_END:
                break
            if block_id == data_id:
                usable = min(record_size, file_size - pos)
                usable -= usable % EVENT_SIZE
                if usable > 0:
                    index.data_spans.append(DataSpan(pos, usable, tile))
                pos += record_size
            elif block_id == BLOCK_TILE:
                payload = handle.read(_TILE_STRUCT.size)
                if len(payload) < _TILE_STRUCT.size:
                    break
                tile = ty.cast(tuple[int, int], _TILE_STRUCT.unpack(payload))
                if tile not in index.tiles:
                    index.tiles.append(tile)
                pos += _TILE_STRUCT.size
            elif block_id == BLOCK_FRAME_END:
                index.n_frames += 1
            else:
                if block_id == BLOCK_ASCII_INFO and record_size:
                    info = _parse_ascii_info(handle.read(min(record_size, ASCII_INFO_LIMIT)))
                    if info:
                        index.appended.append(info)
                pos += record_size

    index.clean = pos == file_size
    if not index.clean:
        logger.warning(f"PHI block chain of '{path.name}' ended at byte {pos} of {file_size}.")
    if not index.data_spans:
        raise ValueError(f"No ion event blocks (id {data_id}) found in '{path}'. Is this a PHI imaging acquisition?")
    return index


def decode_events(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw `uint64` records into `(x, y, tof_ps)`, dropping invalid records."""
    valid = ((words >> _HIGH_SHIFT) == 0) & ((words & _BIT31) != 0)
    good = words[valid]
    return (
        ((good >> _X_SHIFT) & _COORD_MASK).astype(np.int64),
        ((good >> _Y_SHIFT) & _COORD_MASK).astype(np.int64),
        (good & _TOF_MASK).astype(np.int64),
    )


def iter_events(
    path: Path, index: BlockIndex, pixels_per_tile: int, max_records: int = 4_000_000
) -> ty.Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Stream decoded `(x, y, tof_ps, n_records)` batches with mosaic tile offsets applied.

    Adjacent event blocks sharing a tile origin are merged up to `max_records` records. Events past their own tile's
    raster get `x = -1` so the caller drops them as off-raster. `n_records` is the number of raw records in the batch,
    including the invalid ones that were dropped.
    """

    def _decode(chunks: list[np.ndarray], tile: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        words = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        x, y, tof = decode_events(words)
        off_tile = (x >= pixels_per_tile) | (y >= pixels_per_tile)
        x += tile[0] * pixels_per_tile
        y += tile[1] * pixels_per_tile
        x[off_tile] = -1
        return x, y, tof, words.size

    with Path(path).open("rb") as handle:
        pending: list[np.ndarray] = []
        pending_tile = (0, 0)
        pending_count = 0
        for span in index.data_spans:
            if pending and (span.tile != pending_tile or pending_count >= max_records):
                yield _decode(pending, pending_tile)
                pending, pending_count = [], 0
            handle.seek(span.offset)
            buf = handle.read(span.size)
            n = len(buf) // EVENT_SIZE
            if n == 0:
                continue
            pending.append(np.frombuffer(buf, dtype="<u8", count=n))
            pending_tile = span.tile
            pending_count += n
        if pending:
            yield _decode(pending, pending_tile)


# ----------------------------------------------------------------------------------------------------------------------
# mass axis
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PhiMassAxis:
    """Common m/z axis on the detector time-channel grid.

    Attributes
    ----------
    mz : np.ndarray
        Strictly increasing m/z values, one per retained channel.
    channel_offset : int
        Index of `mz[0]` within the full channel grid.
    bin_ps : float
        Channel width in picoseconds.
    start_ps : float
        Flight time of channel zero of the full grid.
    """

    mz: np.ndarray
    channel_offset: int
    bin_ps: float
    start_ps: float

    @property
    def tof_us(self) -> np.ndarray:
        """Flight time (microseconds) of each channel of `mz`."""
        channels = self.channel_offset + np.arange(self.mz.size, dtype=np.float64)
        return (self.start_ps + channels * self.bin_ps) * 1e-6

    def to_channel(self, tof_ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bin flight times (picoseconds) onto the axis, returning channel indices and a mask of in-axis events."""
        channel = np.rint((tof_ps - self.start_ps) / self.bin_ps).astype(np.int64) - self.channel_offset
        return channel, (channel >= 0) & (channel < self.mz.size)


def build_mass_axis(
    slope: float,
    offset: float,
    start_flight_time_us: float,
    stop_flight_time_us: float,
    bin_size_ns: float,
    start_mz: float = 0.0,
    stop_mz: float = 0.0,
) -> PhiMassAxis:
    """Build the common m/z axis from the calibration and the flight time range.

    Only channels with a positive root and within the declared mass range (when non-zero) are kept.
    """
    if bin_size_ns <= 0:
        raise ValueError(f"SpecBinSize must be positive, got {bin_size_ns}")
    bin_ps = bin_size_ns * 1000.0
    start_ps = start_flight_time_us * 1e6
    stop_ps = stop_flight_time_us * 1e6
    n_full = math.floor((stop_ps - start_ps) / bin_ps) + 1
    if n_full <= 0:
        raise ValueError(f"Flight time range {start_flight_time_us}-{stop_flight_time_us} us yields no channels.")

    t_us = (start_ps + np.arange(n_full, dtype=np.float64) * bin_ps) * 1e-6
    root = slope * t_us + offset
    usable = root > 0.0
    mz_full = np.where(usable, root * root, 0.0)
    if start_mz > 0:
        usable &= mz_full >= start_mz
    if stop_mz > 0:
        usable &= mz_full <= stop_mz
    hits = np.flatnonzero(usable)
    if hits.size == 0:
        raise ValueError(
            f"No time channel maps into the mass range {start_mz}-{stop_mz} with slope={slope}, offset={offset}."
        )
    lo, hi = int(hits[0]), int(hits[-1])
    mz = mz_full[lo : hi + 1]
    if not (np.all(np.isfinite(mz)) and np.all(np.diff(mz) > 0)):
        raise ValueError(f"Calibration Mass/Time={slope}, MassOffset={offset} does not give an increasing m/z axis.")
    return PhiMassAxis(mz=mz, channel_offset=lo, bin_ps=bin_ps, start_ps=start_ps)
