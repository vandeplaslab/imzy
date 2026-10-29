"""Parser for SCiLS Lab `.slx` files.

An SLX file is a SQLite database that accompanies the `.sbd` (HDF5) file. It is *not* needed to extract ion images - the
SBD file is self-sufficient - but it carries useful acquisition metadata (polarity, instrument method, source path...).

The `MetaInformation` table stores each object's metadata as a single string made of concatenated length-prefixed
tokens (`<length>:<text>`), alternating between keys and values, e.g.::

    4:Date17:25-10-09 09:37:2319:Source/BaseGeometry20:MTP Slide Adapter II

The length is the number of **UTF-8 bytes** (not characters), so the string must be parsed as bytes.
"""

from __future__ import annotations

import sqlite3
import typing as ty
from pathlib import Path

from koyo.typing import PathLike

SQLITE_MAGIC = b"SQLite format 3\x00"


def is_sqlite(path: PathLike) -> bool:
    """Check whether file starts with the SQLite magic header."""
    path = Path(path)
    try:
        with open(path, "rb") as f:
            return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def parse_length_prefixed(data: bytes | str | None) -> dict[str, str]:
    """Parse SCiLS length-prefixed key/value string into a dictionary.

    Parsing stops silently at the first malformed token and returns whatever was parsed up to that point.
    """
    if not data:
        return {}
    if isinstance(data, str):
        data = data.encode("utf-8")
    tokens: list[str] = []
    pos, n = 0, len(data)
    while pos < n:
        sep = data.find(b":", pos)
        if sep == -1:
            break
        length_bytes = data[pos:sep]
        if not length_bytes.isdigit():
            break
        length = int(length_bytes)
        end = sep + 1 + length
        if end > n:
            break
        tokens.append(data[sep + 1 : end].decode("utf-8", errors="replace"))
        pos = end
    return dict(zip(tokens[0::2], tokens[1::2], strict=False))


def _text(value: ty.Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def read_slx_metadata(path: PathLike) -> dict[str, ty.Any]:
    """Read metadata from SCiLS `.slx` file.

    The file is opened read-only. If the file cannot be read or has an unexpected layout, an empty dictionary (or
    partial result) is returned rather than raising, because this metadata is optional.

    Returns
    -------
    metadata : dict
        Dictionary with the following keys: `master` (key/value pairs), `data_sources` (list of dicts with
        `id`, `source_path`, `original_import_path`, `type`, `polarity`, `spot_name_format` and parsed `meta`),
        `rasters` (list of dicts with raster indices and x/y offsets) and `meta` (parsed MetaInformation per object id).
    """
    path = Path(path)
    result: dict[str, ty.Any] = {"master": {}, "data_sources": [], "rasters": [], "meta": {}}
    if not path.exists() or not is_sqlite(path):
        return result
    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return result
    conn.text_factory = bytes  # some SCiLS strings are not valid UTF-8
    try:
        for name, query, handler in (
            ("master", "SELECT KeyColumn, ValueColumn FROM Master", _handle_master),
            ("meta", "SELECT ID, MetaInformationString FROM MetaInformation", _handle_meta),
            (
                "data_sources",
                "SELECT ID, SourcePath, OriginalImportPath, Type, Polarity, SpotNameFormat FROM DataSources",
                _handle_data_sources,
            ),
            (
                "rasters",
                "SELECT ID, DataSourceId, ScilsRasterIdx, OriginalRasterIdx, RasterXOffset, RasterYOffset FROM Rasters",
                _handle_rasters,
            ),
        ):
            try:
                rows = conn.execute(query).fetchall()
            except sqlite3.Error:
                continue
            result[name] = handler(rows)
    finally:
        conn.close()

    # attach parsed metadata to the data sources
    for source in result["data_sources"]:
        source["meta"] = result["meta"].get(source["id"], {})
    return result


def _handle_master(rows: list[tuple]) -> dict[str, str]:
    return {_text(k): _text(v) for k, v in rows}


def _handle_meta(rows: list[tuple]) -> dict[str, dict[str, str]]:
    meta = {}
    for id_, string in rows:
        id_ = _text(id_)
        if id_ == "secure":  # opaque (encrypted) blob
            continue
        meta[id_] = parse_length_prefixed(string)
    return meta


def _handle_data_sources(rows: list[tuple]) -> list[dict[str, ty.Any]]:
    keys = ("id", "source_path", "original_import_path", "type", "polarity", "spot_name_format")
    return [dict(zip(keys, map(_text, row), strict=True)) for row in rows]


def _handle_rasters(rows: list[tuple]) -> list[dict[str, ty.Any]]:
    def _num(value: ty.Any) -> float | int | None:
        try:
            number = float(_text(value))
        except ValueError:
            return None
        return int(number) if number.is_integer() else number

    return [
        {
            "id": _text(id_),
            "data_source_id": _text(ds),
            "scils_raster_index": _num(scils_idx),
            "original_raster_index": _num(orig_idx),
            "x_offset": _num(x_off),
            "y_offset": _num(y_off),
        }
        for id_, ds, scils_idx, orig_idx, x_off, y_off in rows
    ]
