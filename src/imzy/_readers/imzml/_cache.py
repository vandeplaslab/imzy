"""Versioned, source-checked imzML acquisition caches."""

from __future__ import annotations

import json
import os
import tempfile
import typing as ty
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from imzy._readers.imzml._metadata import Metadata

CACHE_VERSION = 1


def fingerprint(path: Path, ibd_path: Path) -> str:
    """Identify source paths and filesystem versions without hashing binary data."""
    return json.dumps(
        [(str(source.resolve()), source.stat().st_size, source.stat().st_mtime_ns) for source in (path, ibd_path)]
    )


def load_cache(cache_path: Path, source_fingerprint: str) -> Metadata | None:
    """Load a current cache, or request reparsing of stale, legacy, or corrupt data."""
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if int(cache["cache_version"].item()) != CACHE_VERSION:
                return None
            if str(cache["source_fingerprint"].item()) != source_fingerprint:
                return None
            header = json.loads(str(cache["metadata_json"].item()))
            offsets = cache["byte_offsets"]
            coordinates = cache["xyz_coordinates"]
            spans = cache["spans"]
            if (
                offsets.dtype != np.dtype("int64")
                or offsets.ndim != 2
                or offsets.shape[1] != 4
                or coordinates.dtype != np.dtype("int64")
                or coordinates.shape != (len(offsets), 3)
                or spans.dtype != np.dtype("int64")
                or spans.ndim != 2
                or spans.shape[1] != 6
                or np.any(coordinates < 1)
            ):
                return None
            base = tuple(header["coordinate_base"])
            if len(base) != 3 or any(value not in (0, 1) for value in base[:2]) or base[2] not in (0, 1, None):
                return None
            metadata = Metadata(
                None,
                header["mz_precision"],
                header["int_precision"],
                offsets,
                coordinates,
                spans,
                header["kinds"],
                header["values"],
                header["mode"],
                header["mz_min"],
                header["mz_max"],
                ty.cast(tuple[int, int, int | None], base),
                header["uuid"],
            )
            if metadata.mz_precision not in {"f", "d", "i", "l"} or metadata.int_precision not in {"f", "d", "i", "l"}:
                return None
            if metadata.mode not in (None, "centroid", "profile"):
                return None
            if len(spans) and (np.any(spans[:, 2] < 0) or np.any(spans[:, 2] >= len(metadata.kinds))):
                return None
            if not _valid_physical_metadata(metadata):
                return None
            return metadata
    except (OSError, ValueError, TypeError, KeyError, IndexError, EOFError, BadZipFile):
        return None


def _valid_physical_metadata(metadata: Metadata) -> bool:
    for i, axis in enumerate(("x", "y")):
        dimension = metadata.values[f"max count of pixels {axis}"]
        if not isinstance(dimension, int) or dimension < 1 or np.any(metadata.coordinates[:, i] > dimension):
            return False
        size = metadata.values[f"pixel size {axis}"]
        if size is not None and (not np.isfinite(size) or size <= 0):
            return False
        if not isinstance(metadata.values[f"pixel size {axis} reason"], str):
            return False
    for value in (metadata.mz_min, metadata.mz_max):
        if value is not None and (not isinstance(value, (float, int)) or not np.isfinite(value)):
            return False
    return not (metadata.mz_min is not None and metadata.mz_max is not None and metadata.mz_min > metadata.mz_max)


def save_cache(cache_path: Path, metadata: Metadata, source_fingerprint: str) -> None:
    """Atomically replace a complete acquisition cache using a unique temporary file."""
    header = {
        "mz_precision": metadata.mz_precision,
        "int_precision": metadata.int_precision,
        "values": metadata.values,
        "mode": metadata.mode,
        "mz_min": metadata.mz_min,
        "mz_max": metadata.mz_max,
        "coordinate_base": metadata.coordinate_base,
        "uuid": metadata.uuid,
        "kinds": metadata.kinds,
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=f".{cache_path.name}.", delete=False) as stream:
            temp_path = Path(stream.name)
            np.savez(
                stream,
                cache_version=CACHE_VERSION,
                source_fingerprint=source_fingerprint,
                metadata_json=json.dumps(header),
                byte_offsets=metadata.offsets,
                xyz_coordinates=metadata.coordinates,
                spans=metadata.spans,
                # Retain scalar keys used by existing cache inspection tools.
                mz_precision=metadata.mz_precision,
                int_precision=metadata.int_precision,
                mz_min=np.nan if metadata.mz_min is None else metadata.mz_min,
                mz_max=np.nan if metadata.mz_max is None else metadata.mz_max,
                spectrum_mode=metadata.mode or "",
            )
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
