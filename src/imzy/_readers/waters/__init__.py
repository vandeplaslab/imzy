"""Waters reader package."""

from koyo.system import IS_WIN

# the MassLynx SDK (MassLynxRaw.dll) is loaded on import and only exists for Windows
if IS_WIN:
    from ._raw import WatersReader, is_waters
else:
    WatersReader = is_waters = None  # type: ignore[misc,assignment]

__all__ = ("WatersReader", "is_waters")
