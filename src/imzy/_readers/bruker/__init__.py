"""Init."""

from koyo.system import IS_MAC, is_installed

# BAF data needs Bruker's baf2sql library, which only exists for Windows/Linux (it's loaded on first use)
from imzy._readers.bruker._baf import HAS_BAF_BACKEND

# rapifleX data is read with pure Python, so it's available everywhere
from imzy._readers.bruker._rapiflex import RapiflexReader, is_rapiflex

# Bruker ships no timsdata library for macOS, so there we use the pure-Python `pytsf` package when it's installed.
HAS_BRUKER_BACKEND = not IS_MAC or is_installed("pytsf")

if not IS_MAC:
    from imzy._readers.bruker import _tdf, _tsf
    from imzy._readers.bruker._tdf import TDFReader, is_tdf
    from imzy._readers.bruker._tsf import TSFReader, is_tsf
elif HAS_BRUKER_BACKEND:
    from imzy._readers.bruker import _tdf_pytsf as _tdf
    from imzy._readers.bruker import _tsf_pytsf as _tsf
    from imzy._readers.bruker._tdf_pytsf import TDFReader, is_tdf
    from imzy._readers.bruker._tsf_pytsf import TSFReader, is_tsf
else:
    _tdf = _tsf = TDFReader = TSFReader = is_tdf = is_tsf = None  # type: ignore[assignment,misc]

if HAS_BRUKER_BACKEND:
    # Since it's based on TSFReader, we import it here to make sure it's registered as well.
    from imzy._readers.bruker import _neoflex  # isort: skip
    from imzy._readers.bruker._neoflex import NeoFlexReader, is_neoflex  # isort: skip
else:
    _neoflex = NeoFlexReader = is_neoflex = None  # type: ignore[assignment,misc]

if HAS_BAF_BACKEND:
    from imzy._readers.bruker._baf import BAFReader, is_baf  # isort: skip
else:
    BAFReader = is_baf = None  # type: ignore[assignment,misc]

__all__ = (
    "HAS_BAF_BACKEND",
    "HAS_BRUKER_BACKEND",
    "BAFReader",
    "NeoFlexReader",
    "RapiflexReader",
    "TDFReader",
    "TSFReader",
    "is_baf",
    "is_neoflex",
    "is_rapiflex",
    "is_tdf",
    "is_tsf",
)
