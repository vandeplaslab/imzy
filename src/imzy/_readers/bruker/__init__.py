"""Init."""

from importlib.util import find_spec

from koyo.system import IS_MAC

# Bruker ships no timsdata library for macOS, so there we use the pure-Python `pytsf` package when it's installed.
HAS_BRUKER_BACKEND = not IS_MAC or find_spec("pytsf") is not None

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

__all__ = ("HAS_BRUKER_BACKEND", "NeoFlexReader", "TDFReader", "TSFReader", "is_neoflex", "is_tdf", "is_tsf")
