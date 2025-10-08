"""Init."""

from imzy._readers.bruker._tdf import TDFReader, is_tdf
from imzy._readers.bruker._tsf import TSFReader, is_tsf

# Since it's based on TSFReader, we import it here to make sure it's registered as well.
from imzy._readers.bruker._neoflex import NeoFlexReader, is_neoflex  # isort: skip

__all__ = ("NeoFlexReader", "TDFReader", "TSFReader", "is_neoflex", "is_tdf", "is_tsf")
