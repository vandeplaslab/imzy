"""Waters
MassLynx Python SDK.
"""

from ctypes import POINTER, c_byte, c_float, c_int, c_void_p, cast

import numpy as np

from imzy._readers.waters.MassLynxRawReader import MassLynxBaseType, MassLynxRawProcessor, MassLynxRawReader, DLL


class MassLynxRawScanReader(MassLynxRawReader):
    """Read masslynx scan data."""

    def __init__(self, source):
        super().__init__(source, MassLynxBaseType.SCAN)

    def ReadScan(self, whichFunction, whichScan):
        #         masses = []
        #         intensities = []

        # create the retrun values
        size = c_int(0)
        pMasses = c_void_p()
        pIntensities = c_void_p()

        # read scan
        readScan = DLL.readScan
        readScan.argtypes = [c_void_p, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p), POINTER(c_int)]
        super().CheckReturnCode(readScan(self._getReader(), whichFunction, whichScan, pMasses, pIntensities, size))

        # fill the array
        pM = cast(pMasses, POINTER(c_float))
        pI = cast(pIntensities, POINTER(c_float))

        masses = pM[0 : size.value]
        intensities = pI[0 : size.value]

        # dealocate memory
        # MassLynxRawReader.ReleaseMemory( pMasses)
        # MassLynxRawReader.ReleaseMemory( pIntensities)

        return np.asarray(masses), np.asarray(intensities)

    def CombineScan(self, whichFunction, scans):
        # create the retrun values
        size = c_int(0)
        pMasses = c_void_p()
        pIntensities = c_void_p()

        # Create the input data
        nScans = len(scans)
        pScans = (c_int * len(scans))(*scans)

        self.processor = MassLynxRawProcessor(self)
        # read create the function and arguments
        combineScan = DLL.combineScan
        combineScan.argtypes = [c_void_p, c_int, POINTER(c_int), c_int]
        # run combinescan and check errors
        out = combineScan(self.processor._getProcessor(), whichFunction, pScans, nScans)
        self.processor._codeHandler.CheckReturnCode(out)

        # combineScan.argtypes = [c_void_p, c_int, c_int]
        # out2 = combineScan(self.processor._getProcessor(), whichFunction, 1)
        # print(out2)
        # self.processor._codeHandler.CheckReturnCode(out2)

        # Get scan from the combined
        getScan = DLL.getScan
        getScan.argtypes = [c_void_p, POINTER(c_void_p), POINTER(c_void_p), POINTER(c_int)]
        out3 = getScan(self.processor._getProcessor(), pMasses, pIntensities, size)
        self.processor._codeHandler.CheckReturnCode(out3)

        # fill the array
        pM = cast(pMasses, POINTER(c_float))
        pI = cast(pIntensities, POINTER(c_float))

        masses = pM[0 : size.value]
        intensities = pI[0 : size.value]

        # dealocate memory
        # MassLynxRawReader.ReleaseMemory( pMasses)
        # MassLynxRawReader.ReleaseMemory( pIntensities)

        return np.asarray(masses), np.asarray(intensities)

    def ReadScanFlags(self, whichFunction, whichScan):
        flags = []

        # create the retrun values
        size = c_int(0)
        pMasses = c_void_p()
        pIntensities = c_void_p()
        pFlags = c_void_p()

        # read scan
        readScanFlags = DLL.readScanFlags
        readScanFlags.argtypes = [
            c_void_p,
            c_int,
            c_int,
            POINTER(c_void_p),
            POINTER(c_void_p),
            POINTER(c_void_p),
            POINTER(c_int),
        ]
        super().CheckReturnCode(
            readScanFlags(self._getReader(), whichFunction, whichScan, pMasses, pIntensities, pFlags, size)
        )

        # fill the array
        pM = cast(pMasses, POINTER(c_float))
        pI = cast(pIntensities, POINTER(c_float))

        masses = pM[0 : size.value]
        intensities = pI[0 : size.value]

        # check for flags
        if pFlags.value is not None:
            pF = cast(pFlags, POINTER(c_byte))
            flags = pF[0 : size.value]

        return np.asarray(masses), np.asarray(intensities), flags

    def ReadDriftScan(self, whichFunction, whichScan, whichDrift):
        # create the return values
        size = c_int(0)
        pMasses = c_void_p()
        pIntensities = c_void_p()

        # read scan
        readDriftScan = DLL.readDriftScan
        readDriftScan.argtypes = [c_void_p, c_int, c_int, c_int, POINTER(c_void_p), POINTER(c_void_p), POINTER(c_int)]
        super().CheckReturnCode(
            readDriftScan(self._getReader(), whichFunction, whichScan, whichDrift, pMasses, pIntensities, size)
        )

        # fill the array
        pM = cast(pMasses, POINTER(c_float))
        pI = cast(pIntensities, POINTER(c_float))

        masses = pM[0 : size.value]
        intensities = pI[0 : size.value]

        # dealocate memory
        # MassLynxRawReader.ReleaseMemory( pMasses)
        # MassLynxRawReader.ReleaseMemory( pIntensities)

        return np.asarray(masses), np.asarray(intensities)
