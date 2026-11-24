import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def get_imzml_data(pattern: str = "*.imzML") -> list[Path]:
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))


def get_tsf_data(pattern: str = "*_tsf.d") -> list[Path]:
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))


def get_tdf_data(pattern: str = "*_tdf.d"):
    """Get data from the `_test_data` folder."""
    test_dir = Path(__file__).parent / "_test_data"
    return list(test_dir.glob(pattern))


# ---------------------------------------------------------------------------------------------------------------------
# SCiLS synthetic data
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class SyntheticSCiLS:
    """Synthetic SCiLS dataset together with the dense ground truth it was generated from."""

    sbd_path: Path
    slx_path: Path
    dense: np.ndarray  # (n_spots, n_bins) intensities
    mz_axis: np.ndarray
    x: np.ndarray  # raw coordinates
    y: np.ndarray
    rasters: np.ndarray
    norm_values: dict[str, np.ndarray]
    norm_targets: dict[str, float]
    spectra_mz: list[np.ndarray]  # precise m/z of the peaks of each spot (ascending, same order as the intensities)


def _encode_deltas(deltas: np.ndarray, width: int | None = None) -> tuple[bytes, int]:
    """Byte-plane shuffle (least-significant bytes first) unsigned integers using `width` (default: minimal) bytes."""
    peak = int(deltas.max()) if len(deltas) else 0
    needed = max(1, -(-peak.bit_length() // 8))
    width = needed if width is None else width
    assert width >= needed, "width is too small for the deltas"
    columns = deltas.astype("<u8").view(np.uint8).reshape(-1, 8)[:, :width]
    return np.ascontiguousarray(columns.T).tobytes(), width


def encode_spot_indices(spots: np.ndarray, width: int | None = None) -> bytes:
    """Encode sorted spot indices the way SCiLS does: delta -> byte-planes -> zlib."""
    data, _ = _encode_deltas(np.diff(spots.astype(np.int64), prepend=0), width)
    return zlib.compress(data)


def encode_mz(mz: np.ndarray, scale: float = 500_000.0, width: int | None = None) -> bytes:
    """Encode ascending m/z values the way SCiLS does: fixed-point deltas -> byte-planes -> zlib + 17-byte trailer."""
    mz = np.asarray(mz, dtype=np.float64)
    quantized = np.rint((mz - mz[0]) * scale).astype(np.int64)
    data, width = _encode_deltas(np.diff(quantized, prepend=quantized[0]), width)
    return zlib.compress(data) + struct.pack("<Bdd", width, scale, mz[0])


def encode_length_prefixed(items: dict[str, str]) -> str:
    """Encode key/value pairs as SCiLS' length-prefixed (UTF-8 byte count) string."""
    return "".join(f"{len(k.encode())}:{k}{len(v.encode())}:{v}" for k, v in items.items())


def write_synthetic_scils(
    directory: Path,
    name: str = "synthetic",
    n_rasters: int = 1,
    with_tic_normalization: bool = True,
    seed: int = 0,
    empty_spots: tuple[int, ...] = (),
) -> SyntheticSCiLS:
    """Write a small SCiLS .sbd (HDF5) + .slx (SQLite) pair with random sparse data on an irregular grid."""
    import sqlite3

    import h5py

    rng = np.random.default_rng(seed)
    width, height, n_bins = 30, 20, 800
    yy, xx = np.mgrid[:height, :width]
    keep = rng.random(width * height) > 0.15
    x = (xx.ravel()[keep] + 2513).astype(np.int64)
    y = (yy.ravel()[keep] + 721).astype(np.int64)
    n_spots = len(x)
    rasters = np.repeat(np.arange(n_rasters), int(np.ceil(n_spots / n_rasters)))[:n_spots]
    mz_axis = np.linspace(300.0, 600.0, n_bins)

    # sparse intensities; vary bin density so that both 1- and 2-byte deltas are produced
    density = rng.choice([0.002, 0.02, 0.3], size=n_bins, p=[0.3, 0.4, 0.3])
    dense = (rng.random((n_spots, n_bins)) < density).astype(np.float32)
    dense *= rng.integers(1, 5000, size=(n_spots, n_bins)).astype(np.float32)
    dense[list(empty_spots)] = 0

    # image-major
    spot_chunks, spot_offsets, intensity_offsets, image_intensities = [], [], [], []
    spot_total = intensity_total = 0
    for j in range(n_bins):
        spots = np.flatnonzero(dense[:, j])
        spot_offsets.append(spot_total)
        intensity_offsets.append(intensity_total)
        if len(spots):
            chunk = encode_spot_indices(spots)
            spot_chunks.append(chunk)
            spot_total += len(chunk)
            image_intensities.append(dense[spots, j])
            intensity_total += len(spots)
    compressed = np.frombuffer(b"".join(spot_chunks), dtype=np.int8)
    image_intensities = np.concatenate(image_intensities).astype(np.float32)

    # spectrum-major: precise m/z = axis value + jitter of less than half a bin, so the nearest bin is the true bin
    spacing = mz_axis[1] - mz_axis[0]
    mz_blocks, spectra_mz, mz_offsets_spot = [], [], []
    mz_total = 0
    for i in range(n_spots):
        bins = np.flatnonzero(dense[i])
        mz_offsets_spot.append(mz_total)
        if len(bins) == 0:
            spectra_mz.append(np.empty(0))
            continue
        mz = mz_axis[bins] + rng.uniform(-0.4, 0.4, len(bins)) * spacing
        block = encode_mz(mz)
        mz_blocks.append(block)
        mz_total += len(block)
        # what the reader is expected to return (fixed-point quantization of the deltas)
        quantized = np.rint((mz - mz[0]) * 500_000.0).astype(np.int64)
        spectra_mz.append(mz[0] + quantized / 500_000.0)
    compressed_mz = np.frombuffer(b"".join(mz_blocks), dtype=np.int8)
    intensities = np.concatenate([dense[i][dense[i] > 0] for i in range(n_spots)]).astype(np.float32)
    counts = (dense > 0).sum(axis=1)
    intensity_offsets_spot = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.uint64)

    # normalizations: Values = norm / target, as in SCiLS
    tic = dense.sum(axis=1).astype(np.float64)
    rms = np.sqrt((dense.astype(np.float64) ** 2).sum(axis=1))
    norm_targets = {"Total Ion Count": float(tic.mean()), "Root Mean Square": float(rms.mean())}
    norm_values = {
        "Total Ion Count": (tic / norm_targets["Total Ion Count"]).astype(np.float32),
        "Root Mean Square": (rms / norm_targets["Root Mean Square"]).astype(np.float32),
    }
    if not with_tic_normalization:
        norm_values.pop("Total Ion Count")
        norm_targets.pop("Total Ion Count")

    sbd_path = directory / f"{name}.sbd"
    with h5py.File(sbd_path, "w") as f:
        f.attrs["SpectraType"] = np.array([b"Peaks"])
        f.attrs["nSpots"] = np.array([str(n_spots).encode()])
        f.create_dataset("SamplePositions/GlobalMassAxis/SamplePositions", data=mz_axis)
        group = f.create_group("SpectraGroup")
        group.create_dataset("imageSpotOffsets", data=np.asarray(spot_offsets, dtype=np.uint64))
        group.create_dataset("imageIntensityOffsets", data=np.asarray(intensity_offsets, dtype=np.uint64))
        group.create_dataset("compressedImageSpots", data=compressed)
        group.create_dataset("imageIntensities", data=image_intensities)
        group.create_dataset("intensityOffsets", data=intensity_offsets_spot)
        group.create_dataset("intensities", data=intensities)
        group.create_dataset("mzOffsets", data=np.asarray(mz_offsets_spot, dtype=np.uint64))
        group.create_dataset("compressedMz", data=compressed_mz)
        for i, (norm_name, values) in enumerate(norm_values.items()):
            norm = group.create_group(f"Normalizations/{i:032x}")
            norm.attrs["name"] = np.array([norm_name.encode()])
            norm.create_dataset("Values", data=values)
            norm.create_dataset("TargetValue", data=np.array([norm_targets[norm_name]]))
        f.create_dataset("MeasurementRasterIndices", data=rasters.reshape(1, -1).astype(np.int32))
        registration = f.create_group("Registrations/0")
        registration.create_dataset("OriginalCoordinates", data=np.vstack([x, y]).astype(np.uint64))
        for r in range(n_rasters):
            matrix = np.eye(4)
            matrix[0, 0], matrix[1, 1] = 10.0, -10.0
            registration.create_dataset(f"Rasters/{r}/PixelToWorld", data=matrix)
        region = f.create_group("Regions/aaaa")
        region.attrs["Source/Polarity"] = np.array([b"negative"])

    slx_path = directory / f"{name}.slx"
    conn = sqlite3.connect(slx_path)
    conn.executescript(
        """
        CREATE TABLE Master (KeyColumn TEXT NOT NULL UNIQUE, ValueColumn TEXT NOT NULL);
        CREATE TABLE MetaInformation (ID TEXT NOT NULL UNIQUE, MetaInformationString TEXT);
        CREATE TABLE DataSources (ID TEXT NOT NULL UNIQUE, SourcePath TEXT NOT NULL, OriginalImportPath TEXT NOT NULL,
            Rasters BLOB, Type TEXT NOT NULL, Polarity TEXT NOT NULL, SpotNameFormat TEXT NOT NULL,
            RecalibrationObjId TEXT);
        CREATE TABLE Rasters (ID TEXT NOT NULL UNIQUE, DataSourceId TEXT NOT NULL, ScilsRasterIdx NUMERIC NOT NULL,
            OriginalRasterIdx NUMERIC NOT NULL, RasterXOffset NUMERIC, RasterYOffset NUMERIC);
        """
    )
    conn.execute("INSERT INTO Master VALUES ('Version', '10')")
    meta = {"Source/Description": "Bruker timsTOF fleX .tsf file", "Source/Polarity": "positive", "Unit/µm²": "Å²"}
    conn.execute("INSERT INTO MetaInformation VALUES ('ds1', ?)", (encode_length_prefixed(meta),))
    conn.execute("INSERT INTO MetaInformation VALUES ('secure', 'T0l1R24y')")
    conn.execute(
        "INSERT INTO DataSources VALUES ('ds1', ?, ?, NULL, 'TSFSource', 'positive', 'fmt', NULL)",
        ("C:\\data.d", "C:\\data.mis"),
    )
    for r in range(n_rasters):
        conn.execute("INSERT INTO Rasters VALUES (?, 'ds1', ?, ?, 2513, 721)", (f"r{r}", r, r))
    conn.commit()
    conn.close()
    return SyntheticSCiLS(sbd_path, slx_path, dense, mz_axis, x, y, rasters, norm_values, norm_targets, spectra_mz)


# ---------------------------------------------------------------------------------------------------------------------
# TOFWERK synthetic data
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class SyntheticTofwerk:
    """Synthetic TOFWERK TofDAQ imaging file together with the ground truth it was generated from."""

    path: Path
    peaks: np.ndarray  # (writes, bufs, 1, n_peaks)
    tof: np.ndarray  # (writes, bufs, 1, n_samples)
    masses: np.ndarray
    mass_axis: np.ndarray
    labels: list[str]
    write_bufs: dict[int, list[tuple[int, int]]]  # roi -> (write, buf) of each valid spot, in pixel order
    xy: dict[int, list[tuple[int, int]]]  # roi -> expected (x, y) grid position of each valid spot


def write_synthetic_tofwerk(
    directory: Path, name: str = "synthetic.h5", duplicate: bool = False, userblock: bool = False
) -> SyntheticTofwerk:
    """Write a small TofDAQ-like file with 2 sample lines, 1 reference line and one invalid spot.

    Layout of `ImageData/SpotData/Data` (the reference line is stored first to exercise scan offsets):
    - reference line: 3 spots along x at y=900.3 -> writes/bufs (2, 0..2)
    - sample line 0: x=100.5, y=200.2, 220.2, 240.2 -> (0, 1), (0, 2), (0, 3)
    - sample line 1: x=120.5, same y -> (1, 0), (1, 1), (1, 2); the last spot is invalid
    """
    import h5py

    rng = np.random.default_rng(42)
    n_writes, n_bufs = 3, 4
    masses = np.array([10.0, 20.0, 30.0, 40.0])
    labels = ["[10A]+", "[20B]+", "[30C]+", "[40D]+"]
    mass_axis = np.linspace(5.0, 45.0, 9)
    peaks = rng.normal(5, 3, size=(n_writes, n_bufs, 1, len(masses))).astype(np.float32)
    tof = rng.normal(5, 3, size=(n_writes, n_bufs, 1, len(mass_axis))).astype(np.float32)

    line1_x = 100.5 if duplicate else 120.5
    spots = np.array(
        [
            # valid, x, y, z, write, buf
            [1, 5000.3, 900.3, 10.0, 2, 0],
            [1, 5020.3, 900.3, 10.0, 2, 1],
            [1, 5040.3, 900.3, 10.0, 2, 2],
            [1, 100.5, 200.2, 10.0, 0, 1],
            [1, 100.5, 220.2, 10.0, 0, 2],
            [1, 100.5, 240.2, 10.0, 0, 3],
            [1, line1_x, 200.2, 10.0, 1, 0],
            [1, line1_x, 220.2, 10.0, 1, 1],
            [0, line1_x, 240.2, 10.0, -1, -1],
        ],
        dtype=np.float64,
    )
    scans = np.array(
        [
            # scan type, sample type, mode, offset, n spots, layer
            [0, 1, 0, 0, 3, 0],
            [0, 0, 0, 3, 3, 0],
            [0, 0, 0, 6, 3, 0],
        ],
        dtype=np.float64,
    )
    peak_table = np.zeros(
        len(masses),
        dtype=[
            ("label", "S64"),
            ("mass", "<f4"),
            ("lower integration limit", "<f4"),
            ("upper integration limit", "<f4"),
        ],
    )
    peak_table["label"] = [label.encode() for label in labels]
    peak_table["mass"] = masses
    peak_table["lower integration limit"] = masses - 0.5
    peak_table["upper integration limit"] = masses + 0.5

    path = Path(directory) / name
    # TofDAQ files start with a user block, which moves the HDF5 signature away from the start of the file
    with h5py.File(path, "w", userblock_size=512 if userblock else 0) as f:
        f.attrs["TofDAQ Version"] = np.array([1.99])
        f.attrs["IonMode"] = np.bytes_(b"positive")
        image = f.create_group("ImageData")
        image.attrs["SpotSpacing"] = np.array([20.0])
        image.attrs["SampleName"] = np.bytes_(b"synthetic")
        scan_group = image.create_group("ScanData")
        scan_group.create_dataset("Data", data=scans)
        for i, scan_name in enumerate(["STD", "Line0", "Line1"]):
            scan_group.attrs[f"ScanName{i:05d}"] = np.bytes_(scan_name.encode())
        image.create_group("SpotData").create_dataset("Data", data=spots)
        f.create_group("PeakData").create_dataset("PeakData", data=peaks)
        f["PeakData"].create_dataset("PeakTable", data=peak_table)
        full = f.create_group("FullSpectra")
        full.create_dataset("TofData", data=tof)
        full.create_dataset("MassAxis", data=mass_axis.astype(np.float32))
    if userblock:
        with open(path, "r+b") as f:
            f.write(b"=" * 512)

    return SyntheticTofwerk(
        path=path,
        peaks=peaks,
        tof=tof,
        masses=masses,
        mass_axis=mass_axis.astype(np.float32).astype(np.float64),
        labels=labels,
        write_bufs={0: [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1)], 1: [(2, 0), (2, 1), (2, 2)]},
        xy={0: [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], 1: [(0, 0), (1, 0), (2, 0)]},
    )


# ---------------------------------------------------------------------------------------------------------------------
# PHI SmartSoft-TOF
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class SyntheticPhi:
    """Synthetic PHI SmartSoft-TOF `.raw` file together with the ground truth it was generated from."""

    path: Path
    events: np.ndarray  # (n, 3) x, y, channel of each event that should be kept
    nx: int
    ny: int
    slope: float  # calibration that should be used by default
    offset: float
    n_invalid: int
    n_off_raster: int
    n_off_axis: int

    def mz(self, channel: np.ndarray, slope: float | None = None, offset: float | None = None) -> np.ndarray:
        """m/z of channel indices (channel 0 = StartFlightTime)."""
        slope = self.slope if slope is None else slope
        offset = self.offset if offset is None else offset
        t_us = 0.5 + np.asarray(channel, dtype=np.float64) * 0.001
        return (slope * t_us + offset) ** 2

    def dense(self) -> dict[int, dict[int, int]]:
        """Pixel index -> {channel: count}."""
        out: dict[int, dict[int, int]] = {}
        for x, y, channel in self.events:
            pixel = out.setdefault(int(y * self.nx + x), {})
            pixel[int(channel)] = pixel.get(int(channel), 0) + 1
        return out


PHI_SLOPE, PHI_OFFSET = 2.0, 0.0
PHI_RECAL_SLOPE, PHI_RECAL_OFFSET = 2.01, 0.002


def _phi_block(block_id: int, payload: bytes = b"", size: int | None = None) -> bytes:
    return struct.pack("<H10sI", block_id, b"\x00" * 10, len(payload) if size is None else size) + payload


def _phi_event(x: int, y: int, tof_ps: int, valid: bool = True) -> int:
    return (y << 43) | (x << 32) | ((1 << 31) if valid else 0) | tof_ps


def write_synthetic_phi(  # noqa: C901
    directory: Path,
    name: str = "synthetic.raw",
    recalibrated: bool = False,
    mosaic: bool = False,
    header_overrides: dict[str, str | None] | None = None,
    terminate_header: bool = True,
) -> SyntheticPhi:
    """Write a small PHI-like file: 4x4 raster (8x4 for `mosaic`), 2 frames, pixel (3, 3) without any ions.

    Calibration: `sqrt(m/z) = 2 * t_us`, flight times 0.5-10 us in 1 ns channels, mass range 0.9-300, so channel 0
    (m/z 1) is the first channel of the axis. Contains invalid records, off-raster and off-axis events, an unknown
    block and optionally an appended recalibration block.
    """
    rng = np.random.default_rng(7)
    n_pixels = 4
    header_size = 2048
    start_ps, bin_ps = 500_000, 1_000

    tiles = [(0, 0), (1, 0)] if mosaic else [(0, 0)]
    kept = []
    frames: list[list[tuple[tuple[int, int], list[int]]]] = [[], []]
    for frame in frames:
        for tile in tiles:
            words = []
            for _ in range(150):
                x, y = rng.integers(0, n_pixels, size=2)
                if x == 3 and y == 3:
                    continue
                channel = int(rng.choice([10, 11, 500, 501, 3000, 7000]))
                jitter = int(rng.integers(-400, 401))  # rounds back onto the channel
                words.append(_phi_event(int(x), int(y), start_ps + channel * bin_ps + jitter))
                kept.append((int(x) + tile[0] * n_pixels, int(y) + tile[1] * n_pixels, channel))
            words += [
                _phi_event(1, 1, start_ps + 10 * bin_ps, valid=False),  # missing valid bit
                0xADADADADADADADAD,  # uninitialised fill, high bits set
                _phi_event(n_pixels, 0, start_ps + 10 * bin_ps),  # off raster
                _phi_event(0, 0, start_ps - 200 * bin_ps),  # before the flight time range
                _phi_event(0, 0, start_ps + 9000 * bin_ps),  # above the mass range
            ]
            frame.append((tile, words))

    header = {
        "HeaderSize": str(header_size),
        "ImagePixels": str(n_pixels),
        "Mass/Time": str(PHI_SLOPE),
        "MassOffset": str(PHI_OFFSET),
        "StartFlightTime": "0.5",
        "StopFlightTime": "10.0",
        "SpecBinSize": "1.0",
        "SecIonPolarity": "Negative",
        "NoFrames": "2",
        "Calibrated": "no",
    }
    for key, value in (header_overrides or {}).items():
        if value is None:
            header.pop(key, None)
        else:
            header[key] = value
    lines = ["SOFH", *(f"{key}: {value}" for key, value in header.items())]
    lines += ["[Acq Base]", "Start Mass (amu)=0.9", "End Mass (amu)=300.0", "MSMS Active=Inactive"]
    lines += ["[PHI LMIG]", "Raster Size (um)=100.0", "Raster Size Calibration=1.550, 1.550"]
    if terminate_header:
        lines.append("EOFH")
    text = ("\r\n".join(lines) + "\r\n").encode("latin-1")
    assert len(text) < header_size
    content = bytearray(text + b"\x00" * (header_size - len(text)))

    for frame in frames:
        for tile, words in frame:
            if mosaic:
                content += _phi_block(6, struct.pack("<II", *tile), size=0)
            # split into two event blocks to exercise merging
            data = np.asarray(words, dtype="<u8").tobytes()
            half = (len(words) // 2) * 8
            content += _phi_block(1, data[:half]) + _phi_block(1, data[half:])
        content += _phi_block(2)
    content += _phi_block(99, b"\x01" * 12)  # unknown block, skipped
    if recalibrated:
        info = f"AsciiInfoBlock01\r\nMass/Time: {PHI_RECAL_SLOPE}\r\nMassOffset: {PHI_RECAL_OFFSET}\r\n"
        content += _phi_block(14, info.encode("latin-1"))
    content += _phi_block(0)

    path = Path(directory) / name
    path.write_bytes(bytes(content))
    events = np.asarray(kept, dtype=np.int64)
    n_tiles = len(tiles) * len(frames)
    return SyntheticPhi(
        path=path,
        events=events,
        nx=n_pixels * len(tiles),
        ny=n_pixels,
        slope=PHI_RECAL_SLOPE if recalibrated else PHI_SLOPE,
        offset=PHI_RECAL_OFFSET if recalibrated else PHI_OFFSET,
        n_invalid=2 * n_tiles,
        n_off_raster=n_tiles,
        n_off_axis=2 * n_tiles,
    )
