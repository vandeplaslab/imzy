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
