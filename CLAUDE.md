# imzy
Reads, inspects and exports imaging MS datasets (imzML, Bruker TDF/TSF, Waters .raw, SCiLS) through one reader interface, and writes imzML.

## Role in ecosystem
Foundation IO library for imaging mass spectrometry files.
Internal deps: ims-utils, koyo, yoki5
Depended on by: annotine, image2image-io, imspy-data, pyspa, qucee (imported, undeclared: autoims-core, ims-utils)

## Key abstractions
- `get_reader(path)` — picks the right reader, discovered through `imzy_reader` plugin hooks
- `BaseReader` → `IMZMLReader`, `TDFReader`, `TSFReader`, `NeoFlexReader`, `WatersReader`, `SCiLSReader`
- `IMZMLWriter`, `write_imzml` (+ `IMZMLWriterWarning`, `SkippedSpectrumWarning`, `EmptySpectrumWarning`)
- Centroid stores: `H5CentroidsStore`, `ZarrCentroidsStore`, `InMemoryStore` (base `BaseCentroids`)
- `H5NormalizationStore`, `get_normalizations`
- `discover_plugins()` / `plugins.ImzyPluginManager`, `hookspec.py` — add new readers as plugins

## Do not re-explain
When working in this repo, assume familiarity with: ims-utils, koyo, yoki5
