# imzML backend benchmark

Recorded on 2026-10-26. These are local warm-cache measurements, not universal speed guarantees.

Machine: arm64, 10 logical CPUs, macOS-27.0.1-arm64-arm-64bit-Mach-O.
Python 3.14.0, NumPy 2.4.6, rustc 1.99.1 (58cf8f9c3 2026-10-13). Rust release build, default Rayon worker count.

```bash
python scripts/benchmark_imzml.py --pixels 4096 --points 2048 --targets 32 --repeats 5 --output benchmark.json
```

Both datasets contain varied lognormal float32 intensities. The continuous profile dataset shares a float64 axis
with 2048 samples. Processed centroid spectra have independent sorted float64 axes with 1024-2048 samples.
Random reads select 1024 pixels with replacement. One ion image uses m/z 150; batch extraction uses 32 masses
between 150 and 950. Both use a 0.3 Da tolerance.

Times are medians of five runs per backend, with alternating order. Data generation, metadata parsing, correctness
checks, and Numba warmup are excluded. Numba JIT is enabled. Every timed spectrum consumes both complete arrays;
images and TICs are also consumed. TIC caching is reset on every call. Complete spectra, ion tables, and TIC vectors
are checked against the Python implementation before timing. OS file pages are warm; no cold-storage claim is made.

| Dataset | Operation | Python (ms) | Rust (ms) | Speedup |
| --- | --- | ---: | ---: | ---: |
| continuous_profile | random_spectra | 42.810 | 3.956 | 10.82x |
| continuous_profile | iterate | 67.206 | 13.457 | 4.99x |
| continuous_profile | image | 58.497 | 0.586 | 99.88x |
| continuous_profile | images | 227.607 | 1.972 | 115.42x |
| continuous_profile | tic | 61.278 | 1.675 | 36.59x |
| processed_centroid | random_spectra | 40.116 | 4.198 | 9.56x |
| processed_centroid | iterate | 28.533 | 14.688 | 1.94x |
| processed_centroid | image | 30.555 | 1.797 | 17.00x |
| processed_centroid | images | 58.362 | 4.421 | 13.20x |
| processed_centroid | tic | 21.425 | 2.281 | 9.39x |

Raw timings, input sizes, and pixel throughput are in [imzml-macos-arm64.json](imzml-macos-arm64.json).
