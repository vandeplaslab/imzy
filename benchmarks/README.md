# Performance benchmarks

[Documentation](https://vandeplaslab.github.io/imzy/) |
[Interactive results](https://vandeplaslab.github.io/imzy/benchmarks/) |
[Raw history archive](https://vandeplaslab.github.io/imzy/asv-results.tar.gz) |
[ASV documentation](https://asv.readthedocs.io/en/stable/)

The ASV suite tracks imzy's public reader and export workflows across main revisions. Charts are hosted under
`/imzy/benchmarks/`; documentation occupies the site root. The raw history archive keeps its original root URL.
The separate [local backend report](imzml.md) and `scripts/benchmark_imzml.py` remain available.

## Local setup and runs

Use Python 3.12.13 and a Rust compiler. CI pins Rust 1.99.1; install that toolchain to match its compiler.
Create a separate environment so benchmarking does not change your normal development dependencies:

```bash
uv venv --python 3.12.13 .asv/runner
uv pip install --python .asv/runner/bin/python -r benchmarks/requirements.txt
IMZY_BUILD_RUST=1 uv pip install --python .asv/runner/bin/python --no-deps --no-build-isolation .

export NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

machine=$(.asv/runner/bin/python scripts/benchmark_history.py machine --label local)
.asv/runner/bin/python -m asv check --python=same
.asv/runner/bin/python -m asv run --python=same --quick --dry-run --machine "$machine"
```

Run these commands from the repository root. Machine registration supports Linux and macOS, matching the initial
benchmark hosting target.
Both backends and the HDF5/Zarr dependencies are required; unavailable components fail instead of silently dropping cases.

For a full run of the installed checkout, followed by a website preview:

```bash
revision=$(git rev-parse HEAD)
.asv/runner/bin/python -m asv run --python=same --set-commit-hash "$revision" --record-samples --machine "$machine"
.asv/runner/bin/python -m asv publish
.asv/runner/bin/python scripts/benchmark_history.py pack
.asv/runner/bin/python -m asv preview
```

Only label an installed wheel with the revision it was built from. Reinstall it after changing runtime code. For ASV to
build a particular committed revision in its own pinned environment, omit `--python=same` and use a single-revision range:

```bash
.asv/runner/bin/python -m asv run "$revision^!" --record-samples --machine "$machine"
```

Environments, fixtures, results, and generated HTML live under the ignored `.asv/` directory. Local results stay local.
Do not merge measurements from different machines under the same machine name.
Publication charts the branches listed in `asv.conf.json`, initially `main`. To preview results from a feature branch,
use a temporary local configuration with that branch added to `branches`; keep the production configuration on `main`.

## Workloads and timing methodology

There are 88 parameter combinations: 11 operations, two backends, two layouts, and two pixel counts (256 and 4096).
Continuous profile spectra share a float64 m/z axis with 2048 points. Processed centroid spectra have independently
sorted float64 axes with 1024-2048 points. Intensities are seeded, varied float32 lognormal values. The data are synthetic;
vendor SDK performance and cold-storage throughput are outside this suite.

- Random access reads up to 1024 seeded pixel indices with replacement. Iteration reads every spectrum. Both consume
  both complete arrays using the same float64 checksum reduction for each backend.
- One ion image uses m/z 150. Batch extraction and storage exports use 32 masses between 150 and 950, with a 0.3 Da
  inclusive tolerance. Timed reads consume the complete output arrays; exports synchronously write complete stores.
- TIC caching is reset on every timed call. Normalizations compute all supported factors for every pixel.
- Cached initialization includes reader open and close. Uncached initialization additionally includes XML parsing and
  metadata-cache creation. Its cache is removed before each sample, outside the timer. Neither claims cold OS caches.
- Fixture generation, backend loading, Numba warmup, temporary-directory setup, and cleanup are outside the timer.
  Export timings include extraction, serialization, finalization, and close; Zarr also includes rechunking. They measure
  writes through the OS page cache, without forcing disk synchronization.
- Every sample calls its operation once. Three samples in each of two rounds produce six recorded samples per case.
  Automatic timed warmup is disabled so it cannot populate a TIC or export cache before measurement.
- Correctness tests use generated spectra and independent NumPy references for ion sums and normalization factors.
  They also check backend agreement, complete export round trips, and repeated uncached work.

The benchmark requirements file pins the full resolved software environment, including build and test tools. CI fixes
Numba/BLAS/OpenMP to one worker and Rayon to two workers. Machine names include a fingerprint of hardware, Python,
Rust, installed dependencies, the requirements file, and thread settings. Changed hardware or environments therefore
produce separate series. Environment descriptions are retained in the archive's `metadata/` directory.

GitHub-hosted runners share infrastructure and can vary in load and performance. Use the charts as trends, inspect raw
samples before drawing conclusions, and reproduce important changes locally. Timing changes do not fail CI.

To deliberately update the pinned environment, recompile it and rerun correctness and smoke checks:

```bash
uv pip compile pyproject.toml --group benchmark --extra hdf5 --extra zarr --python-version 3.12 \
  --universal --output-file benchmarks/requirements.txt --no-annotate --upgrade
```

## First GitHub Pages deployment

After these files are on `main`:

1. In repository Settings > Pages, choose GitHub Actions as the publishing source.
2. In Actions > Benchmarks > Run workflow, select `main` and enable `initialize_history` for the first run.
3. After it succeeds, follow the deployment URL for documentation, or open its `benchmarks/` subdirectory for charts.

The workflow builds documentation in an isolated environment and publishes both sites in one artifact. Existing
sites migrate without reinitialization: leave `initialize_history` disabled to restore the root history archive.

Normal runs leave `initialize_history` disabled. Main pushes and the weekly Sunday 00:00 UTC schedule benchmark the
checked-out main revision and publish the site. Re-running a revision replaces its measurements for that environment;
each Actions run also keeps its own recovery artifact. PRs run read-only correctness, discovery, and quick smoke checks
without deploying. Manual runs on other branches also perform smoke checks only. History begins with the first run;
older revisions are not automatically backfilled.

## History preservation and recovery

The workflow restores `asv-results.tar.gz` from the live Pages site before benchmarking. After a successful run it
publishes a complete replacement archive containing raw ASV JSON, machine records, and environment descriptions.
The whole restore/run/deploy cycle is serialized. No result branches, Git commits, PRs, or external storage are used.

Missing, inaccessible, malformed, or incomplete archives stop publication, leaving the previously deployed site intact.
Initialization accepts only a missing archive and no prior successful Pages deployment; it never bypasses corruption.
Each publication attempt uploads a `benchmark-history-<run id>-<attempt>` Actions artifact, retained for 90 days. This is
a recovery copy; the live Pages archive is the durable source between runs.

If the published archive is lost or damaged, find a successful main run of Benchmarks with a retained recovery artifact.
Run Benchmarks manually on `main`, leave initialization disabled, and enter that run's numeric ID in `recovery_run_id`.
The workflow verifies the source, restores its archive, benchmarks the current main revision, and republishes the site.
Leave recovery blank again for subsequent runs.

To inspect a downloaded recovery archive locally, restore it into an empty history directory and rebuild:

```bash
.asv/runner/bin/python scripts/benchmark_history.py restore --archive /path/to/asv-results.tar.gz
.asv/runner/bin/python scripts/benchmark_history.py check
.asv/runner/bin/python -m asv publish
.asv/runner/bin/python scripts/benchmark_history.py pack
```

Do not initialize a replacement empty database after results have been published.
