# Documentation development

## Build and preview

From a repository checkout, install only the existing docs group into a separate environment:

```bash
uv venv .asv/docs-env --python 3.12
uv pip install --python .asv/docs-env/bin/python --group docs
.asv/docs-env/bin/python -m mkdocs serve
```

Open the local URL printed by MkDocs. On Windows, use `.asv/docs-env/Scripts/python.exe` in place of
`.asv/docs-env/bin/python`. A strict production build is:

```bash
.asv/docs-env/bin/python -m mkdocs build --strict
```

Generated documentation is written to the ignored `site/` directory. The build does not import imzy or require
a Rust compiler or vendor SDK. Benchmark reports and downloadable example scripts are included from their
repository sources by a small MkDocs hook. Edit those original files to update their published versions.
Run commands from the repository root so example snippet paths resolve.

## Examples and checks

Use a separate package environment for tests and examples:

```bash
uv venv .asv/examples-env --python 3.12
uv pip install --python .asv/examples-env/bin/python ".[test]"
.asv/examples-env/bin/python -m pytest -q tests/test_docs.py tests/test_benchmark_history.py
```

The tests use temporary datasets and check pixel intensities, image shapes, normalization multipliers,
conversion round trips, optional storage read-back, and combined-site assembly. Plot tests save PNGs without
opening a window. Install the relevant extras when running storage examples separately.

## GitHub Pages

Documentation and ASV charts are published together by the existing **Benchmarks** workflow.
Pull requests build docs strictly and run correctness and benchmark smoke checks without deploying.
Main pushes, the weekly schedule, and manual main runs publish only after checks and measurements succeed.
Docs dependencies are installed separately from the pinned benchmark environment.

The deployment layout is:

| URL under `https://vandeplaslab.github.io/imzy/` | Content |
| --- | --- |
| `/` | MkDocs documentation |
| `benchmarks/` | Interactive ASV charts and assets |
| `asv-results.tar.gz` | Durable raw history archive at its original URL |

In repository **Settings > Pages**, select **GitHub Actions** as the source. For an initial deployment,
follow the [benchmark initialization instructions](performance/index.md#first-github-pages-deployment).
When migrating an existing benchmark site, leave initialization disabled: its root archive is restored and retained.
Recovery artifact paths and archive validation remain unchanged.

To preview the combined artifact after a local ASV publication and history pack:

```bash
.asv/docs-env/bin/python -m mkdocs build --strict
python scripts/build_docs_site.py --docs site --benchmarks .asv/html --output .asv/site
python -m http.server 8000 --directory .asv/site
```

Assembly requires a fresh output directory. Remove only the generated `.asv/site` directory before repeating it.
Open `http://localhost:8000/` for docs and `/benchmarks/` for charts. The navigation link to live charts targets
the published site. For ASV setup, measurement methodology, archive creation, and recovery, use the
[benchmark guide](performance/index.md).
