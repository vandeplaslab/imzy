# Command line

Installing imzy adds the `imzy` command. Convert a supported dataset to imzML:

```bash
imzy convert path/to/input path/to/output --silent
```

The output path may be a base name or an `.imzML` filename. Both XML and binary files are written.
The command prints the resulting imzML path.

```bash
imzy convert sample.imzML normalized \
  --normalization TIC \
  --spectrum-type auto \
  --ibd-mode auto
```

## Options

| Option | Default | Accepted values / behavior |
| --- | --- | --- |
| `--spectrum-type` | `auto` | `auto`, `centroid`, `profile` |
| `--ibd-mode` | `auto` | `auto`, `continuous`, `processed` |
| `--normalization NAME` | None | A name returned by Python's `imzy.get_normalizations()`; case-insensitive |
| `--roi INTEGER` | None | Bruker TSF/TDF/NeoFlex ROI ID |
| `--overwrite / --no-overwrite` | No overwrite | Replace existing output files when enabled |
| `--on-error` | `error` | `error` stops conversion; `warn` skips unreadable/invalid spectra with warnings |
| `--silent / --progress` | Progress | Hide or show progress bars |

For Bruker input on a supported platform:

```bash
imzy convert sample.d roi_export --roi 0
```

Choose an ROI ID exposed by the dataset. Other formats reject `--roi`. TDF conversion prints a warning that
ion mobility is collapsed to the m/z axis. Empty spectra are skipped with warnings, as described in
[writing imzML](writing.md#finalization-and-skipped-spectra).

```bash
imzy --help
imzy convert --help
```

The [Python API](api.md#writers) additionally supports selecting pixel indices, coordinate origin,
output precision, metadata overrides, and custom spectrum transforms.
