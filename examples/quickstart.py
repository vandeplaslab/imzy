"""Generate portable IMS data, extract images, normalize, and round-trip an export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from imzy import IMZMLReader, IMZMLWriter, write_imzml


def create_sample(directory: Path) -> Path:
    """Write a two-by-two centroid dataset with three masses and known intensities."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sample.imzML"
    with IMZMLWriter(path, spectrum_type="centroid", ibd_mode="processed") as writer:
        for index in range(4):
            intensities = np.array([1.0, 2.0, 3.0]) * (index + 1)
            writer.add_spectrum([100.0, 200.0, 300.0], intensities, (index % 2 + 1, index // 2 + 1))
    return path


def run(directory: Path, *, plot: bool = False) -> Path:
    """Create sample data and a TIC-normalized copy, optionally saving a plot."""
    path = create_sample(directory)
    with IMZMLReader(path) as reader:
        mzs, intensities = reader.get_spectrum(0)
        tic = reader.get_tic(silent=True)
        images = reader.get_ion_images([100.0, 200.0], tol=0.1, silent=True)
        scales = reader.get_normalizations(silent=True)["TIC"]
        normalized = images[1] * reader.reshape(scales)
        output = write_imzml(reader, directory / "normalized.imzML", normalization="TIC", silent=True)

        if plot:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].plot(mzs, intensities)
            axes[0].set(xlabel="m/z", ylabel="Intensity", title="Pixel 0")
            for ax, image, title in zip(
                axes[1:], [reader.reshape(tic), normalized], ["TIC", "TIC-normalized m/z 200"], strict=True
            ):
                artist = ax.imshow(image)
                ax.set(xlabel="x", ylabel="y", title=title)
                fig.colorbar(artist, ax=ax)
            fig.tight_layout()
            fig.savefig(directory / "quickstart.png", dpi=150)
            plt.close(fig)

    with IMZMLReader(output) as converted:
        # Each normalized pixel has the same total intensity for this proportional input.
        np.testing.assert_allclose(converted.get_tic(silent=True), np.full(4, 15.0), rtol=1e-6)
    print(f"Wrote {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for new datasets; use a fresh directory for each run.")
    parser.add_argument("--plot", action="store_true", help="Save quickstart.png (requires imzy[plot]).")
    args = parser.parse_args()
    run(args.output, plot=args.plot)
