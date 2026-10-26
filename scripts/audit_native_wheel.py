"""Audit the Python extension without interpreting vendor binaries for other OSs."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main() -> None:
    """Extract and audit the imzML extension against the Python 3.10 stable ABI."""
    with zipfile.ZipFile(sys.argv[1]) as archive, tempfile.TemporaryDirectory(prefix="imzy-abi-") as directory:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("imzy/_imzml_rust.") and name.endswith((".so", ".pyd"))
        ]
        if len(names) != 1:
            raise RuntimeError("Expected exactly one imzML extension in the wheel.")
        extension = Path(directory) / Path(names[0]).name
        extension.write_bytes(archive.read(names[0]))
        subprocess.run(
            [sys.executable, "-m", "abi3audit", "--strict", "--assume-minimum-abi3", "3.10", str(extension)],
            check=True,
        )


if __name__ == "__main__":
    main()
