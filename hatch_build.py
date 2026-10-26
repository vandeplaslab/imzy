"""Compile the optional Rust extension while retaining Hatch's package metadata."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import typing as ty
import zipfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RustBuildHook(BuildHookInterface):
    """Include Maturin's native artifact in regular and editable Hatch wheels."""

    def initialize(self, version: str, build_data: dict[str, ty.Any]) -> None:
        """Build when requested or when the Rust toolchain is available."""
        policy = os.environ.get("IMZY_BUILD_RUST", "auto")
        if policy not in {"auto", "0", "1"}:
            raise ValueError("IMZY_BUILD_RUST must be auto, 0, or 1.")
        if policy == "0":
            return
        if shutil.which("cargo") is None:
            if policy == "1":
                raise RuntimeError("IMZY_BUILD_RUST=1 requires the Rust toolchain (cargo).")
            return

        with tempfile.TemporaryDirectory(prefix="imzy-native-") as directory:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "maturin",
                    "build",
                    "--release",
                    "--locked",
                    "--interpreter",
                    sys.executable,
                    "--out",
                    directory,
                ],
                cwd=self.root,
                check=True,
            )
            wheels = list(Path(directory).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected one native wheel from Maturin.")
            wheel = wheels[0]
            with zipfile.ZipFile(wheel) as archive:
                extensions = [
                    name
                    for name in archive.namelist()
                    if name.startswith("imzy/_imzml_rust.") and name.endswith((".so", ".pyd"))
                ]
                if len(extensions) != 1:
                    raise RuntimeError("Native wheel does not contain the imzML extension.")
                name = extensions[0]
                destination = Path(self.root) / "build" / "native" / Path(name).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                build_data["force_include"][str(destination)] = name
                if version == "editable":
                    shutil.copy2(destination, Path(self.root) / "src" / name)

            build_data["pure_python"] = False
            build_data["tag"] = "-".join(wheel.stem.split("-")[-3:])
