"""Plugin manager tests."""

import shutil
from pathlib import Path

import pytest

import imzy

from .utilities import get_imzml_data


@pytest.mark.parametrize("path", get_imzml_data())
def test_init(path: Path, tmp_path: Path) -> None:
    imzml_path = tmp_path / path.name
    shutil.copyfile(path, imzml_path)
    shutil.copyfile(path.with_suffix(".ibd"), imzml_path.with_suffix(".ibd"))
    pm = imzy.discover_plugins()
    assert imzy._plugin_manager is not None
    reader = pm.get_reader(imzml_path)
    assert reader is not None
    reader.close()
