"""Plugin manager tests."""

import pytest

import imzy

from .utilities import get_imzml_data


@pytest.mark.parametrize("path", get_imzml_data())
def test_init(path):
    pm = imzy.discover_plugins()
    assert imzy._plugin_manager is not None
    reader = pm.get_reader(path)
    assert reader is not None
