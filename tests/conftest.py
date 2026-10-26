"""Shared reader backend coverage."""

from __future__ import annotations

import importlib.util
import os
import typing as ty

import pytest

from imzy import IMZMLReader


@pytest.fixture(params=["python", "rust"])
def imzml_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run existing imzML reader and writer tests against both implementations."""
    available = importlib.util.find_spec("imzy._imzml_rust") is not None
    if request.param == "rust" and not available:
        if os.environ.get("IMZY_REQUIRE_RUST") == "1":
            pytest.fail("The native extension is required for this test run.")
        pytest.skip("Native extension is not installed.")
    original = IMZMLReader.__init__

    def initialize(self: IMZMLReader, *args: ty.Any, **kwargs: ty.Any) -> None:
        kwargs.setdefault("backend", request.param)
        original(self, *args, **kwargs)

    monkeypatch.setattr(IMZMLReader, "__init__", initialize)
    return str(request.param)
