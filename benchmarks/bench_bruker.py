"""Bruker TDF/TSF frame decoding on the bundled example datasets; reader setup and JIT warmup are untimed."""

from __future__ import annotations

import os
from pathlib import Path

from imzy._readers.bruker import HAS_BRUKER_BACKEND, TDFReader, TSFReader

DATA_DIR = Path(__file__).parents[1] / "tests" / "_test_data"
TDF_PATH = DATA_DIR / "example_tdf.d"
TSF_PATH = DATA_DIR / "example_tsf.d"

number = 1
repeat = 5
warmup_time = 0


def _check() -> None:
    if os.environ.get("NUMBA_DISABLE_JIT") == "1":
        raise RuntimeError("Benchmarks require Numba JIT to be enabled.")
    if not HAS_BRUKER_BACKEND:
        # asv reports NotImplementedError as a skipped benchmark (no timsdata library / pytsf available)
        raise NotImplementedError("No Bruker backend available.")


class TDFFrames:
    """Decode every frame of a TIMS dataset into the (m/z, mobility) matrix or the summed spectrum."""

    def setup(self) -> None:
        _check()
        self.reader = TDFReader(TDF_PATH)
        self.frames = [int(frame_id) for frame_id in self.reader.frame_indices]
        lo, hi = self.reader.mz_to_index(1, [700.0, 900.0])
        self.mz_window = (round(lo), round(hi))
        # compile kernels and warm the file pages
        self.reader.read_frame_csc(self.frames[0], *self.mz_window)
        self.reader.read_frame_profile(self.frames[0])
        for frame_id in self.frames:
            self.reader.read_frame_csc(frame_id)

    def teardown(self) -> None:
        self.reader.close()

    def time_read_frame_csc(self) -> None:
        for frame_id in self.frames:
            self.reader.read_frame_csc(frame_id)

    def time_read_frame_csc_mz_window(self) -> None:
        for frame_id in self.frames:
            self.reader.read_frame_csc(frame_id, *self.mz_window)

    def time_read_frame_profile(self) -> None:
        for frame_id in self.frames:
            self.reader.read_frame_profile(frame_id)


class TSFFrames:
    """Read every profile spectrum of a TSF dataset."""

    def setup(self) -> None:
        _check()
        self.reader = TSFReader(TSF_PATH)
        self.frames = [int(frame_id) for frame_id in self.reader.frame_indices]
        for frame_id in self.frames:
            self.reader.read_profile_frame(frame_id)

    def teardown(self) -> None:
        self.reader.close()

    def time_read_profile_frame(self) -> None:
        for frame_id in self.frames:
            self.reader.read_profile_frame(frame_id)
