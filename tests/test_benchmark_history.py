"""History retention, initialization, and failure checks without remote mutations."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest
from scripts import benchmark_history as history


def _results(directory: Path, revision: str = "a" * 40) -> None:
    machine = directory / "results" / "machine"
    machine.mkdir(parents=True, exist_ok=True)
    (machine.parent / "benchmarks.json").write_text(json.dumps({"version": 2, "Read.time_tic": {}}))
    (machine / "machine.json").write_text(json.dumps({"version": 1, "machine": "machine"}))
    (machine / f"{revision[:8]}-environment.json").write_text(
        json.dumps(
            {
                "version": 2,
                "commit_hash": revision,
                "result_columns": ["result", "params"],
                "results": {"Read.time_tic": [[0.01, 0.02], [["python", "rust"]]]},
            }
        )
    )
    metadata = directory / "metadata"
    metadata.mkdir(exist_ok=True)
    (metadata / "machine.json").write_text(json.dumps({"python": "3.12", "threads": 2}))


def test_successive_publications_preserve_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = tmp_path / "original"
    _results(original)
    archive = tmp_path / history.ARCHIVE
    history.pack_history(original, archive)
    monkeypatch.setattr(history, "urlopen", lambda *args, **kwargs: archive.open("rb"))
    restored = tmp_path / "second"
    history.restore_history("https://example.com/imzy/", restored)
    _results(restored, "b" * 40)
    history.pack_history(restored, archive)
    final = tmp_path / "third"
    history.restore_history("https://example.com/imzy/", final)
    assert len(list((final / "results" / "machine").glob("*-environment.json"))) == 2
    assert json.loads((final / "metadata" / "machine.json").read_text())["threads"] == 2


@pytest.mark.parametrize("initialize", [False, True])
@pytest.mark.parametrize("published", [False, True])
def test_empty_history_requires_explicit_first_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, initialize: bool, published: bool
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise HTTPError("https://example.com/asv-results.tar.gz", 404, "missing", {}, None)

    monkeypatch.setattr(history, "urlopen", missing)
    monkeypatch.setattr(history, "has_publication", lambda repository: published)
    directory = tmp_path / "history"
    if initialize and not published:
        history.restore_history("https://example.com/", directory, initialize=True, repository="owner/repo")
        assert directory.is_dir()
        assert list(directory.iterdir()) == []
    else:
        with pytest.raises(RuntimeError):
            history.restore_history("https://example.com/", directory, initialize=initialize, repository="owner/repo")
        assert not directory.exists()


def test_network_errors_cannot_initialize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise HTTPError("https://example.com/", 503, "unavailable", {}, None)

    monkeypatch.setattr(history, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="stopped"):
        history.restore_history("https://example.com/", tmp_path / "history", initialize=True)


@pytest.mark.parametrize("contents", [b"invalid gzip", b"\x1f\x8b\x08\x00"])
def test_corrupt_archives_do_not_replace_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes
) -> None:
    monkeypatch.setattr(history, "urlopen", lambda *args, **kwargs: io.BytesIO(contents))
    directory = tmp_path / "history"
    with pytest.raises((tarfile.TarError, EOFError)):
        history.restore_history("https://example.com/", directory, initialize=True)
    assert not directory.exists()


@pytest.mark.parametrize("member", ["../outside.json", "/outside.json", "results/link.json", "unexpected.json"])
def test_archive_cannot_escape_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        info = tarfile.TarInfo(member)
        if "link" in member:
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside.json"
            archive.addfile(info)
        else:
            info.size = 2
            archive.addfile(info, io.BytesIO(b"{}"))
    monkeypatch.setattr(history, "urlopen", lambda *args, **kwargs: io.BytesIO(data.getvalue()))
    with pytest.raises(ValueError, match="member"):
        history.restore_history("https://example.com/", tmp_path / "history")
    assert not (tmp_path / "history").exists()


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_failed_results_cannot_be_published(tmp_path: Path, value: float | None) -> None:
    _results(tmp_path)
    path = next((tmp_path / "results" / "machine").glob("*-environment.json"))
    data = json.loads(path.read_text())
    data["results"]["Read.time_tic"][0] = [0.01, value]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="measurement"):
        history.pack_history(tmp_path, tmp_path / "output.tar.gz")
    assert not (tmp_path / "output.tar.gz").exists()


def test_restore_refuses_nonempty_destination(tmp_path: Path) -> None:
    _results(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        history.restore_history("https://example.com/", tmp_path)


def test_local_recovery_preserves_results_and_adds_site_links(tmp_path: Path) -> None:
    original = tmp_path / "original"
    _results(original)
    site = tmp_path / "site"
    site.mkdir()
    index = site / "index.html"
    index.write_text("<html><body>Charts</body></html>")
    archive = site / history.ARCHIVE
    history.pack_history(original, archive)
    history.pack_history(original, archive)
    assert index.read_text().count('id="imzy-benchmark-links"') == 1
    assert history.ARCHIVE in index.read_text()
    restored = tmp_path / "restored"
    history.restore_archive(archive, restored)
    history.validate_history(restored)
    with pytest.raises(ValueError, match="empty"):
        history.restore_archive(archive, restored)


def test_prior_deployment_pages_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    def response(path: str) -> list[dict]:
        if "/statuses" in path:
            return [{"state": "success"}]
        if "page=1" in path:
            return [{"id": 123}]
        return []

    monkeypatch.setattr(history, "_github_json", response)
    assert history.has_publication("owner/repo")
    monkeypatch.setattr(history, "_github_json", lambda path: [])
    assert not history.has_publication("owner/repo")


def test_invalid_result_metadata_cannot_be_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode="w:gz") as archive:
        member = tarfile.TarInfo("results/benchmarks.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    monkeypatch.setattr(history, "urlopen", lambda *args, **kwargs: io.BytesIO(contents.getvalue()))
    with pytest.raises(ValueError, match="metadata"):
        history.restore_history("https://example.com/", tmp_path / "history")


def test_incomplete_parameters_cannot_be_published(tmp_path: Path) -> None:
    _results(tmp_path)
    path = next((tmp_path / "results" / "machine").glob("*-environment.json"))
    data = json.loads(path.read_text())
    data["results"]["Read.time_tic"][0] = [0.01]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete"):
        history.validate_history(tmp_path)


def test_machine_identity_tracks_environment_without_splitting_per_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = json.loads((Path(__file__).parents[1] / "asv.conf.json").read_text())
    config = tmp_path / "asv.conf.json"
    config.write_text(json.dumps(configuration))
    requirements = tmp_path / "benchmarks" / "requirements.txt"
    requirements.parent.mkdir()
    requirements.write_text("numpy==2.5.4\n")
    for name, values in configuration["matrix"]["env_nobuild"].items():
        monkeypatch.setenv(name, values[0])
    package = SimpleNamespace(metadata={"Name": "numpy"}, version="2.5.4")
    project = SimpleNamespace(metadata={"Name": "imzy"}, version="0.3.1")
    monkeypatch.setattr(history.importlib.metadata, "distributions", lambda: [package, project])
    monkeypatch.setattr(history.platform, "platform", lambda: "test-os")
    monkeypatch.setattr(history.platform, "processor", lambda: "test-cpu")
    monkeypatch.setattr(history.subprocess, "check_output", lambda *args, **kwargs: "test-toolchain")
    monkeypatch.setattr(history.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(history.os, "cpu_count", lambda: 4)
    pages = [16 * 1024**3 // 4096]
    monkeypatch.setattr(history.os, "sysconf", lambda key: pages[0] if key == "SC_PHYS_PAGES" else 4096, raising=False)
    original_read = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/cpuinfo":
            return "model name: test-cpu\n"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    directory = tmp_path / "history"
    initial = history.register_machine("test", directory, config)
    project.version = "0.3.2"
    pages[0] += 128  # Half a MiB of differing VM memory reservations.
    assert history.register_machine("test", directory, config) == initial
    package.version = "2.5.5"
    assert history.register_machine("test", directory, config) != initial
    package.version = "2.5.4"
    requirements.write_text("numpy==2.5.4\n# New build constraints\n")
    assert history.register_machine("test", directory, config) != initial
    monkeypatch.setenv("RAYON_NUM_THREADS", "8")
    with pytest.raises(ValueError, match="thread variables"):
        history.register_machine("test", directory, config)
