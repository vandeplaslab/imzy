"""Preserve ASV history in Pages artifacts without creating Git commits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ARCHIVE = "asv-results.tar.gz"
THREAD_VARIABLES = (
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _github_json(path: str) -> list[dict]:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    token = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
    request = Request(
        f"{api}/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def has_publication(repository: str) -> bool:
    """Check GitHub deployment history before allowing a new empty results database."""
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repository):
        raise ValueError("Initialization requires a GitHub owner/repository.")
    page = 1
    while deployments := _github_json(
        f"repos/{repository}/deployments?environment=github-pages&per_page=100&page={page}"
    ):
        for deployment in deployments:
            statuses = _github_json(f"repos/{repository}/deployments/{deployment['id']}/statuses?per_page=100")
            if any(status["state"] in {"success", "inactive"} for status in statuses):
                return True
        page += 1
    return False


def validate_history(directory: Path) -> None:
    """Require valid ASV metadata and successful, finite measurements before publication."""
    results = directory / "results"
    metadata = json.loads((results / "benchmarks.json").read_text())
    if not isinstance(metadata, dict) or metadata.get("version") != 2:
        raise ValueError("Invalid ASV benchmark metadata.")
    count = 0
    for path in results.glob("*/*.json"):
        data = json.loads(path.read_text())
        if path.name == "machine.json":
            if data.get("machine") != path.parent.name:
                raise ValueError(f"Invalid machine metadata: {path}")
            if not (directory / "metadata" / f"{path.parent.name}.json").is_file():
                raise ValueError(f"Missing environment description: {path.parent.name}")
            continue
        _validate_result(path, data)
        count += 1
    if count == 0:
        raise ValueError("History contains no benchmark results.")
    for path in (directory / "metadata").glob("*.json"):
        json.loads(path.read_text())


def _validate_result(path: Path, data: dict) -> None:
    if not (path.parent / "machine.json").is_file() or not data.get("commit_hash"):
        raise ValueError(f"Invalid ASV result: {path}")
    index = data["result_columns"].index("result")
    if not data["results"]:
        raise ValueError(f"No measurements in {path}")
    for name, values in data["results"].items():
        samples = values[index]
        if not isinstance(samples, list):
            samples = [samples]
        parameters = values[data["result_columns"].index("params")] or []
        if len(samples) != math.prod(len(group) for group in parameters):
            raise ValueError(f"Incomplete parameter results: {name} in {path}")
        if not samples or any(type(value) not in {int, float} or not math.isfinite(value) for value in samples):
            raise ValueError(f"Failed or missing measurement: {name} in {path}")


def restore_history(site_url: str, directory: Path, *, initialize: bool = False, repository: str = "") -> None:
    """Restore the published archive, failing closed on missing or damaged history."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Restore into an empty history directory to avoid replacing local results.")
    with tempfile.TemporaryDirectory(prefix="imzy-asv-history-") as temporary:
        archive_path = Path(temporary) / ARCHIVE
        try:
            with urlopen(f"{site_url.rstrip('/')}/{ARCHIVE}", timeout=60) as response:
                with archive_path.open("wb") as output:
                    shutil.copyfileobj(response, output)
        except HTTPError as error:
            error.close()
            if error.code != 404 or not initialize:
                raise RuntimeError("Cannot restore benchmark history; publication stopped. See benchmarks/README.md.")
            if has_publication(repository):
                raise RuntimeError("This site has already been published; recover its archive instead of initializing.")
            directory.mkdir(parents=True, exist_ok=True)
            return
        restore_archive(archive_path, directory)


def restore_archive(archive_path: Path, directory: Path) -> None:
    """Recover a validated archive without overwriting any existing local history."""
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Restore into an empty history directory to avoid replacing local results.")
    with tempfile.TemporaryDirectory(prefix="imzy-asv-restore-") as temporary:
        restored = Path(temporary) / "history"
        restored.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative.parts[0] not in {"results", "metadata"}
                    or relative.suffix != ".json"
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                target = restored.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        validate_history(restored)
        directory.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(restored, directory, dirs_exist_ok=True)


def pack_history(directory: Path, output: Path) -> None:
    """Package validated history alongside the website and add download/methodology links."""
    validate_history(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(directory.rglob("*.json")):
            if path.is_symlink() or path.relative_to(directory).parts[0] not in {"results", "metadata"}:
                raise ValueError(f"Unexpected history file: {path}")
            archive.add(path, arcname=path.relative_to(directory), recursive=False)
    index = output.parent / "index.html"
    if index.exists():
        html = index.read_text()
        if 'id="imzy-benchmark-links"' not in html:
            links = (
                '<p id="imzy-benchmark-links" class="text-center">'
                f'<a href="{output.name}">Download raw benchmark history</a> | '
                '<a href="https://github.com/vandeplaslab/imzy/blob/main/benchmarks/README.md">'
                "Methodology and reproduction</a></p>\n"
            )
            index.write_text(html.replace("</body>", links + "</body>"))


def register_machine(label: str, directory: Path, config: Path) -> str:
    """Record hardware and pinned software, separating changed environments in ASV charts."""
    configuration = json.loads(config.read_text())
    limits = {name: values[0] for name, values in configuration["matrix"]["env_nobuild"].items()}
    actual_limits = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    if actual_limits != limits:
        raise ValueError(f"Set benchmark thread variables before registering a machine: {limits}")
    cpu = platform.processor() or platform.machine()
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    elif sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    packages = {
        distribution.metadata["Name"].lower(): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"].lower() != "imzy"
    }
    info = {
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram": ram,
        "python": sys.version,
        "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "requirements_sha256": hashlib.sha256((config.parent / "benchmarks/requirements.txt").read_bytes()).hexdigest(),
        "packages": packages,
        "threads": actual_limits,
    }
    # Small VM memory reservations must not split otherwise identical hosted runners.
    identity = {**info, "ram": round(ram / 1024**3)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    name = f"{label}-{fingerprint}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "asv",
            "machine",
            "--yes",
            "--config",
            str(config),
            "--machine",
            name,
            "--cpu",
            cpu,
            "--num_cpu",
            str(os.cpu_count()),
            "--ram",
            str(ram),
            "--os",
            platform.platform(),
        ],
        check=True,
        capture_output=True,
    )
    metadata = directory / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / f"{name}.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return name


def main() -> None:
    """Restore, validate, package, or register a benchmark run for the Pages workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    restore = subparsers.add_parser("restore")
    source = restore.add_mutually_exclusive_group(required=True)
    source.add_argument("--site-url")
    source.add_argument("--archive", type=Path)
    restore.add_argument("--initialize", action="store_true")
    restore.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    pack = subparsers.add_parser("pack")
    pack.add_argument("--output", type=Path, default=Path(".asv/html") / ARCHIVE)
    subparsers.add_parser("check")
    machine = subparsers.add_parser("machine")
    machine.add_argument("--label", default=platform.node())
    machine.add_argument("--config", type=Path, default=Path("asv.conf.json"))
    for subparser in subparsers.choices.values():
        subparser.add_argument("--directory", type=Path, default=Path(".asv/history"))
    args = parser.parse_args()
    if args.command == "restore":
        if args.archive is not None:
            if args.initialize:
                parser.error("Archive recovery cannot initialize empty history.")
            restore_archive(args.archive, args.directory)
        else:
            restore_history(args.site_url, args.directory, initialize=args.initialize, repository=args.repository)
    elif args.command == "pack":
        pack_history(args.directory, args.output)
    elif args.command == "check":
        validate_history(args.directory)
    else:
        print(register_machine(args.label, args.directory, args.config))


if __name__ == "__main__":
    main()
