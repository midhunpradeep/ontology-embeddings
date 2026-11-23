"""Atomic local run registration and lightweight JSONL discovery index."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _implementation_source_root(module_file: str | Path | None = None) -> Path | None:
    """Return the checkout root only when this module is under its source tree."""
    module_path = Path(__file__ if module_file is None else module_file).resolve()
    if len(module_path.parents) < 3:
        return None
    root = module_path.parents[2]
    expected_module = root / "src" / "ontology_embeddings" / "run_store.py"
    return root if module_path == expected_module else None


def _unavailable_provenance(reason: str) -> dict[str, Any]:
    return {
        "commit_sha": None,
        "dirty": None,
        "relevant_source_sha256": {},
        "source_fingerprint_sha256": None,
        "availability": "unavailable",
        "fingerprint_note": f"git provenance unavailable: {reason}",
    }


def source_provenance(root: Path | None) -> dict[str, Any]:
    if root is None:
        return _unavailable_provenance("installed package is not inside a source checkout")
    root = Path(root).resolve()
    expected_module = root / "src" / "ontology_embeddings" / "run_store.py"
    if not (root / "pyproject.toml").is_file() or not expected_module.is_file():
        return _unavailable_provenance("source checkout identity could not be validated")
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        if Path(git_root).resolve() != root:
            return _unavailable_provenance("source directory is not the Git repository root")
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        changed: dict[str, str] = {}
        paths = [line[3:] for line in status.splitlines() if len(line) > 3]
        for relative in paths:
            if relative.startswith(("src/", "tests/")) or relative == "pyproject.toml":
                if not (root / relative).is_file():
                    continue
                changed[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        fingerprint = (
            hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest()
            if changed
            else None
        )
        return {
            "commit_sha": sha,
            "dirty": bool(status),
            "relevant_source_sha256": changed,
            "source_fingerprint_sha256": fingerprint,
            "availability": "available",
            "fingerprint_note": "working-tree source paths hashed; non-source diff details omitted",
        }
    except Exception as error:  # provenance is best effort and must not mask training
        return _unavailable_provenance(f"{type(error).__name__}: {error}")


class RunRecord:
    def __init__(
        self,
        root: Path | str,
        *,
        purpose: str,
        invocation: str,
        config: dict[str, Any],
        dataset: dict[str, Any] | None = None,
        seed: int | None = None,
        issue: str | None = None,
    ):
        self.store = Path(root).expanduser().resolve()
        self.run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        )
        self.path = self.store / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "purpose": purpose,
            "invocation": invocation,
            # Editable checkouts are identified from this module path, never from cwd.
            "source": source_provenance(_implementation_source_root()),
            "dataset": dataset or {},
            "config": config,
            "seed": seed,
            "issue": issue,
            "python": sys.version,
            "environment": {"platform": sys.platform, "python": sys.version},
            "platform": sys.platform,
        }
        try:
            self._persist()
            self._index()
        except BaseException as original_error:
            # The run directory/manifest exists even if initial registration fails.
            # Persist its terminal state and retry indexing best-effort, while preserving
            # the registration error as the exception observed by the caller.
            import traceback

            self.manifest.update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(original_error).__name__}: {original_error}",
                traceback="".join(traceback.format_exception(original_error)),
            )
            try:
                self._persist()
            except BaseException:
                pass
            try:
                self._index()
            except BaseException:
                pass
            raise

    def _persist(self) -> None:
        _atomic_json(self.path / "manifest.json", self.manifest)

    def _index(self) -> None:
        index = self.store / "index.jsonl"
        entry = _index_entry(self.manifest)
        with index.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def update(self, **values: Any) -> None:
        self.manifest.update(values)
        self._persist()
        self._index()

    def fail(self, error: BaseException) -> None:
        import traceback

        self.update(
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(error)),
        )

    def interrupt(self, error: BaseException) -> None:
        import traceback

        self.update(
            status="interrupted",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(error)),
        )

    def metric(self, value: dict[str, Any]) -> None:
        path = self.path / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _index_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    """Project a run manifest onto the searchable index fields."""
    entry = {
        key: manifest.get(key)
        for key in (
            "run_id",
            "started_at",
            "status",
            "purpose",
            "invocation",
            "source",
            "dataset",
            "config",
            "issue",
        )
    }
    source = manifest.get("source")
    dataset = manifest.get("dataset")
    config = manifest.get("config")
    entry.update(
        source_sha=source.get("commit_sha") if isinstance(source, dict) else None,
        dataset_id=dataset.get("dataset_id") if isinstance(dataset, dict) else None,
        model=config.get("model") if isinstance(config, dict) else None,
    )
    return entry


def find_runs(root: Path | str, **filters: str) -> list[dict[str, Any]]:
    """Retrieve run state from the index, reconciled with persisted manifests."""
    store = Path(root)
    index = store / "index.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if index.exists():
        for line in index.read_bytes().decode("utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                # Keep earlier durable index rows; manifests below may have later state.
                continue
            if not isinstance(item, dict) or not isinstance(item.get("run_id"), str):
                continue
            latest[item["run_id"]] = item

    # The manifest is persisted before each index append, so it may contain a newer
    # terminal status after a torn append. It also recovers runs registered before an
    # index row could be written. Corrupt/unreadable manifests do not mask index data.
    if store.is_dir():
        for run_dir in store.iterdir():
            manifest_path = run_dir / "manifest.json"
            if not run_dir.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(manifest, dict) or not isinstance(manifest.get("run_id"), str):
                continue
            latest[manifest["run_id"]] = _index_entry(manifest)

    def matches(item: dict[str, Any]) -> bool:
        for key, value in filters.items():
            candidate = item.get(key)
            if key == "source_sha":
                # Missing SHA is not a searchable identity; never turn it into "None".
                if not isinstance(candidate, str) or not candidate or candidate != value:
                    return False
            elif str(candidate) != value:
                return False
        return True

    return [item for item in latest.values() if matches(item)]


def export_run(root: Path | str, run_id: str, destination: Path | str) -> dict[str, str]:
    """Copy a complete run bundle and return a SHA-256 of its deterministic file listing."""
    source = Path(root) / run_id
    if not source.is_dir():
        raise FileNotFoundError(f"run {run_id!r} not found under {root}")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copytree(source, staging)
        digests = {
            path.relative_to(staging).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        checksum = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
        _atomic_json(staging / "checksums.json", {"files": digests, "bundle_sha256": checksum})
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"path": str(target), "bundle_sha256": checksum}


def verify_export(path: Path | str) -> bool:
    """Verify an export's file hashes and aggregate bundle digest."""
    root = Path(path)
    record = json.loads((root / "checksums.json").read_text(encoding="utf-8"))
    files = record["files"]
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
    return (
        actual == files
        and hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        == record["bundle_sha256"]
    )
