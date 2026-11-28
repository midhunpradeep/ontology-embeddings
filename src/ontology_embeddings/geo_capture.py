"""Synchronous immutable temporal capture for validated GEO model checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .run_store import _atomic_json

CAPTURE_SCHEMA_VERSION = "geo-temporal-capture-v1"
_CAPTURE_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def expected_temporal_positions(epochs: int, batches_per_epoch: int) -> list[dict[str, Any]]:
    """Return initialization, every first-epoch update, then later epoch boundaries."""
    _positive_int(epochs, "epochs")
    _positive_int(batches_per_epoch, "batches_per_epoch")
    positions: list[dict[str, Any]] = [
        {"accepted_updates": 0, "epoch": 0, "batch": 0, "epoch_boundary": False}
    ]
    for batch in range(1, batches_per_epoch + 1):
        positions.append(
            {
                "accepted_updates": batch,
                "epoch": 1,
                "batch": batch,
                "epoch_boundary": batch == batches_per_epoch,
            }
        )
    for epoch in range(2, epochs + 1):
        positions.append(
            {
                "accepted_updates": epoch * batches_per_epoch,
                "epoch": epoch,
                "batch": batches_per_epoch,
                "epoch_boundary": True,
            }
        )
    return positions


def state_dict_sha256(state: dict[str, Any]) -> str:
    """Hash a model state by sorted tensor identity and exact stored tensor bytes."""
    if not isinstance(state, dict) or not state:
        raise ValueError("model state must be a nonempty tensor mapping")
    digest = hashlib.sha256(b"ontology-embeddings-model-state-v1\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(name, str) or not name:
            raise ValueError("model state keys must be nonempty strings")
        if (
            not hasattr(value, "detach")
            or not hasattr(value, "dtype")
            or not hasattr(value, "shape")
        ):
            raise ValueError(f"model state entry {name!r} is not a tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        try:
            raw = tensor.numpy().tobytes()
        except (TypeError, RuntimeError) as error:
            raise ValueError(f"model state entry {name!r} has unsupported dtype") from error
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(raw)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    """Copy an already validated native checkpoint and atomically publish the copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            descriptor = -1
            while block := input_stream.read(1024 * 1024):
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


class TemporalCapture:
    """Publish a prefix-ordered, checksummed sequence of immutable BOX checkpoints."""

    def __init__(self, run_path: Path | str, run_id: str, *, epochs: int, batches_per_epoch: int):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a nonempty string")
        self.run_path = Path(run_path).resolve()
        self.run_id = run_id
        self.epochs = _positive_int(epochs, "epochs")
        self.batches_per_epoch = _positive_int(batches_per_epoch, "batches_per_epoch")
        self.positions = expected_temporal_positions(epochs, batches_per_epoch)
        self.temporal_path = self.run_path / "temporal"
        self.snapshot_path = self.temporal_path / "snapshots"
        if self.temporal_path.exists():
            raise FileExistsError(self.temporal_path)
        self.snapshot_path.mkdir(parents=True)
        self.index_path = self.temporal_path / "index.json"
        self.snapshots: list[dict[str, Any]] = []
        _atomic_json(self.index_path, self._index_payload())

    def _index_payload(self) -> dict[str, Any]:
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "epochs": self.epochs,
            "batches_per_epoch": self.batches_per_epoch,
            "snapshots": self.snapshots,
        }

    def capture(
        self,
        source_checkpoint: Path | str,
        *,
        accepted_updates: int,
        epoch: int,
        batch: int,
        epoch_boundary: bool,
        model_state_sha256: str,
    ) -> dict[str, Any]:
        """Atomically retain the next expected producer checkpoint before publishing its index."""
        position_index = len(self.snapshots)
        if position_index >= len(self.positions):
            raise ValueError("all expected temporal states have already been captured")
        expected = self.positions[position_index]
        actual = {
            "accepted_updates": accepted_updates,
            "epoch": epoch,
            "batch": batch,
            "epoch_boundary": epoch_boundary,
        }
        if actual != expected:
            raise ValueError(f"capture is out of order; next expected position is {expected}")
        if (
            isinstance(accepted_updates, bool)
            or isinstance(epoch, bool)
            or isinstance(batch, bool)
            or not isinstance(epoch_boundary, bool)
            or not isinstance(model_state_sha256, str)
            or _CAPTURE_DIGEST.fullmatch(model_state_sha256) is None
        ):
            raise ValueError("capture identity or model-state digest is invalid")

        source = Path(source_checkpoint)
        if source.is_symlink() or not source.is_file():
            raise ValueError("producer checkpoint must be a regular file")
        try:
            source_relative = source.resolve().relative_to(self.run_path).as_posix()
        except ValueError as error:
            raise ValueError("producer checkpoint must be inside its authoritative run") from error
        expected_source = (
            "checkpoints/initial.pt" if accepted_updates == 0 else "checkpoints/latest.pt"
        )
        if source_relative != expected_source:
            raise ValueError(f"producer checkpoint must be {expected_source}")
        source_digest = _file_sha256(source)
        filename = f"update-{accepted_updates:08d}-epoch-{epoch:03d}-batch-{batch:03d}.pt"
        destination = self.snapshot_path / filename
        _atomic_copy(source, destination)
        snapshot_digest = _file_sha256(destination)
        if source_digest != snapshot_digest:
            raise OSError("immutable checkpoint copy checksum mismatch")
        entry = {
            **expected,
            "checkpoint_path": (Path("temporal") / "snapshots" / filename).as_posix(),
            "checkpoint_sha256": snapshot_digest,
            "source_checkpoint_path": source_relative,
            "source_checkpoint_sha256": source_digest,
            "model_state_sha256": model_state_sha256,
        }
        updated = [*self.snapshots, entry]
        _atomic_json(self.index_path, {**self._index_payload(), "snapshots": updated})
        self.snapshots = updated
        return entry

    def finalize(self) -> dict[str, Any]:
        """Require the complete planned sequence and return measured local retention costs."""
        verification = verify_temporal_capture(
            self.run_path,
            expected_run_id=self.run_id,
            expected_epochs=self.epochs,
            batches_per_epoch=self.batches_per_epoch,
        )
        snapshots = verification["snapshots"]
        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "complete": True,
            "snapshot_count": len(snapshots),
            "snapshot_bytes": sum(
                (self.run_path / entry["checkpoint_path"]).stat().st_size for entry in snapshots
            ),
            "index_path": "temporal/index.json",
            "index_sha256": verification["index_sha256"],
        }


def verify_temporal_capture(
    run_path: Path | str,
    *,
    expected_run_id: str,
    expected_epochs: int,
    batches_per_epoch: int,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Fail closed on missing, reordered, tampered, duplicate or unindexed snapshots."""
    if not isinstance(require_complete, bool):
        raise ValueError("require_complete must be a boolean")
    if not isinstance(expected_run_id, str) or not expected_run_id:
        raise ValueError("expected_run_id must be a nonempty string")
    expected = expected_temporal_positions(expected_epochs, batches_per_epoch)
    root = Path(run_path).resolve()
    index_path = root / "temporal" / "index.json"
    try:
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"temporal capture index is missing or invalid: {error}") from error
    if not isinstance(index, dict) or set(index) != {
        "schema_version",
        "run_id",
        "epochs",
        "batches_per_epoch",
        "snapshots",
    }:
        raise ValueError("temporal capture index schema is invalid")
    if (
        index["schema_version"] != CAPTURE_SCHEMA_VERSION
        or index["run_id"] != expected_run_id
        or type(index["epochs"]) is not int
        or index["epochs"] != expected_epochs
        or type(index["batches_per_epoch"]) is not int
        or index["batches_per_epoch"] != batches_per_epoch
        or not isinstance(index["snapshots"], list)
    ):
        raise ValueError("temporal capture identity or schedule does not match the run")
    entries = index["snapshots"]
    if len(entries) > len(expected) or (require_complete and len(entries) != len(expected)):
        raise ValueError("temporal capture is incomplete or contains excess snapshots")
    snapshots_dir = root / "temporal" / "snapshots"
    expected_names: set[str] = set()
    for offset, (entry, position) in enumerate(zip(entries, expected, strict=False)):
        fields = {
            "accepted_updates",
            "epoch",
            "batch",
            "epoch_boundary",
            "checkpoint_path",
            "checkpoint_sha256",
            "source_checkpoint_path",
            "source_checkpoint_sha256",
            "model_state_sha256",
        }
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"temporal snapshot {offset} has an invalid index record")
        if any(entry[key] != position[key] for key in position):
            raise ValueError(f"temporal snapshot {offset} is missing or out of order")
        if (
            any(
                isinstance(entry[key], bool) or type(entry[key]) is not int
                for key in ("accepted_updates", "epoch", "batch")
            )
            or type(entry["epoch_boundary"]) is not bool
        ):
            raise ValueError(f"temporal snapshot {offset} has an invalid temporal identity")
        update = position["accepted_updates"]
        filename = (
            f"update-{update:08d}-epoch-{position['epoch']:03d}-batch-{position['batch']:03d}.pt"
        )
        relative_path = (Path("temporal") / "snapshots" / filename).as_posix()
        source_path = "checkpoints/initial.pt" if update == 0 else "checkpoints/latest.pt"
        if (
            entry["checkpoint_path"] != relative_path
            or entry["source_checkpoint_path"] != source_path
        ):
            raise ValueError(f"temporal snapshot {offset} path identity is invalid")
        for key in ("checkpoint_sha256", "source_checkpoint_sha256", "model_state_sha256"):
            if not isinstance(entry[key], str) or _CAPTURE_DIGEST.fullmatch(entry[key]) is None:
                raise ValueError(f"temporal snapshot {offset} digest is invalid")
        if entry["checkpoint_sha256"] != entry["source_checkpoint_sha256"]:
            raise ValueError(f"temporal snapshot {offset} differs from its producer checkpoint")
        snapshot_path = root / relative_path
        if snapshot_path.is_symlink() or not snapshot_path.is_file():
            raise ValueError(f"temporal snapshot {offset} file is missing or not regular")
        if _file_sha256(snapshot_path) != entry["checkpoint_sha256"]:
            raise ValueError(f"temporal snapshot {offset} checksum verification failed")
        expected_names.add(filename)
    if not snapshots_dir.is_dir():
        raise ValueError("temporal snapshot directory is missing")
    actual_pt_names = {path.name for path in snapshots_dir.iterdir() if path.suffix == ".pt"}
    if actual_pt_names != expected_names:
        raise ValueError("temporal capture contains missing or unindexed snapshot files")
    if any(path.name.endswith(".tmp") for path in snapshots_dir.iterdir()):
        raise ValueError("temporal capture contains an uncommitted temporary snapshot")
    return {
        "complete": len(entries) == len(expected),
        "snapshot_count": len(entries),
        "snapshots": entries,
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
    }
