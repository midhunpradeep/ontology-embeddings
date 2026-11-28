from __future__ import annotations

import hashlib
import json

import pytest

from ontology_embeddings.geo_capture import (
    CAPTURE_SCHEMA_VERSION,
    TemporalCapture,
    expected_temporal_positions,
    verify_temporal_capture,
)
from ontology_embeddings.run_store import RunRecord


def _record(tmp_path):
    return RunRecord(
        tmp_path / "runs",
        purpose="geo-capture-test",
        invocation="test_geo_capture",
        config={"model": "box-union-two-adapters"},
        seed=23,
        issue="30",
    )


def _position_args(position):
    return {
        "accepted_updates": position["accepted_updates"],
        "epoch": position["epoch"],
        "batch": position["batch"],
        "epoch_boundary": position["epoch_boundary"],
    }


def test_temporal_positions_deduplicate_epoch_one_boundary_and_retain_later_boundaries():
    positions = expected_temporal_positions(epochs=60, batches_per_epoch=12)

    assert len(positions) == 72
    assert positions[0] == {
        "accepted_updates": 0,
        "epoch": 0,
        "batch": 0,
        "epoch_boundary": False,
    }
    assert positions[1] == {
        "accepted_updates": 1,
        "epoch": 1,
        "batch": 1,
        "epoch_boundary": False,
    }
    assert positions[12] == {
        "accepted_updates": 12,
        "epoch": 1,
        "batch": 12,
        "epoch_boundary": True,
    }
    assert positions[13] == {
        "accepted_updates": 24,
        "epoch": 2,
        "batch": 12,
        "epoch_boundary": True,
    }
    assert positions[-1] == {
        "accepted_updates": 720,
        "epoch": 60,
        "batch": 12,
        "epoch_boundary": True,
    }


def test_temporal_capture_copies_then_indexes_each_immutable_state(tmp_path):
    run = _record(tmp_path)
    checkpoints = run.path / "checkpoints"
    checkpoints.mkdir()
    capture = TemporalCapture(run.path, run.run_id, epochs=2, batches_per_epoch=2)

    for index, position in enumerate(expected_temporal_positions(2, 2)):
        source_name = "initial.pt" if index == 0 else "latest.pt"
        source = checkpoints / source_name
        source.write_bytes(f"producer-state-{index}".encode())
        capture.capture(
            source,
            **_position_args(position),
            model_state_sha256=hashlib.sha256(f"state-{index}".encode()).hexdigest(),
        )

    result = capture.finalize()
    verification = verify_temporal_capture(
        run.path,
        expected_run_id=run.run_id,
        expected_epochs=2,
        batches_per_epoch=2,
    )
    index = json.loads((run.path / "temporal" / "index.json").read_text())

    assert result["snapshot_count"] == 4
    assert result["snapshot_bytes"] == sum(
        (run.path / item["checkpoint_path"]).stat().st_size for item in index["snapshots"]
    )
    assert index["schema_version"] == CAPTURE_SCHEMA_VERSION
    assert index["run_id"] == run.run_id
    assert [item["accepted_updates"] for item in index["snapshots"]] == [0, 1, 2, 4]
    assert index["snapshots"][2]["epoch_boundary"] is True
    assert index["snapshots"][3]["epoch"] == 2
    for item in index["snapshots"]:
        path = run.path / item["checkpoint_path"]
        assert path.read_bytes() == (run.path / item["source_checkpoint_path"]).read_bytes() or (
            item["source_checkpoint_path"] == "checkpoints/latest.pt"
        )
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["checkpoint_sha256"]
    assert verification["complete"] is True
    assert verification["snapshot_count"] == 4


def test_temporal_verifier_rejects_tampering_and_unindexed_snapshots(tmp_path):
    run = _record(tmp_path)
    checkpoints = run.path / "checkpoints"
    checkpoints.mkdir()
    capture = TemporalCapture(run.path, run.run_id, epochs=1, batches_per_epoch=1)
    for index, position in enumerate(expected_temporal_positions(1, 1)):
        source = checkpoints / ("initial.pt" if index == 0 else "latest.pt")
        source.write_bytes(f"producer-state-{index}".encode())
        capture.capture(
            source,
            **_position_args(position),
            model_state_sha256=hashlib.sha256(f"state-{index}".encode()).hexdigest(),
        )
    capture.finalize()
    snapshot = next((run.path / "temporal" / "snapshots").glob("*.pt"))
    original_bytes = snapshot.read_bytes()
    snapshot.write_bytes(original_bytes + b"tampered")

    with pytest.raises(ValueError, match="checksum"):
        verify_temporal_capture(
            run.path,
            expected_run_id=run.run_id,
            expected_epochs=1,
            batches_per_epoch=1,
        )

    snapshot.write_bytes(original_bytes)
    extra = run.path / "temporal" / "snapshots" / "orphan.pt"
    extra.write_bytes(b"unindexed")
    with pytest.raises(ValueError, match="unindexed"):
        verify_temporal_capture(
            run.path,
            expected_run_id=run.run_id,
            expected_epochs=1,
            batches_per_epoch=1,
        )


def test_temporal_capture_failure_keeps_committed_prefix_and_exposes_orphan(tmp_path, monkeypatch):
    import ontology_embeddings.geo_capture as geo_capture

    run = _record(tmp_path)
    checkpoints = run.path / "checkpoints"
    checkpoints.mkdir()
    capture = TemporalCapture(run.path, run.run_id, epochs=1, batches_per_epoch=1)
    positions = expected_temporal_positions(1, 1)
    initial = checkpoints / "initial.pt"
    initial.write_bytes(b"initial")
    capture.capture(
        initial,
        **_position_args(positions[0]),
        model_state_sha256=hashlib.sha256(b"initial-state").hexdigest(),
    )
    latest = checkpoints / "latest.pt"
    latest.write_bytes(b"accepted")
    monkeypatch.setattr(
        geo_capture,
        "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("index write failed")),
    )

    with pytest.raises(OSError, match="index write failed"):
        capture.capture(
            latest,
            **_position_args(positions[1]),
            model_state_sha256=hashlib.sha256(b"accepted-state").hexdigest(),
        )

    with pytest.raises(ValueError, match="unindexed"):
        verify_temporal_capture(
            run.path,
            expected_run_id=run.run_id,
            expected_epochs=1,
            batches_per_epoch=1,
            require_complete=False,
        )


def test_temporal_schedule_rejects_invalid_dimensions_and_capture_order(tmp_path):
    with pytest.raises(ValueError):
        expected_temporal_positions(0, 12)
    with pytest.raises(ValueError):
        expected_temporal_positions(2, 0)

    run = _record(tmp_path)
    (run.path / "checkpoints").mkdir()
    capture = TemporalCapture(run.path, run.run_id, epochs=1, batches_per_epoch=2)
    latest = run.path / "checkpoints" / "latest.pt"
    latest.write_bytes(b"state")
    with pytest.raises(ValueError, match="next expected"):
        capture.capture(
            latest,
            accepted_updates=1,
            epoch=1,
            batch=1,
            epoch_boundary=False,
            model_state_sha256=hashlib.sha256(b"state").hexdigest(),
        )
