import hashlib
import json
from pathlib import Path


def test_tracked_pr2_evidence_bundle_matches_manifest_and_seals_test():
    bundle = Path("docs/experiments/pr2-affine-evidence")
    checksums = json.loads((bundle / "checksums.json").read_text())
    files = checksums["files"]
    assert set(files) == {"README.md", "best.pt", "metrics.json"}
    for name, expected in files.items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == (
        checksums["bundle_sha256"]
    )

    report = json.loads((bundle / "metrics.json").read_text())
    assert report["source"]["experiment_source_sha"] == "522c3f30240e65a431abdfc7694d0006866b0874"
    assert report["source"]["experiment_source_clean"] is True
    assert report["test_sealing"]["test_metrics_or_predictions"] is False
    assert report["runs"]["larger_learning"]["best_reload_equivalence"] is True
    assert report["incomplete_timeout_run"]["run_id"] == "20261123T111432-30a6ffe1ea8e"
    assert report["incomplete_timeout_run"]["last_epoch_persisted"] == 38
    assert "incomplete" in report["incomplete_timeout_run"]["interpretation"].lower()
