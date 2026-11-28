import copy
import json

import pytest

from ontology_embeddings.geo_campaign import (
    AttemptLedger, CONFIGURATIONS, MANIFEST, validate_manifest,
)


def test_frozen_matrix_count_order_settings_and_pairing():
    assert len(CONFIGURATIONS) == 12
    assert [(c["family"], c["dimension"], c["carrier"]) for c in CONFIGURATIONS] == [
        (family, dim, carrier)
        for family in ("BOX", "TOWN")
        for dim in (2, 8, 16)
        for carrier in ("unbounded", "bounded")
    ]
    assert len({c["config_id"] for c in CONFIGURATIONS}) == 12
    for config in CONFIGURATIONS:
        s = config["settings"]
        assert s["seed"] == 23 and s["epochs"] == 60 and s["batch_size_pizzas"] == 64
        assert s["dtype"] == "float32" and s["tau"] == 0.1
        assert s["learning_rate_all_groups"] == 0.02 and s["global_gradient_clip"] == 1.0
        assert config["dataset"]["test_access"] == "sealed; not decoded"
    for family in ("BOX", "TOWN"):
        for dim in (2, 8, 16):
            pair = [c for c in CONFIGURATIONS if c["family"] == family and c["dimension"] == dim]
            assert len({c["preparation_id"] for c in pair}) == 1
            assert len({c["pair_id"] for c in pair}) == 1
    validate_manifest(MANIFEST)


def test_manifest_mutation_rejected():
    changed = copy.deepcopy(MANIFEST)
    changed["configurations"][0]["settings"]["epochs"] = 61
    with pytest.raises(ValueError, match="frozen"):
        validate_manifest(changed)


def test_attempts_are_distinct_append_only_and_retry_is_bounded(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[0]["config_id"]
    ledger.append(config_id=config_id, run_id="run-1", status="failed",
                  outcome_class="external_operational", evidence_references=["log://failure"])
    original = (tmp_path / "attempts.jsonl").read_text()
    with pytest.raises(ValueError, match="already recorded"):
        ledger.append(config_id=config_id, run_id="run-1", status="completed",
                      evidence_references=["log://later"])
    with pytest.raises(ValueError, match="not authorized"):
        ledger.append(config_id=config_id, run_id="run-2", status="completed",
                      evidence_references=["log://run2"])
    ledger.append(config_id=config_id, run_id="run-2", status="completed",
                  evidence_references=["log://run2"], recovery_evidence=["review://cause-corrected"])
    with pytest.raises(ValueError, match="not authorized"):
        ledger.append(config_id=config_id, run_id="run-3", status="failed",
                      outcome_class="external_operational", evidence_references=["log://run3"],
                      recovery_evidence=["review://cause-corrected"])
    rows = ledger.records()
    assert [r["run_id"] for r in rows] == ["run-1", "run-2"]
    assert [r["attempt"] for r in rows] == [1, 2]
    assert rows[0]["config_id"] != rows[0]["run_id"]
    assert (tmp_path / "attempts.jsonl").read_text().startswith(original)


def test_invalid_attempt_data_rejected(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    with pytest.raises(ValueError):
        ledger.append(config_id="bogus", run_id="run", status="failed", evidence_references=["x"])
    with pytest.raises(ValueError):
        ledger.append(config_id=CONFIGURATIONS[0]["config_id"], run_id="run",
                      status="pending", evidence_references=["x"])
    with pytest.raises(ValueError):
        ledger.append(config_id=CONFIGURATIONS[0]["config_id"], run_id="run",
                      status="failed", evidence_references=[])
