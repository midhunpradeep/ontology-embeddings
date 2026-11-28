import copy
import hashlib
import json

import pytest

import ontology_embeddings.geo_campaign as campaign
from ontology_embeddings.geo_campaign import (
    CONFIGURATIONS,
    MANIFEST,
    AttemptLedger,
    validate_manifest,
    validate_ordered_selections,
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
        data = config["dataset"]
        assert data["test_access"] == "sealed; not decoded"
        assert data["train_count"] == 720 and data["validation_count"] == 240
        assert data["train_selection_sha256"] == "d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219"
        assert data["validation_selection_sha256"] == "a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c"
        assert config["execution_qualification"]["status"] == "blocked_pending_family_stage_qualification"
        assert config["execution_qualification"]["source_revision"]["commit_sha"] is None
        assert config["execution_qualification"]["environment"]["qualification"] == "unresolved"
        assert "carrier pair" in config["initialization"]["pairing"]
        assert (config["initialization"]["town_window_protocol"] is not None) == (config["family"] == "TOWN")
    for family in ("BOX", "TOWN"):
        for dim in (2, 8, 16):
            pair = [c for c in CONFIGURATIONS if c["family"] == family and c["dimension"] == dim]
            assert len({c["preparation_id"] for c in pair}) == 1
            assert len({c["pair_id"] for c in pair}) == 1
    validate_manifest(MANIFEST)


def test_selection_validation_checks_count_identity_and_disjointness(monkeypatch):
    train = [f"train-{i}" for i in range(720)]
    validation = [f"validation-{i}" for i in range(240)]
    def encode(values):
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(campaign, "TRAIN_SELECTION_SHA256", encode(train))
    monkeypatch.setattr(campaign, "VALIDATION_SELECTION_SHA256", encode(validation))
    validate_ordered_selections(train, validation)
    with pytest.raises(ValueError, match="overlap"):
        validate_ordered_selections(train, [train[0], *validation[1:]])
    with pytest.raises(ValueError, match="counts"):
        validate_ordered_selections(train[:-1], validation)
    with pytest.raises(ValueError, match="identity"):
        validate_ordered_selections([*train[:-1], "changed"], validation)


def test_manifest_mutation_rejected():
    changed = copy.deepcopy(MANIFEST)
    changed["configurations"][0]["settings"]["epochs"] = 61
    with pytest.raises(ValueError, match="frozen"):
        validate_manifest(changed)


def test_pending_to_dependency_blocked_without_run_id(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[0]["config_id"]
    ledger.append(config_id=config_id, status="pending", evidence_references=["qualification://ready"])
    ledger.append(config_id=config_id, status="blocked", outcome_class="dependency_blocked",
                  evidence_references=["dependency://town-stage"], run_id=None)
    rows = ledger.records()
    assert [r["status"] for r in rows] == ["pending", "blocked"]
    assert all(r["run_id"] is None for r in rows)
    assert rows[0]["attempt_id"] == rows[1]["attempt_id"]
    assert rows[0]["config_id"] == rows[1]["config_id"]


def test_external_operational_failure_allows_one_fresh_attempt(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[0]["config_id"]
    ledger.append(config_id=config_id, status="pending", evidence_references=["qualification://ready"])
    ledger.append(config_id=config_id, status="failed", run_id="run-1", outcome_class="external_operational",
                  evidence_references=["log://failure"])
    with pytest.raises(ValueError, match="correction evidence"):
        ledger.append(config_id=config_id, status="pending", evidence_references=["retry://intent"])
    ledger.append(config_id=config_id, status="pending", evidence_references=["retry://intent"],
                  recovery_evidence=["review://cause-corrected"])
    ledger.append(config_id=config_id, status="completed", run_id="run-2",
                  evidence_references=["run://evidence"], outcome_class="scientific_negative")
    rows = ledger.records()
    assert [r["attempt_id"] for r in rows] == [f"{config_id}-A01"] * 2 + [f"{config_id}-A02"] * 2
    assert [r["run_id"] for r in rows] == [None, "run-1", None, "run-2"]
    with pytest.raises(ValueError, match="not retryable"):
        ledger.append(config_id=config_id, status="pending", evidence_references=["retry://third"])


def test_dependency_unblock_and_pre_run_validation(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[1]["config_id"]
    ledger.append(config_id=config_id, status="blocked", outcome_class="dependency_blocked",
                  evidence_references=["dependency://blocked"])
    ledger.append(config_id=config_id, status="pending", evidence_references=["dependency://resolved"])
    ledger.append(config_id=config_id, status="completed", run_id="run-valid",
                  evidence_references=["run://valid"])
    assert [r["status"] for r in ledger.records()] == ["blocked", "pending", "completed"]
    other = AttemptLedger(tmp_path / "invalid.jsonl")
    with pytest.raises(ValueError, match="RunRecord ID"):
        other.append(config_id=config_id, status="completed", evidence_references=["x"])
    with pytest.raises(ValueError, match="dependency_blocked"):
        other.append(config_id=config_id, status="blocked", evidence_references=["x"])


def test_numerical_and_implementation_failures_never_retry(tmp_path):
    for i, kind in enumerate(("numerical_failure", "implementation_correctness")):
        ledger = AttemptLedger(tmp_path / f"failure-{i}.jsonl")
        config_id = CONFIGURATIONS[i]["config_id"]
        ledger.append(config_id=config_id, status="pending", evidence_references=["qualification://ready"])
        ledger.append(config_id=config_id, status="failed", run_id=f"run-{i}", outcome_class=kind,
                      evidence_references=["run://failure"])
        with pytest.raises(ValueError, match="not retryable"):
            ledger.append(config_id=config_id, status="pending", evidence_references=["retry://forbidden"],
                          recovery_evidence=["review://anything"])


def test_second_external_retry_is_rejected_and_history_is_append_only(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[2]["config_id"]
    ledger.append(config_id=config_id, status="pending", evidence_references=["qualification://ready"])
    ledger.append(config_id=config_id, status="failed", run_id="r1", outcome_class="external_operational",
                  evidence_references=["run://one"])
    ledger.append(config_id=config_id, status="pending", evidence_references=["retry://one"],
                  recovery_evidence=["review://corrected"])
    ledger.append(config_id=config_id, status="failed", run_id="r2", outcome_class="external_operational",
                  evidence_references=["run://two"])
    before = (tmp_path / "attempts.jsonl").read_text()
    with pytest.raises(ValueError, match="limited to one"):
        ledger.append(config_id=config_id, status="pending", evidence_references=["retry://two"],
                      recovery_evidence=["review://corrected-again"])
    assert (tmp_path / "attempts.jsonl").read_text() == before


def test_scientific_negative_is_not_retried(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    config_id = CONFIGURATIONS[3]["config_id"]
    ledger.append(config_id=config_id, status="pending", evidence_references=["qualification://ready"])
    ledger.append(config_id=config_id, status="completed", run_id="scientific-run",
                  outcome_class="scientific_negative", evidence_references=["metrics://valid-negative"])
    with pytest.raises(ValueError, match="not retryable"):
        ledger.append(config_id=config_id, status="pending", evidence_references=["retry://tuning"])


def test_invalid_attempt_data_rejected(tmp_path):
    ledger = AttemptLedger(tmp_path / "attempts.jsonl")
    with pytest.raises(ValueError):
        ledger.append(config_id="bogus", status="pending", evidence_references=["x"])
    with pytest.raises(ValueError):
        ledger.append(config_id=CONFIGURATIONS[0]["config_id"], status="failed",
                      evidence_references=["x"], outcome_class="mystery", run_id="run")
    with pytest.raises(ValueError):
        ledger.append(config_id=CONFIGURATIONS[0]["config_id"], status="pending",
                      evidence_references=[])
