"""Frozen GEO-03 configuration manifest and append-only attempt ledger.

This is an auditable campaign record, not a launcher or retry scheduler.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

DATASET_ID = "pizza-6a33710d13eb90fdde3e"
DECLARED_MANIFEST_ID = "c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd"
SOURCE_REFERENCES = [
    "GEO-SPEC #28", "GEO-01-SPEC #30 O1-O2/A3", "GEO-03-SPEC #32 E1-E5",
    "GEO-WORKFLOW #33", "dataset preflight #33 comments 7236705004 and 7236767355",
]
TRAIN_SELECTION_SHA256 = "d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219"
VALIDATION_SELECTION_SHA256 = "a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c"


def _configurations() -> list[dict[str, Any]]:
    configs = []
    for family in ("BOX", "TOWN"):
        for dimension in (2, 8, 16):
            for carrier in ("unbounded", "bounded"):
                index = len(configs) + 1
                config = {
                    "config_id": f"GEO03-{index:02d}-{family}-D{dimension}-{carrier.upper()}",
                    "order": index,
                    "family": family,
                    "dimension": dimension,
                    "carrier": carrier,
                    "pair_id": f"GEO03-{family}-D{dimension}",
                    "preparation_id": f"GEO03-PREP-{family}-D{dimension}",
                    "dataset": {"dataset_id": DATASET_ID, "declared_manifest_id": DECLARED_MANIFEST_ID,
                                "train_count": 720, "validation_count": 240, "test_count": 240,
                                "train_selection_sha256": TRAIN_SELECTION_SHA256,
                                "validation_selection_sha256": VALIDATION_SELECTION_SHA256,
                                "test_access": "sealed; not decoded"},
                    "initialization": {
                        "protocol": "existing family-native bounded initializer per dimension; native frame/BOX width sampling q in [0.05,0.25], per-axis half-width q^(1/d), bounded-feasible centers and native TOWN hosting reserve; unbounded inherits bounded-feasible targets",
                        "pairing": "common requested physical primitive targets and raw affine adapter weights within family/dimension carrier pair; separate independent family streams; record target identity, class order, seeded provenance, decoded targets and reconstruction deviations",
                        "family_streams_independent": True,
                        "town_window_protocol": ("volume-first sampled window fraction 0.05-0.10; initially contained; decoded host-frame exclusion >=0.05; joint validation of both carriers; deterministic finite retries up to 64; fail closed" if family == "TOWN" else None),
                        "status": "planned; actual preparation identity, sampled targets, decoded values, and qualification evidence unresolved",
                    },
                    "execution_qualification": {
                        "status": "blocked_pending_family_stage_qualification",
                        "source_revision": {"commit_sha": None, "clean": None, "qualification": "unresolved"},
                        "environment": {"device": None, "software_versions": None, "platform": None, "qualification": "unresolved"},
                    },
                    "settings": {
                        "seed": 23, "box_k": 1 if family == "BOX" else None,
                        "town_h": 1 if family == "TOWN" else None,
                        "town_w": 1 if family == "TOWN" else None,
                        "town_windows_independent": True if family == "TOWN" else None,
                        "carrier_bounds": [-1.0, 1.0] if carrier == "bounded" else None,
                        "point_activation": "tanh" if carrier == "bounded" else "identity",
                        "clearance_objective": "logistic", "tau": 0.1,
                        "optimizer": "Adam", "learning_rate_all_groups": 0.02,
                        "betas": [0.9, 0.999], "optimizer_epsilon": 1e-8,
                        "amsgrad": False, "weight_decay": 0.0, "lr_schedule": "constant",
                        "global_gradient_clip": 1.0, "geometry_epsilon": 1e-4,
                        "town_reserve_delta": 1e-4 if family == "TOWN" else None,
                        "dtype": "float32", "batch_size_pizzas": 64, "epochs": 60,
                        "optimizer_fresh": True,
                    },
                    "source_evidence_references": SOURCE_REFERENCES,
                }
                configs.append(config)
    return configs


CONFIGURATIONS = _configurations()
MANIFEST = {"manifest_version": 1, "dataset": {"dataset_id": DATASET_ID,
    "declared_manifest_id": DECLARED_MANIFEST_ID}, "source_evidence_references": SOURCE_REFERENCES,
    "configurations": CONFIGURATIONS}
MANIFEST_SHA256 = hashlib.sha256(json.dumps(MANIFEST, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _canonical_manifest() -> dict[str, Any]:
    return {"manifest_version": 1, "dataset": {"dataset_id": DATASET_ID,
        "declared_manifest_id": DECLARED_MANIFEST_ID}, "source_evidence_references": SOURCE_REFERENCES,
        "configurations": _configurations()}


def validate_manifest(value: dict[str, Any]) -> None:
    """Reject changes even if callers have mutated the exported convenience constants."""
    if value != _canonical_manifest():
        raise ValueError("GEO-03 manifest differs from the frozen approved manifest")


def validate_ordered_selections(train_ids: list[str], validation_ids: list[str]) -> None:
    """Check the exact published ordered selections without loading the sealed test split."""
    if len(train_ids) != 720 or len(validation_ids) != 240:
        raise ValueError("train/validation selection counts differ from the approved dataset")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("train/validation selections contain duplicate IDs")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation selections overlap")
    def digest(values: list[str]) -> str:
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    if digest(train_ids) != TRAIN_SELECTION_SHA256 or digest(validation_ids) != VALIDATION_SELECTION_SHA256:
        raise ValueError("ordered train/validation selection identity mismatch")


class AttemptLedger:
    """Append-only status history; each attempt is distinct from config and RunRecord IDs."""
    STATUSES = {"pending", "blocked", "failed", "interrupted", "completed"}
    FAILURE_CLASSES = {"external_operational", "numerical_failure", "implementation_correctness"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result: list[dict[str, Any]] = []
        latest: dict[str, dict[str, Any]] = {}
        run_ids: set[str] = set()
        expected_keys = {"config_id", "config_sha256", "attempt_id", "attempt", "status",
                         "run_id", "outcome_class", "evidence_references", "recovery_evidence"}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed ledger JSON on line {line_number}") from error
            if not isinstance(record, dict) or set(record) != expected_keys:
                raise ValueError(f"malformed ledger record on line {line_number}")
            config_id = record["config_id"]
            config = next((c for c in CONFIGURATIONS if c["config_id"] == config_id), None)
            if config is None or record["config_sha256"] != _config_hash(config):
                raise ValueError(f"unknown configuration or changed config hash on line {line_number}")
            status, outcome = record["status"], record["outcome_class"]
            attempt, run_id = record["attempt"], record["run_id"]
            evidence, recovery = record["evidence_references"], record["recovery_evidence"]
            if (status not in self.STATUSES or type(attempt) is not int or attempt < 1 or
                    not isinstance(evidence, list) or not evidence or
                    any(not isinstance(ref, str) or not ref for ref in evidence) or
                    not isinstance(recovery, list) or
                    any(not isinstance(ref, str) or not ref for ref in recovery)):
                raise ValueError(f"invalid status, attempt, or evidence on line {line_number}")
            if status in {"pending", "blocked"} and run_id is not None:
                raise ValueError(f"pre-run status has RunRecord ID on line {line_number}")
            if status in {"failed", "interrupted", "completed"} and (
                    not isinstance(run_id, str) or not run_id.strip()):
                raise ValueError(f"terminal status lacks RunRecord ID on line {line_number}")
            if status == "pending" and outcome is not None:
                raise ValueError(f"pending status has outcome on line {line_number}")
            if recovery and status != "pending":
                raise ValueError(f"recovery evidence is only valid on a fresh pending attempt on line {line_number}")
            if status == "blocked" and outcome != "dependency_blocked":
                raise ValueError(f"invalid blocked outcome on line {line_number}")
            if status in {"failed", "interrupted"} and outcome not in self.FAILURE_CLASSES:
                raise ValueError(f"invalid failure outcome on line {line_number}")
            if status == "completed" and outcome not in {None, "scientific_negative"}:
                raise ValueError(f"invalid completed outcome on line {line_number}")
            if run_id is not None:
                if run_id in run_ids:
                    raise ValueError(f"duplicate RunRecord ID on line {line_number}")
                run_ids.add(run_id)
            previous = latest.get(config_id)
            if previous is None:
                valid = status in {"pending", "blocked"} and attempt == 1
            elif status == "pending" and previous["status"] == "blocked":
                valid = attempt == previous["attempt"] and not recovery
            elif status == "pending" and previous["status"] == "failed" and previous["outcome_class"] == "external_operational":
                valid = attempt == previous["attempt"] + 1 <= 2 and bool(recovery)
            else:
                valid = (previous["status"] == "pending" and
                         status in {"blocked", "failed", "interrupted", "completed"} and
                         attempt == previous["attempt"])
                if previous["status"] == "blocked" and status == "blocked":
                    valid = attempt == previous["attempt"]
            expected_attempt_id = f"{config_id}-A{attempt:02d}"
            if not valid or record["attempt_id"] != expected_attempt_id:
                raise ValueError(f"illegal lifecycle or attempt identity on line {line_number}")
            latest[config_id] = record
            result.append(record)
        return result

    def append(self, *, config_id: str, status: str, evidence_references: list[str],
               run_id: str | None = None, attempt_id: str | None = None,
               outcome_class: str | None = None,
               recovery_evidence: list[str] | None = None) -> None:
        config = next((c for c in CONFIGURATIONS if c["config_id"] == config_id), None)
        if config is None or status not in self.STATUSES:
            raise ValueError("unknown configuration ID or status")
        if (not isinstance(evidence_references, list) or not evidence_references or
                any(not isinstance(x, str) or not x for x in evidence_references)):
            raise ValueError("status event requires evidence references")
        if recovery_evidence is not None and (
                not isinstance(recovery_evidence, list) or
                any(not isinstance(x, str) or not x for x in recovery_evidence)):
            raise ValueError("recovery evidence must be a list of nonempty references")
        if recovery_evidence and status != "pending":
            raise ValueError("recovery evidence is only valid on a fresh pending attempt")
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("RunRecord ID must be nonempty or null")
        if status in {"pending", "blocked"} and run_id is not None:
            raise ValueError("pre-run pending/blocked states cannot have a RunRecord ID")
        if status in {"failed", "interrupted", "completed"} and run_id is None:
            raise ValueError("terminal run outcome requires a RunRecord ID")
        if status == "pending" and outcome_class is not None:
            raise ValueError("pending state cannot have an outcome class")
        if status == "blocked" and outcome_class != "dependency_blocked":
            raise ValueError("pre-run blocked state requires dependency_blocked classification")
        if status in {"failed", "interrupted"} and outcome_class not in self.FAILURE_CLASSES:
            raise ValueError("run failure requires a recognized outcome class")
        if status == "completed" and outcome_class not in {None, "scientific_negative"}:
            raise ValueError("completed run may only record a valid scientific outcome")
        prior = self.records()
        if run_id is not None and any(r.get("run_id") == run_id for r in prior):
            raise ValueError("RunRecord ID already recorded; history is append-only")
        events = [r for r in prior if r.get("config_id") == config_id]
        last = events[-1] if events else None
        if last is not None and last["config_sha256"] != _config_hash(config):
            raise ValueError("scientific configuration changed after ledger entry")
        if last is None:
            if status not in {"pending", "blocked"}:
                raise ValueError("configuration must first be recorded pending or dependency-blocked")
            attempt = 1
        elif status == "pending":
            if last["status"] == "blocked":
                if recovery_evidence:
                    raise ValueError("dependency unblock does not accept retry evidence")
                attempt = last["attempt"]
            elif last["status"] == "failed" and last.get("outcome_class") == "external_operational":
                attempt = last["attempt"] + 1
                if attempt > 2 or not recovery_evidence:
                    raise ValueError("retry requires correction evidence and is limited to one fresh attempt")
            else:
                raise ValueError("invalid transition to pending; previous outcome is not retryable")
        else:
            attempt = last["attempt"]
            valid = ((last["status"] == "pending" and status in {"blocked", "failed", "interrupted", "completed"}) or
                     (last["status"] == "blocked" and status == "blocked"))
            if not valid:
                raise ValueError("invalid status transition; append-only history cannot be overwritten")
        expected_attempt_id = f"{config_id}-A{attempt:02d}"
        if attempt_id is not None and attempt_id != expected_attempt_id:
            raise ValueError("attempt ID does not match configuration and attempt sequence")
        record = {"config_id": config_id, "config_sha256": _config_hash(config),
                  "attempt_id": expected_attempt_id, "attempt": attempt, "status": status,
                  "run_id": run_id, "outcome_class": outcome_class,
                  "evidence_references": evidence_references,
                  "recovery_evidence": recovery_evidence or []}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
