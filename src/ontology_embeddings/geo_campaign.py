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
    """Append-only JSONL history; each attempt is distinct from its config and RunRecord ID."""
    STATUSES = {"pending", "blocked", "failed", "interrupted", "completed"}

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("invalid ledger record")
            result.append(record)
        return result

    def append(self, *, config_id: str, run_id: str, status: str,
               evidence_references: list[str], outcome_class: str | None = None,
               recovery_evidence: list[str] | None = None) -> None:
        config = next((c for c in CONFIGURATIONS if c["config_id"] == config_id), None)
        if config is None:
            raise ValueError("unknown configuration ID")
        if not isinstance(run_id, str) or not run_id.strip() or status not in self.STATUSES:
            raise ValueError("invalid RunRecord ID or status")
        if status == "pending":
            raise ValueError("pending is a manifest state, not an attempt outcome")
        if not evidence_references or any(not isinstance(x, str) or not x for x in evidence_references):
            raise ValueError("attempt requires evidence references")
        prior = self.records()
        if any(r.get("run_id") == run_id for r in prior):
            raise ValueError("RunRecord ID already recorded; history is append-only")
        attempts = [r for r in prior if r.get("config_id") == config_id]
        if attempts:
            last = attempts[-1]
            allowed_recovery = (last["status"] == "failed" and
                last.get("outcome_class") == "external_operational" and
                recovery_evidence)
            if not allowed_recovery or len(attempts) >= 2:
                raise ValueError("retry not authorized; retain outcome without another attempt")
            if last["config_sha256"] != _config_hash(config):
                raise ValueError("configuration changed between attempts")
        record = {"config_id": config_id, "config_sha256": _config_hash(config),
                  "run_id": run_id, "attempt": len(attempts) + 1, "status": status,
                  "outcome_class": outcome_class, "evidence_references": evidence_references,
                  "recovery_evidence": recovery_evidence or []}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
