"""Fail-closed PR3 BOX controls, paired-run driver, assessment and evidence export.

Only the original train and validation records are loaded by the shared safe loader;
test JSONL rows are skipped before decoding. This module does not launch runs on import.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch

from .affine_smoke import load_train_validation
from .box import BoxGeometry, box_clearance, box_membership, box_objective, decode_geometry
from .box_training import BoxMultiLabelModel, train_box
from .evaluation import class_frequency_baseline
from .features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from .run_store import RunRecord, source_provenance
from .training import _pack

DATASET_ID = "pizza-6a33710d13eb90fdde3e"
HISTORICAL_RUN = "20261123T105917-c00ea4941c34"
HISTORICAL_SOURCE = "522c3f30240e65a431abdfc7694d0006866b0874"
PAIR_TOLERANCE = 2e-5
MIN_PREFLIGHT_FREE_BYTES = 64 * 1024 * 1024
CONFIGS = {
    "bounded": {"domain": "bounded", "activation": "tanh"},
    "unbounded": {"domain": "unbounded", "activation": "identity"},
}
COMMON = {
    "latent_dim": 16,
    "boxes": 1,
    "epochs": 60,
    "batch_size": 64,
    "seed": 23,
    "learning_rate": 0.02,
    "tau": 0.1,
    "clip_norm": 1.0,
    "epsilon": 1e-4,
    "weight_decay": 0,
    "objective": "logistic",
}


def verify_cuda_runtime(device: str) -> dict[str, Any]:
    """Validate an available requested CUDA device and record the actual runtime."""
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("PR3 substantial runs require an available CUDA device")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    actual = torch.device("cuda", index)
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(actual),
        "device_name": torch.cuda.get_device_name(actual),
    }


def require_ignored_artifact_path(path: Path | str) -> None:
    """Reject local run/preparation artifacts under the checkout unless Git ignores them."""
    target = Path(path).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    try:
        target.relative_to(source_root)
    except ValueError:
        return
    check = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(target)],
        cwd=source_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check.returncode != 0:
        raise ValueError(f"artifact path inside checkout is not ignored: {target}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_ids(ids: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


_PREPARED_DATASET_FIELDS = {
    "dataset_id",
    "manifest_sha256",
    "class_manifest_sha256",
    "class_ids",
    "class_names",
    "class_iris",
    "fixed_positive_class_ids",
    "unsupported_class_ids",
    "train_ids_sha256",
    "validation_ids_sha256",
    "train_count",
    "validation_count",
}
_RUN_DATASET_FIELDS = {
    "dataset_id",
    "dataset_manifest_sha256",
    "class_manifest_sha256",
    "class_ids",
    "class_names",
    "class_iris",
    "fixed_positive_class_ids",
    "unsupported_class_ids",
    "train_sample_ids",
    "validation_sample_ids",
}


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(type(item) is int for item in value)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _required_config_matches(config: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        return False
    integer_keys = {"latent_dim", "boxes", "epochs", "batch_size", "seed"}
    numeric_keys = {
        "learning_rate",
        "tau",
        "epsilon",
        "clip_norm",
        "weight_decay",
        "margin",
        "optimizer_epsilon",
    }
    for key, value in expected.items():
        actual = config.get(key)
        if key in integer_keys and type(actual) is not int:
            return False
        if key in numeric_keys and (
            not isinstance(actual, (int, float)) or isinstance(actual, bool)
        ):
            return False
        if key == "amsgrad" and type(actual) is not bool:
            return False
        if key in {"betas"} and (
            not isinstance(actual, list)
            or any(not isinstance(item, (int, float)) or isinstance(item, bool) for item in actual)
        ):
            return False
        if key == "optimizer_groups" and not _is_string_list(actual):
            return False
        if key in {"model", "domain", "objective", "optimizer", "device"} and not isinstance(
            actual, str
        ):
            return False
        if actual != value:
            return False
    return True


def _valid_prepared_dataset_identity(identity: Any) -> bool:
    if not isinstance(identity, dict) or set(identity) != _PREPARED_DATASET_FIELDS:
        return False
    class_ids = identity["class_ids"]
    class_names = identity["class_names"]
    class_iris = identity["class_iris"]
    return (
        isinstance(identity["dataset_id"], str)
        and bool(identity["dataset_id"])
        and _is_sha256(identity["manifest_sha256"])
        and _is_sha256(identity["class_manifest_sha256"])
        and _is_int_list(class_ids)
        and bool(class_ids)
        and _is_string_list(class_names)
        and len(class_names) == len(class_ids)
        and _is_string_list(class_iris)
        and len(class_iris) == len(class_ids)
        and _is_int_list(identity["fixed_positive_class_ids"])
        and _is_int_list(identity["unsupported_class_ids"])
        and _is_sha256(identity["train_ids_sha256"])
        and _is_sha256(identity["validation_ids_sha256"])
        and type(identity["train_count"]) is int
        and identity["train_count"] == 720
        and type(identity["validation_count"]) is int
        and identity["validation_count"] == 240
    )


def _normalize_run_dataset_identity(run_dataset: Any) -> dict[str, Any] | None:
    if not isinstance(run_dataset, dict) or set(run_dataset) != _RUN_DATASET_FIELDS:
        return None
    class_ids = run_dataset["class_ids"]
    class_names = run_dataset["class_names"]
    class_iris = run_dataset["class_iris"]
    train_ids = run_dataset["train_sample_ids"]
    validation_ids = run_dataset["validation_sample_ids"]
    if (
        not isinstance(run_dataset["dataset_id"], str)
        or not run_dataset["dataset_id"]
        or not _is_sha256(run_dataset["dataset_manifest_sha256"])
        or not _is_sha256(run_dataset["class_manifest_sha256"])
        or not _is_int_list(class_ids)
        or not class_ids
        or not _is_string_list(class_names)
        or len(class_names) != len(class_ids)
        or not _is_string_list(class_iris)
        or len(class_iris) != len(class_ids)
        or not _is_int_list(run_dataset["fixed_positive_class_ids"])
        or not _is_int_list(run_dataset["unsupported_class_ids"])
        or not _is_string_list(train_ids)
        or len(train_ids) != 720
        or not _is_string_list(validation_ids)
        or len(validation_ids) != 240
    ):
        return None
    return {
        "dataset_id": run_dataset["dataset_id"],
        "manifest_sha256": run_dataset["dataset_manifest_sha256"],
        "class_manifest_sha256": run_dataset["class_manifest_sha256"],
        "class_ids": class_ids,
        "class_names": class_names,
        "class_iris": class_iris,
        "fixed_positive_class_ids": run_dataset["fixed_positive_class_ids"],
        "unsupported_class_ids": run_dataset["unsupported_class_ids"],
        "train_ids_sha256": _hash_ids(train_ids),
        "validation_ids_sha256": _hash_ids(validation_ids),
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
    }


def _dataset_identity(dataset, train_records, validation_records, selections):
    return {
        "dataset_id": dataset.dataset_id,
        "manifest_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "class_ids": list(dataset.canonical_class_ids),
        "class_names": list(dataset.class_names),
        "class_iris": list(dataset.class_iris),
        "fixed_positive_class_ids": list(dataset.fixed_positive_class_ids),
        "unsupported_class_ids": list(dataset.unsupported_class_ids),
        "train_ids_sha256": selections["train_ids_sha256"],
        "validation_ids_sha256": selections["validation_ids_sha256"],
        "train_count": len(train_records),
        "validation_count": len(validation_records),
    }


def validate_data_identity(dataset, train_records, validation_records, historical_manifest):
    """Require the exact immutable dataset and historical train/validation selection."""
    historical = historical_manifest
    if historical.get("run_id") != HISTORICAL_RUN:
        raise ValueError("historical selection run ID mismatch")
    source = historical.get("source", {})
    if source.get("commit_sha") != HISTORICAL_SOURCE or source.get("dirty") is not False:
        raise ValueError("historical selection source is not the approved clean revision")
    old_data = historical.get("dataset", {})
    if old_data.get("dataset_id") != DATASET_ID:
        raise ValueError("historical run dataset ID mismatch")
    if dataset.dataset_manifest_sha256 != old_data.get("manifest_sha256"):
        raise ValueError("dataset manifest differs from the historical run")
    if dataset.class_manifest_sha256 != old_data.get("class_manifest_sha256"):
        raise ValueError("class manifest differs from the historical run")
    if dataset.dataset_id != DATASET_ID:
        raise ValueError(f"expected dataset {DATASET_ID}, got {dataset.dataset_id}")
    train_ids = [r.sample_id for r in train_records]
    validation_ids = [r.sample_id for r in validation_records]
    if train_ids != old_data.get("train_sample_ids"):
        raise ValueError("train IDs differ from the historical PR2 selection")
    if validation_ids != old_data.get("validation_sample_ids"):
        raise ValueError("validation IDs differ from the historical PR2 selection")
    if len(train_ids) != 720 or len(validation_ids) != 240:
        raise ValueError("required original split sizes are 720 train and 240 validation")
    if (
        set(train_ids) & set(validation_ids)
        or any(r.split != "train" for r in train_records)
        or any(r.split != "validation" for r in validation_records)
    ):
        raise ValueError("train/validation split contamination")
    return {
        "train_ids_sha256": _hash_ids(train_ids),
        "validation_ids_sha256": _hash_ids(validation_ids),
    }


def _task_baselines(dataset, train_records, validation_records):
    result = {}
    for task in ("pizza", "component"):
        tx = _pack(train_records, dataset, torch.device("cpu"))
        vx = _pack(validation_records, dataset, torch.device("cpu"))
        if task == "pizza":
            train_y, train_known, val_y, val_known = tx[1], tx[2], vx[1], vx[2]
            parent_ids = None
            applicable_ids = dataset.pizza_supervision_class_ids
        else:
            train_y, train_known, val_y, val_known = tx[4], tx[5], vx[4], vx[5]
            parent_ids = vx[6]
            applicable_ids = dataset.component_supervision_class_ids
        train_app = torch.tensor([i in applicable_ids for i in dataset.canonical_class_ids])
        val_app = train_app.expand_as(val_y)
        result[task] = class_frequency_baseline(
            train_y,
            train_known,
            train_app.expand_as(train_y),
            val_y,
            val_known,
            val_app,
            dataset.class_names,
            parent_ids,
            dataset.canonical_class_ids,
        )
    return result


def preflight(
    dataset_path: Path | str,
    *,
    historical_manifest_path: Path | str,
    device: str,
    export_destination: Path | str,
    artifact_root: Path | str | None = None,
):
    """Perform every expensive-run gate before registering or executing training."""
    destination = Path(export_destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"evidence destination already exists: {destination}")
    parent = destination.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        raise ValueError(f"evidence destination parent must exist and be writable: {parent}")
    free_bytes = shutil.disk_usage(parent).free
    if free_bytes < MIN_PREFLIGHT_FREE_BYTES:
        raise ValueError(
            f"evidence destination parent has insufficient free space: {free_bytes} bytes"
        )
    artifact_preflight = None
    if artifact_root is not None:
        root = Path(artifact_root).expanduser().resolve()
        require_ignored_artifact_path(root)
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
            raise ValueError(f"run artifact root must be writable: {root}")
        if shutil.disk_usage(root).free < MIN_PREFLIGHT_FREE_BYTES:
            raise ValueError("run artifact root has insufficient free space")
        artifact_preflight = {
            "path": str(root),
            "writable": True,
            "free_bytes": shutil.disk_usage(root).free,
        }
    runtime = verify_cuda_runtime(device)
    dataset, train_records, validation_records = load_train_validation(Path(dataset_path))
    historical = json.loads(Path(historical_manifest_path).read_text(encoding="utf-8"))
    selections = validate_data_identity(dataset, train_records, validation_records, historical)
    # Count split tokens only; never decode test rows or consume their labels/statistics.
    counts = {"train": 0, "validation": 0, "test": 0}
    with (Path(dataset_path) / "samples.jsonl").open(encoding="utf-8") as samples:
        for line in samples:
            match = re.search(r'"split"\s*:\s*"(train|validation|test)"', line)
            if match is None:
                raise ValueError("sample line lacks an explicit split token")
            counts[match.group(1)] += 1
    if counts != {"train": 720, "validation": 240, "test": 240}:
        raise ValueError("dataset line split counts differ from the approved 720/240/240")
    baselines = _task_baselines(dataset, train_records, validation_records)
    for task, report in baselines.items():
        if report["metrics"]["micro"]["f1"] is None:
            raise ValueError(f"train-only {task} baseline has undefined validation micro-F1")
    return {
        "dataset": dataset,
        "train_records": train_records,
        "validation_records": validation_records,
        "baselines": baselines,
        "runtime": runtime,
        "selections": selections,
        "destination": destination,
        "artifact_preflight": artifact_preflight,
        "dataset_hashes": {
            "manifest": _sha(Path(dataset_path) / "manifest.json"),
            "samples": _sha(Path(dataset_path) / "samples.jsonl"),
            "vocabulary": _sha(Path(dataset_path) / "vocabularies/ontology_classes.json"),
        },
    }


CONTROL_EPOCHS = 350
_CONTROL_POINTS = (
    (-0.15, -0.12),
    (0.12, 0.10),
    (0.55, 0.0),
    (0.0, 0.55),
    (-0.55, 0.0),
    (0.0, -0.55),
)
_CONTROL_TARGETS = ((1.0,), (1.0,), (0.0,), (0.0,), (0.0,), (0.0,))
_CONTROL_MEMBERSHIP = [True, True, False, False, False, False]


def _geometry_control_spec(domain: str, objective: str, epochs: int):
    points = torch.tensor(_CONTROL_POINTS, dtype=torch.float32)
    targets = torch.tensor(_CONTROL_TARGETS, dtype=torch.float32)
    config = {
        "model": "fixed-coordinate-geometry-only-control",
        "domain": domain,
        "objective": objective,
        "dimension": 2,
        "boxes": 1,
        "epsilon": 1e-4,
        "dtype": str(points.dtype),
        "epochs": epochs,
        "learning_rate": 0.04,
        "optimizer": "Adam",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "weight_decay": 0,
        "batch_size": len(points),
        "clip_norm": 1.0,
        "seed": 23,
        "device": "cpu",
        "points": points.tolist(),
        "targets": targets.tolist(),
        "expected_membership": _CONTROL_MEMBERSHIP,
        "tau": 0.1 if objective == "logistic" else None,
        "margin": 0.1 if objective == "hinge" else None,
        "points_sha256": hashlib.sha256(points.numpy().tobytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(targets.numpy().tobytes()).hexdigest(),
    }
    return points, targets, config


def _save_control(geometry, path: Path, epoch: int, loss: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "domain": geometry.domain,
            "state_dict": geometry.state_dict(),
        },
        path,
    )


def run_geometry_controls(
    *, artifact_root: Path | str = "artifacts/runs", epochs: int = CONTROL_EPOCHS
) -> list[dict[str, Any]]:
    """Run the four prescribed fixed-coordinate geometry-only recovery controls."""
    require_ignored_artifact_path(artifact_root)
    results = []
    for domain in ("bounded", "unbounded"):
        for objective in ("logistic", "hinge"):
            points, targets, config = _geometry_control_spec(domain, objective, epochs)
            rec = RunRecord(
                artifact_root,
                purpose="pr3-geometry-recovery-control",
                invocation="python -m ontology_embeddings.pr3_acceptance controls",
                config=config,
                dataset={"synthetic_fixed_coordinates": True},
                seed=23,
                issue="14",
            )
            rec.update(
                environment={
                    "python": sys.version,
                    "torch": str(torch.__version__),
                    "cuda": torch.version.cuda,
                    "device": "cpu",
                    "platform": sys.platform,
                }
            )
            try:
                torch.manual_seed(23)
                initial_lo = torch.tensor([[[-0.7, -0.7]]], dtype=torch.float32)
                initial_hi = torch.tensor([[[0.7, 0.7]]], dtype=torch.float32)
                from .box import inverse_geometry

                raw_a, raw_b = inverse_geometry(initial_lo, initial_hi, domain=domain, epsilon=1e-4)
                geometry = BoxGeometry(raw_a, raw_b, domain=domain, epsilon=1e-4)
                optimizer = torch.optim.Adam(geometry.parameters(), lr=0.04)
                checkpoint_dir = rec.path / "checkpoints"
                initial_parameters = {
                    key: value.detach().clone() for key, value in geometry.state_dict().items()
                }
                initial_lower, initial_upper = geometry.endpoints()
                initial_corners = {
                    "lower": initial_lower.detach().tolist(),
                    "upper": initial_upper.detach().tolist(),
                }
                max_gradient_norm = 0.0
                initial_scores = box_clearance(
                    points, *geometry.endpoints(), domain=domain, epsilon=1e-4
                )
                initial_loss = float(
                    box_objective(initial_scores, targets, kind=objective, tau=0.1, margin=0.1)
                    .mean()
                    .detach()
                )
                _save_control(geometry, checkpoint_dir / "initial.pt", 0, initial_loss)
                _save_control(geometry, checkpoint_dir / "latest.pt", 0, initial_loss)
                rec.update(progress={"epoch": 0, "loss": initial_loss}, last_epoch=0)
                last_loss = initial_loss
                for epoch in range(1, epochs + 1):
                    optimizer.zero_grad(set_to_none=True)
                    loss = box_objective(
                        box_clearance(points, *geometry.endpoints(), domain=domain, epsilon=1e-4),
                        targets,
                        kind=objective,
                        tau=0.1,
                        margin=0.1,
                    ).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite control objective")
                    loss.backward()
                    gradients = [parameter.grad for parameter in geometry.parameters()]
                    if not gradients or any(
                        not torch.isfinite(gradient).all() for gradient in gradients
                    ):
                        raise FloatingPointError("missing or non-finite geometry gradient")
                    grad_norm = float(
                        torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in gradients))
                    )
                    max_gradient_norm = max(max_gradient_norm, grad_norm)
                    torch.nn.utils.clip_grad_norm_(geometry.parameters(), 1.0)
                    optimizer.step()
                    last_loss = float(loss.detach())
                    _save_control(geometry, checkpoint_dir / "latest.pt", epoch, last_loss)
                    rec.metric(
                        {
                            "epoch": epoch,
                            "loss": last_loss,
                            "geometry_gradient_norm": grad_norm,
                        }
                    )
                    rec.update(last_epoch=epoch, progress={"epoch": epoch, "loss": last_loss})
                lower, upper = geometry.endpoints()
                prediction = box_membership(points, lower, upper, domain=domain, epsilon=1e-4)
                recovered = prediction.squeeze(1).tolist() == _CONTROL_MEMBERSHIP
                result = {
                    "run_id": rec.run_id,
                    "run_path": str(rec.path),
                    "config": config,
                    "domain": domain,
                    "objective": objective,
                    "status": "completed",
                    "epochs": epochs,
                    "initial_loss": initial_loss,
                    "final_loss": last_loss,
                    "target_recovered": recovered,
                    "membership": prediction.squeeze(1).tolist(),
                    "geometry_updated": any(
                        not torch.equal(initial_parameters[key], value)
                        for key, value in geometry.state_dict().items()
                    ),
                    "geometry_gradient_norm_max": max_gradient_norm,
                    "geometry_gradient_finite": torch.isfinite(
                        torch.tensor(max_gradient_norm)
                    ).item(),
                    "initial_corners": initial_corners,
                    "final_corners": {
                        "lower": lower.detach().tolist(),
                        "upper": upper.detach().tolist(),
                    },
                    "lower": lower.detach().tolist(),
                    "upper": upper.detach().tolist(),
                    "checkpoint_paths": {
                        kind: str(checkpoint_dir / f"{kind}.pt") for kind in ("initial", "latest")
                    },
                }
                _save_control(geometry, checkpoint_dir / "checkpoint.pt", epochs, last_loss)
                result["checkpoint_paths"]["checkpoint"] = str(checkpoint_dir / "checkpoint.pt")
                result["metrics_sha256"] = _sha(rec.path / "metrics.jsonl")
                rec.update(status="completed", result=result)
                results.append(result)
            except KeyboardInterrupt as error:
                rec.interrupt(error)
                raise
            except BaseException as error:
                checkpoint_dir = rec.path / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                receipt = {
                    "status": "failed_before_recovery_state",
                    "error": f"{type(error).__name__}: {error}",
                }
                for name in ("initial.pt", "latest.pt", "checkpoint.pt"):
                    target = checkpoint_dir / name
                    if not target.exists():
                        torch.save(receipt, target)
                rec.fail(error)
                results.append(
                    {
                        "run_id": rec.run_id,
                        "config": config,
                        "domain": domain,
                        "objective": objective,
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return results


def create_preparation(
    data: Path | str,
    *,
    historical_manifest: Path | str,
    device: str,
    export_destination: Path | str,
    work_dir: Path | str,
    artifact_root: Path | str,
) -> dict[str, Any]:
    """Validate all gates and save recoverable paired initial adapter/physical bounds."""
    plan = preflight(
        data,
        historical_manifest_path=historical_manifest,
        device=device,
        export_destination=export_destination,
        artifact_root=artifact_root,
    )
    work = Path(work_dir).resolve()
    require_ignored_artifact_path(work)
    require_ignored_artifact_path(artifact_root)
    if work.exists():
        raise FileExistsError(f"preparation work directory already exists: {work}")
    work.mkdir(parents=True)
    torch.manual_seed(COMMON["seed"])
    reference = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        plan["dataset"].canonical_class_ids,
        plan["dataset"].class_manifest_sha256,
        latent_dim=16,
        boxes=1,
        domain="bounded",
        seed=23,
        class_names=plan["dataset"].class_names,
        class_iris=plan["dataset"].class_iris,
    )
    shared = {
        key: value.detach().cpu().clone()
        for key, value in reference.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    lower, upper = reference.endpoints()
    torch.save(
        {"adapters": shared, "lower": lower.detach().cpu(), "upper": upper.detach().cpu()},
        work / "paired_initial.pt",
    )
    prep = {
        "schema": "pr3-preparation-v1",
        "created": True,
        "data_path_local_only": str(Path(data).resolve()),
        "device": device,
        "runtime": plan["runtime"],
        "destination": str(plan["destination"]),
        "artifact_root": str(Path(artifact_root).resolve()),
        "dataset_id": plan["dataset"].dataset_id,
        "dataset_identity": _dataset_identity(
            plan["dataset"], plan["train_records"], plan["validation_records"], plan["selections"]
        ),
        "dataset_hashes": plan["dataset_hashes"],
        "selection_hashes": plan["selections"],
        "source": source_provenance(Path(__file__).resolve().parents[2]),
        "artifact_preflight": plan["artifact_preflight"],
        "config": COMMON,
        "baselines": plan["baselines"],
        "paired_initial_sha256": _sha(work / "paired_initial.pt"),
        "limitations": [
            "No test rows decoded, used, tuned, or evaluated.",
            "Source data is not copied into the evidence bundle.",
        ],
    }
    (work / "prepare.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")
    # Persist train/validation-only decoded records for separate coordinator run invocations.
    torch.save(
        {
            "dataset": plan["dataset"],
            "train_records": plan["train_records"],
            "validation_records": plan["validation_records"],
        },
        work / "train_validation.pt",
    )
    prep["train_validation_sha256"] = _sha(work / "train_validation.pt")
    (work / "prepare.json").write_text(json.dumps(prep, indent=2, sort_keys=True) + "\n")
    return prep


def run_domain(work_dir: Path | str, domain: str, *, device: str = "cuda") -> dict[str, Any]:
    if domain not in CONFIGS:
        raise ValueError("domain must be bounded or unbounded")
    work = Path(work_dir).resolve()
    prep = json.loads((work / "prepare.json").read_text())
    run_config = {**COMMON, **CONFIGS[domain], "margin": 0.1, "model": "box-union-two-adapters"}
    rec = RunRecord(
        prep["artifact_root"],
        purpose="pr3-required-metadata-run",
        invocation=f"python -m ontology_embeddings.pr3_acceptance run-domain --domain {domain}",
        config=run_config,
        dataset=prep["dataset_identity"],
        seed=23,
        issue="14",
    )
    try:
        if _sha(work / "paired_initial.pt") != prep["paired_initial_sha256"]:
            raise ValueError("paired initialization artifact checksum mismatch")
        prepared_data_path = work / "train_validation.pt"
        if _sha(prepared_data_path) != prep.get("train_validation_sha256"):
            raise ValueError("train/validation preparation bundle checksum mismatch")
        records = torch.load(prepared_data_path, map_location="cpu", weights_only=False)
        observed = _dataset_identity(
            records["dataset"],
            records["train_records"],
            records["validation_records"],
            {
                "train_ids_sha256": _hash_ids([r.sample_id for r in records["train_records"]]),
                "validation_ids_sha256": _hash_ids(
                    [r.sample_id for r in records["validation_records"]]
                ),
            },
        )
        prepared_identity = prep.get("dataset_identity")
        if (
            not _valid_prepared_dataset_identity(prepared_identity)
            or prepared_identity.get("dataset_id") != DATASET_ID
            or observed != prepared_identity
        ):
            raise ValueError("train/validation data identity differs from preparation manifest")
        paired = torch.load(work / "paired_initial.pt", map_location="cpu", weights_only=False)
        runtime = verify_cuda_runtime(device)
        if device != prep["device"] or runtime != prep["runtime"]:
            raise RuntimeError("requested device/runtime does not match preflight CUDA runtime")
        bounds = (paired["lower"].to(device), paired["upper"].to(device))
        from .box import inverse_geometry

        for paired_domain in CONFIGS:
            decoded = decode_geometry(
                *inverse_geometry(*bounds, domain=paired_domain), domain=paired_domain
            )
            error = max(
                float((decoded[0] - bounds[0]).abs().max()),
                float((decoded[1] - bounds[1]).abs().max()),
            )
            if error > PAIR_TOLERANCE:
                raise ValueError(
                    f"paired {paired_domain} bounds reconstruction exceeds tolerance: {error}"
                )
        return train_box(
            records["dataset"],
            records["train_records"],
            records["validation_records"],
            artifact_root=prep["artifact_root"],
            purpose="pr3-required-metadata-run",
            invocation=f"python -m ontology_embeddings.pr3_acceptance run-domain --domain {domain}",
            seed=23,
            epochs=60,
            batch_size=64,
            learning_rate=0.02,
            latent_dim=16,
            boxes=1,
            domain=domain,
            epsilon=1e-4,
            objective="logistic",
            tau=0.1,
            margin=0.1,
            device=device,
            run_record=rec,
            shared_adapter_initial_state={
                key: value.to(device) for key, value in paired["adapters"].items()
            },
            initial_box_bounds=bounds,
            initial_box_class_ids=records["dataset"].canonical_class_ids,
        )
    except KeyboardInterrupt as error:
        if rec.manifest.get("status") == "running":
            rec.interrupt(error)
        raise
    except BaseException as error:
        checkpoint_dir = rec.path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        receipt = {
            "status": "failed_before_checkpoint",
            "error": f"{type(error).__name__}: {error}",
        }
        for name in ("initial.pt", "latest.pt", "checkpoint.pt"):
            target = checkpoint_dir / name
            if not target.exists():
                torch.save(receipt, target)
        if rec.manifest.get("status") == "running":
            rec.fail(error)
        raise


def validate_paired_results(run_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Hash and compare persisted adapter tensors and validate physical snapshots."""
    failed_domains = [
        domain
        for domain in CONFIGS
        if not isinstance(run_results.get(domain), dict)
        or run_results[domain].get("status") != "completed"
    ]
    if failed_domains:
        return {
            "status": "not-verified-incomplete-run",
            "failed_domains": failed_domains,
            "reconstruction_tolerance_abs": PAIR_TOLERANCE,
        }
    adapter_keys = {
        "pizza_adapter.weight",
        "pizza_adapter.bias",
        "component_adapter.weight",
        "component_adapter.bias",
    }
    checkpoints = {}
    for domain in CONFIGS:
        result = run_results[domain]
        run_path_value = result.get("run_path")
        run_id = result.get("run_id")
        checkpoint_paths = result.get("checkpoint_paths")
        if (
            not isinstance(run_path_value, str)
            or not isinstance(run_id, str)
            or result.get("status") != "completed"
            or not isinstance(checkpoint_paths, dict)
        ):
            raise ValueError(f"{domain} initial checkpoint lacks completed run identity")
        run_path = Path(run_path_value).resolve()
        path = Path(checkpoint_paths.get("initial", "")).resolve()
        expected_initial_path = (run_path / "checkpoints" / "initial.pt").resolve()
        if path != expected_initial_path or not path.is_file():
            raise ValueError(f"{domain} initial checkpoint is not bound to its run directory")
        try:
            manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{domain} initial checkpoint run provenance is unreadable: {error}")
        if (
            not isinstance(manifest, dict)
            or not isinstance(summary, dict)
            or not isinstance(payload, dict)
            or manifest.get("run_id") != run_id
            or summary.get("run_id") != run_id
            or manifest.get("status") != "completed"
            or summary.get("status") != "completed"
            or not _json_values_equal(manifest.get("summary"), summary)
            or not _json_values_equal(summary.get("run_path"), str(run_path))
            or not _json_values_equal(summary.get("checkpoint_paths"), checkpoint_paths)
            or not _json_values_equal(checkpoint_paths.get("initial"), str(expected_initial_path))
            or not _json_values_equal(result.get("run_path"), summary.get("run_path"))
            or not _json_values_equal(result.get("run_id"), summary.get("run_id"))
            or not _json_values_equal(result.get("status"), summary.get("status"))
            or not _json_values_equal(manifest.get("purpose"), "pr3-required-metadata-run")
            or not _json_values_equal(manifest.get("issue"), "14")
            or type(manifest.get("seed")) is not int
            or manifest.get("seed") != COMMON["seed"]
        ):
            raise ValueError(f"{domain} initial checkpoint run identity/status/path mismatch")
        run_config = manifest.get("config")
        expected_run_config = {
            "model": "box-union-two-adapters",
            "domain": domain,
            "objective": "logistic",
            "latent_dim": COMMON["latent_dim"],
            "boxes": COMMON["boxes"],
            "epochs": COMMON["epochs"],
            "batch_size": COMMON["batch_size"],
            "seed": COMMON["seed"],
            "learning_rate": COMMON["learning_rate"],
            "tau": COMMON["tau"],
            "epsilon": COMMON["epsilon"],
            "clip_norm": COMMON["clip_norm"],
            "weight_decay": COMMON["weight_decay"],
            "margin": 0.1,
            "optimizer": "Adam",
            "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
            "betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "amsgrad": False,
            "device": run_config.get("device") if isinstance(run_config, dict) else None,
        }
        expected_run_keys = set(expected_run_config) | {"paired_initialization"}
        metadata = payload.get("metadata")
        if (
            not isinstance(run_config, dict)
            or set(run_config) != expected_run_keys
            or not isinstance(run_config.get("device"), str)
            or not _required_config_matches(run_config, expected_run_config)
            or not isinstance(metadata, dict)
            or metadata.get("domain") != domain
            or type(payload.get("epoch")) is not int
            or payload.get("epoch") != 0
            or payload.get("validation_loss") is not None
            or not _json_values_equal(metadata.get("training_config"), run_config)
        ):
            raise ValueError(f"{domain} initial checkpoint domain/config/epoch mismatch")
        dataset = manifest.get("dataset")
        if (
            not isinstance(dataset, dict)
            or not _is_int_list(metadata.get("class_ids"))
            or not _is_string_list(metadata.get("class_names"))
            or not _is_string_list(metadata.get("class_iris"))
            or not _json_values_equal(metadata.get("class_ids"), dataset.get("class_ids"))
            or not _json_values_equal(metadata.get("class_names"), dataset.get("class_names"))
            or not _json_values_equal(metadata.get("class_iris"), dataset.get("class_iris"))
            or not _json_values_equal(
                metadata.get("vocabulary_sha256"), dataset.get("class_manifest_sha256")
            )
        ):
            raise ValueError(f"{domain} initial checkpoint class identity mismatch")
        config = run_config["paired_initialization"]
        latent_dim = metadata.get("latent_dim")
        encoder = metadata.get("encoder", {})
        objective_metadata = metadata.get("objective", {})
        if (
            type(latent_dim) is not int
            or latent_dim != COMMON["latent_dim"]
            or type(metadata.get("boxes")) is not int
            or metadata.get("boxes") != COMMON["boxes"]
            or not _finite_number(metadata.get("epsilon"))
            or metadata.get("epsilon") != COMMON["epsilon"]
            or not isinstance(encoder, dict)
            or type(encoder.get("pizza_features")) is not int
            or encoder.get("pizza_features") != len(PIZZA_FEATURE_NAMES)
            or type(encoder.get("component_features")) is not int
            or encoder.get("component_features") != len(COMPONENT_FEATURE_NAMES)
            or encoder.get("bounded_activation") != CONFIGS[domain]["activation"]
            or not isinstance(objective_metadata, dict)
            or not _json_values_equal(
                objective_metadata, {"kind": "logistic", "tau": COMMON["tau"], "margin": 0.1}
            )
        ):
            raise ValueError(f"{domain} initial checkpoint has invalid model/objective metadata")
        expected_shapes = {
            "pizza_adapter.weight": (latent_dim, encoder.get("pizza_features")),
            "pizza_adapter.bias": (latent_dim,),
            "component_adapter.weight": (latent_dim, encoder.get("component_features")),
            "component_adapter.bias": (latent_dim,),
        }
        if type(latent_dim) is not int or any(
            not isinstance(size, int) or size <= 0
            for shape in expected_shapes.values()
            for size in shape
        ):
            raise ValueError("paired checkpoint adapter dimensions are missing or invalid")
        tensors = payload.get("model_state", {})
        hashes = config.get("adapter_sha256", {})
        if set(hashes) != adapter_keys or not adapter_keys.issubset(tensors):
            raise ValueError("paired adapter checkpoint must contain exact four adapter tensors")
        actual_hashes = {}
        for key in sorted(adapter_keys):
            tensor = tensors[key]
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid or non-finite adapter tensor {key}")
            if tensor.shape != expected_shapes[key]:
                raise ValueError(f"adapter tensor has unexpected shape {key}")
            if not (tensor.is_floating_point() and tensor.dtype == torch.float32):
                raise ValueError(f"adapter tensor has unexpected dtype {key}")
            actual_hashes[key] = hashlib.sha256(
                tensor.detach().contiguous().cpu().numpy().tobytes()
            ).hexdigest()
            if hashes.get(key) != actual_hashes[key]:
                raise ValueError(f"adapter tensor hash disagrees with metadata for {key}")
        checkpoints[domain] = (payload, config, tensors, actual_hashes)
    bounded, unbounded = checkpoints["bounded"], checkpoints["unbounded"]
    for key in sorted(adapter_keys):
        left, right = bounded[2][key], unbounded[2][key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"paired initial adapter tensors differ across domains: {key}")
    bounds_a = bounded[1]["box_bounds"]
    bounds_b = unbounded[1]["box_bounds"]
    if bounds_a["requested_sha256"] != bounds_b["requested_sha256"]:
        raise ValueError("paired requested physical box coordinates differ")
    requested = (
        torch.as_tensor(bounds_a["requested_lower"], dtype=torch.float32),
        torch.as_tensor(bounds_a["requested_upper"], dtype=torch.float32),
    )
    requested_b = (
        torch.as_tensor(bounds_b["requested_lower"], dtype=torch.float32),
        torch.as_tensor(bounds_b["requested_upper"], dtype=torch.float32),
    )
    if any(not torch.equal(a, b) for a, b in zip(requested, requested_b, strict=True)):
        raise ValueError("paired requested physical box coordinates differ")
    if not all(torch.isfinite(value).all() for value in requested):
        raise ValueError("paired requested physical coordinates are non-finite")
    errors = {}
    for domain, (payload, _, _, _) in checkpoints.items():
        snapshot = payload.get("geometry", {})
        lower = torch.as_tensor(snapshot.get("lower"))
        upper = torch.as_tensor(snapshot.get("upper"))
        if (
            lower.dtype != torch.float32
            or upper.dtype != torch.float32
            or lower.shape != requested[0].shape
            or upper.shape != requested[1].shape
            or not torch.isfinite(lower).all()
            or not torch.isfinite(upper).all()
        ):
            raise ValueError(
                f"{domain} initial physical snapshot has invalid shape/dtype/finiteness"
            )
        error = max(
            float((lower - requested[0]).abs().max()), float((upper - requested[1]).abs().max())
        )
        if error > PAIR_TOLERANCE:
            raise ValueError(f"{domain} initial physical bounds mismatch: {error}")
        errors[domain] = error
    return {
        "status": "passed",
        "adapter_sha256": bounded[3],
        "requested_bounds_sha256": bounds_a["requested_sha256"],
        "reconstruction_tolerance_abs": PAIR_TOLERANCE,
        "max_absolute_error_by_domain": errors,
    }


def _validate_main_run_provenance(run_path, result, domain, prep) -> list[str]:
    """Return fail-closed discrepancies across completed manifest, summary, data and checkpoint."""
    errors = []
    run_path = Path(run_path)
    try:
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"run summary/manifest missing or invalid: {error}"]
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        return ["run summary and manifest must be JSON objects"]
    if not isinstance(result, dict):
        return ["supplied run result must be an object"]
    if manifest.get("status") != "completed" or summary.get("status") != "completed":
        errors.append("run summary and manifest are not completed")
    if manifest.get("run_id") != result.get("run_id") or summary.get("run_id") != result.get(
        "run_id"
    ):
        errors.append("run identity mismatch")
    if not _json_values_equal(manifest.get("summary"), summary):
        errors.append("manifest embedded summary differs from durable summary")
    for key in (
        "run_id",
        "status",
        "run_path",
        "history",
        "checkpoint_paths",
        "parameter_updates",
        "best_checkpoint_reload_predictions_equivalent",
        "best_checkpoint_selection",
        "best_mean_available_task_micro_f1_epoch",
        "best_mean_available_task_micro_f1",
        "geometry_snapshots",
    ):
        if key in result and not _json_values_equal(result.get(key), summary.get(key)):
            errors.append(f"supplied run result differs from durable summary: {key}")
    history = summary.get("history")
    try:
        metric_rows = [
            json.loads(line)
            for line in (run_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        metric_epochs = [row.get("epoch") if isinstance(row, dict) else None for row in metric_rows]
        if (
            not isinstance(history, list)
            or len(metric_rows) != COMMON["epochs"]
            or metric_epochs != list(range(1, COMMON["epochs"] + 1))
            or not _json_values_equal(metric_rows, history)
        ):
            errors.append("durable epoch metrics differ from completed run history")
    except (OSError, json.JSONDecodeError, TypeError):
        errors.append("durable epoch metrics are missing or invalid")
    config = manifest.get("config")
    expected = {
        "model": "box-union-two-adapters",
        "domain": domain,
        "objective": "logistic",
        "latent_dim": COMMON["latent_dim"],
        "boxes": COMMON["boxes"],
        "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"],
        "seed": COMMON["seed"],
        "learning_rate": COMMON["learning_rate"],
        "tau": COMMON["tau"],
        "epsilon": COMMON["epsilon"],
        "clip_norm": COMMON["clip_norm"],
        "weight_decay": COMMON["weight_decay"],
        "margin": 0.1,
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "amsgrad": False,
        "device": prep.get("device"),
    }
    if (
        not isinstance(config, dict)
        or set(config) != set(expected) | {"paired_initialization"}
        or not _required_config_matches(config, expected)
    ):
        errors.append("run manifest config differs from required PR3 settings")
    identity = prep.get("dataset_identity")
    normalized_dataset_identity = _normalize_run_dataset_identity(manifest.get("dataset"))
    if (
        not _valid_prepared_dataset_identity(identity)
        or identity.get("dataset_id") != DATASET_ID
        or normalized_dataset_identity != identity
    ):
        errors.append("run manifest dataset/class/split identity differs from preparation")
    source = manifest.get("source")
    prepared_source = prep.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(prepared_source, dict)
        or not isinstance(prepared_source.get("commit_sha"), str)
        or not prepared_source.get("commit_sha")
        or source.get("availability") != "available"
        or prepared_source.get("availability") != "available"
        or source.get("dirty") is not False
        or prepared_source.get("dirty") is not False
        or not _json_values_equal(source, prepared_source)
    ):
        errors.append("run source identity differs from preparation")
    summary_checkpoint_paths = summary.get("checkpoint_paths")
    checkpoint_path = (
        summary_checkpoint_paths.get("best") if isinstance(summary_checkpoint_paths, dict) else None
    )
    if (
        not isinstance(checkpoint_path, str)
        or Path(checkpoint_path).resolve() != (run_path / "checkpoints" / "best.pt").resolve()
    ):
        errors.append("selected checkpoint is not the run's durable best checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("selected checkpoint payload is not an object")
        metadata = checkpoint.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("selected checkpoint metadata is not an object")
        dataset_id = identity if isinstance(identity, dict) else {}
        activation = "tanh" if domain == "bounded" else "identity"
        checkpoint_expected = {
            "domain": domain,
            "latent_dim": COMMON["latent_dim"],
            "boxes": COMMON["boxes"],
            "epsilon": COMMON["epsilon"],
            "class_ids": dataset_id.get("class_ids"),
            "class_names": dataset_id.get("class_names"),
            "class_iris": dataset_id.get("class_iris"),
            "vocabulary_sha256": dataset_id.get("class_manifest_sha256"),
            "objective": {"kind": "logistic", "tau": COMMON["tau"], "margin": 0.1},
            "training_config": config,
        }
        encoder = metadata.get("encoder", {})
        objective = metadata.get("objective", {})
        metadata_types_valid = (
            isinstance(metadata.get("domain"), str)
            and type(metadata.get("latent_dim")) is int
            and type(metadata.get("boxes")) is int
            and isinstance(metadata.get("epsilon"), (int, float))
            and not isinstance(metadata.get("epsilon"), bool)
            and _is_int_list(metadata.get("class_ids"))
            and _is_string_list(metadata.get("class_names"))
            and _is_string_list(metadata.get("class_iris"))
            and isinstance(metadata.get("vocabulary_sha256"), str)
            and isinstance(objective, dict)
            and isinstance(objective.get("kind"), str)
            and isinstance(objective.get("tau"), (int, float))
            and not isinstance(objective.get("tau"), bool)
            and isinstance(objective.get("margin"), (int, float))
            and not isinstance(objective.get("margin"), bool)
            and isinstance(metadata.get("training_config"), dict)
        )
        encoder_matches = (
            isinstance(encoder, dict)
            and type(encoder.get("pizza_features")) is int
            and encoder.get("pizza_features") == len(PIZZA_FEATURE_NAMES)
            and type(encoder.get("component_features")) is int
            and encoder.get("component_features") == len(COMPONENT_FEATURE_NAMES)
            and encoder.get("bounded_activation") == activation
        )
        if (
            not metadata_types_valid
            or any(metadata.get(key) != value for key, value in checkpoint_expected.items())
            or not encoder_matches
        ):
            errors.append(
                "selected checkpoint metadata differs from required config/domain/classes"
            )
        history = summary.get("history", [])
        selectable = [
            row
            for row in history
            if isinstance(row, dict)
            and type(row.get("epoch")) is int
            and isinstance(row.get("validation_loss"), (int, float))
            and not isinstance(row.get("validation_loss"), bool)
            and bool(torch.isfinite(torch.tensor(row["validation_loss"])))
        ]
        if not selectable:
            errors.append("durable run summary has no finite validation-loss selection")
        else:
            selected = min(selectable, key=lambda row: (row["validation_loss"], row["epoch"]))
            checkpoint_epoch = checkpoint.get("epoch")
            checkpoint_loss = checkpoint.get("validation_loss")
            if (
                type(checkpoint_epoch) is not int
                or not isinstance(checkpoint_loss, (int, float))
                or isinstance(checkpoint_loss, bool)
                or not torch.isfinite(torch.tensor(checkpoint_loss))
                or checkpoint_epoch != selected.get("epoch")
                or checkpoint_loss != selected.get("validation_loss")
            ):
                errors.append("selected checkpoint epoch/loss differs from manifest run selection")
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        errors.append(f"selected checkpoint metadata could not be verified: {error}")
    return errors


def _durable_assessment_result(result: Any) -> dict[str, Any] | None:
    """Return C inputs only when caller data matches a completed durable run and metrics log."""
    if not isinstance(result, dict) or result.get("status") != "completed":
        return None
    run_path_value = result.get("run_path")
    run_id = result.get("run_id")
    if not isinstance(run_path_value, str) or not isinstance(run_id, str):
        return None
    run_path = Path(run_path_value).resolve()
    try:
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
        metrics = [
            json.loads(line)
            for line in (run_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        return None
    summary_checkpoint_paths = summary.get("checkpoint_paths")
    if not isinstance(summary_checkpoint_paths, dict):
        return None
    best_path = summary_checkpoint_paths.get("best")
    if (
        manifest.get("status") != "completed"
        or summary.get("status") != "completed"
        or manifest.get("run_id") != run_id
        or summary.get("run_id") != run_id
        or summary.get("run_path") != str(run_path)
        or result.get("run_path") != str(run_path)
        or not _json_values_equal(manifest.get("summary"), summary)
        or not _json_values_equal(result.get("checkpoint_paths"), summary.get("checkpoint_paths"))
        or not _json_values_equal(result.get("history"), summary.get("history"))
        or not _json_values_equal(metrics, summary.get("history"))
        or not isinstance(best_path, str)
        or Path(best_path).resolve() != (run_path / "checkpoints" / "best.pt").resolve()
    ):
        return None
    for key in (
        "best_checkpoint_reload_predictions_equivalent",
        "parameter_updates",
        "best_checkpoint_selection",
        "best_mean_available_task_micro_f1_epoch",
        "best_mean_available_task_micro_f1",
        "geometry_snapshots",
    ):
        if key in result and not _json_values_equal(result.get(key), summary.get(key)):
            return None
    epochs = [row.get("epoch") if isinstance(row, dict) else None for row in metrics]
    if len(metrics) != COMMON["epochs"] or epochs != list(range(1, COMMON["epochs"] + 1)):
        return None
    return summary


def assess_run(
    result: dict[str, Any], baselines: dict[str, Any], *, required_domain: str | None = None
) -> dict[str, Any]:
    """Apply strict per-task C at the earliest minimum native validation-loss epoch."""
    if not isinstance(result, dict) or result.get("status") != "completed":
        return {
            "accepted_C": False,
            "reason": "required metadata run lacks explicit completed status",
        }
    durable_result = _durable_assessment_result(result)
    if durable_result is None:
        return {
            "accepted_C": False,
            "reason": "assessment history is not bound to durable run metrics and manifest",
        }
    result = durable_result
    if result.get("best_checkpoint_reload_predictions_equivalent") is not True:
        return {
            "accepted_C": False,
            "reason": "best checkpoint reload equivalence was not verified",
        }
    best_checkpoint = result.get("checkpoint_paths", {}).get("best")
    if not best_checkpoint or not Path(best_checkpoint).is_file():
        return {"accepted_C": False, "reason": "selected best checkpoint is missing"}
    history = result.get("history", [])
    eligible = [
        row
        for row in history
        if isinstance(row, dict)
        and type(row.get("epoch")) is int
        and isinstance(row.get("validation_loss"), (int, float))
        and not isinstance(row.get("validation_loss"), bool)
        and torch.isfinite(torch.tensor(row["validation_loss"]))
    ]
    if not eligible:
        return {"accepted_C": False, "reason": "no native validation-loss checkpoint"}
    chosen = min(eligible, key=lambda r: (r["validation_loss"], r["epoch"]))
    try:
        checkpoint = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        return {"accepted_C": False, "reason": f"selected best checkpoint is unreadable: {error}"}
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    domain = metadata.get("domain") if isinstance(metadata, dict) else None
    encoder = metadata.get("encoder", {}) if isinstance(metadata, dict) else {}
    objective = metadata.get("objective", {}) if isinstance(metadata, dict) else {}
    training_config = metadata.get("training_config", {}) if isinstance(metadata, dict) else {}
    expected_training_config = {
        "model": "box-union-two-adapters",
        "domain": domain,
        "objective": "logistic",
        "latent_dim": COMMON["latent_dim"],
        "boxes": COMMON["boxes"],
        "epsilon": COMMON["epsilon"],
        "epochs": COMMON["epochs"],
        "batch_size": COMMON["batch_size"],
        "seed": COMMON["seed"],
        "learning_rate": COMMON["learning_rate"],
        "tau": COMMON["tau"],
        "clip_norm": COMMON["clip_norm"],
        "weight_decay": COMMON["weight_decay"],
        "margin": 0.1,
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "amsgrad": False,
        "device": training_config.get("device") if isinstance(training_config, dict) else None,
    }
    metadata_valid = (
        isinstance(metadata, dict)
        and domain in CONFIGS
        and (required_domain is None or domain == required_domain)
        and type(metadata.get("latent_dim")) is int
        and metadata.get("latent_dim") == COMMON["latent_dim"]
        and type(metadata.get("boxes")) is int
        and metadata.get("boxes") == COMMON["boxes"]
        and isinstance(metadata.get("epsilon"), (int, float))
        and not isinstance(metadata.get("epsilon"), bool)
        and metadata.get("epsilon") == COMMON["epsilon"]
        and _is_int_list(metadata.get("class_ids"))
        and bool(metadata.get("class_ids"))
        and _is_string_list(metadata.get("class_names"))
        and len(metadata["class_names"]) == len(metadata["class_ids"])
        and _is_string_list(metadata.get("class_iris"))
        and len(metadata["class_iris"]) == len(metadata["class_ids"])
        and _is_sha256(metadata.get("vocabulary_sha256"))
        and isinstance(objective, dict)
        and objective == {"kind": "logistic", "tau": 0.1, "margin": 0.1}
        and isinstance(training_config, dict)
        and isinstance(encoder, dict)
        and type(encoder.get("pizza_features")) is int
        and encoder.get("pizza_features") == len(PIZZA_FEATURE_NAMES)
        and type(encoder.get("component_features")) is int
        and encoder.get("component_features") == len(COMPONENT_FEATURE_NAMES)
        and encoder.get("bounded_activation") == CONFIGS[domain]["activation"]
        and set(training_config) == set(expected_training_config) | {"paired_initialization"}
        and isinstance(training_config.get("device"), str)
        and training_config["device"].startswith("cuda")
        and _required_config_matches(training_config, expected_training_config)
    )
    if (
        not metadata_valid
        or not isinstance(checkpoint.get("model_state"), dict)
        or not checkpoint["model_state"]
        or type(checkpoint.get("epoch")) is not int
        or not isinstance(checkpoint.get("validation_loss"), (int, float))
        or isinstance(checkpoint.get("validation_loss"), bool)
        or checkpoint.get("epoch") != chosen["epoch"]
        or checkpoint.get("validation_loss") != chosen["validation_loss"]
    ):
        return {
            "accepted_C": False,
            "reason": (
                "selected checkpoint is not self-describing at the selected validation-loss epoch"
            ),
        }
    diagnostics = chosen.get("diagnostics")
    validation_metrics = diagnostics.get("validation") if isinstance(diagnostics, dict) else None
    train_metrics = diagnostics.get("train") if isinstance(diagnostics, dict) else None
    if not isinstance(validation_metrics, dict) or not isinstance(train_metrics, dict):
        return {"accepted_C": False, "reason": "selected epoch diagnostics are malformed"}
    if not isinstance(baselines, dict):
        return {"accepted_C": False, "reason": "train-frequency baselines are malformed"}
    comparisons = {}
    for task in ("pizza", "component"):
        task_validation = validation_metrics.get(task)
        micro = task_validation.get("micro") if isinstance(task_validation, dict) else None
        task_baseline = baselines.get(task)
        baseline_micro = (
            task_baseline.get("metrics", {}).get("micro")
            if isinstance(task_baseline, dict) and isinstance(task_baseline.get("metrics"), dict)
            else None
        )
        if (
            not isinstance(micro, dict)
            or "f1" not in micro
            or not isinstance(baseline_micro, dict)
            or not _finite_number(baseline_micro.get("f1"))
            or not 0 <= baseline_micro["f1"] <= 1
        ):
            return {
                "accepted_C": False,
                "reason": f"selected {task} validation or baseline micro-F1 is malformed",
            }
        value = micro["f1"]
        base = baseline_micro["f1"]
        if value is not None and (not _finite_number(value) or not 0 <= value <= 1):
            return {
                "accepted_C": False,
                "reason": f"selected {task} validation micro-F1 is non-finite or out of range",
            }
        comparisons[task] = {
            "validation_micro_f1": value,
            "train_frequency_baseline_micro_f1": base,
            "strictly_greater": value is not None and value > base,
        }
    return {
        "accepted_C": all(x["strictly_greater"] for x in comparisons.values()),
        "selection": "minimum combined native validation loss; earliest tie",
        "selected_epoch": chosen["epoch"],
        "selected_validation_loss": chosen["validation_loss"],
        "task_comparisons": comparisons,
        "best_mean_available_task_micro_f1_epoch": result.get(
            "best_mean_available_task_micro_f1_epoch"
        ),
        "best_mean_available_task_micro_f1": result.get("best_mean_available_task_micro_f1"),
        "selected_validation_metrics": chosen["diagnostics"]["validation"],
        "selected_train_metrics": chosen["diagnostics"]["train"],
        "feature_signature_diagnostics": chosen["diagnostics"].get("feature_signatures"),
        "selected_geometry_diagnostics": chosen.get("geometry"),
        "selected_checkpoint": result.get("checkpoint_paths", {}).get("best"),
    }


def export_evidence(
    destination: Path | str,
    report: dict[str, Any],
    checkpoint_artifacts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Atomically export compact evidence and selected checkpoint files, without overwrite."""
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.mkdir()
        checkpoint_dir = staging / "checkpoints"
        packaged = {}
        for domain, item in (checkpoint_artifacts or {}).items():
            source = Path(item["path"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"selected {domain} best checkpoint missing: {source}")
            filename = f"{domain}-best.pt"
            checkpoint_dir.mkdir(exist_ok=True)
            shutil.copyfile(source, checkpoint_dir / filename)
            packaged[domain] = {
                "filename": f"checkpoints/{filename}",
                "sha256": _sha(checkpoint_dir / filename),
                "run_id": item.get("run_id"),
                "reference": f"checksummed evidence bundle: checkpoints/{filename}",
            }
        export_report = dict(report)
        export_report["packaged_best_checkpoints"] = packaged
        (staging / "evidence.json").write_text(
            json.dumps(export_report, indent=2, sort_keys=True, default=str) + "\n"
        )
        digests = {
            p.relative_to(staging).as_posix(): _sha(p) for p in staging.rglob("*") if p.is_file()
        }
        checksum = hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()
        (staging / "checksums.json").write_text(
            json.dumps({"files": digests, "bundle_sha256": checksum}, indent=2, sort_keys=True)
            + "\n"
        )
        # Linux renameat2 provides atomic publication with strict no-replace semantics.
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(staging), -100, os.fsencode(target), 1)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(target))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"path": str(target), "bundle_sha256": checksum}


def _checkpoint_parameter_updates(result: dict[str, Any]) -> dict[str, bool]:
    """Independently compare initial/latest persisted model states by parameter group."""
    paths = result.get("checkpoint_paths", {})
    try:
        initial = torch.load(paths["initial"], map_location="cpu", weights_only=True)["model_state"]
        latest = torch.load(paths["latest"], map_location="cpu", weights_only=True)["model_state"]
        if set(initial) != set(latest):
            return {group: False for group in ("pizza_adapter", "component_adapter", "geometry")}
        checks = {}
        for group, prefix in (
            ("pizza_adapter", "pizza_adapter."),
            ("component_adapter", "component_adapter."),
            ("geometry", "geometry."),
        ):
            keys = [key for key in initial if key.startswith(prefix)]
            checks[group] = (
                bool(keys)
                and all(
                    isinstance(initial[key], torch.Tensor)
                    and isinstance(latest[key], torch.Tensor)
                    and initial[key].shape == latest[key].shape
                    and initial[key].dtype == latest[key].dtype
                    and torch.isfinite(initial[key]).all()
                    and torch.isfinite(latest[key]).all()
                    for key in keys
                )
                and any(not torch.equal(initial[key], latest[key]) for key in keys)
            )
        return checks
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return {group: False for group in ("pizza_adapter", "component_adapter", "geometry")}


def _compact_trajectory(history) -> list[dict[str, Any]]:
    """Retain per-epoch learning signals without embedding full diagnostics or examples."""
    compact = []
    for row in history if isinstance(history, list) else []:
        if not isinstance(row, dict):
            continue
        diagnostics = row.get("diagnostics", {})
        validation = diagnostics.get("validation", {}) if isinstance(diagnostics, dict) else {}
        task_f1 = {
            task: validation.get(task, {}).get("micro", {}).get("f1")
            for task in ("pizza", "component")
        }
        geometry = row.get("geometry", {})
        compact.append(
            {
                key: row.get(key)
                for key in (
                    "epoch",
                    "train_loss",
                    "validation_loss",
                    "train_pizza_loss",
                    "train_component_loss",
                    "validation_pizza_loss",
                    "validation_component_loss",
                    "gradient_norm_pre_clip_mean",
                    "encoder_gradient_norm_pre_clip_mean",
                    "geometry_gradient_norm_pre_clip_mean",
                    "geometry_zero_pre_clip_gradient_fraction",
                )
            }
            | {
                "validation_task_micro_f1": task_f1,
                "geometry_scale": {
                    key: geometry.get(key)
                    for key in ("width_min", "width_max", "training_point_coordinate_std_mean")
                }
                if isinstance(geometry, dict)
                else {},
            }
        )
    return compact


def _local_run_reference(result, prep) -> dict[str, Any]:
    run_id = result.get("run_id") if isinstance(result, dict) else None
    run_path = result.get("run_path") if isinstance(result, dict) else None
    relative = None
    if run_path and prep.get("artifact_root"):
        try:
            relative = (
                Path(run_path)
                .resolve()
                .relative_to(Path(prep["artifact_root"]).resolve())
                .as_posix()
            )
        except (OSError, ValueError):
            relative = None
    return {"run_id": run_id, "relative_path": relative, "local_only": True}


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare JSON evidence without Python's bool/int or int/float coercions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _run_mathematical_numerical_tests(expected_source_sha: str | None) -> dict[str, Any]:
    """Run the fixed PR3 mathematical test set and bind evidence to clean source."""
    root = Path(__file__).resolve().parents[2]
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_box.py",
        "tests/test_box_training.py",
    ]
    source = source_provenance(root)
    base = {"command": argv, "source": source, "test_count": None, "returncode": None}
    if (
        source.get("availability") != "available"
        or source.get("dirty") is not False
        or not isinstance(expected_source_sha, str)
        or source.get("commit_sha") != expected_source_sha
    ):
        return {
            **base,
            "status": "failed",
            "reason": "mathematical tests require the clean prepared source revision",
            "output_sha256": None,
        }
    env = os.environ.copy()
    source_path = str(root / "src")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, env.get("PYTHONPATH", "")) if value
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        output = completed.stdout + completed.stderr
        matches = re.findall(r"(?m)^\s*(\d+) passed(?:[ ,]|$)", output)
        test_count = int(matches[-1]) if matches else None
        source_after = source_provenance(root)
        source_stable = (
            source_after.get("availability") == "available"
            and source_after.get("dirty") is False
            and source_after.get("commit_sha") == expected_source_sha
            and _json_values_equal(source, source_after)
        )
        passed = (
            completed.returncode == 0
            and test_count is not None
            and test_count > 0
            and source_stable
        )
        return {
            **base,
            "status": "passed" if passed else "failed",
            "returncode": completed.returncode,
            "test_count": test_count,
            "source_after": source_after,
            "source_stable": source_stable,
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "output_tail": output[-2000:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        output = str(error)
        return {
            **base,
            "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
            "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }


def _numeric_tree(value: Any) -> bool:
    if isinstance(value, list):
        return all(_numeric_tree(item) for item in value)
    return _finite_number(value)


def _control_corners_match(corners: Any, endpoints: tuple[torch.Tensor, torch.Tensor]) -> bool:
    if (
        not isinstance(corners, dict)
        or not _numeric_tree(corners.get("lower"))
        or not _numeric_tree(corners.get("upper"))
    ):
        return False
    try:
        lower = torch.as_tensor(corners["lower"], dtype=torch.float32)
        upper = torch.as_tensor(corners["upper"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError):
        return False
    return torch.equal(lower, endpoints[0]) and torch.equal(upper, endpoints[1])


def _verify_control_result(item: Any, prep: dict[str, Any]) -> dict[str, Any]:
    """Verify each geometry control against its durable run, config, metrics and states."""
    errors = []
    if not isinstance(item, dict):
        return {"status": "failed", "run_id": None, "errors": ["control result is not an object"]}
    run_id = item.get("run_id")
    artifact_root = prep.get("artifact_root")
    if (
        not isinstance(run_id, str)
        or not re.fullmatch(r"[0-9]{8}T[0-9]{6}-[0-9a-f]{12}", run_id)
        or not isinstance(artifact_root, str)
    ):
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": ["control run ID or artifact root is invalid"],
        }
    run_path = (Path(artifact_root).resolve() / run_id).resolve()
    if item.get("run_path") != str(run_path) or not run_path.is_dir():
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": ["control result is not bound to its durable run directory"],
        }
    try:
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": [f"control manifest missing or invalid: {error}"],
        }
    if not isinstance(manifest, dict):
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": ["control manifest is not an object"],
        }
    domain, objective = item.get("domain"), item.get("objective")
    if domain not in CONFIGS or objective not in ("logistic", "hinge"):
        return {"status": "failed", "run_id": run_id, "errors": ["control pair is invalid"]}
    _, _, expected_config = _geometry_control_spec(domain, objective, CONTROL_EPOCHS)
    source = manifest.get("source")
    prepared_source = prep.get("source")
    expected_source_sha = (
        prepared_source.get("commit_sha") if isinstance(prepared_source, dict) else None
    )
    if (
        manifest.get("run_id") != run_id
        or manifest.get("status") != "completed"
        or manifest.get("purpose") != "pr3-geometry-recovery-control"
        or manifest.get("issue") != "14"
        or type(manifest.get("seed")) is not int
        or manifest.get("seed") != 23
        or not _json_values_equal(manifest.get("dataset"), {"synthetic_fixed_coordinates": True})
        or not _json_values_equal(manifest.get("config"), expected_config)
        or not isinstance(source, dict)
        or source.get("availability") != "available"
        or source.get("dirty") is not False
        or not isinstance(prepared_source, dict)
        or prepared_source.get("dirty") is not False
        or not _json_values_equal(source, prepared_source)
        or source.get("commit_sha") != expected_source_sha
    ):
        errors.append("control manifest provenance or prescribed configuration is invalid")
    durable_result = manifest.get("result")
    if not isinstance(durable_result, dict) or not _json_values_equal(item, durable_result):
        errors.append("supplied control result differs from durable run result")
    if errors:
        return {"status": "failed", "run_id": run_id, "errors": errors}

    checkpoint_dir = run_path / "checkpoints"
    expected_paths = {
        key: str(checkpoint_dir / f"{key}.pt") for key in ("initial", "latest", "checkpoint")
    }
    checkpoint_paths = durable_result.get("checkpoint_paths")
    if not _json_values_equal(checkpoint_paths, expected_paths):
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": ["control checkpoints are not the run's durable initial/latest/final files"],
        }
    loaded = {}
    for key, path in expected_paths.items():
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"control {key} checkpoint is unreadable: {error}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"control {key} checkpoint is not an object")
            continue
        expected_epoch = 0 if key == "initial" else CONTROL_EPOCHS
        state = payload.get("state_dict")
        if (
            payload.get("domain") != domain
            or type(payload.get("epoch")) is not int
            or payload.get("epoch") != expected_epoch
            or not _finite_number(payload.get("loss"))
            or not isinstance(state, dict)
            or set(state) != {"raw_a", "raw_b"}
            or any(
                not isinstance(state.get(name), torch.Tensor)
                or state[name].dtype != torch.float32
                or state[name].shape != (1, 1, 2)
                or not torch.isfinite(state[name]).all()
                for name in ("raw_a", "raw_b")
            )
        ):
            errors.append(f"control {key} checkpoint metadata or geometry is invalid")
            continue
        geometry = BoxGeometry(state["raw_a"], state["raw_b"], domain=domain, epsilon=1e-4)
        loaded[key] = (payload, state, geometry.endpoints())
    if errors or len(loaded) != 3:
        return {"status": "failed", "run_id": run_id, "errors": errors}

    initial_payload, initial_state, initial_endpoints = loaded["initial"]
    final_payload, final_state, final_endpoints = loaded["checkpoint"]
    points, _, config = _geometry_control_spec(domain, objective, CONTROL_EPOCHS)
    membership = (
        box_membership(points, *final_endpoints, domain=domain, epsilon=1e-4).squeeze(1).tolist()
    )
    metrics_path = run_path / "metrics.jsonl"
    if not _is_sha256(durable_result.get("metrics_sha256")) or _sha(
        metrics_path
    ) != durable_result.get("metrics_sha256"):
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": ["control epoch metrics checksum differs from durable run result"],
        }
    try:
        metric_rows = [
            json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "run_id": run_id,
            "errors": [f"control epoch metrics are missing or invalid: {error}"],
        }
    norms = []
    if len(metric_rows) != CONTROL_EPOCHS:
        errors.append("control durable metrics do not contain all prescribed epochs")
    for expected_epoch, row in enumerate(metric_rows, 1):
        if (
            not isinstance(row, dict)
            or type(row.get("epoch")) is not int
            or row.get("epoch") != expected_epoch
            or not _finite_number(row.get("loss"))
            or not _finite_number(row.get("geometry_gradient_norm"))
            or row["geometry_gradient_norm"] < 0
        ):
            errors.append("control durable epoch metrics are malformed or non-finite")
            break
        norms.append(row["geometry_gradient_norm"])
    max_norm = max(norms) if norms else 0.0
    if (
        not norms
        or max_norm <= 0
        or not _finite_number(durable_result.get("geometry_gradient_norm_max"))
        or not math.isclose(durable_result.get("geometry_gradient_norm_max"), max_norm)
        or durable_result.get("geometry_gradient_finite") is not True
        or durable_result.get("geometry_updated") is not True
        or not any(
            not torch.equal(initial_state[name], final_state[name]) for name in initial_state
        )
        or durable_result.get("target_recovered") is not True
        or not _json_values_equal(durable_result.get("membership"), config["expected_membership"])
        or membership != config["expected_membership"]
        or not _control_corners_match(durable_result.get("initial_corners"), initial_endpoints)
        or not _control_corners_match(durable_result.get("final_corners"), final_endpoints)
        or not _finite_number(durable_result.get("initial_loss"))
        or durable_result.get("initial_loss") != initial_payload["loss"]
        or not _finite_number(durable_result.get("final_loss"))
        or durable_result.get("final_loss") != final_payload["loss"]
        or metric_rows[-1].get("loss") != durable_result.get("final_loss")
    ):
        errors.append("control recovery, gradient, state-update or checkpoint evidence is invalid")
    return {
        "status": "passed" if not errors else "failed",
        "run_id": run_id,
        "relative_path": run_id,
        "local_only": True,
        "errors": errors,
        "manifest_sha256": _sha(run_path / "manifest.json"),
        "metrics_sha256": durable_result["metrics_sha256"],
        "initial_checkpoint_sha256": _sha(Path(expected_paths["initial"])),
        "final_checkpoint_sha256": _sha(Path(expected_paths["checkpoint"])),
    }


def finalize(
    work_dir: Path | str,
    run_results: dict[str, dict[str, Any]],
    control_results: list[dict[str, Any]] | None = None,
    verification_evidence: dict[str, Any] | None = None,
):
    """Export truthful acceptance evidence; missing inputs can never become a pass."""
    work = Path(work_dir).resolve()
    prep = json.loads((work / "prepare.json").read_text())
    run_results = run_results if isinstance(run_results, dict) else {}
    run_results = {
        domain: run_results.get(domain) if isinstance(run_results.get(domain), dict) else {}
        for domain in CONFIGS
        if domain in run_results
    }
    verification_evidence = (
        verification_evidence if isinstance(verification_evidence, dict) else None
    )
    assessments = {}
    for domain in CONFIGS:
        try:
            assessments[domain] = assess_run(
                run_results.get(domain, {}), prep["baselines"], required_domain=domain
            )
        except (KeyError, TypeError, ValueError) as error:
            assessments[domain] = {
                "accepted_C": False,
                "reason": f"invalid assessment input: {error}",
            }
    try:
        pairing = validate_paired_results(run_results)
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
        pairing = {"status": "failed-validation", "error": f"{type(error).__name__}: {error}"}
    manifests = {}
    provenance_valid = True
    provenance_invalid = False
    for domain in CONFIGS:
        if domain not in run_results:
            manifests[domain] = {"status": "not-provided"}
            provenance_valid = False
            continue
        result = run_results.get(domain, {})
        run_path = result.get("run_path")
        try:
            manifest = json.loads((Path(run_path) / "manifest.json").read_text())
            validated = _read_run_result(run_path)
            discrepancies = _validate_main_run_provenance(run_path, result, domain, prep)
            if (
                manifest.get("run_id") != result.get("run_id")
                or manifest.get("status") != result.get("status")
                or manifest.get("status") != "completed"
                or validated.get("status") != "completed"
                or validated.get("run_id") != result.get("run_id")
                or discrepancies
            ):
                provenance_valid = False
                provenance_invalid = True
                manifest["acceptance_validation_errors"] = discrepancies
        except (TypeError, OSError, json.JSONDecodeError):
            manifest = {"status": "missing-provenance"}
            provenance_valid = False
            provenance_invalid = True
        manifests[domain] = manifest
    if provenance_invalid:
        for assessment in assessments.values():
            assessment["accepted_C"] = False
            assessment["reason"] = "run summary/manifest provenance is invalid"

    review_references = (verification_evidence or {}).get("review_references")
    valid_review_references = (
        isinstance(review_references, list)
        and bool(review_references)
        and all(isinstance(reference, str) and reference.strip() for reference in review_references)
    )
    prepared_source = prep.get("source")
    expected_source_sha = (
        prepared_source.get("commit_sha") if isinstance(prepared_source, dict) else None
    )
    mathematical_tests = _run_mathematical_numerical_tests(expected_source_sha)
    evidence = {
        "mathematical_numerical_tests": mathematical_tests,
        "review_references": review_references if valid_review_references else [],
    }
    if mathematical_tests.get("status") == "failed":
        status_a = "failed"
    elif mathematical_tests.get("status") == "passed":
        status_a = "passed"
    else:
        status_a = "not-established"

    required_pairs = {
        (domain, objective) for domain in CONFIGS for objective in ("logistic", "hinge")
    }
    control_results = control_results if isinstance(control_results, list) else []
    actual_pairs = [
        (item.get("domain"), item.get("objective"))
        for item in control_results
        if isinstance(item, dict)
    ]
    exact_controls = len(actual_pairs) == 4 and set(actual_pairs) == required_pairs
    control_verification = [_verify_control_result(item, prep) for item in control_results]
    controls_success = (
        exact_controls
        and len(control_verification) == 4
        and all(item.get("status") == "passed" for item in control_verification)
    )
    controls_status = (
        "passed" if controls_success else "failed" if exact_controls else "not-established"
    )

    def positive_finite(value):
        return (
            isinstance(value, (int, float))
            and bool(torch.isfinite(torch.tensor(value)))
            and value > 0
        )

    joint_run_checks = {}
    checkpoint_parameter_updates = {}
    joint_gradient_evidence = {}
    for domain in CONFIGS:
        result = run_results.get(domain, {})
        updates = result.get("parameter_updates", {})
        actual_updates = _checkpoint_parameter_updates(result)
        checkpoint_parameter_updates[domain] = actual_updates
        history = result.get("history", [])
        joint_gradient_evidence[domain] = {}
        for group, field in (
            ("encoder", "encoder_gradient_norm_pre_clip_mean"),
            ("geometry", "geometry_gradient_norm_pre_clip_mean"),
        ):
            values = [
                row.get(field)
                for row in history
                if isinstance(row, dict) and positive_finite(row.get(field))
            ]
            joint_gradient_evidence[domain][group] = {
                "positive_finite_epochs": len(values),
                "maximum_positive_finite_norm": max(values) if values else None,
            }
        joint_run_checks[domain] = (
            result.get("status") == "completed"
            and result.get("best_checkpoint_reload_predictions_equivalent") is True
            and all(
                updates.get(group) is True and actual_updates.get(group) is True
                for group in ("pizza_adapter", "component_adapter", "geometry")
            )
            and any(
                positive_finite(row.get("encoder_gradient_norm_pre_clip_mean"))
                and positive_finite(row.get("geometry_gradient_norm_pre_clip_mean"))
                for row in history
                if isinstance(row, dict)
            )
        )
    b_inputs_present = exact_controls and all(domain in run_results for domain in CONFIGS)
    status_b = (
        "passed"
        if controls_success
        and all(joint_run_checks.values())
        and pairing.get("status") == "passed"
        and provenance_valid
        else "failed"
        if (exact_controls and not controls_success) or b_inputs_present
        else "not-established"
    )
    status_c = (
        "passed"
        if provenance_valid and all(item.get("accepted_C") is True for item in assessments.values())
        else "failed"
        if all(domain in run_results for domain in CONFIGS)
        else "not-established"
    )
    if all(status == "passed" for status in (status_a, status_b, status_c)):
        overall = "passed"
    elif "failed" in (status_a, status_b, status_c) or provenance_invalid:
        overall = "failed"
    else:
        overall = "not-established"

    checkpoint_artifacts = {}
    for domain in CONFIGS:
        result = run_results.get(domain, {})
        path = result.get("checkpoint_paths", {}).get("best")
        if (
            provenance_valid
            and assessments[domain].get("selected_epoch") is not None
            and path
            and Path(path).is_file()
        ):
            checkpoint_artifacts[domain] = {"path": path, "run_id": result.get("run_id")}
        assessments[domain]["selected_checkpoint"] = (
            f"checksummed evidence bundle: checkpoints/{domain}-best.pt"
            if domain in checkpoint_artifacts
            else None
        )
    control_evidence = [
        {
            key: item.get(key)
            for key in (
                "run_id",
                "config",
                "domain",
                "objective",
                "status",
                "epochs",
                "target_recovered",
                "membership",
                "geometry_updated",
                "geometry_gradient_norm_max",
                "geometry_gradient_finite",
                "initial_corners",
                "final_corners",
            )
        }
        for item in control_results or []
        if isinstance(item, dict)
    ]
    for control, validation in zip(control_evidence, control_verification, strict=True):
        control["verification"] = validation
        control["raw_run_reference"] = {
            "run_id": validation.get("run_id"),
            "relative_path": validation.get("relative_path"),
            "local_only": True,
        }
    report = {
        "schema": "pr3-acceptance-evidence-v1",
        "dataset_id": prep.get("dataset_id"),
        "historical_selection": {"run_id": HISTORICAL_RUN, "source_sha": HISTORICAL_SOURCE},
        "dataset_hashes": prep.get("dataset_hashes"),
        "selection_hashes": prep.get("selection_hashes"),
        "preflight_artifact_storage": {
            key: prep.get("artifact_preflight", {}).get(key) for key in ("writable", "free_bytes")
        }
        if isinstance(prep.get("artifact_preflight"), dict)
        else None,
        "pairing_verification": pairing,
        "source": {domain: manifests[domain].get("source") for domain in CONFIGS},
        "run_configs": {domain: manifests[domain].get("config") for domain in CONFIGS},
        "run_environments": {domain: manifests[domain].get("environment") for domain in CONFIGS},
        "run_failures_and_progress": {
            domain: {
                key: manifests[domain].get(key)
                for key in ("status", "error", "last_epoch", "latest_metrics")
            }
            for domain in CONFIGS
        },
        "config": COMMON,
        "baselines": prep.get("baselines"),
        "assessments": assessments,
        "runs": {
            domain: {
                "run_id": run_results.get(domain, {}).get("run_id"),
                "status": run_results.get(domain, {}).get("status"),
                "history_epoch_count": len(run_results.get(domain, {}).get("history", [])),
                "selected_epoch": assessments[domain].get("selected_epoch"),
                "selected_validation_loss": assessments[domain].get("selected_validation_loss"),
                "parameter_updates": run_results.get(domain, {}).get("parameter_updates"),
                "best_checkpoint_reload_predictions_equivalent": run_results.get(domain, {}).get(
                    "best_checkpoint_reload_predictions_equivalent"
                ),
                "raw_run_reference": _local_run_reference(run_results.get(domain, {}), prep),
                "physical_geometry_snapshots": run_results.get(domain, {}).get(
                    "geometry_snapshots"
                ),
                "epoch_trajectory": _compact_trajectory(
                    run_results.get(domain, {}).get("history", [])
                ),
            }
            for domain in CONFIGS
        },
        "historical_affine_comparison": {
            "run_id": HISTORICAL_RUN,
            "source_sha": HISTORICAL_SOURCE,
            "selected_epoch": 28,
            "combined_validation_bce": 0.01401536,
            "validation_micro_f1": {"pizza": 0.9894, "component": 1.0},
            "train_frequency_baseline_micro_f1": {"pizza": 0.6476, "component": 0.7320},
            "limitations": [
                "Historical values are rounded affine PR2 references, not matched BOX controls.",
                "The representation and geometry differ, so comparison cannot isolate geometry.",
            ],
        },
        "joint_gradient_evidence": joint_gradient_evidence,
        "controls": control_evidence,
        "control_verification": control_verification,
        "verification_evidence": evidence,
        "acceptance_A": {"status": status_a, "evidence": evidence, "ci_status_is_separate": True},
        "acceptance_B": {
            "status": status_b,
            "controls_status": controls_status,
            "control_combinations_exact": exact_controls,
            "joint_run_checks": joint_run_checks,
            "checkpoint_parameter_updates": checkpoint_parameter_updates,
        },
        "acceptance_C": {
            "status": status_c,
            "by_domain": {domain: item.get("accepted_C") for domain, item in assessments.items()},
        },
        "scientific_acceptance_status": overall,
        "provenance_valid": provenance_valid,
        "limitations": prep.get("limitations", []),
    }
    exported = export_evidence(prep["destination"], report, checkpoint_artifacts)
    return {
        "report": {
            **report,
            "packaged_best_checkpoints": json.loads(
                (Path(exported["path"]) / "evidence.json").read_text()
            ).get("packaged_best_checkpoints", {}),
        },
        "export": exported,
    }


def _read_run_result(path: Path | str) -> dict[str, Any]:
    """Load only a summary whose identity/status agree with its durable manifest."""
    candidate = Path(path)
    if candidate.is_dir():
        run_path = candidate.resolve()
    else:
        try:
            supplied = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return {"status": "invalid-provenance", "provenance_errors": [str(error)]}
        run_path = Path(supplied.get("run_path", ""))
        if not run_path.is_dir():
            return {
                **supplied,
                "status": "invalid-provenance",
                "provenance_errors": ["summary has no recoverable run directory"],
            }
    summary_path = run_path / "summary.json"
    manifest_path = run_path / "manifest.json"
    errors = []
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary, errors = {}, ["run summary is missing or invalid"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest, errors = {}, errors + ["run manifest is missing or invalid"]
    if summary and manifest:
        if summary.get("run_id") != manifest.get("run_id"):
            errors.append("summary and manifest run_id mismatch")
        if summary.get("status") != manifest.get("status"):
            errors.append("summary and manifest status mismatch")
        if summary.get("run_id") is None or manifest.get("run_id") is None:
            errors.append("run identity is missing")
        summary_run_path = summary.get("run_path")
        if summary_run_path and Path(summary_run_path).resolve() != run_path:
            errors.append("summary and run directory identity mismatch")
        embedded = manifest.get("summary")
        if isinstance(embedded, dict) and (
            embedded.get("run_id") != summary.get("run_id")
            or embedded.get("status") != summary.get("status")
        ):
            errors.append("manifest summary identity/status mismatch")
    result = {**summary, "run_path": str(run_path), "manifest_status": manifest.get("status")}
    if errors:
        result["status"] = "invalid-provenance"
        result["provenance_errors"] = errors
    else:
        result["status"] = manifest["status"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("controls")
    p.add_argument("--artifact-root", default="artifacts/runs")
    p.add_argument("--output")
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--historical-manifest", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--export-destination", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument("--artifact-root", default="artifacts/runs")
    p = sub.add_parser("run-domain")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--domain", choices=tuple(CONFIGS), required=True)
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("finalize")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--bounded-summary", required=True)
    p.add_argument("--unbounded-summary", required=True)
    p.add_argument("--controls-json", required=True)
    p.add_argument("--verification-evidence", help="coordinator-provided A evidence JSON")
    args = parser.parse_args(argv)
    if args.command == "controls":
        result = run_geometry_controls(artifact_root=args.artifact_root)
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "prepare":
        result = create_preparation(
            args.data,
            historical_manifest=args.historical_manifest,
            device=args.device,
            export_destination=args.export_destination,
            work_dir=args.work_dir,
            artifact_root=args.artifact_root,
        )
    elif args.command == "run-domain":
        result = run_domain(args.work_dir, args.domain, device=args.device)
    else:
        result = finalize(
            args.work_dir,
            {
                "bounded": _read_run_result(args.bounded_summary),
                "unbounded": _read_run_result(args.unbounded_summary),
            },
            json.loads(Path(args.controls_json).read_text(encoding="utf-8")),
            json.loads(Path(args.verification_evidence).read_text(encoding="utf-8"))
            if args.verification_evidence
            else None,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
