"""Offline BOX checkpoint reconstruction and GEO-01 producer/consumer verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .box import box_clearance, box_membership
from .box_training import BoxMultiLabelModel
from .dataset.image_data import targets_in_trainable_order
from .evaluation import membership_metrics
from .features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
    component_features,
    pizza_features,
)
from .geo_capture import state_dict_sha256, verify_temporal_capture
from .run_store import _atomic_json


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _feature_signature_sha256(kind: str, values: Sequence[float]) -> str:
    tensor = torch.tensor(values, dtype=torch.float32).contiguous()
    digest = hashlib.sha256(b"geo-feature-signature-v1\0")
    digest.update(kind.encode("ascii") + b"\0")
    digest.update(FEATURE_SCHEMA["sha256"].encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _new_box_model(
    dataset: Any, config: dict[str, Any], device: torch.device
) -> BoxMultiLabelModel:
    required = {
        "domain",
        "latent_dim",
        "boxes",
        "epsilon",
        "objective",
        "tau",
        "margin",
    }
    if not required <= set(config):
        raise ValueError("checkpoint training config lacks BOX reconstruction settings")
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        dataset.canonical_class_ids,
        dataset.class_manifest_sha256,
        latent_dim=config["latent_dim"],
        boxes=config["boxes"],
        domain=config["domain"],
        epsilon=config["epsilon"],
        seed=config.get("seed", 23),
        class_names=dataset.class_names,
        class_iris=dataset.class_iris,
        fixed_positive_class_ids=getattr(dataset, "fixed_positive_class_ids", ()),
        device=device,
    ).to(device)
    return model


def _reconstruct_population(
    model: BoxMultiLabelModel,
    dataset: Any,
    records: Sequence[Any],
    *,
    split: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> dict[str, Any]:
    class_ids = tuple(dataset.canonical_class_ids)
    pizza_supervision = set(dataset.pizza_supervision_class_ids)
    component_supervision = set(dataset.component_supervision_class_ids)
    pizza_applicable_values = [class_id in pizza_supervision for class_id in class_ids]
    component_applicable_values = [class_id in component_supervision for class_id in class_ids]
    pizza_vectors = [pizza_features(record) for record in records]
    pizza_tensor = torch.tensor(pizza_vectors, dtype=torch.float32, device=lower.device).reshape(
        len(records), len(PIZZA_FEATURE_NAMES)
    )
    pizza_coordinates = model.encode(pizza_tensor, "pizza")
    pizza_predictions = box_membership(
        pizza_coordinates, lower, upper, domain=model.domain, epsilon=model.epsilon
    )
    pizza_clearances = box_clearance(
        pizza_coordinates, lower, upper, domain=model.domain, epsilon=model.epsilon
    )
    pizza_occurrences: list[dict[str, Any]] = []
    pizza_signatures: dict[str, dict[str, Any]] = {}
    component_inputs: list[tuple[Any, int, int, Any, tuple[float, ...]]] = []
    for parent_index, record in enumerate(records):
        for component_index, component in enumerate(record.components):
            component_inputs.append(
                (record, parent_index, component_index, component, component_features(component))
            )
    component_vectors = [row[4] for row in component_inputs]
    component_tensor = torch.tensor(
        component_vectors, dtype=torch.float32, device=lower.device
    ).reshape(len(component_inputs), len(COMPONENT_FEATURE_NAMES))
    component_coordinates = model.encode(component_tensor, "component")
    component_predictions = box_membership(
        component_coordinates, lower, upper, domain=model.domain, epsilon=model.epsilon
    )
    component_clearances = box_clearance(
        component_coordinates, lower, upper, domain=model.domain, epsilon=model.epsilon
    )
    component_occurrences: list[dict[str, Any]] = []
    component_signatures: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        pizza_targets, pizza_known, _, _ = targets_in_trainable_order(dataset, record)
        signature = _feature_signature_sha256("pizza", pizza_vectors[index])
        occurrence_id = record.sample_id
        row = {
            "occurrence_id": occurrence_id,
            "sample_id": record.sample_id,
            "split": split,
            "parent_index": index,
            "feature_signature_sha256": signature,
            "coordinates": pizza_coordinates[index].detach().cpu().tolist(),
            "targets": list(pizza_targets),
            "known_mask": [bool(value) for value in pizza_known],
            "applicable_mask": pizza_applicable_values,
            "predictions": pizza_predictions[index].detach().cpu().tolist(),
            "clearance": pizza_clearances[index].detach().cpu().tolist(),
        }
        pizza_occurrences.append(row)
        group = pizza_signatures.setdefault(
            signature, {"occurrence_ids": [], "occurrence_count": 0, "signature_weight": 1.0}
        )
        group["occurrence_ids"].append(occurrence_id)
        group["occurrence_count"] += 1
    for index, (record, parent_index, component_index, component, vector) in enumerate(
        component_inputs
    ):
        signature = _feature_signature_sha256("component", vector)
        occurrence_id = component.component_id
        row = {
            "occurrence_id": occurrence_id,
            "sample_id": record.sample_id,
            "parent_sample_id": record.sample_id,
            "parent_index": parent_index,
            "component_index": component_index,
            "split": split,
            "feature_signature_sha256": signature,
            "coordinates": component_coordinates[index].detach().cpu().tolist(),
            "targets": [component.ontology_targets[class_id] for class_id in class_ids],
            "known_mask": [bool(component.ontology_known_mask[class_id]) for class_id in class_ids],
            "applicable_mask": component_applicable_values,
            "predictions": component_predictions[index].detach().cpu().tolist(),
            "clearance": component_clearances[index].detach().cpu().tolist(),
        }
        component_occurrences.append(row)
        group = component_signatures.setdefault(
            signature, {"occurrence_ids": [], "occurrence_count": 0, "signature_weight": 1.0}
        )
        group["occurrence_ids"].append(occurrence_id)
        group["occurrence_count"] += 1
    return {
        "pizza_occurrences": pizza_occurrences,
        "component_occurrences": component_occurrences,
        "distinct_feature_signatures": {
            "pizza": pizza_signatures,
            "component": component_signatures,
        },
    }


def reconstruct_box_checkpoint(
    checkpoint_path: Path | str,
    dataset: Any,
    train_records: Sequence[Any],
    validation_records: Sequence[Any],
    *,
    training_config: dict[str, Any],
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Reconstruct exact BOX endpoints, entity observations, masks and populations offline."""
    if any(record.split != "train" for record in train_records) or any(
        record.split != "validation" for record in validation_records
    ):
        raise ValueError("BOX reconstruction accepts train and validation records only")
    train_ids = [record.sample_id for record in train_records]
    validation_ids = [record.sample_id for record in validation_records]
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("reconstruction sample identities must be unique within each split")
    if set(train_ids) & set(validation_ids):
        raise ValueError("reconstruction train and validation identities overlap")
    if not train_records:
        raise ValueError("BOX reconstruction requires the original training selection")
    device_obj = torch.device(device)
    model = _new_box_model(dataset, training_config, device_obj)
    checkpoint = Path(checkpoint_path)
    model.load_checkpoint(
        checkpoint,
        objective=training_config["objective"],
        tau=training_config["tau"],
        margin=training_config["margin"],
        map_location=device_obj,
        training_config=training_config,
    )
    payload = torch.load(checkpoint, map_location=device_obj, weights_only=True)
    geometry = payload.get("geometry") if isinstance(payload, dict) else None
    if not isinstance(geometry, dict) or set(geometry) != {"lower", "upper"}:
        raise ValueError("BOX checkpoint lacks authoritative physical endpoints")
    lower, upper = geometry["lower"], geometry["upper"]
    decoded_lower, decoded_upper = model.endpoints()
    if not torch.equal(lower, decoded_lower) or not torch.equal(upper, decoded_upper):
        raise ValueError("BOX checkpoint raw state disagrees with authoritative stored endpoints")
    populations = {
        "train": _reconstruct_population(
            model, dataset, train_records, split="train", lower=lower, upper=upper
        ),
        "validation": _reconstruct_population(
            model, dataset, validation_records, split="validation", lower=lower, upper=upper
        ),
    }
    return {
        "schema_version": "geo-box-reconstruction-v1",
        "family": model.family,
        "device": str(device_obj),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_state_sha256": state_dict_sha256(model.state_dict()),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "feature_schema_sha256": FEATURE_SCHEMA["sha256"],
        },
        "class_ids": list(dataset.canonical_class_ids),
        "class_names": list(dataset.class_names),
        "geometry": {
            "domain": model.domain,
            "dimensions": model.latent_dim,
            "boxes_per_class": model.box_count,
            "lower": lower.detach().cpu().tolist(),
            "upper": upper.detach().cpu().tolist(),
        },
        "populations": populations,
    }


def _metrics_for_population(
    population: dict[str, Any], class_names: Sequence[str]
) -> dict[str, Any]:
    pizza = population["pizza_occurrences"]
    components = population["component_occurrences"]

    def tensors(rows: list[dict[str, Any]]):
        width = len(class_names)
        if not rows:
            empty_bool = torch.empty((0, width), dtype=torch.bool)
            return empty_bool, empty_bool, empty_bool, empty_bool
        return tuple(
            torch.tensor([row[key] for row in rows], dtype=torch.bool)
            for key in ("predictions", "targets", "known_mask", "applicable_mask")
        )

    pp, pt, pk, pa = tensors(pizza)
    cp, ct, ck, ca = tensors(components)
    return {
        "pizza": membership_metrics(pp, pt, pk, pa, class_names),
        "component": membership_metrics(
            cp, ct, ck, ca, class_names, [row["parent_index"] for row in components]
        ),
    }


def _compare_portable_observations(
    authoritative: dict[str, Any], portable: dict[str, Any]
) -> dict[str, Any]:
    if authoritative["class_ids"] != portable["class_ids"]:
        raise ValueError("portable reconstruction changed canonical class order")
    comparisons: dict[str, Any] = {}
    mismatch_count = 0
    maximum_endpoint_difference = 0.0
    maximum_coordinate_difference = 0.0
    maximum_clearance_difference = 0.0
    for key in ("lower", "upper"):
        for authoritative_class, portable_class in zip(
            authoritative["geometry"][key], portable["geometry"][key], strict=True
        ):
            for authoritative_box, portable_box in zip(
                authoritative_class, portable_class, strict=True
            ):
                for authoritative_value, portable_value in zip(
                    authoritative_box, portable_box, strict=True
                ):
                    maximum_endpoint_difference = max(
                        maximum_endpoint_difference,
                        abs(authoritative_value - portable_value),
                    )
    boundary_disagreements: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        comparisons[split] = {}
        for task, row_key in (
            ("pizza", "pizza_occurrences"),
            ("component", "component_occurrences"),
        ):
            left = authoritative["populations"][split][row_key]
            right = portable["populations"][split][row_key]
            if [row["occurrence_id"] for row in left] != [row["occurrence_id"] for row in right]:
                raise ValueError("portable reconstruction changed occurrence identity/order")
            task_mismatches = 0
            for authoritative_row, portable_row in zip(left, right, strict=True):
                for identity_key in (
                    "sample_id",
                    "parent_sample_id",
                    "parent_index",
                    "component_index",
                    "feature_signature_sha256",
                    "targets",
                    "known_mask",
                    "applicable_mask",
                ):
                    if authoritative_row.get(identity_key) != portable_row.get(identity_key):
                        raise ValueError(
                            f"portable reconstruction changed {identity_key} for an occurrence"
                        )
                coordinate_differences = [
                    abs(a - b)
                    for a, b in zip(
                        authoritative_row["coordinates"], portable_row["coordinates"], strict=True
                    )
                ]
                clearance_differences = [
                    abs(a - b)
                    for a, b in zip(
                        authoritative_row["clearance"], portable_row["clearance"], strict=True
                    )
                ]
                if coordinate_differences:
                    maximum_coordinate_difference = max(
                        maximum_coordinate_difference, max(coordinate_differences)
                    )
                if clearance_differences:
                    maximum_clearance_difference = max(
                        maximum_clearance_difference, max(clearance_differences)
                    )
                differing_classes = [
                    index
                    for index, (left_prediction, right_prediction) in enumerate(
                        zip(
                            authoritative_row["predictions"],
                            portable_row["predictions"],
                            strict=True,
                        )
                    )
                    if left_prediction != right_prediction
                ]
                task_mismatches += len(differing_classes)
                if differing_classes:
                    for class_index in differing_classes:
                        boundary_disagreements.append(
                            {
                                "split": split,
                                "task": task,
                                "occurrence_id": authoritative_row["occurrence_id"],
                                "class_id": authoritative["class_ids"][class_index],
                                "authoritative_clearance": authoritative_row["clearance"][
                                    class_index
                                ],
                                "portable_clearance": portable_row["clearance"][class_index],
                            }
                        )
            mismatch_count += task_mismatches
            comparisons[split][task] = {
                "occurrence_count": len(left),
                "exact_membership_mismatch_count": task_mismatches,
            }
    return {
        "status": "exact-membership-match"
        if mismatch_count == 0
        else "authoritative-device-required",
        "membership_mismatch_count": mismatch_count,
        "maximum_endpoint_absolute_difference": maximum_endpoint_difference,
        "maximum_coordinate_absolute_difference": maximum_coordinate_difference,
        "maximum_clearance_absolute_difference": maximum_clearance_difference,
        "population_comparisons": comparisons,
        "boundary_sensitive_disagreements": boundary_disagreements,
    }


def verify_geo_box_run(
    run_path: Path | str,
    data_root: Path | str,
    *,
    expected_campaign_config_sha256: str,
    device: torch.device | str = "cuda:0",
    portable_device: torch.device | str | None = "cpu",
) -> dict[str, Any]:
    """Validate all captured snapshots and reconstruct real train/validation observations."""
    from .geo01 import (
        GEO01_BATCH_SIZE,
        GEO01_DECLARED_TEST_COUNT,
        GEO01_EPOCHS,
        GEO01_LEARNING_RATE,
        GEO01_SEED,
        GEO03_CONFIG_SOURCE_COMMIT,
        geo_campaign_identity,
        geo_preparation_identity,
        load_approved_train_validation,
    )

    if str(device) != "cuda:0":
        raise ValueError("the authoritative GEO-01 replay device is cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("the approved producer-device replay requires CUDA:0")
    started = time.perf_counter()
    root = Path(run_path).resolve()
    try:
        run_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"GEO run manifest or summary is missing/invalid: {error}") from error
    if not isinstance(run_manifest, dict) or not isinstance(summary, dict):
        raise ValueError("GEO run manifest and summary must be JSON objects")
    publication_fields = json.dumps(
        {
            "invocation": run_manifest.get("invocation"),
            "dataset": run_manifest.get("dataset"),
            "summary": summary,
        },
        sort_keys=True,
        default=str,
    )
    private_paths = {
        str(Path.home().resolve()),
        Path(data_root).expanduser().resolve().as_posix(),
        root.as_posix(),
    }
    if any(path and path in publication_fields for path in private_paths):
        raise ValueError("GEO RunRecord invocation/dataset/summary exposes a local path")
    summary_run_path = summary.get("run_path")
    summary_checkpoint_paths = summary.get("checkpoint_paths")
    if (
        not isinstance(summary_run_path, str)
        or not isinstance(summary_checkpoint_paths, dict)
        or any(not isinstance(path, str) for path in summary_checkpoint_paths.values())
        or Path(summary_run_path).is_absolute()
        or any(Path(path).is_absolute() for path in summary_checkpoint_paths.values())
    ):
        raise ValueError("GEO summary artifact paths must be relative strings")
    if run_manifest.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError("GEO-01 run is not complete")
    config = run_manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("GEO run config is missing")
    expected_config = {
        "model": "box-union-two-adapters",
        "domain": "unbounded",
        "latent_dim": 2,
        "boxes": 1,
        "epsilon": 1e-4,
        "objective": "logistic",
        "tau": 0.1,
        "margin": 0.1,
        "learning_rate": GEO01_LEARNING_RATE,
        "epochs": GEO01_EPOCHS,
        "batch_size": GEO01_BATCH_SIZE,
        "device": str(device),
        "clip_norm": 1.0,
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "seed": GEO01_SEED,
        "temporal_capture": {
            "schema_version": "geo-temporal-capture-v1",
            "policy": "initial+every-first-epoch-update+later-epoch-boundaries-v1",
        },
    }
    if any(
        key not in config or type(config[key]) is not type(value) or config[key] != value
        for key, value in expected_config.items()
    ):
        raise ValueError("GEO-01 run config differs from the approved 2D unbounded BOX contract")
    if run_manifest.get("issue") != "30":
        raise ValueError("GEO run is not registered against issue 30")
    if (
        not isinstance(expected_campaign_config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_campaign_config_sha256) is None
    ):
        raise ValueError("the exact coordinator-supplied GEO-03 config hash is required")
    expected_campaign_identity = geo_campaign_identity(expected_campaign_config_sha256)
    campaign_identity = run_manifest.get("geo_campaign")
    preparation_identity = run_manifest.get("geo_preparation")
    if campaign_identity != expected_campaign_identity:
        raise ValueError("terminal RunRecord GEO campaign identity does not match GEO-03")
    if preparation_identity != geo_preparation_identity():
        raise ValueError("terminal RunRecord GEO pair/preparation identity does not match GEO-03")
    if run_manifest.get("geo_campaign_source_commit") != GEO03_CONFIG_SOURCE_COMMIT:
        raise ValueError(
            "RunRecord GEO-03 config source revision differs from the verified candidate"
        )
    if "geo_campaign" in config or "geo_preparation" in config:
        raise ValueError("GEO campaign identity must remain a top-level RunRecord field")
    dataset, train_records, validation_records, data_identity = load_approved_train_validation(
        data_root
    )
    source_record = run_manifest.get("source")
    if (
        not isinstance(source_record, dict)
        or source_record.get("availability") != "available"
        or not isinstance(source_record.get("commit_sha"), str)
        or source_record.get("dirty") is not False
    ):
        raise ValueError("GEO run source provenance is incomplete or dirty")
    dataset_record = run_manifest.get("dataset")
    if not isinstance(dataset_record, dict):
        raise ValueError("GEO run dataset provenance is missing")
    for key in (
        "dataset_id",
        "manifest_id",
        "dataset_manifest_sha256",
        "class_manifest_sha256",
        "class_ids",
        "class_names",
        "class_iris",
        "train_count",
        "validation_count",
        "train_sample_ids_sha256",
        "validation_sample_ids_sha256",
    ):
        if dataset_record.get(key) != data_identity[key]:
            raise ValueError(f"GEO run dataset provenance mismatch for {key}")
    if (
        dataset_record.get("declared_test_count") != GEO01_DECLARED_TEST_COUNT
        or dataset_record.get("test_split_decoded") is not False
        or dataset_record.get("test_statistics_used") is not False
    ):
        raise ValueError("GEO run violates the declared sealed-test isolation metadata")
    if dataset_record.get("train_sample_ids") != [record.sample_id for record in train_records]:
        raise ValueError("GEO run ordered train selection differs from the verified source")
    if dataset_record.get("validation_sample_ids") != [
        record.sample_id for record in validation_records
    ]:
        raise ValueError("GEO run ordered validation selection differs from the verified source")
    batches_per_epoch = math.ceil(len(train_records) / GEO01_BATCH_SIZE)
    paired = config.get("paired_initialization")
    if not isinstance(paired, dict):
        raise ValueError("GEO run lacks paired initialization provenance")
    paired_bounds = paired.get("box_bounds")
    paired_provenance = paired.get("provenance")
    if (
        not isinstance(paired_bounds, dict)
        or not isinstance(paired_provenance, dict)
        or paired_provenance.get("schema_version") != "geo-box-paired-initialization-v1"
        or paired_provenance.get("reference_carrier") != "bounded"
        or paired_provenance.get("target_carrier") != "unbounded"
        or paired_provenance.get("seed") != GEO01_SEED
        or paired_provenance.get("class_ids") != list(dataset.canonical_class_ids)
        or paired_provenance.get("class_manifest_sha256") != dataset.class_manifest_sha256
        or paired_provenance.get("target_sha256") != paired_bounds.get("requested_sha256")
        or paired_provenance.get("adapter_state_sha256") != paired.get("adapter_sha256")
        or paired_bounds.get("reconstruction_tolerance_abs") != 2e-5
        or paired_bounds.get("max_absolute_reconstruction_error", float("inf")) > 2e-5
    ):
        raise ValueError("GEO run paired initialization identity/accuracy is invalid")
    capture = verify_temporal_capture(
        root,
        expected_run_id=run_manifest["run_id"],
        expected_epochs=GEO01_EPOCHS,
        batches_per_epoch=batches_per_epoch,
    )
    if (
        hashlib.sha256((root / "checkpoints" / "initial.pt").read_bytes()).hexdigest()
        != capture["snapshots"][0]["checkpoint_sha256"]
        or hashlib.sha256((root / "checkpoints" / "latest.pt").read_bytes()).hexdigest()
        != capture["snapshots"][-1]["checkpoint_sha256"]
    ):
        raise ValueError(
            "temporal initial/final states differ from conventional producer checkpoints"
        )
    history = summary.get("history")
    if (
        not isinstance(history, list)
        or len(history) != GEO01_EPOCHS
        or summary.get("accepted_updates") != GEO01_EPOCHS * batches_per_epoch
    ):
        raise ValueError("GEO-01 summary must retain 60 epochs and all accepted updates")
    snapshot_reports: list[dict[str, Any]] = []
    authoritative_observations: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(capture["snapshots"]):
        checkpoint = root / entry["checkpoint_path"]
        observation = reconstruct_box_checkpoint(
            checkpoint,
            dataset,
            train_records,
            validation_records,
            training_config=config,
            device=device,
        )
        if observation["model_state_sha256"] != entry["model_state_sha256"]:
            raise ValueError(
                f"captured model state identity differs at update {entry['accepted_updates']}"
            )
        if position == 0 and (
            observation["geometry"]["lower"] != paired_bounds.get("decoded_actual_lower")
            or observation["geometry"]["upper"] != paired_bounds.get("decoded_actual_upper")
        ):
            raise ValueError("initial captured geometry differs from paired decoded target")
        digest = _json_sha256(observation)
        report = {
            "accepted_updates": entry["accepted_updates"],
            "epoch": entry["epoch"],
            "batch": entry["batch"],
            "epoch_boundary": entry["epoch_boundary"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "model_state_sha256": entry["model_state_sha256"],
            "reconstruction_sha256": digest,
            "train_pizza_occurrences": len(
                observation["populations"]["train"]["pizza_occurrences"]
            ),
            "train_component_occurrences": len(
                observation["populations"]["train"]["component_occurrences"]
            ),
            "validation_pizza_occurrences": len(
                observation["populations"]["validation"]["pizza_occurrences"]
            ),
            "validation_component_occurrences": len(
                observation["populations"]["validation"]["component_occurrences"]
            ),
            "producer_epoch_metrics_match": None,
        }
        if entry["epoch_boundary"]:
            epoch_metrics = {
                split: _metrics_for_population(
                    observation["populations"][split], dataset.class_names
                )
                for split in ("train", "validation")
            }
            producer_metrics = history[entry["epoch"] - 1].get("diagnostics")
            if producer_metrics != epoch_metrics:
                raise ValueError(
                    f"reconstructed exact membership metrics disagree at epoch {entry['epoch']}"
                )
            report["producer_epoch_metrics_match"] = True
            authoritative_observations[position] = observation
        elif position in (0, 1):
            authoritative_observations[position] = observation
        snapshot_reports.append(report)
    portable_report: dict[str, Any] = {"status": "not-requested"}
    if portable_device is not None:
        selected_positions = sorted(
            {
                0,
                1,
                batches_per_epoch,
                len(capture["snapshots"]) - 1,
            }
        )
        portable_cases: list[dict[str, Any]] = []
        for position in selected_positions:
            authoritative = authoritative_observations.get(position)
            if authoritative is None:
                entry = capture["snapshots"][position]
                authoritative = reconstruct_box_checkpoint(
                    root / entry["checkpoint_path"],
                    dataset,
                    train_records,
                    validation_records,
                    training_config=config,
                    device=device,
                )
                authoritative_observations[position] = authoritative
            entry = capture["snapshots"][position]
            try:
                portable = reconstruct_box_checkpoint(
                    root / entry["checkpoint_path"],
                    dataset,
                    train_records,
                    validation_records,
                    training_config=config,
                    device=portable_device,
                )
                comparison = _compare_portable_observations(authoritative, portable)
                comparison.update(
                    accepted_updates=entry["accepted_updates"],
                    epoch=entry["epoch"],
                    batch=entry["batch"],
                )
            except (ValueError, RuntimeError, OSError) as error:
                comparison = {
                    "status": "portable-reconstruction-rejected",
                    "accepted_updates": entry["accepted_updates"],
                    "epoch": entry["epoch"],
                    "batch": entry["batch"],
                    "error": f"{type(error).__name__}: {error}",
                    "authoritative_fallback": str(device),
                }
            portable_cases.append(comparison)
        portable_report = {
            "device": str(portable_device),
            "cases": portable_cases,
            "authoritative_device": str(device),
            "membership_is_exact": all(
                case.get("status") == "exact-membership-match" for case in portable_cases
            ),
        }
    report = {
        "schema_version": "geo-box-run-verification-v1",
        "run_id": run_manifest["run_id"],
        "geo_campaign": campaign_identity,
        "geo_preparation": preparation_identity,
        "geo_campaign_source_commit": run_manifest["geo_campaign_source_commit"],
        "source_commit_sha": run_manifest.get("source", {}).get("commit_sha"),
        "dataset_identity": {
            key: data_identity[key]
            for key in (
                "dataset_id",
                "manifest_id",
                "dataset_manifest_sha256",
                "class_manifest_sha256",
                "train_count",
                "validation_count",
                "train_sample_ids_sha256",
                "validation_sample_ids_sha256",
                "declared_test_count",
                "test_split_decoded",
                "test_statistics_used",
            )
        },
        "capture_index_sha256": capture["index_sha256"],
        "snapshot_count": capture["snapshot_count"],
        "epoch_boundary_metric_checks": sum(
            item["producer_epoch_metrics_match"] is True for item in snapshot_reports
        ),
        "snapshots": snapshot_reports,
        "portable_reconstruction": portable_report,
        "verification_runtime_seconds": time.perf_counter() - started,
    }
    derived_path = root / "derived" / "geo-box-run-verification-v1.json"
    _atomic_json(derived_path, report)
    report["derived_report_path"] = "derived/geo-box-run-verification-v1.json"
    report["derived_report_sha256"] = hashlib.sha256(derived_path.read_bytes()).hexdigest()
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--portable-device", choices=("cpu", "none"), default="cpu")
    args = parser.parse_args(argv)
    result = verify_geo_box_run(
        args.run_path,
        args.data_root,
        expected_campaign_config_sha256=args.config_sha256,
        device=args.device,
        portable_device=None if args.portable_device == "none" else args.portable_device,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
