"""Reproducible issue #9 training/control invocation and compact evidence export.

All support calculations and training inputs are restricted to the original train and
validation partitions. The test partition is neither parsed nor passed into the trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from .affine_smoke import load_train_validation
from .dataset.constants import TOPPING_COLUMNS
from .dataset.schema import ComponentRecord, DatasetRecord
from .features import PIZZA_FEATURE_NAMES
from .run_store import source_provenance
from .training import train, train_shuffled_label_control


def support_report(dataset, records: Sequence[DatasetRecord]) -> dict[str, Any]:
    """Return per-task known support for exactly the supplied train/validation records."""
    output: dict[str, Any] = {}
    for task in ("pizza", "component"):
        applicable_ids = set(
            dataset.pizza_supervision_class_ids
            if task == "pizza"
            else dataset.component_supervision_class_ids
        )
        counts = {
            class_id: {
                "known_positive": 0,
                "known_negative": 0,
                "unknown": 0,
                "not_applicable": 0,
            }
            for class_id in dataset.canonical_class_ids
        }
        entity_count = 0
        eligible_entities = 0
        eligible_pizzas: set[int] = set()
        for parent, record in enumerate(records):
            if task == "pizza":
                entities = ((record.pizza_targets, record.pizza_known_mask),)
            else:
                entities = tuple(
                    (component.ontology_targets, component.ontology_known_mask)
                    for component in record.components
                )
            entity_count += len(entities)
            pizza_eligible = False
            for targets, known_mask in entities:
                entity_eligible = False
                for class_id in dataset.canonical_class_ids:
                    item = counts[class_id]
                    if class_id not in applicable_ids:
                        item["not_applicable"] += 1
                    elif not known_mask[class_id]:
                        item["unknown"] += 1
                    elif targets[class_id]:
                        item["known_positive"] += 1
                        entity_eligible = True
                    else:
                        item["known_negative"] += 1
                        entity_eligible = True
                if entity_eligible:
                    eligible_entities += 1
                    pizza_eligible = True
            if pizza_eligible:
                eligible_pizzas.add(parent)
        output[task] = {
            "entity_count": entity_count,
            "eligible_entity_count": eligible_entities,
            "eligible_pizza_count": len(eligible_pizzas),
            "applicable_class_count": len(applicable_ids),
            "per_class": {
                dataset.class_names[index]: {
                    "class_id": class_id,
                    "applicable": class_id in applicable_ids,
                    **counts[class_id],
                    "status": (
                        "not_applicable"
                        if class_id not in applicable_ids
                        else "supported"
                        if counts[class_id]["known_positive"] or counts[class_id]["known_negative"]
                        else "no_known_support"
                    ),
                }
                for index, class_id in enumerate(dataset.canonical_class_ids)
            },
        }
    return output


def _tiny_linear_records() -> tuple[Any, list[DatasetRecord]]:
    """Same deliberately linearly separable 2-task control as the durable unit test."""
    class_ids = tuple(range(60))
    data = SimpleNamespace(
        dataset_id="synthetic-linear-control",
        dataset_manifest_sha256="synthetic",
        class_manifest_sha256="classes",
        schema_version="synthetic-v1",
        canonical_class_ids=class_ids,
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(1,),
    )
    examples = []
    for i, positive in enumerate((False, True, False, True)):
        components = []
        for j in range(2):
            category = "AnchovyTopping" if positive else "GarlicTopping"
            targets = [0] * 66
            known = [0] * 66
            targets[1] = int(positive)
            known[1] = 1
            components.append(
                ComponentRecord(
                    component_id=f"{i}:{j}",
                    detector_category_id=j + 1,
                    detector_category=category,
                    bbox_xyxy_original=(0, 0, 1, 1),
                    ontology_targets=tuple(targets),
                    ontology_known_mask=tuple(known),
                    annotation_path="synthetic",
                )
            )
        targets = [0] * 66
        known = [0] * 66
        targets[0] = int(positive)
        known[0] = 1
        raw = {
            "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
            "has_base": "ThinAndCrispyBase",
            "country_of_origin": None,
        }
        raw["topping_counts"][TOPPING_COLUMNS[0]] = int(positive)
        examples.append(
            DatasetRecord(
                sample_id=f"synthetic-{i}",
                split="train",
                generator_type="not-a-feature",
                image_path="",
                image_width=1,
                image_height=1,
                pizza_targets=tuple(targets),
                pizza_known_mask=tuple(known),
                pizza_source_labels=(),
                components=tuple(components),
                audit={"raw_generator": raw},
                source_annotations={},
            )
        )
    return data, examples


def _best_validation_entry(result: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [item for item in result["history"] if item["validation_loss"] is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["validation_loss"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_acceptance(dataset_path: Path, *, evidence_dir: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    dataset, train_records, validation_records = load_train_validation(dataset_path)
    support = {
        "train": support_report(dataset, train_records),
        "validation": support_report(dataset, validation_records),
    }
    command = "python -m ontology_embeddings.pr2_acceptance --dataset " + str(dataset_path)
    common = {
        "artifact_root": "artifacts/runs",
        "epochs": 60,
        "batch_size": 64,
        "learning_rate": 0.02,
        "latent_dim": 16,
        "issue": "#9",
    }
    main = train(
        dataset,
        train_records,
        validation_records,
        purpose="pr2-larger-support-driven-learning",
        invocation=command,
        seed=23,
        **common,
    )
    control = train_shuffled_label_control(
        dataset,
        train_records,
        validation_records,
        seed=29,
        purpose="shuffled-label-control",
        invocation=command + " (training labels shuffled; validation labels unchanged)",
        **common,
    )
    tiny_data, tiny_records = _tiny_linear_records()
    tiny = train(
        tiny_data,
        tiny_records,
        (),
        artifact_root="artifacts/runs",
        purpose="pr2-tiny-linearly-learnable-overfit-control",
        invocation=command + " (persistent tiny linearly learnable control)",
        seed=11,
        epochs=60,
        batch_size=4,
        learning_rate=0.04,
        latent_dim=16,
        issue="#9",
    )

    evidence_dir.mkdir(parents=True, exist_ok=True)
    if (evidence_dir / "metrics.json").exists() or (evidence_dir / "best.pt").exists():
        raise FileExistsError(
            f"evidence already exists in {evidence_dir}; choose a new output path"
        )
    main_best = Path(main["run_path"]) / "checkpoints" / "best.pt"
    if not main_best.is_file():
        raise RuntimeError("larger training run produced no best-by-validation-BCE checkpoint")
    shutil.copy2(main_best, evidence_dir / "best.pt")
    main_best_entry = _best_validation_entry(main)
    control_best_entry = _best_validation_entry(control)
    payload = {
        "schema_version": "pr2-affine-acceptance-v1",
        "claim_scope": (
            "metadata-oracle engineering evidence; not image recognition or "
            "learned ontology geometry"
        ),
        "sealed_test": (
            "No test JSONL rows/targets were parsed, passed to training, scored, or used for "
            "selection. The train/validation loader skips test lines before JSON parsing."
        ),
        "source": source_provenance(Path(__file__).resolve().parents[2]),
        "dataset": {
            "path_local_only": str(dataset_path),
            "dataset_id": dataset.dataset_id,
            "manifest_sha256": dataset.dataset_manifest_sha256,
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "samples_sha256": _sha256(dataset_path / "samples.jsonl"),
            "train_split_sha256": _sha256(dataset_path / "splits" / "train.json"),
            "validation_split_sha256": _sha256(dataset_path / "splits" / "validation.json"),
            "config": json.loads((dataset_path / "manifest.json").read_text())["resolved_config"],
            "selection": (
                "full original train and validation partitions; no subset; original "
                "deterministic 60/20/20 split policy retained"
            ),
        },
        "vocabulary": {
            "canonical_trainable_classes": len(dataset.canonical_class_ids),
            "fixed_positive_class_ids": list(dataset.fixed_positive_class_ids),
            "unsupported_class_ids": list(dataset.unsupported_class_ids),
            "pizza_supervision_class_ids": list(dataset.pizza_supervision_class_ids),
            "component_supervision_class_ids": list(dataset.component_supervision_class_ids),
        },
        "support": support,
        "runs": {
            "larger_learning": {
                "run_id": main["run_id"],
                "local_run_path": main["run_path"],
                "seed": 23,
                "config": {
                    key: common[key]
                    for key in ("epochs", "batch_size", "learning_rate", "latent_dim")
                },
                "status": main["status"],
                "parameter_updates": main["parameter_updates"],
                "parameter_count": main["parameter_count"],
                "best_epoch_by_combined_validation_bce": main_best_entry["epoch"]
                if main_best_entry
                else None,
                "best_validation_bce": main_best_entry["validation_loss"]
                if main_best_entry
                else None,
                "best_validation_metrics": main_best_entry["diagnostics"]["validation"]
                if main_best_entry
                else None,
                "feature_signature_diagnostics": main_best_entry["diagnostics"][
                    "feature_signatures"
                ]
                if main_best_entry
                else None,
                "class_frequency_baseline": main["baseline"],
                "reload_equivalence": main["best_checkpoint_reload_metrics_equivalent"],
                "curves": [
                    {
                        key: item[key]
                        for key in (
                            "epoch",
                            "train_loss",
                            "train_pizza_loss",
                            "train_component_loss",
                            "validation_loss",
                            "validation_pizza_loss",
                            "validation_component_loss",
                            "gradient_norm_mean",
                            "learning_rate",
                        )
                    }
                    for item in main["history"]
                ],
                "best_checkpoint": "best.pt",
            },
            "shuffled_training_label_control": {
                "run_id": control["run_id"],
                "local_run_path": control["run_path"],
                "seed": 29,
                "scope": (
                    "Independently shuffle each class's known pizza targets over eligible "
                    "training pizzas and component targets over eligible training occurrences; "
                    "preserve all masks; validation records and targets are unchanged."
                ),
                "status": control["status"],
                "best_epoch_by_combined_validation_bce": control_best_entry["epoch"]
                if control_best_entry
                else None,
                "best_validation_bce": control_best_entry["validation_loss"]
                if control_best_entry
                else None,
                "best_validation_metrics": control_best_entry["diagnostics"]["validation"]
                if control_best_entry
                else None,
                "class_frequency_baseline": control["baseline"],
                "reload_equivalence": control["best_checkpoint_reload_metrics_equivalent"],
                "curves": [
                    {
                        key: item[key]
                        for key in (
                            "epoch",
                            "train_loss",
                            "train_pizza_loss",
                            "train_component_loss",
                            "validation_loss",
                            "validation_pizza_loss",
                            "validation_component_loss",
                            "gradient_norm_mean",
                            "learning_rate",
                        )
                    }
                    for item in control["history"]
                ],
            },
            "tiny_linear_overfit": {
                "run_id": tiny["run_id"],
                "local_run_path": tiny["run_path"],
                "seed": 11,
                "status": tiny["status"],
                "final_train_metrics": tiny["history"][-1],
                "parameter_updates": tiny["parameter_updates"],
                "final_train_loss": tiny["history"][-1]["train_loss"],
                "interpretation": (
                    "synthetic, deliberately linearly realizable engineering control; "
                    "not predictive task evidence"
                ),
            },
        },
        "limits": [
            (
                "Pizza metadata and component detector-category one-hot inputs are oracle "
                "inputs, not pixels or image-derived predictions."
            ),
            (
                "The generator's closed-world masks are task-specific synthetic supervision; "
                "unsupported classes remain outside task applicability."
            ),
            "No F1 acceptance threshold was set; report support and metrics as measured.",
            (
                "Run store and ignored dataset/raw histories are local-only; this tracked "
                "evidence bundle preserves the compact report and selected checkpoint but "
                "does not archive imagery or full histories."
            ),
        ],
    }
    report_path = evidence_dir / "metrics.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksums = {
        path.name: _sha256(path)
        for path in sorted(evidence_dir.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }
    (evidence_dir / "checksums.json").write_text(
        json.dumps({"files": checksums}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "evidence_dir": str(evidence_dir),
        "runs": {name: item["run_id"] for name, item in payload["runs"].items()},
        "checksums": checksums,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--evidence-dir", type=Path, default=Path("docs/experiments/pr2-affine-evidence")
    )
    arguments = parser.parse_args(argv)
    result = run_acceptance(arguments.dataset, evidence_dir=arguments.evidence_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
