"""Run the earliest CPU metadata-oracle smoke on train/validation fixture records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .dataset.constants import DETECTOR_CATEGORIES
from .dataset.image_data import PreparedDataset
from .dataset.schema import DatasetRecord
from .run_store import RunRecord
from .training import train

INVOCATION = "python -m ontology_embeddings.affine_smoke"
PURPOSE = "smoke"
PROVENANCE_NOTES = [
    "Coordinator disclosure: earlier investigation parsed the full JSON object for "
    "img_00000 whose split is test to inspect its raw-generator metadata and split. "
    "Its target fields were present in the parsed object but their values were not "
    "accessed, printed, or used. No other test rows or targets "
    "were read in implementation commands. This smoke reads train/validation lines "
    "only and skips test lines before JSON parsing. No test predictions or metrics.",
]


def load_train_validation(
    root: Path,
) -> tuple[PreparedDataset, list[DatasetRecord], list[DatasetRecord]]:
    """Read only train/validation JSONL records; skip test lines before JSON parsing."""
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    vocab_path = root / "vocabularies" / "ontology_classes.json"
    vocab_bytes = vocab_path.read_bytes()
    vocabulary = json.loads(vocab_bytes)
    classes = vocabulary["classes"]
    trainable = [item for item in classes if item.get("trainable") is True]
    ids = tuple(item["class_id"] for item in trainable)
    if ids != tuple(vocabulary["trainable_class_ids"]) or len(ids) != 60:
        raise ValueError("fixture does not provide the canonical trainable 60-class vocabulary")
    detector = json.loads((root / "vocabularies" / "detector_categories.json").read_text())
    if tuple(item["name"] for item in detector["categories"]) != DETECTOR_CATEGORIES:
        raise ValueError("detector vocabulary order differs from feature schema")
    selected: dict[str, list[DatasetRecord]] = {"train": [], "validation": []}
    with (root / "samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            split_match = re.search(r'"split"\s*:\s*"(train|validation|test)"', line)
            if split_match is None:
                raise ValueError("dataset JSONL line lacks an explicit split")
            split = split_match.group(1)
            if split == "test":
                continue
            selected[split].append(DatasetRecord.from_dict(json.loads(line)))
    dataset = PreparedDataset(
        root=root,
        dataset_id=str(manifest["dataset_id"]),
        manifest_id=str(manifest["manifest_id"]),
        schema_version=str(manifest["schema_version"]),
        records=tuple((*selected["train"], *selected["validation"])),
        canonical_class_ids=ids,
        class_names=tuple(item["name"] for item in trainable),
        class_iris=tuple(item["iri"] for item in trainable),
        pizza_supervision_class_ids=tuple(
            item["class_id"] for item in trainable if item.get("pizza_supervision")
        ),
        component_supervision_class_ids=tuple(
            item["class_id"] for item in trainable if item.get("component_supervision")
        ),
        fixed_positive_class_ids=tuple(vocabulary.get("fixed_positive_class_ids", ())),
        unsupported_class_ids=tuple(vocabulary.get("unsupported_class_ids", ())),
        class_manifest_sha256=hashlib.sha256(vocab_bytes).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    return dataset, selected["train"], selected["validation"]


def run_smoke(
    data: Path | str = Path("examples/pizza-48"),
    *,
    artifact_root: Path | str = "artifacts/runs",
) -> dict[str, Any]:
    """Register first, then load and train under the same durable run identity."""
    rec = RunRecord(
        artifact_root,
        purpose=PURPOSE,
        invocation=INVOCATION,
        config={
            "model": "two-affine-adapters-shared-affine-head",
            "latent_dim": 16,
            "loss": "known-applicable-unweighted-bce-per-entity-task-mean-v1",
            "optimizer": "Adam",
            "learning_rate": 0.02,
            "epochs": 2,
            "batch_size": None,
            "device": "cpu",
        },
        dataset={"requested_path": str(data)},
        seed=7,
    )
    try:
        dataset, train_records, validation_records = load_train_validation(Path(data))
        return train(
            dataset,
            train_records,
            validation_records,
            artifact_root=artifact_root,
            purpose=PURPOSE,
            epochs=2,
            batch_size=max(1, min(8, len(train_records))),
            invocation=INVOCATION,
            provenance_notes=PROVENANCE_NOTES,
            run_record=rec,
        )
    except KeyboardInterrupt as error:
        rec.interrupt(error)
        raise
    except BaseException as error:
        rec.fail(error)
        raise


def main() -> None:
    print(json.dumps(run_smoke(), indent=2))


if __name__ == "__main__":
    main()
