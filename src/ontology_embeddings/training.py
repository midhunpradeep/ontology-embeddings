"""CPU trainer for pizza-centred batches with occurrence-preserving component rows."""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

from .dataset.image_data import PreparedDataset, targets_in_trainable_order
from .dataset.schema import DatasetRecord
from .evaluation import class_frequency_baseline, membership_metrics
from .features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
    component_features,
    pizza_features,
)
from .model import AffineMultiLabelModel, masked_task_loss
from .run_store import RunRecord, _atomic_json


def _atomic_torch_save(value: Any, path: Path | str) -> None:
    """Write a checkpoint beside its destination, replacing it only after success."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _applicability(dataset: PreparedDataset, task: str) -> list[float]:
    ids = dataset.canonical_class_ids
    available = set(
        dataset.pizza_supervision_class_ids
        if task == "pizza"
        else dataset.component_supervision_class_ids
    )
    return [float(class_id in available) for class_id in ids]


def _pack(records: Sequence[DatasetRecord], dataset: PreparedDataset, device: torch.device):
    ids = dataset.canonical_class_ids
    pizza_x, pizza_y, pizza_m = [], [], []
    component_x, component_y, component_m, parents = [], [], [], []
    for parent, record in enumerate(records):
        pizza_x.append(pizza_features(record))
        py, pm, cy, cm = targets_in_trainable_order(dataset, record)
        pizza_y.append(py)
        pizza_m.append(pm)
        for occurrence, component in enumerate(record.components):
            component_x.append(component_features(component))
            component_y.append(cy[occurrence])
            component_m.append(cm[occurrence])
            parents.append(parent)
    n_classes = len(ids)
    x_p = torch.tensor(pizza_x, dtype=torch.float32, device=device)
    y_p = torch.tensor(pizza_y, dtype=torch.float32, device=device)
    m_p = torch.tensor(pizza_m, dtype=torch.bool, device=device)
    if component_x:
        x_c = torch.tensor(component_x, dtype=torch.float32, device=device)
        y_c = torch.tensor(component_y, dtype=torch.float32, device=device)
        m_c = torch.tensor(component_m, dtype=torch.bool, device=device)
    else:
        x_c = torch.empty((0, len(COMPONENT_FEATURE_NAMES)), dtype=torch.float32, device=device)
        y_c = torch.empty((0, n_classes), dtype=torch.float32, device=device)
        m_c = torch.empty((0, n_classes), dtype=torch.bool, device=device)
    return x_p, y_p, m_p, x_c, y_c, m_c, parents


def _loss(
    model: AffineMultiLabelModel,
    records: Sequence[DatasetRecord],
    dataset: PreparedDataset,
    device: torch.device,
):
    x_p, y_p, m_p, x_c, y_c, m_c, parents = _pack(records, dataset, device)
    a_p = torch.tensor(_applicability(dataset, "pizza"), dtype=torch.bool, device=device).expand_as(
        y_p
    )
    a_c = torch.tensor(
        _applicability(dataset, "component"), dtype=torch.bool, device=device
    ).expand_as(y_c)
    p_logits = model.logits(x_p, "pizza")
    c_logits = model.logits(x_c, "component")
    total, tasks = masked_task_loss(p_logits, y_p, m_p, a_p, c_logits, y_c, m_c, a_c, parents)
    return total, tasks, (p_logits, c_logits, y_p, m_p, y_c, m_c, a_p, a_c, parents)


def _task_metric(
    name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    known: torch.Tensor,
    applicable: torch.Tensor,
    class_names: Sequence[str],
    component_parents: Sequence[int],
) -> dict[str, Any]:
    return membership_metrics(
        logits > 0,
        targets,
        known,
        applicable,
        class_names,
        component_parents if name == "component" else None,
    )


def _task_metrics(batch, class_names: Sequence[str]) -> dict[str, Any]:
    p_logits, c_logits, y_p, m_p, y_c, m_c, a_p, a_c, parents = batch
    return {
        "pizza": _task_metric("pizza", p_logits, y_p, m_p, a_p, class_names, parents),
        "component": _task_metric("component", c_logits, y_c, m_c, a_c, class_names, parents),
    }


def _signature_diagnostics(
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord],
    validation_batch,
    class_names: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task, train_vectors, validation_vectors in (
        (
            "pizza",
            [pizza_features(x) for x in train_records],
            [pizza_features(x) for x in validation_records],
        ),
        (
            "component",
            [component_features(c) for x in train_records for c in x.components],
            [component_features(c) for x in validation_records for c in x.components],
        ),
    ):
        train_signatures = {tuple(vector) for vector in train_vectors}
        validation_signatures = [tuple(vector) for vector in validation_vectors]
        seen_mask = [vector in train_signatures for vector in validation_signatures]
        unseen_mask = [not item for item in seen_mask]
        _, _, y_p, m_p, y_c, m_c, a_p, a_c, component_parents = validation_batch
        targets, known, applicable = (y_p, m_p, a_p) if task == "pizza" else (y_c, m_c, a_c)
        signature_tensor = torch.tensor(seen_mask, dtype=torch.bool, device=targets.device)
        logits = validation_batch[0] if task == "pizza" else validation_batch[1]
        if task == "component":
            seen_component_parents = [
                parent for parent, seen in zip(component_parents, seen_mask, strict=True) if seen
            ]
            unseen_component_parents = [
                parent
                for parent, seen in zip(component_parents, seen_mask, strict=True)
                if not seen
            ]
        else:
            seen_component_parents = unseen_component_parents = []
        result[task] = {
            "train_entity_count": len(train_vectors),
            "validation_entity_count": len(validation_vectors),
            "train_distinct_signatures": len(train_signatures),
            "train_repeated_signature_count": len(train_vectors) - len(train_signatures),
            "validation_distinct_signatures": len(set(validation_signatures)),
            "validation_repeated_signature_count": len(validation_signatures)
            - len(set(validation_signatures)),
            "overlap_distinct_signatures": len(train_signatures & set(validation_signatures)),
            "validation_seen_count": sum(seen_mask),
            "validation_unseen_count": sum(unseen_mask),
            "validation_seen_metrics": _task_metric(
                task,
                logits[signature_tensor],
                targets[signature_tensor],
                known[signature_tensor],
                applicable[signature_tensor],
                class_names,
                seen_component_parents if task == "component" else (),
            ),
            "validation_unseen_metrics": _task_metric(
                task,
                logits[~signature_tensor],
                targets[~signature_tensor],
                known[~signature_tensor],
                applicable[~signature_tensor],
                class_names,
                unseen_component_parents if task == "component" else (),
            ),
        }
    return result


def _shuffle_known_columns(
    targets: Sequence[Sequence[int]], known_masks: Sequence[Sequence[int]], rng: random.Random
) -> list[tuple[int, ...]]:
    shuffled = [list(row) for row in targets]
    if not shuffled:
        return []
    for column in range(len(shuffled[0])):
        eligible_rows = [index for index, mask in enumerate(known_masks) if mask[column]]
        values = [shuffled[index][column] for index in eligible_rows]
        rng.shuffle(values)
        for index, value in zip(eligible_rows, values, strict=True):
            shuffled[index][column] = value
    return [tuple(row) for row in shuffled]


def shuffle_training_labels(records: Sequence[DatasetRecord], seed: int) -> list[DatasetRecord]:
    """Shuffle each known class's training targets among eligible entities; preserve masks."""
    rng = random.Random(seed)
    pizza_targets = _shuffle_known_columns(
        [item.pizza_targets for item in records],
        [item.pizza_known_mask for item in records],
        rng,
    )
    components = [component for item in records for component in item.components]
    component_targets = _shuffle_known_columns(
        [item.ontology_targets for item in components],
        [item.ontology_known_mask for item in components],
        rng,
    )
    offset = 0
    output = []
    for index, item in enumerate(records):
        shuffled_components = []
        for component in item.components:
            shuffled_components.append(
                replace(component, ontology_targets=component_targets[offset])
            )
            offset += 1
        output.append(
            replace(item, pizza_targets=pizza_targets[index], components=tuple(shuffled_components))
        )
    return output


def train_shuffled_label_control(
    dataset: PreparedDataset,
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord],
    *,
    seed: int = 7,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train a separately indexed global-training-row shuffle against untouched validation."""
    shuffled = shuffle_training_labels(train_records, seed)
    kwargs.setdefault("purpose", "shuffled-label-control")
    kwargs.setdefault("invocation", "python training.train_shuffled_label_control")
    kwargs.setdefault(
        "provenance_notes",
        [
            "Control independently shuffles each class's known pizza/component targets across "
            "eligible training entities/occurrences only; masks and all original validation "
            "records/labels remain unchanged."
        ],
    )
    result = train(dataset, shuffled, validation_records, seed=seed, **kwargs)
    return result


def train(
    dataset: PreparedDataset,
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord] = (),
    *,
    artifact_root: Path | str = "artifacts/runs",
    purpose: str = "smoke",
    invocation: str = "python -m ontology_embeddings.affine_smoke",
    seed: int = 7,
    epochs: int = 1,
    batch_size: int | None = None,
    learning_rate: float = 0.02,
    latent_dim: int = 16,
    provenance_notes: Sequence[str] = (),
    run_record: RunRecord | None = None,
    issue: str | None = None,
) -> dict[str, Any]:
    """Train and automatically register every invocation, including failures."""
    config = {
        "model": "two-affine-adapters-shared-affine-head",
        "latent_dim": latent_dim,
        "loss": "known-applicable-unweighted-bce-per-entity-task-mean-v1",
        "optimizer": "Adam",
        "optimizer_config": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
        },
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": "cpu",
        "hard_prediction_rule": "logit > 0",
        "python_random_seed": seed,
        "torch_seed": seed,
    }
    rec = run_record or RunRecord(
        artifact_root,
        purpose=purpose,
        invocation=invocation,
        config=config,
        dataset={},
        seed=seed,
        issue=issue,
    )
    try:
        rec.update(
            provenance_notes=list(provenance_notes),
            feature_schema=FEATURE_SCHEMA,
            environment={
                "torch": torch.__version__,
                "device": "cpu",
                "cuda": torch.version.cuda,
                "python": __import__("sys").version,
                "platform": __import__("platform").platform(),
            },
        )
        rec.update(
            dataset={
                "dataset_id": dataset.dataset_id,
                "manifest_sha256": dataset.dataset_manifest_sha256,
                "class_manifest_sha256": dataset.class_manifest_sha256,
                "schema_version": dataset.schema_version,
                "class_ids": list(dataset.canonical_class_ids),
                "class_names": list(getattr(dataset, "class_names", ())),
                "class_iris": list(getattr(dataset, "class_iris", ())),
                "pizza_supervision_class_ids": list(dataset.pizza_supervision_class_ids),
                "component_supervision_class_ids": list(dataset.component_supervision_class_ids),
                "train_sample_ids": [x.sample_id for x in train_records],
                "validation_sample_ids": [x.sample_id for x in validation_records],
                "train_selection_sha256": __import__("hashlib")
                .sha256(
                    json.dumps([x.sample_id for x in train_records], separators=(",", ":")).encode()
                )
                .hexdigest(),
                "validation_selection_sha256": __import__("hashlib")
                .sha256(
                    json.dumps(
                        [x.sample_id for x in validation_records], separators=(",", ":")
                    ).encode()
                )
                .hexdigest(),
                "train_split": sorted({x.split for x in train_records}),
                "validation_split": sorted({x.split for x in validation_records}),
            }
        )
        if not train_records:
            raise ValueError("training split must contain at least one pizza")
        if any(item.split == "test" for item in (*train_records, *validation_records)):
            raise ValueError("sealed test records cannot be passed to training or validation")
        if any(item.split != "train" for item in train_records):
            raise ValueError("all train_records must have split='train'")
        if any(item.split != "validation" for item in validation_records):
            raise ValueError("all validation_records must have split='validation'")
        train_sample_ids = {item.sample_id for item in train_records}
        validation_sample_ids = {item.sample_id for item in validation_records}
        if train_sample_ids & validation_sample_ids:
            raise ValueError("overlapping sample IDs between train and validation are not allowed")
        if epochs < 1 or learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        torch.manual_seed(seed)
        random.seed(seed)
        device = torch.device("cpu")
        model = AffineMultiLabelModel(
            len(PIZZA_FEATURE_NAMES),
            len(COMPONENT_FEATURE_NAMES),
            len(dataset.canonical_class_ids),
            latent_dim,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
        )
        initial_parameters = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        chunk = len(train_records) if batch_size is None else batch_size
        if chunk < 1:
            raise ValueError("batch_size must be positive")
        config["batch_size"] = chunk
        rec.update(config=config)
        history: list[dict[str, Any]] = []
        best_validation = float("inf")
        best_validation_metrics = None
        best_validation_predictions = None
        class_names = getattr(dataset, "class_names", [str(x) for x in dataset.canonical_class_ids])
        baseline = None
        if validation_records:
            with torch.no_grad():
                _, _, train_batch = _loss(model, train_records, dataset, device)
                _, _, validation_batch = _loss(model, validation_records, dataset, device)
            baseline = {
                task: class_frequency_baseline(
                    train_batch[2] if task == "pizza" else train_batch[4],
                    train_batch[3] if task == "pizza" else train_batch[5],
                    train_batch[6] if task == "pizza" else train_batch[7],
                    validation_batch[2] if task == "pizza" else validation_batch[4],
                    validation_batch[3] if task == "pizza" else validation_batch[5],
                    validation_batch[6] if task == "pizza" else validation_batch[7],
                    class_names,
                    validation_batch[8] if task == "component" else None,
                    dataset.canonical_class_ids,
                )
                for task in ("pizza", "component")
            }
            rec.update(baseline=baseline)
        rec.path.joinpath("checkpoints").mkdir()
        for epoch in range(epochs):
            model.train()
            order = list(train_records)
            random.shuffle(order)
            gradient_norms: list[float] = []
            for start in range(0, len(order), chunk):
                batch = order[start : start + chunk]
                optimizer.zero_grad(set_to_none=True)
                loss, tasks, _ = _loss(model, batch, dataset, device)
                if loss is None or not torch.isfinite(loss):
                    raise ValueError(
                        "training batch has no eligible supervision or non-finite loss"
                    )
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise ValueError("missing or non-finite model gradients")
                gradient_norms.append(
                    float(torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in gradients)))
                )
                optimizer.step()
            model.eval()
            # Match validation's full-partition objective; minibatch losses are only
            # optimization signals and must not be averaged equally across batch sizes.
            with torch.no_grad():
                train_loss, train_tasks, _ = _loss(model, train_records, dataset, device)
            if validation_records:
                with torch.no_grad():
                    val_loss, val_tasks, _ = _loss(model, validation_records, dataset, device)
            else:
                val_loss, val_tasks = None, {"pizza": None, "component": None}
            diagnostics = {}
            if validation_records:
                with torch.no_grad():
                    _, _, train_batch = _loss(model, train_records, dataset, device)
                    _, _, validation_batch = _loss(model, validation_records, dataset, device)
                diagnostics = {
                    "train": _task_metrics(train_batch, class_names),
                    "validation": _task_metrics(validation_batch, class_names),
                    "feature_signatures": _signature_diagnostics(
                        train_records, validation_records, validation_batch, class_names
                    ),
                }
            current = {
                "epoch": epoch + 1,
                "train_loss": None if train_loss is None else float(train_loss),
                "train_pizza_loss": (
                    None if train_tasks["pizza"] is None else float(train_tasks["pizza"])
                ),
                "train_component_loss": (
                    None if train_tasks["component"] is None else float(train_tasks["component"])
                ),
                "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_loss": None if val_loss is None else float(val_loss),
                "validation_pizza_loss": None
                if val_tasks["pizza"] is None
                else float(val_tasks["pizza"]),
                "validation_component_loss": None
                if val_tasks["component"] is None
                else float(val_tasks["component"]),
                "diagnostics": diagnostics,
            }
            history.append(current)
            rec.metric(current)
            _atomic_torch_save(model.state_dict(), rec.path / "checkpoints" / "latest.pt")
            if (
                current["validation_loss"] is not None
                and current["validation_loss"] < best_validation
            ):
                best_validation = current["validation_loss"]
                best_validation_metrics = diagnostics.get("validation")
                if validation_records:
                    best_validation_predictions = (
                        (validation_batch[0] > 0).cpu(),
                        (validation_batch[1] > 0).cpu(),
                    )
                _atomic_torch_save(model.state_dict(), rec.path / "checkpoints" / "best.pt")
            rec.update(last_epoch=epoch + 1, latest_metrics=current)
        reload_equivalence = None
        best_path = rec.path / "checkpoints" / "best.pt"
        if validation_records and best_path.exists():
            with torch.no_grad():
                _, _, validation_batch = _loss(model, validation_records, dataset, device)
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            model.eval()
            with torch.no_grad():
                _, _, validation_batch = _loss(model, validation_records, dataset, device)
                after = _task_metrics(validation_batch, class_names)
            predictions_match = all(
                torch.equal(expected, actual)
                for expected, actual in zip(
                    best_validation_predictions,
                    ((validation_batch[0] > 0).cpu(), (validation_batch[1] > 0).cpu()),
                    strict=True,
                )
            )
            reload_equivalence = best_validation_metrics == after and predictions_match
        update_by_module = {
            module: any(
                not torch.equal(initial_parameters[name], parameter.detach())
                for name, parameter in model.named_parameters()
                if name.startswith(module + ".")
            )
            for module in ("pizza_adapter", "component_adapter", "head")
        }
        result = {
            "run_id": rec.run_id,
            "run_path": str(rec.path),
            "history": history,
            "status": "completed",
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_updates": update_by_module,
            "provenance_notes": list(provenance_notes),
            "baseline": baseline,
            "best_checkpoint_reload_metrics_equivalent": reload_equivalence,
            "best_checkpoint_selection": "minimum combined validation BCE",
        }
        _atomic_json(rec.path / "summary.json", result)
        rec.update(
            status="completed",
            finished_at=__import__("datetime").datetime.now().astimezone().isoformat(),
            summary=result,
        )
        return result
    except KeyboardInterrupt as error:
        try:
            rec.interrupt(error)
        except BaseException:
            pass
        raise
    except BaseException as error:
        try:
            rec.fail(error)
        except BaseException:
            # Failure recording is best-effort; retain the original training exception.
            pass
        raise
