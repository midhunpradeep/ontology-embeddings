"""CPU trainer for pizza-centred batches with occurrence-preserving component rows."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Sequence

import torch

from .dataset.image_data import PreparedDataset, targets_in_trainable_order
from .dataset.schema import DatasetRecord
from .features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
    component_features,
    pizza_features,
)
from .model import AffineMultiLabelModel, masked_task_loss
from .run_store import RunRecord


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
    return total, tasks, (p_logits, c_logits, y_p, m_p, y_c, m_c, a_p, a_c)


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
) -> dict[str, Any]:
    """Train and automatically register every invocation, including failures."""
    config = {
        "model": "two-affine-adapters-shared-affine-head",
        "latent_dim": latent_dim,
        "loss": "known-applicable-unweighted-bce-per-entity-task-mean-v1",
        "optimizer": "Adam",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": "cpu",
    }
    rec = run_record or RunRecord(
        artifact_root,
        purpose=purpose,
        invocation=invocation,
        config=config,
        dataset={},
        seed=seed,
    )
    try:
        rec.update(
            provenance_notes=list(provenance_notes),
            feature_schema=FEATURE_SCHEMA,
            environment={"torch": torch.__version__, "device": "cpu", "cuda": torch.version.cuda},
        )
        rec.update(
            dataset={
                "dataset_id": dataset.dataset_id,
                "manifest_sha256": dataset.dataset_manifest_sha256,
                "class_manifest_sha256": dataset.class_manifest_sha256,
                "schema_version": dataset.schema_version,
                "class_ids": list(dataset.canonical_class_ids),
                "train_sample_ids": [x.sample_id for x in train_records],
                "validation_sample_ids": [x.sample_id for x in validation_records],
            }
        )
        if not train_records:
            raise ValueError("training split must contain at least one pizza")
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
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        rec.path.joinpath("checkpoints").mkdir()
        for epoch in range(epochs):
            model.train()
            order = list(train_records)
            random.shuffle(order)
            epoch_losses = []
            train_task_values: dict[str, list[float]] = {"pizza": [], "component": []}
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
                for name, value in tasks.items():
                    if value is not None:
                        train_task_values[name].append(float(value.detach()))
                optimizer.step()
                epoch_losses.append(float(loss.detach()))
            model.eval()
            if validation_records:
                with torch.no_grad():
                    val_loss, val_tasks, _ = _loss(model, validation_records, dataset, device)
            else:
                val_loss, val_tasks = None, {"pizza": None, "component": None}
            current = {
                "epoch": epoch + 1,
                "train_loss": sum(epoch_losses) / len(epoch_losses),
                "train_pizza_loss": (
                    sum(train_task_values["pizza"]) / len(train_task_values["pizza"])
                    if train_task_values["pizza"]
                    else None
                ),
                "train_component_loss": (
                    sum(train_task_values["component"]) / len(train_task_values["component"])
                    if train_task_values["component"]
                    else None
                ),
                "gradient_norm_mean": sum(gradient_norms) / len(gradient_norms),
                "validation_loss": None if val_loss is None else float(val_loss),
                "validation_pizza_loss": None
                if val_tasks["pizza"] is None
                else float(val_tasks["pizza"]),
                "validation_component_loss": None
                if val_tasks["component"] is None
                else float(val_tasks["component"]),
            }
            history.append(current)
            rec.metric(current)
            _atomic_torch_save(model.state_dict(), rec.path / "checkpoints" / "latest.pt")
            if (
                current["validation_loss"] is not None
                and current["validation_loss"] < best_validation
            ):
                best_validation = current["validation_loss"]
                _atomic_torch_save(model.state_dict(), rec.path / "checkpoints" / "best.pt")
            rec.update(last_epoch=epoch + 1, latest_metrics=current)
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
        }
        rec.update(
            status="completed",
            finished_at=__import__("datetime").datetime.now().astimezone().isoformat(),
            summary=result,
        )
        (rec.path / "summary.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result
    except BaseException as error:
        try:
            rec.fail(error)
        except BaseException:
            # Failure recording is best-effort; retain the original training exception.
            pass
        raise
