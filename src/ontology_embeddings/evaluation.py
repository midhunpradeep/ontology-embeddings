"""Support-aware binary membership metrics and train-only constant baselines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


def membership_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    known: torch.Tensor,
    applicable: torch.Tensor,
    class_names: Sequence[str] | None = None,
    entity_parent_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Report exact known supports; metrics with zero denominator are ``None``."""
    predictions = predictions.bool()
    targets = targets.bool()
    known = known.bool()
    applicable = applicable.bool()
    if not (predictions.shape == targets.shape == known.shape == applicable.shape):
        raise ValueError("predictions, targets, known and applicable must have identical shapes")
    if predictions.ndim != 2:
        raise ValueError("membership inputs must be [entities, classes]")
    n_classes = predictions.shape[1]
    names = list(class_names) if class_names is not None else [str(i) for i in range(n_classes)]
    if len(names) != n_classes:
        raise ValueError("class_names length must match class dimension")
    if entity_parent_ids is not None and len(entity_parent_ids) != predictions.shape[0]:
        raise ValueError("entity_parent_ids length must match entity dimension")
    per_class: dict[str, Any] = {}
    totals = {key: 0 for key in ("tp", "fp", "fn")}
    defined_f1: list[float] = []
    weighted_f1: list[tuple[int, float]] = []
    weighted_precision: list[tuple[int, float]] = []
    weighted_recall: list[tuple[int, float]] = []
    for index, name in enumerate(names):
        valid = known[:, index] & applicable[:, index]
        positive = valid & targets[:, index]
        negative = valid & ~targets[:, index]
        predicted = predictions[:, index] & valid
        tp = int((predicted & positive).sum())
        fp = int((predicted & negative).sum())
        fn = int((~predictions[:, index] & positive).sum())
        pos, neg = int(positive.sum()), int(negative.sum())
        pred_pos = int(predicted.sum())
        precision = tp / pred_pos if pred_pos else None
        recall = tp / pos if pos else None
        f1_denominator = 2 * tp + fp + (pos - tp)
        f1 = 2 * tp / f1_denominator if f1_denominator else None
        unknown = int((applicable[:, index] & ~known[:, index]).sum())
        per_class[name] = {
            "known_positive": pos,
            "known_negative": neg,
            "unknown": unknown,
            "not_applicable": int((~applicable[:, index]).sum()),
            "predicted_positive": pred_pos,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        if f1 is not None:
            defined_f1.append(f1)
            weighted_f1.append((pos, f1))
        if precision is not None:
            weighted_precision.append((pos, precision))
        if recall is not None:
            weighted_recall.append((pos, recall))
    micro_precision = (
        totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else None
    )
    micro_recall = (
        totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else None
    )
    micro_f1_denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
    micro_f1 = (
        2 * totals["tp"] / micro_f1_denominator if micro_f1_denominator else None
    )
    f1_weight = sum(weight for weight, _ in weighted_f1)
    precision_weight = sum(weight for weight, _ in weighted_precision)
    recall_weight = sum(weight for weight, _ in weighted_recall)
    eligible_entities = (known & applicable).any(dim=1)
    eligible_pizza_count = int(eligible_entities.sum())
    if entity_parent_ids is not None:
        eligible_pizza_count = len(
            {
                parent
                for parent, is_eligible in zip(
                    entity_parent_ids, eligible_entities.tolist(), strict=True
                )
                if is_eligible
            }
        )
    return {
        "known_positive": sum(item["known_positive"] for item in per_class.values()),
        "known_negative": sum(item["known_negative"] for item in per_class.values()),
        "unknown": sum(item["unknown"] for item in per_class.values()),
        "entity_count": int(predictions.shape[0]),
        "eligible_entity_count": int(eligible_entities.sum()),
        "eligible_pizza_count": eligible_pizza_count,
        "per_class": per_class,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro_precision": sum(
            item["precision"] for item in per_class.values() if item["precision"] is not None
        )
        / sum(item["precision"] is not None for item in per_class.values())
        if any(item["precision"] is not None for item in per_class.values())
        else None,
        "macro_recall": sum(
            item["recall"] for item in per_class.values() if item["recall"] is not None
        )
        / sum(item["recall"] is not None for item in per_class.values())
        if any(item["recall"] is not None for item in per_class.values())
        else None,
        "macro_f1": sum(defined_f1) / len(defined_f1) if defined_f1 else None,
        "weighted_precision": sum(weight * score for weight, score in weighted_precision)
        / precision_weight
        if precision_weight
        else None,
        "weighted_recall": sum(weight * score for weight, score in weighted_recall) / recall_weight
        if recall_weight
        else None,
        "weighted_f1": sum(weight * f1 for weight, f1 in weighted_f1) / f1_weight
        if f1_weight
        else None,
        "macro_defined_classes": len(defined_f1),
    }


def class_frequency_baseline(
    train_targets: torch.Tensor,
    train_known: torch.Tensor,
    train_applicable: torch.Tensor,
    validation_targets: torch.Tensor,
    validation_known: torch.Tensor,
    validation_applicable: torch.Tensor,
    class_names: Sequence[str] | None = None,
    validation_parent_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit constant predictions from training frequencies only; score validation labels."""
    train_valid = train_known.bool() & train_applicable.bool()
    positives = (train_valid & train_targets.bool()).sum(dim=0)
    support = train_valid.sum(dim=0)
    frequency = torch.where(support > 0, positives.float() / support.clamp_min(1), 0.0)
    predictions = (frequency >= 0.5).expand_as(validation_targets)
    return {
        "prediction_rule": "training_frequency >= 0.5 (ties positive)",
        "training_positive": positives.tolist(),
        "training_known": support.tolist(),
        "training_frequency": frequency.tolist(),
        "metrics": membership_metrics(
            predictions,
            validation_targets,
            validation_known,
            validation_applicable,
            class_names,
            validation_parent_ids,
        ),
    }
