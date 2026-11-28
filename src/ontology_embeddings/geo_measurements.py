"""Standalone offline measurements over validated GEO numeric observations.

This module intentionally accepts tensors and endpoint snapshots, never trainable models or
application objects. Membership and clearance are delegated to the producer geometry kernels;
all statistical and scale-aware summaries are supplementary offline measurements.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
import torch

from .box import box_clearance, box_membership, validate_endpoints
from .town_geometry import (
    town_clearance,
    town_membership,
    validate_town_endpoints,
)

Domain = Literal["bounded", "unbounded"]
Split = Literal["train", "validation"]
EntityType = Literal["pizza", "component"]
Population = Literal["occurrence", "input_signature"]
MetricStatus = Literal["defined", "undefined", "unavailable"]
MetricProvenance = Literal["producer", "offline", "REL"]

__all__ = [
    "BoxEndpoints",
    "NumericObservations",
    "RecordedDiagnostic",
    "TownEndpoints",
    "compare_entity_positions",
    "measure_clearance",
    "measure_embedding_spread",
    "measure_geometry",
    "measure_predictive",
]


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identities(values: Sequence[str], name: str, *, unique: bool = False) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an ordered sequence of strings")
    result = tuple(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _numeric_tensor(value: Any, name: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.requires_grad:
        raise ValueError(f"{name} must be a detached numeric observation, not a trainable tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class RecordedDiagnostic:
    """A producer/REL measurement carried through without recomputation or relabeling."""

    name: str
    value: float | int | str | bool | None
    unit: str
    scope: str
    provenance: Literal["producer", "REL"] = "producer"
    denominator: float | None = None
    support: float | None = None
    status: MetricStatus = "defined"
    reason: str | None = None

    def __post_init__(self) -> None:
        _identity(self.name, "diagnostic name")
        _identity(self.unit, "diagnostic unit")
        _identity(self.scope, "diagnostic scope")
        if self.provenance not in ("producer", "REL"):
            raise ValueError("recorded diagnostics must retain producer or REL provenance")
        if self.status not in ("defined", "undefined", "unavailable"):
            raise ValueError("recorded diagnostic status must be defined, undefined or unavailable")
        for name, value in (("denominator", self.denominator), ("support", self.support)):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value < 0
            ):
                raise ValueError(f"diagnostic {name} must be finite and nonnegative")
        if self.status == "defined":
            if self.value is None:
                raise ValueError("a defined diagnostic requires a value")
            if isinstance(self.value, (float, int)) and not isinstance(self.value, bool):
                if not math.isfinite(float(self.value)):
                    raise ValueError("a defined diagnostic value must be finite")
        elif self.value is not None:
            raise ValueError("undefined or unavailable diagnostics must not substitute a value")
        elif not self.reason:
            raise ValueError("undefined or unavailable diagnostics require an explicit reason")


def _validate_endpoints_common(
    run_id: str, state_id: str, class_ids: Sequence[str]
) -> tuple[str, str, tuple[str, ...]]:
    return (
        _identity(run_id, "run_id"),
        _identity(state_id, "state_id"),
        _identities(class_ids, "class_ids", unique=True),
    )


@dataclass(frozen=True, slots=True)
class BoxEndpoints:
    """Validated physical BOX endpoints for one committed state: ``[C,K,d]``."""

    run_id: str
    state_id: str
    class_ids: tuple[str, ...]
    domain: Domain
    lower: torch.Tensor
    upper: torch.Tensor
    epsilon: float = 1e-4
    recorded_diagnostics: tuple[RecordedDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        run_id, state_id, class_ids = _validate_endpoints_common(
            self.run_id, self.state_id, self.class_ids
        )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "class_ids", class_ids)
        _numeric_tensor(self.lower, "BOX lower endpoints", dtype=torch.float32)
        _numeric_tensor(self.upper, "BOX upper endpoints", dtype=torch.float32)
        if self.lower.ndim != 3 or self.lower.shape[0] != len(class_ids):
            raise ValueError("BOX endpoints must have shape [class, box, dimension]")
        validate_endpoints(self.lower, self.upper, domain=self.domain, epsilon=self.epsilon)
        object.__setattr__(self, "lower", self.lower.detach().clone())
        object.__setattr__(self, "upper", self.upper.detach().clone())
        diagnostics = tuple(self.recorded_diagnostics)
        if any(not isinstance(item, RecordedDiagnostic) for item in diagnostics):
            raise TypeError("recorded_diagnostics must contain RecordedDiagnostic values")
        object.__setattr__(self, "recorded_diagnostics", diagnostics)


@dataclass(frozen=True, slots=True)
class TownEndpoints:
    """Validated physical TOWN frame/window endpoints for one committed state."""

    run_id: str
    state_id: str
    class_ids: tuple[str, ...]
    domain: Domain
    frame_lower: torch.Tensor
    frame_upper: torch.Tensor
    window_lower: torch.Tensor
    window_upper: torch.Tensor
    epsilon: float = 1e-4
    delta: float = 1e-4
    recorded_diagnostics: tuple[RecordedDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        run_id, state_id, class_ids = _validate_endpoints_common(
            self.run_id, self.state_id, self.class_ids
        )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "state_id", state_id)
        object.__setattr__(self, "class_ids", class_ids)
        for name, tensor in (
            ("frame_lower", self.frame_lower),
            ("frame_upper", self.frame_upper),
            ("window_lower", self.window_lower),
            ("window_upper", self.window_upper),
        ):
            _numeric_tensor(tensor, f"TOWN {name}", dtype=torch.float32)
        if self.frame_lower.ndim != 3 or self.frame_lower.shape[0] != len(class_ids):
            raise ValueError("TOWN frame endpoints must have shape [class, house, dimension]")
        if self.window_lower.ndim != 4:
            raise ValueError(
                "TOWN window endpoints must have shape [class, house, window, dimension]"
            )
        hosting_reserve = 2.0 * self.delta if self.window_lower.shape[2] else 0.0
        validate_town_endpoints(
            self.frame_lower,
            self.frame_upper,
            domain=self.domain,
            epsilon=self.epsilon,
            reserve=hosting_reserve,
        )
        validate_town_endpoints(
            self.window_lower,
            self.window_upper,
            domain=self.domain,
            epsilon=self.epsilon,
            reserve=0.0,
        )
        if (
            self.window_lower.shape[:2] != self.frame_lower.shape[:2]
            or self.window_lower.shape[-1] != self.frame_lower.shape[-1]
        ):
            raise ValueError("TOWN frame and window axes must match")
        if self.window_upper.shape != self.window_lower.shape:
            raise ValueError("TOWN window endpoint shapes must match")
        if any(
            endpoint.device != self.frame_lower.device
            for endpoint in (self.frame_upper, self.window_lower, self.window_upper)
        ):
            raise ValueError("TOWN endpoints must share a device")
        for name in ("frame_lower", "frame_upper", "window_lower", "window_upper"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())
        diagnostics = tuple(self.recorded_diagnostics)
        if any(not isinstance(item, RecordedDiagnostic) for item in diagnostics):
            raise TypeError("recorded_diagnostics must contain RecordedDiagnostic values")
        object.__setattr__(self, "recorded_diagnostics", diagnostics)


GeometrySnapshot: TypeAlias = BoxEndpoints | TownEndpoints


@dataclass(frozen=True, slots=True)
class NumericObservations:
    """Validated point, supervision, identity and geometry data for one split/state.

    Only train and validation are accepted. A caller must not decode or pass sealed test
    observations. Feature-signature IDs refer to exact model-input signatures; they are
    never inferred from labels, parent IDs or latent-coordinate coincidence.
    """

    run_id: str
    state_id: str
    split: Split
    entity_type: EntityType
    occurrence_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    signature_ids: tuple[str, ...]
    class_ids: tuple[str, ...]
    points: torch.Tensor
    targets: torch.Tensor
    known: torch.Tensor
    applicable: torch.Tensor
    scoring_eligibility: torch.Tensor
    geometry: GeometrySnapshot
    recorded_diagnostics: tuple[RecordedDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        _identity(self.run_id, "run_id")
        _identity(self.state_id, "state_id")
        if self.split not in ("train", "validation"):
            raise ValueError("split must be train or validation; test observations are excluded")
        if self.entity_type not in ("pizza", "component"):
            raise ValueError("entity_type must be pizza or component")
        for name in ("occurrence_ids", "parent_ids", "signature_ids", "class_ids"):
            object.__setattr__(
                self,
                name,
                _identities(
                    getattr(self, name), name, unique=(name in ("occurrence_ids", "class_ids"))
                ),
            )
        count, class_count = len(self.occurrence_ids), len(self.class_ids)
        if not (len(self.parent_ids) == len(self.signature_ids) == count):
            raise ValueError("occurrence, parent and signature identities must have equal lengths")
        _numeric_tensor(self.points, "points", dtype=torch.float32)
        if self.points.ndim != 2 or self.points.shape[0] != count:
            raise ValueError("points must have shape [occurrences, dimension]")
        if self.points.shape[1] < 1:
            raise ValueError("points must have at least one latent dimension")
        for name in ("targets", "known", "applicable"):
            tensor = _numeric_tensor(getattr(self, name), name, dtype=torch.bool)
            if tensor.shape != (count, class_count):
                raise ValueError(f"{name} must have shape [occurrences, classes]")
            if tensor.device != self.points.device:
                raise ValueError("points, targets and masks must share a device")
        eligibility = _numeric_tensor(
            self.scoring_eligibility, "scoring_eligibility", dtype=torch.bool
        )
        if eligibility.shape != (class_count,) or eligibility.device != self.points.device:
            raise ValueError("scoring_eligibility must be a class mask on the observation device")
        if not isinstance(self.geometry, (BoxEndpoints, TownEndpoints)):
            raise TypeError("geometry must be a validated BoxEndpoints or TownEndpoints snapshot")
        if self.geometry.run_id != self.run_id or self.geometry.state_id != self.state_id:
            raise ValueError("numeric observations and geometry must identify the same run/state")
        if self.geometry.class_ids != self.class_ids:
            raise ValueError("geometry class identities/order must match supervision exactly")
        geometry_dimension = (
            self.geometry.lower.shape[-1]
            if isinstance(self.geometry, BoxEndpoints)
            else self.geometry.frame_lower.shape[-1]
        )
        if geometry_dimension != self.points.shape[1]:
            raise ValueError("point and geometry dimensions differ")
        tensors = (
            (self.geometry.lower, self.geometry.upper)
            if isinstance(self.geometry, BoxEndpoints)
            else (
                self.geometry.frame_lower,
                self.geometry.frame_upper,
                self.geometry.window_lower,
                self.geometry.window_upper,
            )
        )
        if any(tensor.device != self.points.device for tensor in tensors):
            raise ValueError("points and geometry must share a device")
        diagnostics = tuple(self.recorded_diagnostics)
        if any(not isinstance(item, RecordedDiagnostic) for item in diagnostics):
            raise TypeError("recorded_diagnostics must contain RecordedDiagnostic values")
        object.__setattr__(self, "recorded_diagnostics", diagnostics)
        for name in (
            "points",
            "targets",
            "known",
            "applicable",
            "scoring_eligibility",
        ):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    @property
    def predictions(self) -> torch.Tensor:
        """Recompute authoritative hard membership from stored physical endpoints."""
        geometry = self.geometry
        if isinstance(geometry, BoxEndpoints):
            return box_membership(
                self.points,
                geometry.lower,
                geometry.upper,
                domain=geometry.domain,
                epsilon=geometry.epsilon,
            )
        if geometry.window_lower.shape[2] == 0:
            return box_membership(
                self.points,
                geometry.frame_lower,
                geometry.frame_upper,
                domain=geometry.domain,
                epsilon=geometry.epsilon,
            )
        return town_membership(
            self.points,
            geometry.frame_lower,
            geometry.frame_upper,
            geometry.window_lower,
            geometry.window_upper,
            domain=geometry.domain,
            epsilon=geometry.epsilon,
            delta=geometry.delta,
        )

    @property
    def clearances(self) -> torch.Tensor:
        """Recompute producer-native raw physical clearance for each class."""
        geometry = self.geometry
        if isinstance(geometry, BoxEndpoints):
            scores = box_clearance(
                self.points,
                geometry.lower,
                geometry.upper,
                domain=geometry.domain,
                epsilon=geometry.epsilon,
            )
        elif geometry.window_lower.shape[2] == 0:
            # H boxes with W=0 are exactly a closed BOX union by TOWN's contract.
            scores = box_clearance(
                self.points,
                geometry.frame_lower,
                geometry.frame_upper,
                domain=geometry.domain,
                epsilon=geometry.epsilon,
            )
        else:
            scores = town_clearance(
                self.points,
                geometry.frame_lower,
                geometry.frame_upper,
                geometry.window_lower,
                geometry.window_upper,
                domain=geometry.domain,
                epsilon=geometry.epsilon,
                delta=geometry.delta,
            )
        if not torch.isfinite(scores).all():
            raise ValueError("native clearance produced non-finite values for this observation")
        return scores


def _metric(
    value: Any,
    unit: str,
    *,
    provenance: MetricProvenance = "offline",
    support: float | int | None = None,
    numerator: float | int | None = None,
    denominator: float | int | None = None,
    status: MetricStatus = "defined",
    reason: str | None = None,
) -> dict[str, Any]:
    if status == "defined":
        if value is None:
            raise ValueError("defined measurement requires a value")
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError("defined scalar measurements must be finite")
    elif value is not None:
        raise ValueError("undefined/unavailable measurements must not be coerced to a value")
    return {
        "value": value,
        "unit": unit,
        "provenance": provenance,
        "status": status,
        "support": support,
        "numerator": numerator,
        "denominator": denominator,
        "reason": reason,
    }


def _undefined(unit: str, reason: str, *, support: float | int | None = None) -> dict[str, Any]:
    return _metric(None, unit, status="undefined", reason=reason, support=support)


def _unavailable(unit: str, reason: str) -> dict[str, Any]:
    return _metric(None, unit, status="unavailable", reason=reason)


def _scope(observation: NumericObservations, population: Population) -> dict[str, str]:
    return {
        "run_id": observation.run_id,
        "state_id": observation.state_id,
        "split": observation.split,
        "family": "BOX" if isinstance(observation.geometry, BoxEndpoints) else "TOWN",
        "entity_type": observation.entity_type,
        "dimension": int(observation.points.shape[1]),
        "domain": observation.geometry.domain,
        "population": "occurrence" if population == "occurrence" else "input_feature_signature",
    }


def _signature_weights(
    observation: NumericObservations,
    eligible: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    """Class-specific weights: each signature's eligible occurrences sum to one."""
    weights = np.zeros(eligible.shape, dtype=np.float64)
    signature_ids = observation.signature_ids
    for class_index in range(eligible.shape[1]):
        grouped: dict[str, list[int]] = {}
        for row in row_indices.tolist():
            if eligible[row, class_index]:
                grouped.setdefault(signature_ids[row], []).append(row)
        for rows in grouped.values():
            weights[rows, class_index] = 1.0 / len(rows)
    return weights


def _weights_for(
    observation: NumericObservations,
    eligible: np.ndarray,
    row_indices: np.ndarray,
    population: Population,
) -> np.ndarray:
    weights = np.zeros(eligible.shape, dtype=np.float64)
    if population == "occurrence":
        weights[row_indices] = eligible[row_indices].astype(np.float64)
        return weights
    if population == "input_signature":
        return _signature_weights(observation, eligible, row_indices)
    raise ValueError(f"unsupported population {population!r}")


def _counts_and_metrics(
    tp: float, fp: float, fn: float, tn: float, *, support: float
) -> dict[str, Any]:
    precision_den = tp + fp
    recall_den = tp + fn
    f1_den = 2 * tp + fp + fn
    precision = (
        _metric(
            tp / precision_den, "fraction", support=support, numerator=tp, denominator=precision_den
        )
        if precision_den > 0
        else _undefined("fraction", "no predicted positives", support=support)
    )
    recall = (
        _metric(tp / recall_den, "fraction", support=support, numerator=tp, denominator=recall_den)
        if recall_den > 0
        else _undefined("fraction", "no known positive support", support=support)
    )
    f1 = (
        _metric(2 * tp / f1_den, "fraction", support=support, numerator=2 * tp, denominator=f1_den)
        if f1_den > 0
        else _undefined("fraction", "no positive labels or predictions", support=support)
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def _unavailable_class_metrics(reason: str) -> dict[str, Any]:
    return {
        "precision": _unavailable("fraction", reason),
        "recall": _unavailable("fraction", reason),
        "f1": _unavailable("fraction", reason),
        "confusion": None,
    }


def _weighted_macro(
    per_class: dict[str, dict[str, Any]], metric_name: str, *, weighted: bool
) -> dict[str, Any]:
    selected = [
        (class_data["metrics"][metric_name], class_data["scored_support"]["known_positive"])
        for class_data in per_class.values()
        if class_data["metrics"][metric_name]["status"] == "defined"
    ]
    if weighted:
        denominator = sum(float(weight) for _, weight in selected)
        if denominator <= 0:
            return _undefined("fraction", "no positive support for weighted classes", support=0)
        numerator = sum(float(metric["value"]) * float(weight) for metric, weight in selected)
        return _metric(
            numerator / denominator,
            "fraction",
            support=denominator,
            numerator=numerator,
            denominator=denominator,
        )
    if not selected:
        return _undefined("fraction", "no classes with a defined metric", support=0)
    numerator = sum(float(metric["value"]) for metric, _ in selected)
    denominator = len(selected)
    return _metric(
        numerator / denominator,
        "fraction",
        support=denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _predictive_block(
    observation: NumericObservations,
    population: Population,
    row_indices: np.ndarray,
    *,
    include_scope: bool = True,
) -> dict[str, Any]:
    n, class_count = observation.targets.shape
    targets = observation.targets.detach().cpu().numpy()
    known = observation.known.detach().cpu().numpy()
    applicable = observation.applicable.detach().cpu().numpy()
    predictions = observation.predictions.detach().cpu().numpy()
    eligibility_classes = observation.scoring_eligibility.detach().cpu().numpy()
    eligible = known & applicable & eligibility_classes[None, :]
    weights = _weights_for(observation, eligible, row_indices, population)
    signature_weights = _signature_weights(observation, eligible, row_indices)
    per_class: dict[str, dict[str, Any]] = {}
    micro = np.zeros(4, dtype=np.float64)  # TP, FP, FN, TN over eligible task labels.

    for class_index, class_id in enumerate(observation.class_ids):
        observed = row_indices
        valid_label = known[observed, class_index] & applicable[observed, class_index]
        positive = valid_label & targets[observed, class_index]
        negative = valid_label & ~targets[observed, class_index]
        unknown = applicable[observed, class_index] & ~known[observed, class_index]
        not_applicable = ~applicable[observed, class_index]
        occurrence_support = {
            "entities": int(len(observed)),
            "known_positive": int(np.count_nonzero(positive)),
            "known_negative": int(np.count_nonzero(negative)),
            "unknown": int(np.count_nonzero(unknown)),
            "not_applicable": int(np.count_nonzero(not_applicable)),
        }
        class_eligible = eligible[:, class_index]
        pos_mask = class_eligible & targets[:, class_index]
        neg_mask = class_eligible & ~targets[:, class_index]
        scored_support = {
            "eligible": float(weights[:, class_index].sum()),
            "known_positive": float(weights[pos_mask, class_index].sum()),
            "known_negative": float(weights[neg_mask, class_index].sum()),
        }
        signature_support = {
            "eligible": float(signature_weights[:, class_index].sum()),
            "known_positive": float(signature_weights[pos_mask, class_index].sum()),
            "known_negative": float(signature_weights[neg_mask, class_index].sum()),
        }
        selected_eligible = observed[class_eligible[observed]]
        selected_positive = observed[pos_mask[observed]]
        selected_negative = observed[neg_mask[observed]]
        parent_support = {
            "eligible": int(len({observation.parent_ids[index] for index in selected_eligible})),
            "known_positive": int(
                len({observation.parent_ids[index] for index in selected_positive})
            ),
            "known_negative": int(
                len({observation.parent_ids[index] for index in selected_negative})
            ),
        }
        if not bool(eligibility_classes[class_index]):
            metric_set = _unavailable_class_metrics("class excluded by scoring-eligibility mask")
        else:
            predicted = predictions[:, class_index]
            tp = float(weights[:, class_index][pos_mask & predicted].sum())
            fp = float(weights[:, class_index][neg_mask & predicted].sum())
            fn = float(weights[:, class_index][pos_mask & ~predicted].sum())
            tn = float(weights[:, class_index][neg_mask & ~predicted].sum())
            metric_set = _counts_and_metrics(tp, fp, fn, tn, support=scored_support["eligible"])
            micro += (tp, fp, fn, tn)
        per_class[class_id] = {
            "scoring_eligibility": bool(eligibility_classes[class_index]),
            "occurrence_support": occurrence_support,
            "scored_support": scored_support,
            "signature_support": signature_support,
            "parent_support": parent_support,
            "metrics": metric_set,
        }

    aggregate_metrics = _counts_and_metrics(*micro, support=float(micro.sum()))
    for metric_name in ("precision", "recall", "f1"):
        aggregate_metrics[f"macro_{metric_name}"] = _weighted_macro(
            per_class, metric_name, weighted=False
        )
        aggregate_metrics[f"weighted_{metric_name}"] = _weighted_macro(
            per_class, metric_name, weighted=True
        )
    output: dict[str, Any] = {
        "scope": _scope(observation, population),
        "provenance": "offline",
        "weighting": {
            "occurrence": "one weight per eligible entity occurrence",
            "input_signature": (
                "each signature contributes total weight one per class among eligible "
                "known-applicable occurrences; all eligible labels are retained"
            ),
            "weighted_average": "class metrics weighted by eligible known-positive support",
            "signature_population": (
                "input-feature signatures only; not REL's time-varying distinct-embedding "
                "population"
            ),
        },
        "geometry_space": "original latent coordinates; no projection-based evaluation",
        "class_ids": list(observation.class_ids),
        "per_class": per_class,
        "aggregate": aggregate_metrics,
        "support": {
            "occurrences": int(len(row_indices)),
            "distinct_input_signatures": len(
                {observation.signature_ids[index] for index in row_indices.tolist()}
            ),
            "eligible_class_ids": [
                class_id
                for class_id, is_eligible in zip(
                    observation.class_ids, eligibility_classes.tolist(), strict=True
                )
                if is_eligible
            ],
        },
        "recorded_diagnostics": [asdict(item) for item in observation.recorded_diagnostics],
    }
    if not include_scope:
        output.pop("scope")
    return output


def measure_predictive(
    observation: NumericObservations,
    *,
    population: Population,
    training_signature_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Measure exact hard-membership metrics with explicit occurrence/signature weights.

    Signature weights normalize eligible occurrences separately for each class/signature;
    conflicting labels are retained fractionally rather than represented by a chosen row.
    Existing producer metrics remain available verbatim in ``recorded_diagnostics``.
    """
    if population not in ("occurrence", "input_signature"):
        raise ValueError("population must be occurrence or input_signature")
    all_rows = np.arange(len(observation.occurrence_ids), dtype=np.int64)
    report = _predictive_block(observation, population, all_rows)
    if observation.split != "validation":
        report["seen_unseen"] = {
            "status": "unavailable",
            "reason": "seen/unseen signature strata are defined for validation observations",
        }
    elif training_signature_ids is None:
        report["seen_unseen"] = {
            "status": "unavailable",
            "reason": "training signature identities were not supplied",
        }
    else:
        if any(not isinstance(item, str) or not item.strip() for item in training_signature_ids):
            raise ValueError("training_signature_ids must contain nonempty strings")
        seen_set = set(training_signature_ids)
        seen_rows = np.asarray(
            [
                index
                for index, signature in enumerate(observation.signature_ids)
                if signature in seen_set
            ],
            dtype=np.int64,
        )
        unseen_rows = np.asarray(
            [
                index
                for index, signature in enumerate(observation.signature_ids)
                if signature not in seen_set
            ],
            dtype=np.int64,
        )
        report["seen_unseen"] = {
            "status": "defined",
            "reference": "exact input-feature signature identity in training split",
            "training_signature_count": len(seen_set),
            "seen": _predictive_block(observation, population, seen_rows, include_scope=False),
            "unseen": _predictive_block(observation, population, unseen_rows, include_scope=False),
        }
    return report


def _occupancy_weights(observation: NumericObservations, population: Population) -> np.ndarray:
    n, c = observation.targets.shape
    weights = np.ones((n, c), dtype=np.float64)
    if population == "occurrence":
        return weights
    if population != "input_signature":
        raise ValueError("population must be occurrence or input_signature")
    groups: dict[str, list[int]] = {}
    for index, signature in enumerate(observation.signature_ids):
        groups.setdefault(signature, []).append(index)
    weights.fill(0.0)
    for rows in groups.values():
        weights[rows, :] = 1.0 / len(rows)
    return weights


def measure_clearance(
    observation: NumericObservations, *, population: Population
) -> dict[str, Any]:
    """Summarize producer-native clearances and observed occupancy in raw latent units."""
    if population not in ("occurrence", "input_signature"):
        raise ValueError("population must be occurrence or input_signature")
    scores = observation.clearances.detach().cpu().numpy().astype(np.float64, copy=False)
    predictions = observation.predictions.detach().cpu().numpy()
    targets = observation.targets.detach().cpu().numpy()
    known = observation.known.detach().cpu().numpy()
    applicable = observation.applicable.detach().cpu().numpy()
    class_eligibility = observation.scoring_eligibility.detach().cpu().numpy()
    eligible = known & applicable & class_eligibility[None, :]
    rows = np.arange(len(observation.occurrence_ids), dtype=np.int64)
    weights = _weights_for(observation, eligible, rows, population)
    occupancy_weights = _occupancy_weights(observation, population)
    per_class: dict[str, Any] = {}

    def distribution(mask: np.ndarray, class_index: int, *, available: bool) -> dict[str, Any]:
        indices = np.flatnonzero(mask)
        if not available:
            metric = _unavailable(
                "latent-coordinate-units", "class excluded by scoring-eligibility mask"
            )
            return {
                "count": len(indices),
                "weighted_support": 0.0,
                "minimum": metric,
                "maximum": metric,
                "mean": metric,
            }
        weight = weights[:, class_index][indices]
        denominator = float(weight.sum())
        if not len(indices) or denominator <= 0:
            reason = "no eligible observations in clearance group"
            metric = _undefined("latent-coordinate-units", reason, support=denominator)
            return {
                "count": len(indices),
                "weighted_support": denominator,
                "minimum": metric,
                "maximum": metric,
                "mean": metric,
            }
        values = scores[indices, class_index]
        numerator = float(np.dot(values, weight))
        return {
            "count": int(len(indices)),
            "weighted_support": denominator,
            "minimum": _metric(float(values.min()), "latent-coordinate-units", support=denominator),
            "maximum": _metric(float(values.max()), "latent-coordinate-units", support=denominator),
            "mean": _metric(
                numerator / denominator,
                "latent-coordinate-units",
                support=denominator,
                numerator=numerator,
                denominator=denominator,
            ),
        }

    for class_index, class_id in enumerate(observation.class_ids):
        valid = eligible[:, class_index]
        positive = valid & targets[:, class_index]
        negative = valid & ~targets[:, class_index]
        occupancy_weight = occupancy_weights[:, class_index]
        occupancy_denominator = float(occupancy_weight.sum())
        occupied = float(occupancy_weight[predictions[:, class_index]].sum())
        occupancy = (
            _metric(
                occupied / occupancy_denominator,
                "fraction of observed entities",
                support=occupancy_denominator,
                numerator=occupied,
                denominator=occupancy_denominator,
            )
            if occupancy_denominator > 0
            else _undefined(
                "fraction of observed entities", "empty observation population", support=0
            )
        )
        valid_labels = known[:, class_index] & applicable[:, class_index]
        parent_support = {
            "eligible": len({observation.parent_ids[index] for index in np.flatnonzero(valid)}),
            "known_positive": len(
                {observation.parent_ids[index] for index in np.flatnonzero(positive)}
            ),
            "known_negative": len(
                {observation.parent_ids[index] for index in np.flatnonzero(negative)}
            ),
        }
        per_class[class_id] = {
            "occurrence_support": {
                "entities": int(len(rows)),
                "known_positive": int(np.count_nonzero(valid_labels & targets[:, class_index])),
                "known_negative": int(np.count_nonzero(valid_labels & ~targets[:, class_index])),
                "unknown": int(
                    np.count_nonzero(applicable[:, class_index] & ~known[:, class_index])
                ),
                "not_applicable": int(np.count_nonzero(~applicable[:, class_index])),
                "eligible_positive": int(np.count_nonzero(positive)),
                "eligible_negative": int(np.count_nonzero(negative)),
            },
            "weighted_support": {
                "eligible_positive": float(weights[positive, class_index].sum()),
                "eligible_negative": float(weights[negative, class_index].sum()),
            },
            "parent_support": parent_support,
            "positive": distribution(
                positive, class_index, available=bool(class_eligibility[class_index])
            ),
            "negative": distribution(
                negative, class_index, available=bool(class_eligibility[class_index])
            ),
            "observed_occupancy": {
                "count": int(len(rows)),
                "distinct_input_signatures": len(set(observation.signature_ids)),
                "weighted_support": occupancy_denominator,
                "membership_rate": occupancy,
                "population_uses_supervision_mask": False,
            },
            "scoring_eligibility": bool(class_eligibility[class_index]),
        }
    return {
        "scope": _scope(observation, population),
        "provenance": "offline",
        "clearance_unit": "raw latent-coordinate units; not Euclidean distance or probability",
        "weighting": {
            "clearance_groups": "class-specific eligible occurrence weights",
            "occupancy": "all observed occurrences, or one total weight per input signature",
            "signature_population": (
                "input-feature signatures only; not REL's time-varying distinct-embedding "
                "population"
            ),
        },
        "geometry_space": "original latent coordinates; no projection-based evaluation",
        "per_class": per_class,
        "recorded_diagnostics": [asdict(item) for item in observation.recorded_diagnostics],
    }


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64, copy=False)


def _log_volume(widths: np.ndarray) -> float:
    return float(np.log(widths).sum())


def _volume_from_log(log_volume: float, *, support: int = 1) -> dict[str, Any]:
    if log_volume > math.log(sys.float_info.max):
        return _undefined(
            "latent-coordinate-units^d", "physical volume overflows float64", support=support
        )
    if log_volume < math.log(float.fromhex("0x0.0000000000001p-1022")):
        return _undefined(
            "latent-coordinate-units^d", "physical volume underflows float64", support=support
        )
    return _metric(math.exp(log_volume), "latent-coordinate-units^d", support=support)


def _primitive_measurement(
    class_id: str,
    kind: str,
    primitive_index: tuple[int, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    domain: Domain,
    reference: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    widths = upper - lower
    center = (upper + lower) / 2.0
    log_volume = _log_volume(widths)
    characteristic = math.exp(float(np.log(widths).mean()))
    volume = _volume_from_log(log_volume)
    record: dict[str, Any] = {
        "class_id": class_id,
        "kind": kind,
        "primitive_index": list(primitive_index),
        "center": _metric(center.tolist(), "latent-coordinate-units"),
        "widths": _metric(widths.tolist(), "latent-coordinate-units"),
        "minimum_width": _metric(float(widths.min()), "latent-coordinate-units"),
        "maximum_width": _metric(float(widths.max()), "latent-coordinate-units"),
        "aspect_ratio": _metric(float(widths.max() / widths.min()), "dimensionless"),
        "physical_volume": volume,
        "log_volume": _metric(log_volume, "log(latent-coordinate-units^d)"),
        "characteristic_width": _metric(characteristic, "latent-coordinate-units"),
        "scale_aware": {
            "width_fraction_of_bounded_carrier": (
                _metric((widths / 2.0).tolist(), "dimensionless")
                if domain == "bounded"
                else _unavailable(
                    "dimensionless", "unbounded carrier has no finite reference extent"
                )
            ),
            "center_fraction_of_bounded_carrier": (
                _metric((center / 2.0).tolist(), "dimensionless")
                if domain == "bounded"
                else _unavailable(
                    "dimensionless", "unbounded carrier has no finite reference extent"
                )
            ),
        },
        "change_from_reference": _unavailable("mixed", "reference geometry was not supplied"),
    }
    if reference is not None:
        ref_lower, ref_upper = reference
        ref_widths = ref_upper - ref_lower
        ref_center = (ref_lower + ref_upper) / 2.0
        ref_characteristic = float(math.exp(float(np.log(ref_widths).mean())))
        ref_log_volume = _log_volume(ref_widths)
        log_ratio = log_volume - ref_log_volume
        if log_ratio > math.log(sys.float_info.max):
            volume_ratio = _undefined("dimensionless", "volume ratio overflows float64", support=1)
        elif log_ratio < math.log(float.fromhex("0x0.0000000000001p-1022")):
            volume_ratio = _undefined("dimensionless", "volume ratio underflows float64", support=1)
        else:
            volume_ratio = _metric(math.exp(log_ratio), "dimensionless", support=1)
        center_delta = center - ref_center
        center_distance = float(np.linalg.norm(center_delta))
        normalized = (
            _metric(center_distance / ref_characteristic, "dimensionless", support=1)
            if ref_characteristic > 0
            else _undefined("dimensionless", "reference characteristic width is zero", support=1)
        )
        record["change_from_reference"] = {
            "reference_characteristic_width": _metric(
                ref_characteristic, "latent-coordinate-units"
            ),
            "center_displacement_vector": _metric(center_delta.tolist(), "latent-coordinate-units"),
            "center_displacement": _metric(center_distance, "latent-coordinate-units"),
            "normalized_center_displacement": normalized,
            "lower_endpoint_change": _metric(
                (lower - ref_lower).tolist(), "latent-coordinate-units"
            ),
            "upper_endpoint_change": _metric(
                (upper - ref_upper).tolist(), "latent-coordinate-units"
            ),
            "maximum_endpoint_displacement": _metric(
                float(max(np.max(np.abs(lower - ref_lower)), np.max(np.abs(upper - ref_upper)))),
                "latent-coordinate-units",
            ),
            "width_ratio": _metric((widths / ref_widths).tolist(), "dimensionless"),
            "relative_width_change": _metric((widths / ref_widths - 1.0).tolist(), "dimensionless"),
            "characteristic_width_ratio": _metric(
                characteristic / ref_characteristic, "dimensionless"
            ),
            "log_volume_change": _metric(log_ratio, "log(volume ratio)"),
            "volume_ratio": volume_ratio,
            "expansion_contraction": "expansion"
            if log_ratio > 0
            else "contraction"
            if log_ratio < 0
            else "unchanged",
        }
    return record


def _same_geometry_shape(first: GeometrySnapshot, second: GeometrySnapshot) -> None:
    if first.run_id != second.run_id:
        raise ValueError(
            "geometry comparisons require the same run; cross-run axes are not aligned"
        )
    if type(first) is not type(second):
        raise ValueError("geometry comparison requires the same family")
    if first.class_ids != second.class_ids or first.domain != second.domain:
        raise ValueError("geometry comparison class identities/order and carrier must match")
    if isinstance(first, BoxEndpoints) and isinstance(second, BoxEndpoints):
        if first.epsilon != second.epsilon:
            raise ValueError("BOX geometry comparison width floors must match")
        if first.lower.shape != second.lower.shape:
            raise ValueError("BOX geometry comparison primitive/dimension shapes must match")
    elif isinstance(first, TownEndpoints) and isinstance(second, TownEndpoints):
        if first.epsilon != second.epsilon or first.delta != second.delta:
            raise ValueError("TOWN geometry comparison width floors/reserves must match")
        if (
            first.frame_lower.shape != second.frame_lower.shape
            or first.window_lower.shape != second.window_lower.shape
        ):
            raise ValueError("TOWN geometry comparison primitive/dimension shapes must match")


def _town_region(
    lower: np.ndarray, upper: np.ndarray, win_lower: np.ndarray, win_upper: np.ndarray
) -> dict[str, Any]:
    widths = upper - lower
    intersection_widths = np.maximum(
        0.0, np.minimum(upper, win_upper) - np.maximum(lower, win_lower)
    )
    frame_log_volume = _log_volume(widths)
    intersection_empty = bool(np.any(intersection_widths == 0))
    if intersection_empty:
        intersection_ratio = 0.0
    else:
        intersection_ratio = math.exp(_log_volume(intersection_widths) - frame_log_volume)
        intersection_ratio = min(1.0, max(0.0, intersection_ratio))
    represented_fraction = 1.0 - intersection_ratio
    fully_removed = bool(np.all(win_lower < lower) and np.all(win_upper > upper))
    represented_volume = (
        _metric(0.0, "latent-coordinate-units^d", support=1)
        if represented_fraction == 0.0
        else _volume_from_log(frame_log_volume + math.log(represented_fraction))
    )
    intersection_volume = (
        _metric(0.0, "latent-coordinate-units^d", support=1)
        if intersection_empty
        else _volume_from_log(_log_volume(intersection_widths))
    )
    represented_log = (
        _undefined(
            "log(latent-coordinate-units^d)", "represented geometric volume is zero", support=1
        )
        if represented_fraction == 0.0
        else _metric(
            frame_log_volume + math.log(represented_fraction), "log(latent-coordinate-units^d)"
        )
    )
    return {
        "primitive_volume": _volume_from_log(frame_log_volume),
        "window_intersection_volume": intersection_volume,
        "physical_volume": represented_volume,
        "log_volume": represented_log,
        "volume_fraction_of_frame": _metric(represented_fraction, "dimensionless", support=1),
        "is_empty": _metric(fully_removed, "exact represented-set predicate", support=1),
        "set_semantics": (
            "closed frame minus strict ambient window interior; zero volume does not imply empty"
        ),
    }


def measure_geometry(
    geometry: GeometrySnapshot,
    *,
    initial: GeometrySnapshot | None = None,
) -> dict[str, Any]:
    """Report primitive physical geometry and scale-aware within-run comparisons.

    BOX union volume is reported as represented volume only for K=1. TOWN represented volume
    is computed only for H=1,W=1. No overlapping primitive volumes are summed.
    """
    if initial is not None:
        _same_geometry_shape(geometry, initial)
    classes: dict[str, Any] = {}
    if isinstance(geometry, BoxEndpoints):
        lowers, uppers = _numpy(geometry.lower), _numpy(geometry.upper)
        ref_lower = _numpy(initial.lower) if isinstance(initial, BoxEndpoints) else None
        ref_upper = _numpy(initial.upper) if isinstance(initial, BoxEndpoints) else None
        for class_index, class_id in enumerate(geometry.class_ids):
            primitives = []
            for primitive_index in range(lowers.shape[1]):
                reference = (
                    (
                        ref_lower[class_index, primitive_index],
                        ref_upper[class_index, primitive_index],
                    )
                    if ref_lower is not None and ref_upper is not None
                    else None
                )
                primitives.append(
                    _primitive_measurement(
                        class_id,
                        "box",
                        (primitive_index,),
                        lowers[class_index, primitive_index],
                        uppers[class_index, primitive_index],
                        domain=geometry.domain,
                        reference=reference,
                    )
                )
            represented = (
                primitives[0]["physical_volume"]
                if len(primitives) == 1
                else _unavailable(
                    "latent-coordinate-units^d",
                    "exact union volume for multiple boxes is not computed; "
                    "primitive volumes are not summed",
                )
            )
            classes[class_id] = {
                "primitives": primitives,
                "represented_region_volume": represented,
                "represented_region": {
                    "physical_volume": represented,
                    "is_empty": _metric(False, "exact represented-set predicate"),
                },
            }
    else:
        frame_lower, frame_upper = _numpy(geometry.frame_lower), _numpy(geometry.frame_upper)
        window_lower, window_upper = _numpy(geometry.window_lower), _numpy(geometry.window_upper)
        ref_frame_lower = (
            _numpy(initial.frame_lower) if isinstance(initial, TownEndpoints) else None
        )
        ref_frame_upper = (
            _numpy(initial.frame_upper) if isinstance(initial, TownEndpoints) else None
        )
        ref_window_lower = (
            _numpy(initial.window_lower) if isinstance(initial, TownEndpoints) else None
        )
        ref_window_upper = (
            _numpy(initial.window_upper) if isinstance(initial, TownEndpoints) else None
        )
        for class_index, class_id in enumerate(geometry.class_ids):
            primitives: list[dict[str, Any]] = []
            for house_index in range(frame_lower.shape[1]):
                reference = (
                    (
                        ref_frame_lower[class_index, house_index],
                        ref_frame_upper[class_index, house_index],
                    )
                    if ref_frame_lower is not None and ref_frame_upper is not None
                    else None
                )
                primitives.append(
                    _primitive_measurement(
                        class_id,
                        "frame",
                        (house_index,),
                        frame_lower[class_index, house_index],
                        frame_upper[class_index, house_index],
                        domain=geometry.domain,
                        reference=reference,
                    )
                )
                for window_index in range(window_lower.shape[2]):
                    window_reference = (
                        (
                            ref_window_lower[class_index, house_index, window_index],
                            ref_window_upper[class_index, house_index, window_index],
                        )
                        if ref_window_lower is not None and ref_window_upper is not None
                        else None
                    )
                    primitives.append(
                        _primitive_measurement(
                            class_id,
                            "window",
                            (house_index, window_index),
                            window_lower[class_index, house_index, window_index],
                            window_upper[class_index, house_index, window_index],
                            domain=geometry.domain,
                            reference=window_reference,
                        )
                    )
            if frame_lower.shape[1] == 1 and window_lower.shape[2] == 1:
                represented = _town_region(
                    frame_lower[class_index, 0],
                    frame_upper[class_index, 0],
                    window_lower[class_index, 0, 0],
                    window_upper[class_index, 0, 0],
                )
                if (
                    ref_frame_lower is not None
                    and ref_frame_upper is not None
                    and ref_window_lower is not None
                    and ref_window_upper is not None
                ):
                    reference_region = _town_region(
                        ref_frame_lower[class_index, 0],
                        ref_frame_upper[class_index, 0],
                        ref_window_lower[class_index, 0, 0],
                        ref_window_upper[class_index, 0, 0],
                    )
                    current_volume = represented["physical_volume"]["value"]
                    reference_volume = reference_region["physical_volume"]["value"]
                    current_log = represented["log_volume"]["value"]
                    reference_log = reference_region["log_volume"]["value"]
                    if reference_volume == 0.0:
                        volume_ratio = _undefined(
                            "dimensionless", "reference represented volume is zero", support=1
                        )
                    elif current_volume == 0.0:
                        volume_ratio = _metric(0.0, "dimensionless", support=1)
                    elif current_log is not None and reference_log is not None:
                        log_ratio = current_log - reference_log
                        if log_ratio > math.log(sys.float_info.max):
                            volume_ratio = _undefined(
                                "dimensionless", "volume ratio overflows float64", support=1
                            )
                        elif log_ratio < math.log(float.fromhex("0x0.0000000000001p-1022")):
                            volume_ratio = _undefined(
                                "dimensionless", "volume ratio underflows float64", support=1
                            )
                        else:
                            volume_ratio = _metric(math.exp(log_ratio), "dimensionless", support=1)
                    else:
                        volume_ratio = _unavailable(
                            "dimensionless", "represented volume ratio is unavailable"
                        )
                    log_change = (
                        _metric(current_log - reference_log, "log(volume ratio)")
                        if current_log is not None and reference_log is not None
                        else _undefined(
                            "log(volume ratio)", "zero represented volume has no finite logarithm"
                        )
                    )
                    delta_volume = (
                        _metric(
                            current_volume - reference_volume,
                            "latent-coordinate-units^d",
                        )
                        if current_volume is not None and reference_volume is not None
                        else _unavailable(
                            "latent-coordinate-units^d",
                            "physical volume change is not representable in float64",
                        )
                    )
                    if current_volume is not None and reference_volume is not None:
                        direction = (
                            "expansion"
                            if current_volume > reference_volume
                            else "contraction"
                            if current_volume < reference_volume
                            else "unchanged"
                        )
                    elif current_log is not None and reference_log is not None:
                        direction = (
                            "expansion"
                            if current_log > reference_log
                            else "contraction"
                            if current_log < reference_log
                            else "unchanged"
                        )
                    else:
                        direction = "undefined"
                    represented["change_from_reference"] = {
                        "reference_state_id": initial.state_id,
                        "volume_change": delta_volume,
                        "log_volume_change": log_change,
                        "volume_ratio": volume_ratio,
                        "expansion_contraction": direction,
                    }
                else:
                    represented["change_from_reference"] = _unavailable(
                        "mixed", "reference geometry was not supplied"
                    )
            else:
                represented = {
                    "physical_volume": _unavailable(
                        "latent-coordinate-units^d",
                        "represented volume is computed only for H=1,W=1",
                    ),
                    "is_empty": _unavailable(
                        "exact represented-set predicate", "represented-set solver is out of scope"
                    ),
                    "change_from_reference": _unavailable(
                        "mixed", "represented volume is unavailable"
                    ),
                }
            classes[class_id] = {
                "primitives": primitives,
                "represented_region": represented,
                "represented_region_volume": represented["physical_volume"],
            }
    return {
        "scope": {
            "run_id": geometry.run_id,
            "state_id": geometry.state_id,
            "class_ids": list(geometry.class_ids),
            "family": "BOX" if isinstance(geometry, BoxEndpoints) else "TOWN",
            "dimension": int(
                geometry.lower.shape[-1]
                if isinstance(geometry, BoxEndpoints)
                else geometry.frame_lower.shape[-1]
            ),
            "domain": geometry.domain,
            "reference_state_id": initial.state_id if initial is not None else None,
        },
        "provenance": "offline",
        "recorded_diagnostics": [asdict(item) for item in geometry.recorded_diagnostics],
        "unit_conventions": {
            "center_width_movement": "original latent-coordinate units",
            "primitive_volume": "original latent-coordinate units raised to dimension",
            "normalized_quantities": (
                "dimensionless; bounded carrier scale is width 2, otherwise within-run "
                "primitive reference"
            ),
            "primitive_volume_is_not_union_volume": True,
        },
        "classes": classes,
    }


def _population_points(
    observation: NumericObservations, population: Population
) -> tuple[np.ndarray, list[str]]:
    points = _numpy(observation.points)
    if population == "occurrence":
        return points, list(observation.occurrence_ids)
    if population != "input_signature":
        raise ValueError("population must be occurrence or input_signature")
    groups: dict[str, list[int]] = {}
    for index, signature in enumerate(observation.signature_ids):
        groups.setdefault(signature, []).append(index)
    selected, identities = [], []
    for signature, rows in groups.items():
        coordinates = points[rows]
        if not np.all(coordinates == coordinates[0]):
            raise ValueError(f"signature has inconsistent coordinates: {signature}")
        selected.append(coordinates[0])
        identities.append(signature)
    if not selected:
        return np.empty((0, points.shape[1]), dtype=np.float64), identities
    return np.stack(selected), identities


def measure_embedding_spread(
    observation: NumericObservations, *, population: Population = "occurrence"
) -> dict[str, Any]:
    """Measure original-space spread/covariance; use population covariance (divide by N)."""
    points, identities = _population_points(observation, population)
    dimension = observation.points.shape[1]
    unit = "latent-coordinate-units"
    if len(points) == 0:
        undefined_vector = _undefined(unit, "empty observation population", support=0)
        undefined_matrix = _undefined(
            "latent-coordinate-units^2", "empty observation population", support=0
        )
        undefined_dimensionless = _undefined(
            "dimensionless", "empty observation population", support=0
        )
        return {
            "scope": _scope(observation, population),
            "provenance": "offline",
            "support": {"entity_count": 0, "coordinate_dimension": dimension},
            "mean": undefined_vector,
            "per_axis_spread": undefined_vector,
            "covariance": undefined_matrix,
            "eigenvalues": undefined_vector,
            "normalized_eigenvalues": undefined_vector,
            "participation_ratio": undefined_dimensionless,
            "normalized_effective_dimension": undefined_dimensionless,
            "effective_dimension": _undefined(
                "effective dimensions", "empty observation population", support=0
            ),
        }
    mean = points.mean(axis=0)
    centered = points - mean
    covariance = centered.T @ centered / len(points)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = (
        np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(eigenvalues)))) * dimension * 8
    )
    if np.any(eigenvalues < -tolerance):
        raise ValueError("covariance eigenspectrum is numerically invalid")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(eigenvalues.sum())
    if trace == 0.0:
        effective = _undefined("effective dimensions", "zero covariance", support=len(points))
        normalized_eigenvalues = _undefined("dimensionless", "zero covariance", support=len(points))
        participation = _undefined("dimensionless", "zero covariance", support=len(points))
        normalized_effective = _undefined("dimensionless", "zero covariance", support=len(points))
    else:
        squared_sum = float(np.square(eigenvalues).sum())
        ratio = trace * trace / squared_sum
        normalized = eigenvalues / trace
        effective = _metric(ratio, "effective dimensions", support=len(points))
        normalized_eigenvalues = _metric(normalized.tolist(), "dimensionless", support=len(points))
        participation = _metric(ratio, "dimensionless participation ratio", support=len(points))
        normalized_effective = _metric(ratio / dimension, "dimensionless", support=len(points))
    return {
        "scope": _scope(observation, population),
        "provenance": "offline",
        "covariance_convention": "population covariance, denominator N; eigenvalues ascending",
        "support": {
            "entity_count": len(points),
            "coordinate_dimension": dimension,
            "identity_count": len(identities),
        },
        "mean": _metric(mean.tolist(), unit, support=len(points)),
        "per_axis_spread": _metric(
            np.sqrt(np.diag(covariance)).tolist(), unit, support=len(points)
        ),
        "covariance": _metric(
            covariance.tolist(), "latent-coordinate-units^2", support=len(points)
        ),
        "eigenvalues": _metric(
            eigenvalues.tolist(), "latent-coordinate-units^2", support=len(points)
        ),
        "normalized_eigenvalues": normalized_eigenvalues,
        "participation_ratio": participation,
        "effective_dimension": effective,
        "normalized_effective_dimension": normalized_effective,
    }


def compare_entity_positions(
    previous: NumericObservations,
    current: NumericObservations,
    *,
    population: Population = "occurrence",
) -> dict[str, Any]:
    """Compare stable entities within one run; cross-run latent displacement is forbidden."""
    if previous.run_id != current.run_id:
        raise ValueError("entity movement requires the same run; cross-run axes are not aligned")
    if previous.entity_type != current.entity_type or previous.split != current.split:
        raise ValueError("entity movement requires matching entity type and data split")
    if (
        previous.class_ids != current.class_ids
        or previous.points.shape[1] != current.points.shape[1]
    ):
        raise ValueError("entity movement requires matching class identities and dimension")
    if population not in ("occurrence", "input_signature"):
        raise ValueError("population must be occurrence or input_signature")
    old_points = _numpy(previous.points)
    new_points = _numpy(current.points)
    if population == "occurrence":
        old_index = {identity: index for index, identity in enumerate(previous.occurrence_ids)}
        new_index = {identity: index for index, identity in enumerate(current.occurrence_ids)}
        matched = sorted(set(old_index) & set(new_index))
        for identity in matched:
            oi, ni = old_index[identity], new_index[identity]
            if (
                previous.parent_ids[oi] != current.parent_ids[ni]
                or previous.signature_ids[oi] != current.signature_ids[ni]
            ):
                raise ValueError(
                    "stable occurrence parent/signature identity changed across states"
                )
        old = (
            np.stack([old_points[old_index[item]] for item in matched])
            if matched
            else np.empty((0, old_points.shape[1]))
        )
        new = (
            np.stack([new_points[new_index[item]] for item in matched])
            if matched
            else np.empty((0, new_points.shape[1]))
        )
        identity_count = len(matched)
        identity_label = "matched_occurrences"
    else:
        old_by_signature: dict[str, list[int]] = {}
        new_by_signature: dict[str, list[int]] = {}
        for index, signature in enumerate(previous.signature_ids):
            old_by_signature.setdefault(signature, []).append(index)
        for index, signature in enumerate(current.signature_ids):
            new_by_signature.setdefault(signature, []).append(index)
        matched = sorted(set(old_by_signature) & set(new_by_signature))
        old_rows, new_rows = [], []
        for signature in matched:
            old_group, new_group = (
                old_points[old_by_signature[signature]],
                new_points[new_by_signature[signature]],
            )
            if not np.all(old_group == old_group[0]) or not np.all(new_group == new_group[0]):
                raise ValueError(f"signature has inconsistent coordinates: {signature}")
            old_rows.append(old_group[0])
            new_rows.append(new_group[0])
        old = np.stack(old_rows) if old_rows else np.empty((0, old_points.shape[1]))
        new = np.stack(new_rows) if new_rows else np.empty((0, new_points.shape[1]))
        identity_count = len(matched)
        identity_label = "matched_input_signatures"
    if not len(old):
        undefined_distance = _undefined(
            "latent-coordinate-units", "no matching identities", support=0
        )
        normalized = _undefined("dimensionless", "no matching identities", support=0)
        scale = _undefined("latent-coordinate-units", "no matching identities", support=0)
        return {
            "scope": {
                "run_id": previous.run_id,
                "from_state_id": previous.state_id,
                "to_state_id": current.state_id,
                "entity_type": previous.entity_type,
                "split": previous.split,
                "population": "occurrence"
                if population == "occurrence"
                else "input_feature_signature",
            },
            "provenance": "offline",
            "support": {identity_label: 0},
            "mean_displacement": undefined_distance,
            "root_mean_square_displacement": undefined_distance,
            "maximum_displacement": undefined_distance,
            "reference_spread_scale": scale,
            "normalized_mean_displacement": normalized,
        }
    distances = np.linalg.norm(new - old, axis=1)
    reference_centered = old - old.mean(axis=0)
    reference_scale = float(
        np.sqrt(np.square(reference_centered).sum() / (len(old) * old.shape[1]))
    )
    mean_displacement = float(distances.mean())
    normalized = (
        _metric(mean_displacement / reference_scale, "dimensionless", support=len(old))
        if reference_scale > 0
        else _undefined("dimensionless", "reference embedding spread is zero", support=len(old))
    )
    return {
        "scope": {
            "run_id": previous.run_id,
            "from_state_id": previous.state_id,
            "to_state_id": current.state_id,
            "entity_type": previous.entity_type,
            "split": previous.split,
            "population": "occurrence" if population == "occurrence" else "input_feature_signature",
        },
        "provenance": "offline",
        "normalization": "reference RMS coordinate spread sqrt(trace(covariance)/dimension)",
        "support": {identity_label: identity_count},
        "mean_displacement": _metric(
            mean_displacement, "latent-coordinate-units", support=len(old)
        ),
        "root_mean_square_displacement": _metric(
            float(np.sqrt(np.square(distances).mean())), "latent-coordinate-units", support=len(old)
        ),
        "maximum_displacement": _metric(
            float(distances.max()), "latent-coordinate-units", support=len(old)
        ),
        "reference_spread_scale": _metric(
            reference_scale, "latent-coordinate-units", support=len(old)
        ),
        "normalized_mean_displacement": normalized,
    }
