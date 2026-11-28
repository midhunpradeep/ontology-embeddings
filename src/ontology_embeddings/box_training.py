"""Minimal BOX classifier and fail-closed metadata training integration."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn

from .box import BoxGeometry, box_clearance, box_membership, validate_endpoints
from .dataset.image_data import PreparedDataset
from .dataset.schema import DatasetRecord
from .features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
)
from .geo_capture import CAPTURE_SCHEMA_VERSION, TemporalCapture, state_dict_sha256
from .run_store import RunRecord, _atomic_json
from .shared_training import (
    checked_optimizer_update,
    invalid_optimizer_groups,
    masked_geometric_task_loss,
)
from .training import _atomic_torch_save, _pack, _signature_diagnostics, _task_metrics

LossKind = Literal["logistic", "hinge"]
INITIAL_BOUNDS_RECONSTRUCTION_TOLERANCE = 2e-5


class BoxMultiLabelModel(nn.Module):
    """Separate input adapters sharing one ordered global trainable-class geometry."""

    family = "box-union-classifier"
    checkpoint_version = "box-checkpoint-v1"

    def __init__(
        self,
        pizza_features: int,
        component_features: int,
        class_ids: Sequence[int],
        vocabulary_sha256: str,
        *,
        latent_dim: int = 16,
        boxes: int = 1,
        domain: str = "bounded",
        epsilon: float = 1e-4,
        seed: int = 7,
        class_names: Sequence[str] | None = None,
        class_iris: Sequence[str] | None = None,
        fixed_positive_class_ids: Sequence[int] = (),
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        ids = tuple(class_ids)
        if not ids or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise ValueError("class_ids must be a nonempty ordered sequence of integers")
        if len(set(ids)) != len(ids):
            raise ValueError("class_ids must be unique")
        fixed_ids = tuple(fixed_positive_class_ids)
        if len(set(fixed_ids)) != len(fixed_ids) or set(ids) & set(fixed_ids):
            raise ValueError("fixed-positive IDs must be unique and separate from trainable IDs")
        if any(isinstance(i, bool) or not isinstance(i, int) for i in fixed_ids):
            raise ValueError("fixed-positive IDs must be integers")
        if not isinstance(vocabulary_sha256, str) or not vocabulary_sha256:
            raise ValueError("vocabulary identity is required")
        if domain not in ("bounded", "unbounded"):
            raise ValueError("domain must be bounded or unbounded")
        if min(pizza_features, component_features, latent_dim, boxes) < 1:
            raise ValueError("feature, latent and box dimensions must be positive")
        self.class_ids, self.fixed_positive_class_ids = ids, fixed_ids
        self.vocabulary_sha256 = vocabulary_sha256
        self.class_names = tuple(class_names or (str(i) for i in ids))
        self.class_iris = tuple(class_iris or ("" for _ in ids))
        if len(self.class_names) != len(ids) or len(self.class_iris) != len(ids):
            raise ValueError("class names and IRIs must follow class_ids exactly")
        self.pizza_adapter = nn.Linear(pizza_features, latent_dim)
        self.component_adapter = nn.Linear(component_features, latent_dim)
        self.geometry = BoxGeometry.initialized(
            len(ids), boxes, latent_dim, domain=domain, seed=seed, epsilon=epsilon, device=device
        )
        self.domain, self.epsilon = domain, epsilon
        self.latent_dim, self.box_count = latent_dim, boxes

    def endpoints(self):
        return self.geometry.endpoints()

    def encode(self, features: torch.Tensor, entity: str) -> torch.Tensor:
        if entity == "pizza":
            points = self.pizza_adapter(features)
        elif entity == "component":
            points = self.component_adapter(features)
        else:
            raise ValueError(f"unknown entity type {entity!r}")
        if not torch.isfinite(points).all():
            raise ValueError("encoder produced non-finite pre-activation values")
        if self.domain == "bounded":
            points = torch.tanh(points)
        if not torch.isfinite(points).all():
            raise ValueError("encoder produced non-finite points")
        if self.domain == "bounded" and points.numel() and torch.any((points < -1) | (points > 1)):
            raise ValueError("bounded encoder produced points outside [-1,1]")
        return points

    def clearances(
        self, features: torch.Tensor, entity: str, *, point_chunk_size: int | None = None
    ) -> torch.Tensor:
        points = self.encode(features, entity)
        lower, upper = self.endpoints()
        if point_chunk_size is None:
            return box_clearance(points, lower, upper, domain=self.domain, epsilon=self.epsilon)
        if (
            isinstance(point_chunk_size, bool)
            or not isinstance(point_chunk_size, int)
            or point_chunk_size < 1
        ):
            raise ValueError("point_chunk_size must be a positive integer")
        return (
            torch.cat(
                [
                    box_clearance(
                        points[start : start + point_chunk_size],
                        lower,
                        upper,
                        domain=self.domain,
                        epsilon=self.epsilon,
                    )
                    for start in range(0, len(points), point_chunk_size)
                ],
                dim=0,
            )
            if len(points)
            else points.new_empty((0, len(self.class_ids)))
        )

    def domain_thing_membership(self, points: torch.Tensor) -> torch.Tensor:
        """Parameter-free fixed-positive predicate: true exactly for valid Delta points."""
        if (
            not isinstance(points, torch.Tensor)
            or points.ndim != 2
            or points.shape[1] != self.latent_dim
        ):
            raise ValueError("points must have shape [N, latent_dim]")
        if points.dtype != torch.float32 or not torch.isfinite(points).all():
            raise ValueError("points must be finite float32")
        if self.domain == "bounded":
            return ((points >= -1) & (points <= 1)).all(dim=1)
        return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    def predict(self, features: torch.Tensor, entity: str, *, point_chunk_size: int | None = None):
        points = self.encode(features, entity)
        lower, upper = self.endpoints()
        if point_chunk_size is None:
            size = max(1, len(points))
        elif (
            isinstance(point_chunk_size, bool)
            or not isinstance(point_chunk_size, int)
            or point_chunk_size < 1
        ):
            raise ValueError("point chunk size must be a positive integer")
        else:
            size = point_chunk_size
        return (
            torch.cat(
                [
                    box_membership(
                        points[start : start + size],
                        lower,
                        upper,
                        domain=self.domain,
                        epsilon=self.epsilon,
                    )
                    for start in range(0, len(points), size)
                ],
                dim=0,
            )
            if len(points)
            else torch.empty((0, len(self.class_ids)), dtype=torch.bool, device=points.device)
        )

    def checkpoint_metadata(self, *, objective: str, tau: float, margin: float) -> dict[str, Any]:
        return {
            "format": self.checkpoint_version,
            "family": self.family,
            "parameterization": BoxGeometry.parameterization_version,
            "domain": self.domain,
            "latent_dim": self.latent_dim,
            "boxes": self.box_count,
            "epsilon": self.epsilon,
            "class_ids": list(self.class_ids),
            "fixed_positive_class_ids": list(self.fixed_positive_class_ids),
            "class_names": list(self.class_names),
            "class_iris": list(self.class_iris),
            "vocabulary_sha256": self.vocabulary_sha256,
            "feature_schema": FEATURE_SCHEMA,
            "encoder": {
                "pizza_features": self.pizza_adapter.in_features,
                "component_features": self.component_adapter.in_features,
                "bounded_activation": "tanh" if self.domain == "bounded" else "identity",
            },
            "objective": {"kind": objective, "tau": tau, "margin": margin},
        }

    def save_checkpoint(
        self,
        path: Path | str,
        *,
        objective: str = "logistic",
        tau: float = 0.1,
        margin: float = 0.1,
        epoch: int = 0,
        validation_loss: float | None = None,
        training_config: dict[str, Any] | None = None,
    ):
        lower, upper = self.endpoints()
        validate_endpoints(lower, upper, domain=self.domain, epsilon=self.epsilon)
        _atomic_torch_save(
            {
                "metadata": {
                    **self.checkpoint_metadata(objective=objective, tau=tau, margin=margin),
                    "training_config": training_config or {},
                },
                "model_state": self.state_dict(),
                "epoch": epoch,
                "validation_loss": validation_loss,
                "geometry": {
                    "lower": lower.detach().cpu(),
                    "upper": upper.detach().cpu(),
                },
            },
            path,
        )

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        objective: str = "logistic",
        tau: float = 0.1,
        margin: float = 0.1,
        map_location: torch.device | str = "cpu",
        training_config: dict[str, Any] | None = None,
    ):
        payload = torch.load(path, map_location=map_location, weights_only=True)
        expected_metadata = {
            **self.checkpoint_metadata(objective=objective, tau=tau, margin=margin),
            "training_config": training_config or {},
        }
        if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
            raise ValueError("BOX checkpoint metadata/configuration mismatch")
        state = payload.get("model_state")
        if not isinstance(state, dict) or set(state) != set(self.state_dict()):
            raise ValueError("BOX checkpoint state keys mismatch")
        for key, current in self.state_dict().items():
            candidate = state[key]
            if (
                not isinstance(candidate, torch.Tensor)
                or candidate.shape != current.shape
                or candidate.dtype != current.dtype
            ):
                raise ValueError(f"invalid checkpoint tensor {key}")
            if not torch.isfinite(candidate).all():
                raise ValueError(f"non-finite checkpoint tensor {key}")
        try:
            decoded = self._decode_state(state)
            validate_endpoints(*decoded, domain=self.domain, epsilon=self.epsilon)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ValueError(f"invalid checkpoint geometry: {error}") from error
        snapshot = payload.get("geometry")
        if not isinstance(snapshot, dict) or set(snapshot) != {"lower", "upper"}:
            raise ValueError("BOX checkpoint geometry snapshot is missing")
        for saved, decoded_value in zip(
            (snapshot["lower"], snapshot["upper"]), decoded, strict=True
        ):
            if (
                not isinstance(saved, torch.Tensor)
                or saved.shape != decoded_value.shape
                or saved.dtype != torch.float32
                or not torch.equal(saved.cpu(), decoded_value.detach().cpu())
            ):
                raise ValueError("BOX checkpoint geometry snapshot disagrees with model state")
        # No mutation until metadata, tensor shapes/finiteness and decoded geometry validate.
        self.load_state_dict(state, strict=True)

    def _decode_state(self, state):
        from .box import decode_geometry

        return decode_geometry(
            state["geometry.raw_a"],
            state["geometry.raw_b"],
            domain=self.domain,
            epsilon=self.epsilon,
        )


def _masked_box_entity_mean(scores, targets, known, applicable, *, kind, tau, margin):
    """Compatibility wrapper for the shared known/applicable entity reduction."""
    empty_scores = scores.new_empty((0, scores.shape[1]))
    empty_targets = targets.new_empty((0, targets.shape[1]))
    empty_mask = torch.empty((0, scores.shape[1]), dtype=torch.bool, device=scores.device)
    loss, _ = masked_geometric_task_loss(
        scores,
        targets,
        known,
        applicable,
        empty_scores,
        empty_targets,
        empty_mask,
        empty_mask,
        [],
        kind=kind,
        tau=tau,
        margin=margin,
    )
    return loss


def masked_box_task_loss(
    pizza_scores,
    pizza_targets,
    pizza_known,
    pizza_applicable,
    component_scores,
    component_targets,
    component_known,
    component_applicable,
    component_parent,
    *,
    kind="logistic",
    tau=0.1,
    margin=0.1,
):
    """PR2 known/applicable reduction: entity, components per parent, then tasks equally."""
    return masked_geometric_task_loss(
        pizza_scores,
        pizza_targets,
        pizza_known,
        pizza_applicable,
        component_scores,
        component_targets,
        component_known,
        component_applicable,
        component_parent,
        kind=kind,
        tau=tau,
        margin=margin,
    )


def _box_loss(
    model,
    records,
    dataset,
    device,
    *,
    kind,
    tau,
    margin,
    point_chunk_size=None,
    include_predictions=True,
):
    x_p, y_p, m_p, x_c, y_c, m_c, parents = _pack(records, dataset, device)
    pizza_supervision = set(dataset.pizza_supervision_class_ids)
    component_supervision = set(dataset.component_supervision_class_ids)
    a_p = torch.tensor(
        [i in pizza_supervision for i in dataset.canonical_class_ids],
        dtype=torch.bool,
        device=device,
    ).expand_as(y_p)
    a_c = torch.tensor(
        [i in component_supervision for i in dataset.canonical_class_ids],
        dtype=torch.bool,
        device=device,
    ).expand_as(y_c)
    if point_chunk_size is not None and (
        isinstance(point_chunk_size, bool)
        or not isinstance(point_chunk_size, int)
        or point_chunk_size < 1
    ):
        raise ValueError("point_chunk_size must be a positive integer")
    points_p = model.encode(x_p, "pizza")
    points_c = model.encode(x_c, "component")
    lower, upper = model.endpoints()

    def score_points(points):
        size = max(1, len(points)) if point_chunk_size is None else point_chunk_size
        return (
            torch.cat(
                [
                    box_clearance(
                        points[start : start + size],
                        lower,
                        upper,
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )
                    for start in range(0, len(points), size)
                ],
                dim=0,
            )
            if len(points)
            else points.new_empty((0, len(model.class_ids)))
        )

    def exact_membership(points):
        if not include_predictions:
            return None
        size = max(1, len(points)) if point_chunk_size is None else point_chunk_size
        return (
            torch.cat(
                [
                    box_membership(
                        points[start : start + size],
                        lower,
                        upper,
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )
                    for start in range(0, len(points), size)
                ],
                dim=0,
            )
            if len(points)
            else torch.empty((0, len(model.class_ids)), dtype=torch.bool, device=points.device)
        )

    ps, cs = score_points(points_p), score_points(points_c)
    total, tasks = masked_box_task_loss(
        ps, y_p, m_p, a_p, cs, y_c, m_c, a_c, parents, kind=kind, tau=tau, margin=margin
    )
    # Exact Boolean containment remains authoritative and is evaluated independently.
    pm, cm = exact_membership(points_p), exact_membership(points_c)
    return total, tasks, (pm, cm, y_p, m_p, y_c, m_c, a_p, a_c, parents)


def _geometry_pair_sha256(lower, upper):
    digest = hashlib.sha256(b"ontology-embeddings-f12-endpoints-v1\\0")
    for label, tensor in ((b"lower", lower), (b"upper", upper)):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(label + b"\\0" + str(value.dtype).encode() + b"\\0")
        digest.update(str(tuple(value.shape)).encode() + b"\\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _geometry_diagnostics(model, records, dataset, device):
    """Compact deterministic geometry activity/clearance summaries (training records only)."""
    lower, upper = model.endpoints()
    centers = (lower + upper) / 2
    widths = upper - lower
    x_p, y_p, m_p, x_c, y_c, m_c, _ = _pack(records, dataset, device)
    points = torch.cat((model.encode(x_p, "pizza"), model.encode(x_c, "component")), dim=0)
    targets = torch.cat((y_p, y_c), dim=0)
    pizza_applicable = torch.tensor(
        [i in set(dataset.pizza_supervision_class_ids) for i in dataset.canonical_class_ids],
        device=device,
        dtype=torch.bool,
    ).expand_as(y_p)
    component_applicable = torch.tensor(
        [i in set(dataset.component_supervision_class_ids) for i in dataset.canonical_class_ids],
        device=device,
        dtype=torch.bool,
    ).expand_as(y_c)
    mask = torch.cat((m_p.bool() & pizza_applicable, m_c.bool() & component_applicable), dim=0)
    if points.numel():
        inside = (
            (points[:, None, None, :] >= lower[None]) & (points[:, None, None, :] <= upper[None])
        ).all(dim=-1)
        occupancy = inside.float().mean(dim=0)
        point_scale = points.std(dim=0, unbiased=False)
        faces = torch.stack(
            (points[:, None, None, :] - lower[None], upper[None] - points[:, None, None, :]), dim=-1
        )
        face_min = faces.amin(dim=(-1, -2))
        best_clearance = face_min.amax(dim=-1, keepdim=True)
        winning_box_mask = face_min == best_clearance
        box_ties = winning_box_mask.sum(dim=-1)
        winning_boxes = (winning_box_mask.to(torch.float32) / box_ties.unsqueeze(-1)).sum(
            dim=(0, 1)
        )
        limiting_face_mask = faces == face_min[..., None, None]
        face_tie_count = limiting_face_mask.sum(dim=(-1, -2), keepdim=True)
        winning_box_count = box_ties.clamp_min(1).unsqueeze(-1)
        winning_box_weight = winning_box_mask.to(torch.float32) / winning_box_count
        winning_face_mask = limiting_face_mask & winning_box_mask[..., None, None]
        winning_face = (
            winning_face_mask.to(torch.float32)
            * winning_box_weight[..., None, None]
            / face_tie_count.clamp_min(1)
        ).sum(dim=(0, 1, 2))
        winning_face_tied_box_count = (
            (face_tie_count.squeeze(-1).squeeze(-1) > 1) & winning_box_mask
        ).sum()
    else:
        occupancy = torch.zeros_like(widths[..., 0])
        point_scale = points.new_zeros((model.latent_dim,))
        winning_boxes = torch.zeros((model.box_count,), dtype=torch.float32, device=device)
        box_ties = torch.empty((0, len(model.class_ids)), dtype=torch.int64, device=device)
        winning_face_tied_box_count = torch.tensor(0, dtype=torch.int64, device=device)
        winning_face = torch.zeros((model.latent_dim, 2), dtype=torch.float32, device=device)
    clearances = box_clearance(points, lower, upper, domain=model.domain, epsilon=model.epsilon)
    pos = mask & targets.bool()
    neg = mask & ~targets.bool()
    pizza_mask = m_p.bool() & pizza_applicable
    component_mask = m_c.bool() & component_applicable
    pizza_clearances = clearances[: len(y_p)]
    component_clearances = clearances[len(y_p) :]

    def distribution(values):
        if not values.numel():
            return {"count": 0, "min": None, "mean": None, "max": None}
        return {
            "count": int(values.numel()),
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }

    raw_a, raw_b = model.geometry.raw_a.detach(), model.geometry.raw_b.detach()
    saturation_tolerance = 1e-6
    if model.domain == "bounded":
        sigmoid_width = torch.sigmoid(raw_a)
        tanh_center = torch.tanh(raw_b).abs()
        transform_saturation = {
            "threshold": saturation_tolerance,
            "width_sigmoid_saturated_fraction": float(
                (
                    (sigmoid_width <= saturation_tolerance)
                    | (sigmoid_width >= 1 - saturation_tolerance)
                )
                .float()
                .mean()
            ),
            "center_tanh_saturated_fraction": float(
                (tanh_center >= 1 - saturation_tolerance).float().mean()
            ),
            "encoder_tanh_saturation_fraction": float(
                (points.abs() >= 1 - saturation_tolerance).float().mean()
            )
            if points.numel()
            else 0.0,
            "width_softplus_derivative_near_zero_fraction": None,
            "center_identity_saturation_fraction": None,
        }
    else:
        transform_saturation = {
            "threshold": saturation_tolerance,
            "width_sigmoid_saturated_fraction": None,
            "center_tanh_saturated_fraction": None,
            "encoder_tanh_saturation_fraction": None,
            "width_softplus_derivative_near_zero_fraction": float(
                (torch.sigmoid(raw_a) <= saturation_tolerance).float().mean()
            ),
            "center_identity_saturation_fraction": None,
        }
    return {
        "width_min": float(widths.min()),
        "width_max": float(widths.max()),
        "center_abs_max": float(centers.abs().max()),
        "box_point_occupancy_fraction": occupancy.detach().cpu().tolist(),
        "training_point_coordinate_std_mean": float(point_scale.mean()),
        "clearance_scope": "training pizza/component records; applicable known targets",
        "positive_clearance": distribution(clearances[pos]),
        "negative_clearance": distribution(clearances[neg]),
        "entity_clearances": {
            "pizza": {
                "positive": distribution(pizza_clearances[pizza_mask & y_p.bool()]),
                "negative": distribution(pizza_clearances[pizza_mask & ~y_p.bool()]),
            },
            "component": {
                "positive": distribution(component_clearances[component_mask & y_c.bool()]),
                "negative": distribution(component_clearances[component_mask & ~y_c.bool()]),
            },
        },
        "target_clearances": {
            str(class_id): {
                "positive": distribution(clearances[:, index][pos[:, index]]),
                "negative": distribution(clearances[:, index][neg[:, index]]),
            }
            for index, class_id in enumerate(model.class_ids)
        },
        "winning_box_scope": (
            "training pizza/component points by target; tied winners split equally"
        ),
        "winning_box_counts": winning_boxes.detach().cpu().tolist(),
        "winning_box_exact_tie_count": int((box_ties > 1).sum()),
        "winning_face_scope": (
            "split across tied winning boxes, then each winner's tied limiting faces"
        ),
        "winning_face_exact_tie_count": int(winning_face_tied_box_count),
        "winning_face_exact_tie_scope": "count of tied face minima per winning box and target",
        "winning_face_counts": winning_face.detach().cpu().tolist(),
        "inactive_box_fraction": float((winning_boxes == 0).float().mean()),
        "width_floor_fraction": float((widths <= model.epsilon + 1e-6).float().mean()),
        "bounded_max_width_fraction": None
        if model.domain != "bounded"
        else float((widths >= 2.0 - 1e-6).float().mean()),
        "transform_saturation": transform_saturation,
        "raw_geometry_abs_max": {
            "width": float(raw_a.abs().max()),
            "center": float(raw_b.abs().max()),
        },
    }


_invalid_optimizer_groups = invalid_optimizer_groups


def train_box(
    dataset: PreparedDataset,
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord] = (),
    *,
    artifact_root="artifacts/runs",
    purpose="smoke",
    invocation="python ontology_embeddings.box_training.train_box",
    seed=7,
    epochs=1,
    batch_size=None,
    learning_rate=0.02,
    latent_dim=16,
    boxes=1,
    domain="bounded",
    epsilon=1e-4,
    objective: LossKind = "logistic",
    tau=0.1,
    margin=0.1,
    device="cpu",
    point_chunk_size=None,
    run_record: RunRecord | None = None,
    shared_adapter_initial_state: dict[str, torch.Tensor] | None = None,
    initial_box_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
    initial_box_class_ids: Sequence[int] | None = None,
    paired_initialization_provenance: dict[str, Any] | None = None,
    dataset_provenance: dict[str, Any] | None = None,
    capture_temporal: bool = False,
    portable_artifact_paths: bool = False,
):
    """Train BOX metadata model, persisting only validated states and recording all outcomes."""
    config = {
        "model": "box-union-two-adapters",
        "domain": domain,
        "latent_dim": latent_dim,
        "boxes": boxes,
        "epsilon": epsilon,
        "objective": objective,
        "tau": tau,
        "margin": margin,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "clip_norm": 1.0,
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "seed": seed,
    }
    if capture_temporal:
        config["temporal_capture"] = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "policy": "initial+every-first-epoch-update+later-epoch-boundaries-v1",
        }
    rec = run_record or RunRecord(
        artifact_root,
        purpose=purpose,
        invocation=invocation,
        config=config,
        dataset={},
        seed=seed,
        issue="13",
    )
    try:
        if dataset_provenance is not None and not isinstance(dataset_provenance, dict):
            raise ValueError("dataset provenance must be a mapping")
        dataset_identity = {
            "dataset_id": dataset.dataset_id,
            "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "class_ids": list(dataset.canonical_class_ids),
            "class_names": list(dataset.class_names),
            "class_iris": list(dataset.class_iris),
            "fixed_positive_class_ids": list(getattr(dataset, "fixed_positive_class_ids", ())),
            "unsupported_class_ids": list(getattr(dataset, "unsupported_class_ids", ())),
            "train_sample_ids": [x.sample_id for x in train_records],
            "validation_sample_ids": [x.sample_id for x in validation_records],
        }
        if hasattr(dataset, "manifest_id"):
            dataset_identity["manifest_id"] = dataset.manifest_id
        if dataset_provenance is not None:
            for key, value in dataset_provenance.items():
                if key in dataset_identity and dataset_identity[key] != value:
                    raise ValueError(
                        f"dataset provenance conflicts with verified identity for {key}"
                    )
            dataset_identity.update(dataset_provenance)
        environment = {
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "python": __import__("sys").version,
        }
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            environment["device_name"] = torch.cuda.get_device_name(torch.device(device))
        rec.update(
            config=config,
            feature_schema=FEATURE_SCHEMA,
            environment=environment,
            dataset=dataset_identity,
        )
        if (
            not train_records
            or isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or epochs < 1
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("training requires records, positive epochs and finite learning rate")
        if objective not in ("logistic", "hinge"):
            raise ValueError("objective must be logistic or hinge")
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(margin) or margin < 0:
            raise ValueError("tau must be positive and margin nonnegative, both finite")
        if any(r.split != "train" for r in train_records) or any(
            r.split != "validation" for r in validation_records
        ):
            raise ValueError("BOX training accepts train records and validation records only")
        if {r.sample_id for r in train_records} & {r.sample_id for r in validation_records}:
            raise ValueError("overlapping train and validation samples")
        torch.manual_seed(seed)
        random.seed(seed)
        dev = torch.device(device)
        model = BoxMultiLabelModel(
            len(PIZZA_FEATURE_NAMES),
            len(COMPONENT_FEATURE_NAMES),
            dataset.canonical_class_ids,
            dataset.class_manifest_sha256,
            latent_dim=latent_dim,
            boxes=boxes,
            domain=domain,
            epsilon=epsilon,
            seed=seed,
            class_names=dataset.class_names,
            class_iris=dataset.class_iris,
            fixed_positive_class_ids=getattr(dataset, "fixed_positive_class_ids", ()),
            device=dev,
        ).to(dev)
        if not isinstance(capture_temporal, bool):
            raise ValueError("capture_temporal must be a boolean")
        if not isinstance(portable_artifact_paths, bool):
            raise ValueError("portable_artifact_paths must be a boolean")
        if paired_initialization_provenance is not None and not isinstance(
            paired_initialization_provenance, dict
        ):
            raise ValueError("paired initialization provenance must be a mapping")
        initial_identity = {"adapter_sha256": {}, "box_bounds": None}
        if paired_initialization_provenance is not None:
            initial_identity["provenance"] = paired_initialization_provenance
        if shared_adapter_initial_state is not None:
            expected = {
                "pizza_adapter.weight": model.pizza_adapter.weight,
                "pizza_adapter.bias": model.pizza_adapter.bias,
                "component_adapter.weight": model.component_adapter.weight,
                "component_adapter.bias": model.component_adapter.bias,
            }
            if not isinstance(shared_adapter_initial_state, dict) or set(
                shared_adapter_initial_state
            ) != set(expected):
                raise ValueError("shared adapter state must contain exact adapter weight/bias keys")
            validated = {}
            for key, current in expected.items():
                candidate = shared_adapter_initial_state[key]
                if (
                    not isinstance(candidate, torch.Tensor)
                    or candidate.shape != current.shape
                    or candidate.dtype != current.dtype
                    or candidate.device != dev
                    or not torch.isfinite(candidate).all()
                ):
                    raise ValueError(f"invalid shared adapter tensor {key}")
                validated[key] = candidate.detach().clone()
            initial_identity["adapter_sha256"] = {
                key: hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest()
                for key, value in validated.items()
            }
        elif initial_box_bounds is not None:
            raise ValueError("shared adapter state is required with paired box bounds")
        if initial_box_bounds is not None:
            if initial_box_class_ids is None or tuple(initial_box_class_ids) != tuple(
                dataset.canonical_class_ids
            ):
                raise ValueError(
                    "initial box class order must exactly match dataset canonical class IDs"
                )
            if not isinstance(initial_box_bounds, (tuple, list)) or len(initial_box_bounds) != 2:
                raise ValueError("initial_box_bounds must contain (lower, upper)")
            lower, upper = initial_box_bounds
            try:
                validate_endpoints(lower, upper, domain=domain, epsilon=epsilon)
                if lower.shape != (len(dataset.canonical_class_ids), boxes, latent_dim):
                    raise ValueError(
                        "initial box bounds shape does not match class/box/latent order"
                    )
                if lower.device != dev or upper.device != dev:
                    raise ValueError("initial box bounds must be on the training device")
                from .box import decode_geometry, inverse_geometry

                raw_a, raw_b = inverse_geometry(lower, upper, domain=domain, epsilon=epsilon)
                if raw_a.device != dev:
                    raise ValueError("initial box bounds must be on the training device")
                decoded_lower, decoded_upper = decode_geometry(
                    raw_a, raw_b, domain=domain, epsilon=epsilon
                )
                reconstruction_error = max(
                    float((decoded_lower - lower).abs().max()),
                    float((decoded_upper - upper).abs().max()),
                )
                if reconstruction_error > INITIAL_BOUNDS_RECONSTRUCTION_TOLERANCE:
                    raise ValueError(
                        "initial physical box reconstruction error "
                        f"{reconstruction_error:.9g} exceeds absolute tolerance "
                        f"{INITIAL_BOUNDS_RECONSTRUCTION_TOLERANCE:.9g}"
                    )
            except (ValueError, TypeError, RuntimeError) as error:
                raise ValueError(f"invalid initial physical box bounds: {error}") from error
            with torch.no_grad():
                model.geometry.raw_a.copy_(raw_a)
                model.geometry.raw_b.copy_(raw_b)
            initial_identity["box_bounds"] = {
                "class_ids": list(dataset.canonical_class_ids),
                "requested_lower": lower.detach().cpu().tolist(),
                "requested_upper": upper.detach().cpu().tolist(),
                "decoded_actual_lower": decoded_lower.detach().cpu().tolist(),
                "decoded_actual_upper": decoded_upper.detach().cpu().tolist(),
                "max_absolute_reconstruction_error": reconstruction_error,
                "reconstruction_tolerance_abs": INITIAL_BOUNDS_RECONSTRUCTION_TOLERANCE,
                "requested_sha256": _geometry_pair_sha256(lower, upper),
                "decoded_actual_sha256": _geometry_pair_sha256(decoded_lower, decoded_upper),
            }
        if shared_adapter_initial_state is not None:
            with torch.no_grad():
                for key, target in expected.items():
                    target.copy_(validated[key])
        config["paired_initialization"] = initial_identity
        rec.update(config=config)
        optimizer = torch.optim.Adam(
            [
                {"params": model.pizza_adapter.parameters(), "name": "pizza_adapter"},
                {"params": model.component_adapter.parameters(), "name": "component_adapter"},
                {"params": model.geometry.parameters(), "name": "geometry"},
            ],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
        )
        chunk = len(train_records) if batch_size is None else batch_size
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("batch_size must be a positive integer")
        config["batch_size"] = chunk
        rec.update(config=config)
        checkpoints = rec.path / "checkpoints"
        checkpoints.mkdir()

        def save(kind, epoch, val_loss=None):
            model.save_checkpoint(
                checkpoints / f"{kind}.pt",
                objective=objective,
                tau=tau,
                margin=margin,
                epoch=epoch,
                validation_loss=val_loss,
                training_config=config,
            )

        save("initial", 0)
        save("latest", 0)
        temporal_capture = None
        batches_per_epoch = math.ceil(len(train_records) / chunk)
        if capture_temporal:
            temporal_capture = TemporalCapture(
                rec.path, rec.run_id, epochs=epochs, batches_per_epoch=batches_per_epoch
            )
            temporal_capture.capture(
                checkpoints / "initial.pt",
                accepted_updates=0,
                epoch=0,
                batch=0,
                epoch_boundary=False,
                model_state_sha256=state_dict_sha256(model.state_dict()),
            )
        if validation_records:
            with torch.no_grad():
                initial_validation_loss, _, _ = _box_loss(
                    model,
                    validation_records,
                    dataset,
                    dev,
                    kind=objective,
                    tau=tau,
                    margin=margin,
                    point_chunk_size=point_chunk_size,
                    include_predictions=False,
                )
            if initial_validation_loss is None:
                raise ValueError("validation has no eligible known applicable labels")

        def physical_snapshot():
            lo, hi = model.endpoints()
            return {"lower": lo.detach().cpu().tolist(), "upper": hi.detach().cpu().tolist()}

        initial_geometry_snapshot = physical_snapshot()
        best_geometry_snapshot = None
        history = []
        best = float("inf")
        best_predictions = None
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        accepted_updates = 0
        best_micro_f1 = None
        best_micro_f1_epoch = None
        for epoch in range(epochs):
            order = list(train_records)
            random.shuffle(order)
            gradnorms = []
            clipped_gradnorms = []
            encoder_pre_norms, geometry_pre_norms = [], []
            encoder_post_norms, geometry_post_norms = [], []
            epoch_start = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            for start in range(0, len(order), chunk):
                # Release prior gradient buffers before constructing the next forward graph.
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = _box_loss(
                    model,
                    order[start : start + chunk],
                    dataset,
                    dev,
                    kind=objective,
                    tau=tau,
                    margin=margin,
                    point_chunk_size=point_chunk_size,
                    include_predictions=False,
                )
                stats = checked_optimizer_update(
                    model,
                    optimizer,
                    loss,
                    update_number=accepted_updates + 1,
                    gradient_groups={
                        "encoder": [
                            *model.pizza_adapter.parameters(),
                            *model.component_adapter.parameters(),
                        ],
                        "geometry": list(model.geometry.parameters()),
                    },
                    max_grad_norm=1.0,
                    validate_model=lambda: validate_endpoints(
                        *model.endpoints(), domain=domain, epsilon=epsilon
                    ),
                    model_name="BOX",
                )
                gradnorms.append(stats["global_pre_clip_norm"])
                clipped_gradnorms.append(stats["global_post_clip_norm"])
                encoder_pre_norms.append(stats["group_pre_clip_norms"]["encoder"])
                geometry_pre_norms.append(stats["group_pre_clip_norms"]["geometry"])
                encoder_post_norms.append(stats["group_post_clip_norms"]["encoder"])
                geometry_post_norms.append(stats["group_post_clip_norms"]["geometry"])
                save("latest", epoch + 1)
                accepted_updates += 1
                batch_number = start // chunk + 1
                last_batch = start + chunk >= len(order)
                if temporal_capture is not None and (epoch == 0 or last_batch):
                    temporal_capture.capture(
                        checkpoints / "latest.pt",
                        accepted_updates=accepted_updates,
                        epoch=epoch + 1,
                        batch=batch_number,
                        epoch_boundary=last_batch,
                        model_state_sha256=state_dict_sha256(model.state_dict()),
                    )
            with torch.no_grad():
                train_loss, train_tasks, train_batch = _box_loss(
                    model,
                    train_records,
                    dataset,
                    dev,
                    kind=objective,
                    tau=tau,
                    margin=margin,
                    point_chunk_size=point_chunk_size,
                )
                if validation_records:
                    val_loss, val_tasks, val_batch = _box_loss(
                        model,
                        validation_records,
                        dataset,
                        dev,
                        kind=objective,
                        tau=tau,
                        margin=margin,
                        point_chunk_size=point_chunk_size,
                    )
                    diagnostics = {
                        "train": _task_metrics(train_batch, dataset.class_names),
                        "validation": _task_metrics(val_batch, dataset.class_names),
                        "feature_signatures": _signature_diagnostics(
                            train_records, validation_records, val_batch, dataset.class_names
                        ),
                    }
                else:
                    val_loss, val_tasks, diagnostics = None, {"pizza": None, "component": None}, {}
            task_micro_f1 = [
                diagnostics.get("validation", {}).get(task, {}).get("micro", {}).get("f1")
                for task in ("pizza", "component")
            ]
            available_micro_f1 = [value for value in task_micro_f1 if value is not None]
            mean_micro_f1 = (
                sum(available_micro_f1) / len(available_micro_f1) if available_micro_f1 else None
            )
            if mean_micro_f1 is not None and (
                best_micro_f1 is None or mean_micro_f1 > best_micro_f1
            ):
                best_micro_f1 = mean_micro_f1
                best_micro_f1_epoch = epoch + 1
            row = {
                "epoch": epoch + 1,
                "train_loss": None if train_loss is None else float(train_loss),
                "validation_loss": None if val_loss is None else float(val_loss),
                "train_pizza_loss": None
                if train_tasks["pizza"] is None
                else float(train_tasks["pizza"]),
                "train_component_loss": None
                if train_tasks["component"] is None
                else float(train_tasks["component"]),
                "validation_pizza_loss": None
                if val_tasks["pizza"] is None
                else float(val_tasks["pizza"]),
                "validation_component_loss": None
                if val_tasks["component"] is None
                else float(val_tasks["component"]),
                "validation_mean_available_task_micro_f1": mean_micro_f1,
                "gradient_norm_pre_clip_mean": sum(gradnorms) / len(gradnorms),
                "gradient_norm_post_clip_mean": sum(clipped_gradnorms) / len(clipped_gradnorms),
                "encoder_gradient_norm_pre_clip_mean": sum(encoder_pre_norms)
                / len(encoder_pre_norms),
                "geometry_gradient_norm_pre_clip_mean": sum(geometry_pre_norms)
                / len(geometry_pre_norms),
                "encoder_gradient_norm_post_clip_mean": sum(encoder_post_norms)
                / len(encoder_post_norms),
                "geometry_gradient_norm_post_clip_mean": sum(geometry_post_norms)
                / len(geometry_post_norms),
                "geometry_zero_pre_clip_gradient_fraction": sum(
                    value == 0 for value in geometry_pre_norms
                )
                / len(geometry_pre_norms),
                "parameter_updates": {
                    group: any(
                        not torch.equal(epoch_start[name], model.state_dict()[name])
                        for name in epoch_start
                        if name.startswith(prefix)
                    )
                    for group, prefix in (
                        ("pizza_adapter", "pizza_adapter."),
                        ("component_adapter", "component_adapter."),
                        ("geometry", "geometry."),
                    )
                },
                "diagnostics": diagnostics,
                "geometry": _geometry_diagnostics(model, train_records, dataset, dev),
            }
            history.append(row)
            rec.metric(row)
            if val_loss is not None and float(val_loss) < best:
                best = float(val_loss)
                best_predictions = (val_batch[0] > 0, val_batch[1] > 0)
                best_geometry_snapshot = physical_snapshot()
                save("best", epoch + 1, best)
            rec.update(last_epoch=epoch + 1, latest_metrics=row)
        temporal_capture_summary = None if temporal_capture is None else temporal_capture.finalize()
        final_geometry_snapshot = physical_snapshot()
        if validation_records and (checkpoints / "best.pt").exists():
            model.load_checkpoint(
                checkpoints / "best.pt",
                objective=objective,
                tau=tau,
                margin=margin,
                map_location=dev,
                training_config=config,
            )
            with torch.no_grad():
                _, _, loaded = _box_loss(
                    model, validation_records, dataset, dev, kind=objective, tau=tau, margin=margin
                )
            equal = all(
                torch.equal(a, b)
                for a, b in zip(best_predictions, (loaded[0] > 0, loaded[1] > 0), strict=True)
            )
        else:
            equal = None
        updated = {
            group: any(
                not torch.equal(initial[name], model.state_dict()[name])
                for name in initial
                if name.startswith(prefix)
            )
            for group, prefix in (
                ("pizza_adapter", "pizza_adapter."),
                ("component_adapter", "component_adapter."),
                ("geometry", "geometry."),
            )
        }
        result = {
            "run_id": rec.run_id,
            "run_path": "." if portable_artifact_paths else str(rec.path),
            "status": "completed",
            "history": history,
            "accepted_updates": accepted_updates,
            "parameter_updates": updated,
            "best_checkpoint_reload_predictions_equivalent": equal,
            "best_checkpoint_selection": "minimum combined native validation loss; earliest tie",
            "best_mean_available_task_micro_f1_epoch": best_micro_f1_epoch,
            "best_mean_available_task_micro_f1": best_micro_f1,
            "geometry_snapshots": {
                "initial": initial_geometry_snapshot,
                "best": best_geometry_snapshot,
                "final": final_geometry_snapshot,
            },
            "checkpoint_paths": {
                x: f"checkpoints/{x}.pt"
                if portable_artifact_paths
                else str(checkpoints / f"{x}.pt")
                for x in ("initial", "latest", "best")
                if (checkpoints / f"{x}.pt").exists()
            },
        }
        if temporal_capture_summary is not None:
            result["temporal_capture"] = temporal_capture_summary
        _atomic_json(rec.path / "summary.json", result)
        rec.update(
            status="completed",
            finished_at=__import__("datetime").datetime.now().astimezone().isoformat(),
            summary=result,
        )
        return result
    except KeyboardInterrupt as error:
        rec.interrupt(error)
        raise
    except BaseException as error:
        rec.fail(error)
        raise
