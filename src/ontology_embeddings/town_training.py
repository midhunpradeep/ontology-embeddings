"""Integrated trainable TOWN classifiers and fail-closed metadata training.

TOWN owns its adapters, physical geometry snapshots and family-specific diagnostics;
masked supervision reductions and optimizer update checks are shared with BOX.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn

from .box import initialize_geometry
from .box_training import BoxMultiLabelModel
from .dataset.image_data import PreparedDataset
from .dataset.schema import DatasetRecord
from .evaluation import class_frequency_baseline, membership_metrics
from .features import COMPONENT_FEATURE_NAMES, FEATURE_SCHEMA, PIZZA_FEATURE_NAMES
from .run_store import RunRecord, _atomic_json
from .shared_training import checked_optimizer_update, masked_geometric_task_loss
from .town_geometry import (
    TownGeometry,
    encode_endpoints,
    initialize_volume_first_windows,
    town_clearance,
    town_membership,
    validate_town_endpoints,
)
from .training import _atomic_torch_save, _pack

LossKind = Literal["logistic", "hinge"]
INITIAL_GEOMETRY_RECONSTRUCTION_TOLERANCE = 2e-5
TOWN_BOX_SCORE_PARITY_ATOL = 1e-6
TOWN_BOX_GRADIENT_PARITY_ATOL = 1e-7
TOWN_BOX_UPDATE_PARITY_ATOL = 2e-6


def _sha256_tensor(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode() + b"\0")
    digest.update(str(tuple(cpu.shape)).encode() + b"\0")
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _json_identity(value: Any) -> Any:
    """Restrict checkpoint metadata to portable JSON-like values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("initializer metadata floats must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_identity(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_identity(item) for key, item in value.items()}
    raise TypeError("initializer metadata must contain only JSON-compatible values")


class TownMultiLabelModel(nn.Module):
    """Separate pizza/component adapters over a shared subtractive town classifier."""

    family = "town-subtractive-classifier"
    checkpoint_version = "town-checkpoint-v1"

    def __init__(
        self,
        pizza_features: int,
        component_features: int,
        class_ids: Sequence[int],
        vocabulary_sha256: str,
        *,
        latent_dim: int = 16,
        houses: int = 1,
        windows: int = 1,
        domain: str = "bounded",
        epsilon: float = 1e-4,
        delta: float = 1e-4,
        seed: int = 7,
        class_names: Sequence[str] | None = None,
        class_iris: Sequence[str] | None = None,
        fixed_positive_class_ids: Sequence[int] = (),
        device: torch.device | str = "cpu",
        initial_frame_bounds: tuple[torch.Tensor, torch.Tensor] | None = None,
        initial_adapter_state: dict[str, torch.Tensor] | None = None,
        initializer_source: dict[str, Any] | None = None,
        window_retry_limit: int = 64,
    ) -> None:
        super().__init__()
        ids = tuple(class_ids)
        if not ids or any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise ValueError("class_ids must be a nonempty ordered sequence of integers")
        if len(set(ids)) != len(ids):
            raise ValueError("class_ids must be unique")
        fixed = tuple(fixed_positive_class_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in fixed):
            raise ValueError("fixed-positive IDs must be integers")
        if len(set(fixed)) != len(fixed) or set(ids) & set(fixed):
            raise ValueError("fixed-positive IDs must be unique and separate from trainable IDs")
        if not isinstance(vocabulary_sha256, str) or not vocabulary_sha256:
            raise ValueError("vocabulary identity is required")
        if domain not in ("bounded", "unbounded"):
            raise ValueError("domain must be bounded or unbounded")
        for name, value in (("pizza_features", pizza_features), ("component_features", component_features),
                            ("latent_dim", latent_dim), ("houses", houses)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(windows, bool) or not isinstance(windows, int) or windows < 0:
            raise ValueError("windows must be a nonnegative integer")
        if isinstance(window_retry_limit, bool) or not isinstance(window_retry_limit, int) or window_retry_limit < 1:
            raise ValueError("window_retry_limit must be a positive integer")
        if (isinstance(epsilon, bool) or isinstance(delta, bool)
                or not math.isfinite(epsilon) or epsilon <= 0
                or not math.isfinite(delta) or delta <= 0):
            raise ValueError("epsilon and delta must be finite and positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if domain == "bounded" and epsilon + (2 * delta if windows else 0) >= 2:
            raise ValueError("frame width floor is not representable in the bounded domain")

        self.class_ids, self.fixed_positive_class_ids = ids, fixed
        self.vocabulary_sha256 = vocabulary_sha256
        self.class_names = tuple(class_names or (str(value) for value in ids))
        self.class_iris = tuple(class_iris or ("" for _ in ids))
        if len(self.class_names) != len(ids) or len(self.class_iris) != len(ids):
            raise ValueError("class names and IRIs must follow class_ids exactly")
        if not all(isinstance(value, str) for value in (*self.class_names, *self.class_iris)):
            raise ValueError("class names and IRIs must be strings")
        self.pizza_adapter = nn.Linear(pizza_features, latent_dim)
        self.component_adapter = nn.Linear(component_features, latent_dim)
        self.domain, self.epsilon, self.delta = domain, float(epsilon), float(delta)
        self.latent_dim, self.house_count, self.window_count = latent_dim, houses, windows
        self.seed = seed
        dev = torch.device(device)
        reserve = 2.0 * delta if windows else 0.0
        if initial_frame_bounds is None:
            frame_a, frame_b = initialize_geometry(
                len(ids), houses, latent_dim, domain=domain, seed=seed,
                epsilon=epsilon + reserve, device=dev,
            )
            frame_initialization = {"policy": "dimension-aware-box-v1", "seed": seed}
        else:
            if not isinstance(initial_frame_bounds, (tuple, list)) or len(initial_frame_bounds) != 2:
                raise ValueError("initial_frame_bounds must contain (lower, upper)")
            requested_lower, requested_upper = initial_frame_bounds
            requested_lower, requested_upper = requested_lower.to(dev), requested_upper.to(dev)
            validate_town_endpoints(
                requested_lower, requested_upper, domain=domain, epsilon=epsilon, reserve=reserve
            )
            expected_shape = (len(ids), houses, latent_dim)
            if requested_lower.shape != expected_shape:
                raise ValueError("initial frame bounds shape does not match class/house/latent order")
            frame_a, frame_b = encode_endpoints(
                requested_lower, requested_upper, domain=domain, epsilon=epsilon, reserve=reserve
            )
            decoded = self._decode_frames(frame_a, frame_b)
            error = max(
                float((decoded[0] - requested_lower).abs().max()),
                float((decoded[1] - requested_upper).abs().max()),
            )
            if error > INITIAL_GEOMETRY_RECONSTRUCTION_TOLERANCE:
                raise ValueError(
                    f"initial frame reconstruction error {error:.9g} exceeds "
                    f"{INITIAL_GEOMETRY_RECONSTRUCTION_TOLERANCE:.9g}"
                )
            frame_initialization = {
                "policy": "validated-historical-box-physical-bounds-v1",
                "requested_lower": requested_lower.detach().cpu().tolist(),
                "requested_upper": requested_upper.detach().cpu().tolist(),
                "decoded_lower": decoded[0].detach().cpu().tolist(),
                "decoded_upper": decoded[1].detach().cpu().tolist(),
                "max_absolute_reconstruction_error": error,
                "reconstruction_tolerance_abs": INITIAL_GEOMETRY_RECONSTRUCTION_TOLERANCE,
            }
        frame_lower, frame_upper = self._decode_frames(frame_a, frame_b)
        if windows:
            initialized_windows = initialize_volume_first_windows(
                frame_lower, frame_upper, window_count=windows, domain=domain,
                seed=seed + 1, epsilon=epsilon, delta=delta,
                retry_limit=window_retry_limit,
            )
            window_a, window_b = initialized_windows.raw_a.to(dev), initialized_windows.raw_b.to(dev)
            window_initialization = {
                "policy": "volume-first-contained-independent-windows-v1",
                "seed": seed + 1,
                "retry_limit": window_retry_limit,
                "sampled_volume_fractions": initialized_windows.sampled_fractions.detach().cpu().tolist(),
                "actual_decoded_frame_exclusion_fractions": initialized_windows.actual_fractions.detach().cpu().tolist(),
                "target_lower": initialized_windows.target_lower.detach().cpu().tolist(),
                "target_upper": initialized_windows.target_upper.detach().cpu().tolist(),
                "actual_lower": initialized_windows.lower.detach().cpu().tolist(),
                "actual_upper": initialized_windows.upper.detach().cpu().tolist(),
            }
        else:
            window_a = window_b = None
            window_initialization = {"policy": "no-windows", "count": 0}
        self.geometry = TownGeometry(
            frame_a, frame_b, window_a, window_b, domain=domain, epsilon=epsilon, delta=delta
        )
        expected_adapters = {
            "pizza_adapter.weight": self.pizza_adapter.weight,
            "pizza_adapter.bias": self.pizza_adapter.bias,
            "component_adapter.weight": self.component_adapter.weight,
            "component_adapter.bias": self.component_adapter.bias,
        }
        adapter_hashes = {}
        if initial_adapter_state is not None:
            if not isinstance(initial_adapter_state, dict) or set(initial_adapter_state) != set(expected_adapters):
                raise ValueError("initial adapter state must have exact pizza/component weight and bias keys")
            validated = {}
            for key, current in expected_adapters.items():
                candidate = initial_adapter_state[key]
                if not isinstance(candidate, torch.Tensor) or candidate.shape != current.shape or candidate.dtype != current.dtype:
                    raise ValueError(f"invalid initial adapter tensor {key}")
                if not torch.isfinite(candidate).all():
                    raise ValueError(f"non-finite initial adapter tensor {key}")
                validated[key] = candidate.to(current.device).detach().clone()
                adapter_hashes[key] = _sha256_tensor(validated[key])
            with torch.no_grad():
                for key, current in expected_adapters.items():
                    current.copy_(validated[key])
        source = _json_identity(initializer_source or {})
        self.initializer_identity = {
            "version": "town-initializer-v1",
            "frame": frame_initialization,
            "window": window_initialization,
            "source": source,
            "adapter_sha256": adapter_hashes,
            "epsilon": self.epsilon,
            "delta": self.delta,
        }
        self.to(dev)

    def _decode_frames(self, raw_a, raw_b):
        from .town_geometry import decode_parameters

        return decode_parameters(
            raw_a, raw_b, domain=self.domain, epsilon=self.epsilon,
            reserve=2.0 * self.delta if self.window_count else 0.0,
        )

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

    def clearances(self, features: torch.Tensor, entity: str, *, point_chunk_size: int | None = None):
        points = self.encode(features, entity)
        if point_chunk_size is not None and (
            isinstance(point_chunk_size, bool)
            or not isinstance(point_chunk_size, int)
            or point_chunk_size < 1
        ):
            raise ValueError("point_chunk_size must be a positive integer")
        endpoints = self.endpoints()
        size = max(1, len(points)) if point_chunk_size is None else point_chunk_size
        return (
            torch.cat([
                town_clearance(
                    points[start:start + size], *endpoints, domain=self.domain,
                    epsilon=self.epsilon, delta=self.delta,
                )
                for start in range(0, len(points), size)
            ], dim=0)
            if len(points) else points.new_empty((0, len(self.class_ids)))
        )

    def predict(self, features: torch.Tensor, entity: str, *, point_chunk_size: int | None = None):
        points = self.encode(features, entity)
        if point_chunk_size is None:
            size = max(1, len(points))
        elif isinstance(point_chunk_size, bool) or not isinstance(point_chunk_size, int) or point_chunk_size < 1:
            raise ValueError("point_chunk_size must be a positive integer")
        else:
            size = point_chunk_size
        endpoints = self.endpoints()
        return (
            torch.cat([
                town_membership(
                    points[start:start + size], *endpoints, domain=self.domain,
                    epsilon=self.epsilon, delta=self.delta,
                )
                for start in range(0, len(points), size)
            ], dim=0)
            if len(points) else torch.empty((0, len(self.class_ids)), dtype=torch.bool, device=points.device)
        )

    def domain_thing_membership(self, points: torch.Tensor) -> torch.Tensor:
        if not isinstance(points, torch.Tensor) or points.ndim != 2 or points.shape[1] != self.latent_dim:
            raise ValueError("points must have shape [N, latent_dim]")
        if points.dtype != torch.float32 or not torch.isfinite(points).all():
            raise ValueError("points must be finite float32")
        if self.domain == "bounded":
            return ((points >= -1) & (points <= 1)).all(dim=1)
        return torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

    def optimizer_parameter_groups(self) -> list[dict[str, Any]]:
        groups = [
            {"params": list(self.pizza_adapter.parameters()), "name": "pizza_adapter"},
            {"params": list(self.component_adapter.parameters()), "name": "component_adapter"},
            {"params": [self.geometry.raw_frame_a, self.geometry.raw_frame_b], "name": "frame_geometry"},
        ]
        if self.window_count:
            groups.append({
                "params": [self.geometry.raw_window_a, self.geometry.raw_window_b],
                "name": "window_geometry",
            })
        return groups

    def checkpoint_metadata(self, *, objective: str, tau: float, margin: float) -> dict[str, Any]:
        return {
            "format": self.checkpoint_version,
            "family": self.family,
            "parameterization": TownGeometry.parameterization_version,
            "domain": self.domain,
            "latent_dim": self.latent_dim,
            "houses": self.house_count,
            "windows": self.window_count,
            "width_floors": {"epsilon": self.epsilon, "delta": self.delta,
                             "frame": self.epsilon + (2 * self.delta if self.window_count else 0),
                             "window": self.epsilon},
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
            "initializer": self.initializer_identity,
        }

    def save_checkpoint(
        self, path: Path | str, *, objective: str, tau: float, margin: float,
        epoch: int, validation_loss: float | None, training_config: dict[str, Any],
    ) -> None:
        geometry = self.endpoints()
        for lo, hi, reserve in (
            (geometry[0], geometry[1], 2 * self.delta if self.window_count else 0.0),
            (geometry[2], geometry[3], 0.0),
        ):
            validate_town_endpoints(lo, hi, domain=self.domain, epsilon=self.epsilon, reserve=reserve)
        state = {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}
        _atomic_torch_save({
            "metadata": {
                **self.checkpoint_metadata(objective=objective, tau=tau, margin=margin),
                "training_config": _json_identity(training_config),
            },
            "model_state": state,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "geometry": {
                "frame_lower": geometry[0].detach().cpu(),
                "frame_upper": geometry[1].detach().cpu(),
                "window_lower": geometry[2].detach().cpu(),
                "window_upper": geometry[3].detach().cpu(),
            },
        }, path)

    def _decode_state(self, state):
        from .town_geometry import decode_parameters

        reserve = 2 * self.delta if self.window_count else 0.0
        frame = decode_parameters(
            state["geometry.raw_frame_a"], state["geometry.raw_frame_b"],
            domain=self.domain, epsilon=self.epsilon, reserve=reserve,
        )
        if self.window_count:
            window = decode_parameters(
                state["geometry.raw_window_a"], state["geometry.raw_window_b"],
                domain=self.domain, epsilon=self.epsilon,
            )
        else:
            empty = frame[0].new_empty((*frame[0].shape[:2], 0, self.latent_dim))
            window = (empty, empty.clone())
        return (*frame, *window)

    def load_checkpoint(
        self, path: Path | str, *, objective: str, tau: float, margin: float,
        training_config: dict[str, Any], map_location: torch.device | str = "cpu",
    ) -> None:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        expected_metadata = {
            **self.checkpoint_metadata(objective=objective, tau=tau, margin=margin),
            "training_config": _json_identity(training_config),
        }
        if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
            raise ValueError("TOWN checkpoint metadata/configuration mismatch")
        state = payload.get("model_state")
        current_state = self.state_dict()
        if not isinstance(state, dict) or set(state) != set(current_state):
            raise ValueError("TOWN checkpoint state keys mismatch")
        for key, current in current_state.items():
            candidate = state[key]
            if not isinstance(candidate, torch.Tensor) or candidate.shape != current.shape or candidate.dtype != current.dtype:
                raise ValueError(f"invalid TOWN checkpoint tensor {key}")
            if not torch.isfinite(candidate).all():
                raise ValueError(f"non-finite TOWN checkpoint tensor {key}")
        try:
            decoded = self._decode_state(state)
            validate_town_endpoints(decoded[0], decoded[1], domain=self.domain,
                                    epsilon=self.epsilon, reserve=2 * self.delta if self.window_count else 0)
            validate_town_endpoints(decoded[2], decoded[3], domain=self.domain, epsilon=self.epsilon)
        except (ValueError, TypeError, RuntimeError) as error:
            raise ValueError(f"invalid TOWN checkpoint geometry: {error}") from error
        snapshot = payload.get("geometry")
        keys = ("frame_lower", "frame_upper", "window_lower", "window_upper")
        if not isinstance(snapshot, dict) or set(snapshot) != set(keys):
            raise ValueError("TOWN checkpoint geometry snapshot is missing")
        for key, expected in zip(keys, decoded, strict=True):
            value = snapshot[key]
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != torch.float32:
                raise ValueError(f"invalid TOWN checkpoint geometry snapshot {key}")
            if not torch.equal(value.cpu(), expected.detach().cpu()):
                raise ValueError("TOWN checkpoint geometry snapshot disagrees with raw model state")
        # Validate all identity, tensor and physical predicates before any mutation.
        self.load_state_dict(state, strict=True)


def load_box_initialization(
    path: Path | str,
    dataset: PreparedDataset,
    *,
    domain: str,
    houses: int,
    latent_dim: int,
    epsilon: float = 1e-4,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a verified initial (not trained) BOX producer checkpoint for matched TOWN starts."""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_epoch = payload.get("epoch") if isinstance(payload, dict) else None
    if isinstance(source_epoch, bool) or not isinstance(source_epoch, int) or source_epoch != 0:
        raise ValueError("BOX initialization requires an initial checkpoint with epoch 0")
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("BOX initialization checkpoint lacks metadata")
    objective = metadata.get("objective")
    training_config = metadata.get("training_config")
    if not isinstance(objective, dict) or set(objective) != {"kind", "tau", "margin"}:
        raise ValueError("BOX initialization checkpoint objective metadata is invalid")
    if objective["kind"] not in ("logistic", "hinge"):
        raise ValueError("BOX initialization checkpoint objective is unsupported")
    if not isinstance(training_config, dict):
        raise ValueError("BOX initialization checkpoint training config is invalid")
    if metadata.get("domain") != domain or metadata.get("latent_dim") != latent_dim:
        raise ValueError("BOX initialization domain/latent dimension mismatch")
    if metadata.get("boxes") != houses or metadata.get("epsilon") != epsilon:
        raise ValueError("BOX initialization capacity/width floor mismatch")
    if metadata.get("class_ids") != list(dataset.canonical_class_ids):
        raise ValueError("BOX initialization canonical class identity mismatch")
    if metadata.get("vocabulary_sha256") != dataset.class_manifest_sha256:
        raise ValueError("BOX initialization vocabulary identity mismatch")
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), dataset.canonical_class_ids,
        dataset.class_manifest_sha256, latent_dim=latent_dim, boxes=houses, domain=domain,
        epsilon=epsilon, seed=0, class_names=dataset.class_names, class_iris=dataset.class_iris,
        fixed_positive_class_ids=getattr(dataset, "fixed_positive_class_ids", ()), device="cpu",
    )
    try:
        model.load_checkpoint(
            checkpoint, objective=objective["kind"], tau=objective["tau"],
            margin=objective["margin"], training_config=training_config, map_location="cpu",
        )
    except (ValueError, TypeError, RuntimeError) as error:
        raise ValueError(f"invalid historical BOX initialization producer artifact: {error}") from error
    lower, upper = model.endpoints()
    validate_town_endpoints(
        lower, upper, domain=domain, epsilon=epsilon, reserve=0.0,
    )
    target_device = torch.device(device)
    adapter_state = {
        key: value.detach().to(target_device).clone()
        for key, value in model.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    return {
        "frame_lower": lower.detach().to(target_device),
        "frame_upper": upper.detach().to(target_device),
        "adapter_state": adapter_state,
        "provenance": {
            "producer_family": metadata["family"],
            "producer_format": metadata["format"],
            "producer_training_config": training_config,
            "artifact_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_epoch": source_epoch,
        },
    }


def _town_loss(
    model: TownMultiLabelModel,
    records: Sequence[DatasetRecord],
    dataset: PreparedDataset,
    device: torch.device,
    *,
    kind: LossKind,
    tau: float,
    margin: float,
    point_chunk_size: int | None = None,
    update_only: bool = False,
):
    x_p, y_p, m_p, x_c, y_c, m_c, parents = _pack(records, dataset, device)
    pizza_supervision = set(dataset.pizza_supervision_class_ids)
    component_supervision = set(dataset.component_supervision_class_ids)
    a_p = torch.tensor([value in pizza_supervision for value in dataset.canonical_class_ids],
                       dtype=torch.bool, device=device).expand_as(y_p)
    a_c = torch.tensor([value in component_supervision for value in dataset.canonical_class_ids],
                       dtype=torch.bool, device=device).expand_as(y_c)
    if point_chunk_size is not None and (isinstance(point_chunk_size, bool) or
                                         not isinstance(point_chunk_size, int) or point_chunk_size < 1):
        raise ValueError("point_chunk_size must be a positive integer")
    points_p = model.encode(x_p, "pizza")
    points_c = model.encode(x_c, "component")
    endpoints = model.endpoints()
    size = max(1, len(points_p) + len(points_c)) if point_chunk_size is None else point_chunk_size

    def score(points):
        return (torch.cat([
            town_clearance(
                points[start:start + size], *endpoints, domain=model.domain,
                epsilon=model.epsilon, delta=model.delta,
            )
            for start in range(0, len(points), size)
        ], dim=0) if len(points) else points.new_empty((0, len(model.class_ids))))

    score_p, score_c = score(points_p), score(points_c)
    loss, tasks = masked_geometric_task_loss(
        score_p, y_p, m_p, a_p, score_c, y_c, m_c, a_c, parents,
        kind=kind, tau=tau, margin=margin,
    )
    if update_only:
        membership_p = membership_c = None
    else:
        def exact(points):
            return (torch.cat([
                town_membership(
                    points[start:start + size], *endpoints, domain=model.domain,
                    epsilon=model.epsilon, delta=model.delta,
                )
                for start in range(0, len(points), size)
            ], dim=0) if len(points) else torch.empty(
                (0, len(model.class_ids)), dtype=torch.bool, device=device
            ))
        membership_p, membership_c = exact(points_p), exact(points_c)
    batch = (membership_p, membership_c, y_p, m_p, y_c, m_c, a_p, a_c, parents)
    return loss, tasks, batch


def _town_task_metrics(batch, class_names):
    p_pred, c_pred, y_p, m_p, y_c, m_c, a_p, a_c, parents = batch
    result = {}
    for name, pred, target, known, applicable, component_parent in (
        ("pizza", p_pred, y_p, m_p, a_p, None),
        ("component", c_pred, y_c, m_c, a_c, parents),
    ):
        result[name] = None if pred is None else membership_metrics(
            pred, target, known, applicable, class_names, component_parent
        )
    return result


def _town_geometry_diagnostics(model, records, dataset, device):
    """Layered observed geometry/activity diagnostics without inferring global relations."""
    lower, upper, window_lower, window_upper = model.endpoints()
    x_p, y_p, m_p, x_c, y_c, m_c, _ = _pack(records, dataset, device)
    points = torch.cat((model.encode(x_p, "pizza"), model.encode(x_c, "component")), dim=0)
    if not points.shape[0]:
        return {"point_count": 0, "frames": {"lower": lower.detach().cpu().tolist(),
                "upper": upper.detach().cpu().tolist()},
                "windows": {"lower": window_lower.detach().cpu().tolist(),
                "upper": window_upper.detach().cpu().tolist(), "count": model.window_count}}
    in_frame = ((points[:, None, None, :] >= lower[None]) &
                (points[:, None, None, :] <= upper[None])).all(dim=-1)
    window_interior = ((points[:, None, None, None, :] > window_lower[None]) &
                       (points[:, None, None, None, :] < window_upper[None])).all(dim=-1)
    membership = town_membership(
        points, lower, upper, window_lower, window_upper, domain=model.domain,
        epsilon=model.epsilon, delta=model.delta,
    )
    frame_membership = in_frame.any(dim=2)
    width = upper - lower
    intersection = (torch.minimum(window_upper, upper.unsqueeze(2)) -
                    torch.maximum(window_lower, lower.unsqueeze(2))).clamp_min(0.0)
    volume_fraction = torch.prod(
        intersection.to(torch.float64) / width.unsqueeze(2).to(torch.float64), dim=-1
    ) if model.window_count else points.new_empty((*lower.shape[:2], 0))
    frame_point_count = in_frame.sum(dim=0)
    window_observed_count = (window_interior & in_frame.unsqueeze(-1)).sum(dim=0)
    observed_fraction = window_observed_count.to(torch.float32) / frame_point_count.unsqueeze(-1).clamp_min(1)
    faces = torch.stack((points[:, None, None, :] - lower[None],
                         upper[None] - points[:, None, None, :]), dim=-1)
    frame_score = faces.amin(dim=(-1, -2))
    window_faces = torch.stack((points[:, None, None, None, :] - window_lower[None],
                                window_upper[None] - points[:, None, None, None, :]), dim=-1)
    window_score = window_faces.amin(dim=(-1, -2))
    house_score = torch.amin(torch.cat((frame_score.unsqueeze(-1), -window_score), dim=-1), dim=-1)
    town_score = torch.amax(house_score, dim=-1)
    selected_house = house_score == town_score.unsqueeze(-1)
    window_control_mask = ((-window_score == house_score.unsqueeze(-1)) &
                            selected_house.unsqueeze(-1))
    window_controls = window_control_mask.sum(dim=0)
    window_control_ties = (window_control_mask.sum(dim=-1) > 1).sum(dim=0)
    selected_house_count = selected_house.sum(dim=0)
    frame_control_counts = ((frame_score == house_score) & selected_house).sum(dim=0)
    town_house_tie_counts = (house_score == town_score.unsqueeze(-1)).sum(dim=0)
    return {
        "point_count": int(points.shape[0]),
        "frames": {
            "lower": lower.detach().cpu().tolist(), "upper": upper.detach().cpu().tolist(),
            "width_min": float(width.detach().min()), "width_max": float(width.detach().max()),
            "center_abs_max": float(((lower + upper) / 2).detach().abs().max()),
            "observed_closed_frame_membership_fraction": float(frame_membership.float().mean()),
        },
        "windows": {
            "lower": window_lower.detach().cpu().tolist(), "upper": window_upper.detach().cpu().tolist(),
            "count": model.window_count,
            "contained_in_host_frame": ((window_lower >= lower.unsqueeze(2)) &
                                         (window_upper <= upper.unsqueeze(2))).all(dim=-1).detach().cpu().tolist(),
            "actual_host_frame_volume_exclusion_fraction": volume_fraction.detach().cpu().tolist(),
            "observed_frame_point_exclusion_fraction": observed_fraction.detach().cpu().tolist(),
            "observed_window_interior_counts": window_observed_count.detach().cpu().tolist(),
            "clearance_controlling_window_counts_with_ties": window_controls.detach().cpu().tolist(),
            "points_with_multiple_controlling_windows": window_control_ties.detach().cpu().tolist(),
            "selected_house_counts": selected_house_count.detach().cpu().tolist(),
            "clearance_controlling_frame_counts": frame_control_counts.detach().cpu().tolist(),
            "town_house_tie_counts": town_house_tie_counts.detach().cpu().tolist(),
            "observed_town_membership_exclusion_fraction": float((frame_membership & ~membership).float().mean()),
        },
        "membership_true_fraction": float(membership.float().mean()),
        "clearance_finite": bool(torch.isfinite(town_score).all()),
    }


def _group_parameter_names(model):
    return {
        "pizza_adapter": ("pizza_adapter.",),
        "component_adapter": ("component_adapter.",),
        "frame_geometry": ("geometry.raw_frame_",),
        "window_geometry": ("geometry.raw_window_",),
    }


def train_town(
    dataset: PreparedDataset,
    train_records: Sequence[DatasetRecord],
    validation_records: Sequence[DatasetRecord] = (),
    *,
    artifact_root: Path | str = "artifacts/runs",
    purpose: str = "smoke",
    invocation: str = "python -m ontology_embeddings.town_experiment",
    seed: int = 7,
    epochs: int = 1,
    batch_size: int | None = None,
    learning_rate: float = 0.02,
    latent_dim: int = 16,
    houses: int = 1,
    windows: int = 1,
    domain: str = "bounded",
    epsilon: float = 1e-4,
    delta: float = 1e-4,
    objective: LossKind = "logistic",
    tau: float = 0.1,
    margin: float = 0.1,
    device: torch.device | str = "cpu",
    point_chunk_size: int | None = None,
    initial_box_checkpoint: Path | str | None = None,
    run_record: RunRecord | None = None,
    issue: str = "25",
):
    """Train TOWN with one Adam, global clipping, validated snapshots and run-store evidence."""
    start_time = time.perf_counter()
    config = {
        "model": "town-subtractive-two-adapters",
        "domain": domain, "latent_dim": latent_dim, "houses": houses, "windows": windows,
        "epsilon": epsilon, "delta": delta, "objective": objective, "tau": tau, "margin": margin,
        "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size,
        "device": str(device), "clip_norm": 1.0, "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "frame_geometry"] +
                            (["window_geometry"] if windows else []),
        "betas": [0.9, 0.999], "optimizer_epsilon": 1e-8, "weight_decay": 0,
        "amsgrad": False, "seed": seed,
        "point_chunk_size": point_chunk_size,
        "initial_box_checkpoint_sha256": None,
    }
    rec = run_record or RunRecord(
        artifact_root, purpose=purpose, invocation=invocation, config=config,
        dataset={}, seed=seed, issue=issue,
    )
    try:
        rec.update(
            config=config, feature_schema=FEATURE_SCHEMA,
            environment={"torch": torch.__version__, "device": str(device),
                         "cuda": torch.version.cuda, "python": __import__("sys").version},
            dataset={
                "dataset_id": dataset.dataset_id,
                "dataset_manifest_sha256": dataset.dataset_manifest_sha256,
                "class_manifest_sha256": dataset.class_manifest_sha256,
                "class_ids": list(dataset.canonical_class_ids),
                "class_names": list(dataset.class_names), "class_iris": list(dataset.class_iris),
                "fixed_positive_class_ids": list(getattr(dataset, "fixed_positive_class_ids", ())),
                "unsupported_class_ids": list(getattr(dataset, "unsupported_class_ids", ())),
                "train_sample_ids": [item.sample_id for item in train_records],
                "validation_sample_ids": [item.sample_id for item in validation_records],
            },
        )
        if (not train_records or isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1
                or isinstance(seed, bool) or not isinstance(seed, int)
                or not math.isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError("training requires records, positive epochs and finite learning rate")
        if objective not in ("logistic", "hinge"):
            raise ValueError("objective must be logistic or hinge")
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(margin) or margin < 0:
            raise ValueError("tau must be positive and margin nonnegative, both finite")
        if any(item.split != "train" for item in train_records) or any(
            item.split != "validation" for item in validation_records
        ):
            raise ValueError("TOWN training accepts train and validation records only")
        if {item.sample_id for item in train_records} & {item.sample_id for item in validation_records}:
            raise ValueError("overlapping train and validation sample IDs")
        if any(item.split == "test" for item in (*train_records, *validation_records)):
            raise ValueError("sealed test records cannot be used for training or validation")
        torch.manual_seed(seed)
        random.seed(seed)
        dev = torch.device(device)
        box_initialization = None
        if initial_box_checkpoint is not None:
            box_initialization = load_box_initialization(
                initial_box_checkpoint, dataset, domain=domain, houses=houses,
                latent_dim=latent_dim, epsilon=epsilon, device=dev,
            )
            config["initial_box_checkpoint_sha256"] = box_initialization["provenance"]["artifact_sha256"]
        model = TownMultiLabelModel(
            len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), dataset.canonical_class_ids,
            dataset.class_manifest_sha256, latent_dim=latent_dim, houses=houses, windows=windows,
            domain=domain, epsilon=epsilon, delta=delta, seed=seed,
            class_names=dataset.class_names, class_iris=dataset.class_iris,
            fixed_positive_class_ids=getattr(dataset, "fixed_positive_class_ids", ()), device=dev,
            initial_frame_bounds=None if box_initialization is None else (
                box_initialization["frame_lower"], box_initialization["frame_upper"]
            ),
            initial_adapter_state=None if box_initialization is None else box_initialization["adapter_state"],
            initializer_source={} if box_initialization is None else box_initialization["provenance"],
        )
        config["initializer"] = model.initializer_identity
        if box_initialization is not None:
            config["historical_box_initialization"] = box_initialization["provenance"]
        rec.update(config=config)
        optimizer = torch.optim.Adam(
            model.optimizer_parameter_groups(), lr=learning_rate, betas=(0.9, 0.999),
            eps=1e-8, weight_decay=0, amsgrad=False,
        )
        chunk = len(train_records) if batch_size is None else batch_size
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("batch_size must be a positive integer")
        config["batch_size"] = chunk
        rec.update(config=config)
        checkpoints = rec.path / "checkpoints"
        checkpoints.mkdir()

        def save(kind, epoch, validation_loss=None):
            model.save_checkpoint(
                checkpoints / f"{kind}.pt", objective=objective, tau=tau, margin=margin,
                epoch=epoch, validation_loss=validation_loss, training_config=config,
            )

        save("initial", 0)
        save("latest", 0)
        if validation_records:
            with torch.no_grad():
                initial_val, _, _ = _town_loss(
                    model, validation_records, dataset, dev, kind=objective, tau=tau, margin=margin,
                    point_chunk_size=point_chunk_size, update_only=True,
                )
            if initial_val is None or not torch.isfinite(initial_val):
                raise ValueError("validation has no eligible supervision or non-finite loss")
        initial_geometry = _town_geometry_diagnostics(model, train_records, dataset, dev)
        best_geometry = None
        best_loss = float("inf")
        best_predictions = None
        best_micro_f1 = None
        best_micro_f1_epoch = None
        history = []
        accepted_updates = 0
        initial_parameters = {key: value.detach().clone() for key, value in model.state_dict().items()}
        for epoch in range(epochs):
            order = list(train_records)
            random.shuffle(order)
            per_group_pre, per_group_post, global_pre, global_post = {}, {}, [], []
            epoch_start = {key: value.detach().clone() for key, value in model.state_dict().items()}
            for start in range(0, len(order), chunk):
                loss, _, _ = _town_loss(
                    model, order[start:start + chunk], dataset, dev, kind=objective,
                    tau=tau, margin=margin, point_chunk_size=point_chunk_size, update_only=True,
                )
                gradient_groups = {
                    group["name"]: group["params"] for group in model.optimizer_parameter_groups()
                }
                stats = checked_optimizer_update(
                    model, optimizer, loss, update_number=accepted_updates + 1,
                    gradient_groups=gradient_groups, max_grad_norm=1.0,
                    validate_model=lambda: model.endpoints(), model_name="TOWN",
                )
                global_pre.append(stats["global_pre_clip_norm"])
                global_post.append(stats["global_post_clip_norm"])
                for name, value in stats["group_pre_clip_norms"].items():
                    per_group_pre.setdefault(name, []).append(value)
                for name, value in stats["group_post_clip_norms"].items():
                    per_group_post.setdefault(name, []).append(value)
                accepted_updates += 1
                # Persist only after the shared numerical and physical checks pass.
                save("latest", epoch + 1)
            with torch.no_grad():
                train_loss, train_tasks, train_batch = _town_loss(
                    model, train_records, dataset, dev, kind=objective, tau=tau, margin=margin,
                    point_chunk_size=point_chunk_size,
                )
                if validation_records:
                    val_loss, val_tasks, val_batch = _town_loss(
                        model, validation_records, dataset, dev, kind=objective, tau=tau,
                        margin=margin, point_chunk_size=point_chunk_size,
                    )
                    train_metrics = _town_task_metrics(train_batch, dataset.class_names)
                    validation_metrics = _town_task_metrics(val_batch, dataset.class_names)
                    baseline = class_frequency_baseline(
                        train_batch[2], train_batch[3], train_batch[6],
                        val_batch[2], val_batch[3], val_batch[6], dataset.class_names, None,
                        dataset.canonical_class_ids,
                    )
                    component_baseline = class_frequency_baseline(
                        train_batch[4], train_batch[5], train_batch[7],
                        val_batch[4], val_batch[5], val_batch[7], dataset.class_names,
                        val_batch[8], dataset.canonical_class_ids,
                    )
                    baseline = {"pizza": baseline, "component": component_baseline}
                else:
                    val_loss, val_tasks, val_batch = None, {"pizza": None, "component": None}, None
                    train_metrics, validation_metrics, baseline = {}, {}, None
            available_f1 = [
                value["micro"]["f1"] for value in validation_metrics.values()
                if value is not None and value["micro"]["f1"] is not None
            ]
            mean_f1 = sum(available_f1) / len(available_f1) if available_f1 else None
            if mean_f1 is not None and (best_micro_f1 is None or mean_f1 > best_micro_f1):
                best_micro_f1, best_micro_f1_epoch = mean_f1, epoch + 1
            names = _group_parameter_names(model)
            movement_norms = {}
            parameter_updates = {}
            for group, prefixes in names.items():
                matching = [
                    (epoch_start[key], value) for key, value in model.state_dict().items()
                    if any(key.startswith(prefix) for prefix in prefixes)
                ]
                movement_norms[group] = float(torch.sqrt(sum(
                    ((after - before) ** 2).sum() for before, after in matching
                ))) if matching else 0.0
                parameter_updates[group] = any(not torch.equal(before, after) for before, after in matching)
            row = {
                "epoch": epoch + 1,
                "train_loss": None if train_loss is None else float(train_loss),
                "validation_loss": None if val_loss is None else float(val_loss),
                "train_task_losses": {key: None if value is None else float(value)
                                      for key, value in train_tasks.items()},
                "validation_task_losses": {key: None if value is None else float(value)
                                           for key, value in val_tasks.items()},
                "train_metrics_exact_membership": train_metrics,
                "validation_metrics_exact_membership": validation_metrics,
                "train_class_frequency_baseline": baseline,
                "validation_mean_available_task_micro_f1": mean_f1,
                "gradient_norm_pre_clip_mean": {
                    key: sum(value) / len(value) for key, value in per_group_pre.items()
                },
                "gradient_norm_post_clip_mean": {
                    key: sum(value) / len(value) for key, value in per_group_post.items()
                },
                "global_gradient_norm_pre_clip_mean": sum(global_pre) / len(global_pre),
                "global_gradient_norm_post_clip_mean": sum(global_post) / len(global_post),
                "movement_norms": movement_norms,
                "parameter_updates": parameter_updates,
                "geometry": _town_geometry_diagnostics(model, train_records, dataset, dev),
            }
            history.append(row)
            rec.metric(row)
            if val_loss is not None and float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best_predictions = (val_batch[0].clone(), val_batch[1].clone())
                best_geometry = _town_geometry_diagnostics(model, train_records, dataset, dev)
                save("best", epoch + 1, best_loss)
            rec.update(last_epoch=epoch + 1, latest_metrics=row)
        final_geometry = _town_geometry_diagnostics(model, train_records, dataset, dev)
        save("final", epochs, None if not validation_records else best_loss)
        if validation_records and (checkpoints / "best.pt").is_file():
            model.load_checkpoint(
                checkpoints / "best.pt", objective=objective, tau=tau, margin=margin,
                map_location=dev, training_config=config,
            )
            with torch.no_grad():
                _, _, loaded = _town_loss(
                    model, validation_records, dataset, dev, kind=objective, tau=tau,
                    margin=margin, point_chunk_size=point_chunk_size,
                )
            reload_equal = all(torch.equal(expected, actual) for expected, actual in zip(
                best_predictions, (loaded[0], loaded[1]), strict=True
            ))
        else:
            reload_equal = None
        final_updates = {
            group: any(
                not torch.equal(initial_parameters[key], value)
                for key, value in model.state_dict().items()
                if any(key.startswith(prefix) for prefix in prefixes)
            )
            for group, prefixes in _group_parameter_names(model).items()
        }
        result = {
            "run_id": rec.run_id, "run_path": str(rec.path), "status": "completed",
            "history": history, "parameter_updates": final_updates,
            "best_checkpoint_reload_predictions_equivalent": reload_equal,
            "best_checkpoint_selection": "minimum combined native validation loss; earliest tie",
            "best_mean_available_task_micro_f1_epoch": best_micro_f1_epoch,
            "best_mean_available_task_micro_f1": best_micro_f1,
            "geometry_snapshots": {
                "initial": initial_geometry, "native_loss_selected": best_geometry,
                "final": final_geometry,
            },
            "training_config": config,
            "checkpoint_paths": {
                key: str(checkpoints / f"{key}.pt")
                for key in ("initial", "latest", "best", "final")
                if (checkpoints / f"{key}.pt").is_file()
            },
            "accepted_updates": accepted_updates,
            "runtime_seconds": time.perf_counter() - start_time,
            "peak_rss_kib_process_cumulative": __import__("resource").getrusage(
                __import__("resource").RUSAGE_SELF
            ).ru_maxrss,
        }
        _atomic_json(rec.path / "summary.json", result)
        rec.update(status="completed", finished_at=__import__("datetime").datetime.now().astimezone().isoformat(),
                   summary=result)
        return result
    except KeyboardInterrupt as error:
        rec.interrupt(error)
        raise
    except BaseException as error:
        rec.fail(error)
        raise
