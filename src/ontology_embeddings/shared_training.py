"""Small shared reductions and fail-closed update step for geometric classifiers.

Geometry families own point encoding, physical decoding, exact membership, and model
validation. This module shares only the known/applicable supervision reduction and
optimizer-safety mechanics required by BOX and subtractive geometries.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .box import box_objective


def _masked_entity_means(scores, targets, known, applicable, *, kind, tau, margin):
    if scores.ndim != 2 or any(
        value.shape != scores.shape for value in (targets, known, applicable)
    ):
        raise ValueError(
            "geometric scores, targets, known and applicable must share [entities, classes]"
        )
    eligible = known.bool() & applicable.bool()
    valid_entities = eligible.any(dim=1)
    # Sanitize only ineligible values before applying a penalty: unknown NaNs must not
    # enter an objective (or its backward graph), while eligible values remain checked.
    safe_scores = torch.where(
        eligible, scores, torch.zeros((), dtype=scores.dtype, device=scores.device)
    )
    safe_targets = torch.where(
        eligible,
        targets.to(dtype=torch.float32),
        torch.zeros((), dtype=torch.float32, device=scores.device),
    )
    penalties = box_objective(safe_scores, safe_targets, kind=kind, tau=tau, margin=margin)
    weights = eligible.to(dtype=penalties.dtype)
    counts = weights.sum(dim=1)
    means = (penalties * weights).sum(dim=1) / counts.clamp_min(1)
    task_mean = means[valid_entities].mean() if bool(valid_entities.any()) else None
    return task_mean, means, valid_entities


def masked_geometric_task_loss(
    pizza_scores: torch.Tensor,
    pizza_targets: torch.Tensor,
    pizza_known: torch.Tensor,
    pizza_applicable: torch.Tensor,
    component_scores: torch.Tensor,
    component_targets: torch.Tensor,
    component_known: torch.Tensor,
    component_applicable: torch.Tensor,
    component_parent: Sequence[int],
    *,
    kind: str = "logistic",
    tau: float = 0.1,
    margin: float = 0.1,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor | None]]:
    """Average eligible entity means, component means within each parent, then tasks equally.

    Selection and denominators match the existing BOX reductions. Empty tasks remain
    undefined; when neither task has supervised values, return ``(None, ...)``.
    """
    if component_scores.ndim != 2:
        raise ValueError("component scores must have shape [occurrences, classes]")
    if len(component_parent) != component_scores.shape[0]:
        raise ValueError("component_parent length mismatch")
    if any(
        isinstance(parent, bool)
        or not isinstance(parent, int)
        or parent < 0
        or parent >= pizza_scores.shape[0]
        for parent in component_parent
    ):
        raise ValueError("invalid component parent index")
    pizza_loss, _, _ = _masked_entity_means(
        pizza_scores,
        pizza_targets,
        pizza_known,
        pizza_applicable,
        kind=kind,
        tau=tau,
        margin=margin,
    )
    _, component_entity_means, valid_components = _masked_entity_means(
        component_scores,
        component_targets,
        component_known,
        component_applicable,
        kind=kind,
        tau=tau,
        margin=margin,
    )
    parent_indices = torch.as_tensor(
        component_parent, dtype=torch.long, device=component_scores.device
    )
    parent_count = pizza_scores.shape[0]
    if parent_count:
        # Dense batch-local incidence avoids entity/parent Python loops and atomic scatter
        # accumulation on accelerators; parent means retain their explicit denominator.
        parent_ids = torch.arange(parent_count, dtype=torch.long, device=component_scores.device)
        parent_incidence = parent_ids[:, None] == parent_indices[None, :]
        parent_weights = parent_incidence.to(component_entity_means.dtype)
        valid_weights = valid_components.to(component_entity_means.dtype)
        parent_sums = (
            parent_weights * component_entity_means[None, :] * valid_weights[None, :]
        ).sum(dim=1)
        parent_counts = (parent_weights * valid_weights[None, :]).sum(dim=1)
        eligible_parents = parent_counts > 0
        component_loss = (
            (parent_sums[eligible_parents] / parent_counts[eligible_parents]).mean()
            if bool(eligible_parents.any())
            else None
        )
    else:
        component_loss = None
    available = [value for value in (pizza_loss, component_loss) if value is not None]
    total = torch.stack(available).mean() if available else None
    return total, {"pizza": pizza_loss, "component": component_loss}


def invalid_optimizer_groups(optimizer: torch.optim.Optimizer) -> list[str]:
    """Return named optimizer groups whose floating-point state is nonfinite."""
    invalid = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            for value in state.values():
                if isinstance(value, torch.Tensor) and (
                    value.is_floating_point() or value.is_complex()
                ):
                    if not torch.isfinite(value).all():
                        invalid.append(group.get("name", "unnamed"))
                        break
                elif isinstance(value, float) and not math.isfinite(value):
                    invalid.append(group.get("name", "unnamed"))
                    break
    return sorted(set(invalid))


def _group_norm(parameters: Sequence[nn.Parameter]) -> float:
    values = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(sum(torch.sum(value**2) for value in values))) if values else 0.0


def checked_optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor | None,
    *,
    update_number: int,
    gradient_groups: Mapping[str, Sequence[nn.Parameter]],
    max_grad_norm: float,
    validate_model: Callable[[], Any] | None = None,
    model_name: str = "geometric",
) -> dict[str, Any]:
    """Execute one clipped update, rejecting invalid loss, gradients, state or geometry.

    Family-specific validation is provided as a callback, keeping BOX/TOWN parameter
    transforms and physical endpoint policies outside the shared trainer.
    """
    optimizer.zero_grad(set_to_none=True)
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError(f"invalid loss at update {update_number}; affected=loss")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not grads or any(not torch.isfinite(grad).all() for grad in grads):
        raise FloatingPointError(f"invalid gradients at update {update_number}; affected=gradients")
    pre_norms = {name: _group_norm(parameters) for name, parameters in gradient_groups.items()}
    pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(pre_clip):
        raise FloatingPointError(
            f"non-finite gradient norm at update {update_number}; affected=gradient_norm"
        )
    if any(
        not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ):
        raise FloatingPointError(
            f"invalid clipped gradients at update {update_number}; affected=gradients"
        )
    post_norms = {name: _group_norm(parameters) for name, parameters in gradient_groups.items()}
    post_global = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )
    )
    optimizer.step()
    invalid_parameters = [
        name for name, parameter in model.named_parameters() if not torch.isfinite(parameter).all()
    ]
    invalid_groups = invalid_optimizer_groups(optimizer)
    if invalid_parameters or invalid_groups:
        affected = sorted(set(invalid_parameters + invalid_groups))
        raise FloatingPointError(
            f"invalid {model_name} update {update_number}; affected={affected}"
        )
    if validate_model is not None:
        try:
            validate_model()
        except (ValueError, RuntimeError) as error:
            raise FloatingPointError(
                f"invalid {model_name} update {update_number}; affected=decoded_geometry; "
                f"reason={error}"
            ) from error
    return {
        "global_pre_clip_norm": float(pre_clip),
        "global_post_clip_norm": post_global,
        "group_pre_clip_norms": pre_norms,
        "group_post_clip_norms": post_norms,
    }
