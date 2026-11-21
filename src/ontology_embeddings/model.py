"""Shared latent affine membership model and approved masked reductions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class AffineMultiLabelModel(nn.Module):
    def __init__(
        self, pizza_features: int, component_features: int, classes: int = 60, latent_dim: int = 16
    ):
        super().__init__()
        if min(pizza_features, component_features, classes, latent_dim) < 1:
            raise ValueError("feature, class, and latent dimensions must be positive")
        self.pizza_adapter = nn.Linear(pizza_features, latent_dim)
        self.component_adapter = nn.Linear(component_features, latent_dim)
        self.head = nn.Linear(latent_dim, classes)

    def encode(self, features: torch.Tensor, entity: str) -> torch.Tensor:
        if entity == "pizza":
            return self.pizza_adapter(features)
        if entity == "component":
            return self.component_adapter(features)
        raise ValueError(f"unknown entity type {entity!r}")

    def logits(self, features: torch.Tensor, entity: str) -> torch.Tensor:
        return self.head(self.encode(features, entity))

    def predict(self, features: torch.Tensor, entity: str) -> torch.Tensor:
        """Hard memberships use the exact contract logit > 0."""
        return self.logits(features, entity) > 0


def _masked_entity_mean(
    logits: torch.Tensor, targets: torch.Tensor, known: torch.Tensor, applicable: torch.Tensor
) -> torch.Tensor | None:
    eligible = known.bool() & applicable.bool()
    denominator = eligible.sum(dim=1)
    valid = denominator > 0
    if not torch.any(valid):
        return None
    per_entity: list[torch.Tensor] = []
    for row in range(logits.shape[0]):
        selected = eligible[row]
        if torch.any(selected):
            per_entity.append(
                F.binary_cross_entropy_with_logits(
                    logits[row, selected], targets[row, selected].to(logits.dtype), reduction="mean"
                )
            )
    return torch.stack(per_entity).mean()


def masked_task_loss(
    pizza_logits: torch.Tensor,
    pizza_targets: torch.Tensor,
    pizza_known: torch.Tensor,
    pizza_applicable: torch.Tensor,
    component_logits: torch.Tensor,
    component_targets: torch.Tensor,
    component_known: torch.Tensor,
    component_applicable: torch.Tensor,
    component_parent: Sequence[int],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor | None]]:
    """Reduce each entity, components within parent pizza, then available tasks equally."""
    pizza_loss = _masked_entity_mean(pizza_logits, pizza_targets, pizza_known, pizza_applicable)
    component_valid = component_known.bool() & component_applicable.bool()
    counts = component_valid.sum(dim=1)
    eligible_components = counts > 0
    entity_losses = torch.zeros(
        component_logits.shape[0], dtype=component_logits.dtype, device=component_logits.device
    )
    for row in range(component_logits.shape[0]):
        selected = component_valid[row]
        if torch.any(selected):
            entity_losses[row] = F.binary_cross_entropy_with_logits(
                component_logits[row, selected],
                component_targets[row, selected].to(component_logits.dtype),
                reduction="mean",
            )
    pizza_component_means: list[torch.Tensor] = []
    for parent in range(pizza_logits.shape[0]):
        indices = [i for i, owner in enumerate(component_parent) if owner == parent]
        if indices:
            selected = torch.tensor(indices, device=counts.device)
            valid_indices = selected[eligible_components[selected]]
            if valid_indices.numel():
                pizza_component_means.append(entity_losses[valid_indices].mean())
    component_loss = torch.stack(pizza_component_means).mean() if pizza_component_means else None
    available = [item for item in (pizza_loss, component_loss) if item is not None]
    total = torch.stack(available).mean() if available else None
    return total, {"pizza": pizza_loss, "component": component_loss}
