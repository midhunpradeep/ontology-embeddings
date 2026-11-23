"""Minimal affine metadata-oracle learning slice and CPU smoke entry point."""

from .features import FEATURE_SCHEMA, component_features, pizza_features
from .model import AffineMultiLabelModel, masked_task_loss
from .training import train

__all__ = [
    "AffineMultiLabelModel",
    "FEATURE_SCHEMA",
    "component_features",
    "masked_task_loss",
    "pizza_features",
    "train",
]
