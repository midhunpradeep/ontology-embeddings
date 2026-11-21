"""Deterministic primitive metadata features; labels are never feature inputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .dataset.constants import (
    BASE_CATEGORIES,
    DETECTOR_CATEGORIES,
    FLAG_INDIVIDUALS,
    TOPPING_COLUMNS,
)
from .dataset.schema import ComponentRecord, DatasetRecord

FEATURE_SCHEMA_VERSION = "metadata-oracle-v1"
PIZZA_FEATURE_NAMES = (
    *(f"topping_count:{name}" for name in TOPPING_COLUMNS),
    *(f"base:{name}" for name in BASE_CATEGORIES),
    *(f"origin:{name}" for name in (*FLAG_INDIVIDUALS, "<missing>")),
)
COMPONENT_FEATURE_NAMES = tuple(f"detector_category:{name}" for name in DETECTOR_CATEGORIES)
FEATURE_SCHEMA = {
    "version": FEATURE_SCHEMA_VERSION,
    "purpose": "metadata_oracle_audit_only_not_predictive_input",
    "pizza": list(PIZZA_FEATURE_NAMES),
    "component": list(COMPONENT_FEATURE_NAMES),
    "sha256": hashlib.sha256(
        json.dumps(
            {
                "version": FEATURE_SCHEMA_VERSION,
                "pizza": PIZZA_FEATURE_NAMES,
                "component": COMPONENT_FEATURE_NAMES,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest(),
}
_DETECTOR_INDEX = {name: index for index, name in enumerate(DETECTOR_CATEGORIES)}


def pizza_features(record: DatasetRecord) -> tuple[float, ...]:
    """Build counts/base/origin only, from the unchanged primitive generator audit."""
    raw: Any = record.audit.get("raw_generator")
    if not isinstance(raw, dict):
        raise ValueError(f"sample {record.sample_id} lacks raw generator metadata")
    counts = raw.get("topping_counts")
    if not isinstance(counts, dict):
        raise ValueError(f"sample {record.sample_id} lacks topping counts")
    try:
        topping_values = [float(counts[name]) for name in TOPPING_COLUMNS]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"sample {record.sample_id} has invalid topping counts") from error
    if any(value < 0 or not value.is_integer() for value in topping_values):
        raise ValueError(f"sample {record.sample_id} topping counts must be nonnegative integers")
    base = raw.get("has_base")
    origin = raw.get("country_of_origin")
    if base not in BASE_CATEGORIES:
        raise ValueError(f"sample {record.sample_id} has unsupported base {base!r}")
    if origin in (None, ""):
        origin = "<missing>"
    if origin not in (*FLAG_INDIVIDUALS, "<missing>"):
        raise ValueError(f"sample {record.sample_id} has unsupported origin {origin!r}")
    return (
        tuple(topping_values)
        + tuple(float(base == value) for value in BASE_CATEGORIES)
        + tuple(float(origin == value) for value in (*FLAG_INDIVIDUALS, "<missing>"))
    )


def component_features(component: ComponentRecord) -> tuple[float, ...]:
    """One-hot detector category identity; category IDs are not OWL class IDs."""
    if component.detector_category not in _DETECTOR_INDEX:
        raise ValueError(f"unsupported detector category {component.detector_category!r}")
    return tuple(float(name == component.detector_category) for name in DETECTOR_CATEGORIES)
