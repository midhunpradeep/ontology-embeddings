from __future__ import annotations

import json
import sys

import torch

from ontology_embeddings.geo_explorer import (
    discover_run_metadata,
    native_box_scene,
    select_observation,
)
from ontology_embeddings.geo_measurements import BoxEndpoints, NumericObservations


def _observation(run="run-1", state="u0"):
    geometry = BoxEndpoints(
        run_id=run,
        state_id=state,
        class_ids=("class-a",),
        domain="unbounded",
        lower=torch.tensor([[[0.0, 0.0]]]),
        upper=torch.tensor([[[1.0, 1.0]]]),
    )
    return NumericObservations(
        run_id=run,
        state_id=state,
        split="validation",
        entity_type="pizza",
        occurrence_ids=("p1", "p2"),
        parent_ids=("p1", "p2"),
        signature_ids=("sig1", "sig2"),
        class_ids=("class-a",),
        points=torch.tensor([[0.5, 0.5], [2.0, 0.5]]),
        targets=torch.tensor([[True], [False]]),
        known=torch.ones((2, 1), dtype=torch.bool),
        applicable=torch.ones((2, 1), dtype=torch.bool),
        scoring_eligibility=torch.tensor([True]),
        geometry=geometry,
    )


def test_pure_adapter_selects_committed_numeric_observations_by_run_and_state():
    observations = (
        _observation("run-a", "u0"),
        _observation("run-a", "u4"),
        _observation("run-b", "u0"),
    )
    assert select_observation(observations, "run-a", "u4") is observations[1]
    assert select_observation(observations, "absent", "u0") is None


def test_native_2d_box_scene_uses_original_points_exact_membership_and_endpoints():
    scene = native_box_scene(_observation(), "class-a")
    assert scene["scope"] == {
        "run_id": "run-1",
        "state_id": "u0",
        "split": "validation",
        "entity_type": "pizza",
    }
    assert scene["points"][0]["coordinates"] == [0.5, 0.5]
    assert scene["points"][0]["membership"] is True
    assert scene["points"][1]["membership"] is False
    assert scene["boxes"] == [{"lower": [0.0, 0.0], "upper": [1.0, 1.0]}]


def test_read_only_run_discovery_uses_run_store_and_preserves_index(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    index = root / "index.jsonl"
    row = {
        "run_id": "run-1",
        "started_at": "2026-11-28T00:00:00Z",
        "status": "completed",
        "purpose": "geo",
        "config": {"model": "box", "latent_dim": 2, "domain": "unbounded"},
    }
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    original = index.read_bytes()

    discovered = discover_run_metadata(root)

    assert discovered[0]["run_id"] == "run-1"
    assert discovered[0]["status"] == "completed"
    assert discovered[0]["family"] == "box"
    assert discovered[0]["dimension"] == 2
    assert index.read_bytes() == original


def test_module_import_is_dash_lazy():
    assert "dash" not in sys.modules
    assert "plotly" not in sys.modules


def test_scene_rejects_non_box_and_non_2d_observations():
    import pytest

    obs = _observation()
    assert obs.geometry.domain == "unbounded"
    with pytest.raises(ValueError, match="BOX observations"):
        native_box_scene(object(), "class-a")
