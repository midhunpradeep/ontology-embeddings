from __future__ import annotations

import math

import pytest
import torch

from ontology_embeddings.box import box_clearance, box_membership
from ontology_embeddings.evaluation import membership_metrics
from ontology_embeddings.geo_measurements import (
    BoxEndpoints,
    NumericObservations,
    RecordedDiagnostic,
    TownEndpoints,
    compare_entity_positions,
    measure_clearance,
    measure_embedding_spread,
    measure_geometry,
    measure_predictive,
)
from ontology_embeddings.town_geometry import town_clearance, town_membership


def _box_observation(
    *, run_id="run-1", state_id="u0", split="validation", points=None, diagnostics=()
):
    points = (
        points
        if points is not None
        else torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
    )
    lower = torch.tensor([[[0.0, 0.0]], [[-1.0, -1.0]]], dtype=torch.float32)
    upper = torch.tensor([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32)
    geometry = BoxEndpoints(
        run_id=run_id,
        state_id=state_id,
        class_ids=("class-a", "class-b"),
        domain="unbounded",
        lower=lower,
        upper=upper,
        recorded_diagnostics=diagnostics,
    )
    return NumericObservations(
        run_id=run_id,
        state_id=state_id,
        split=split,
        entity_type="pizza",
        occurrence_ids=("pizza-1", "pizza-2", "pizza-3", "pizza-4"),
        parent_ids=("pizza-1", "pizza-2", "pizza-3", "pizza-4"),
        signature_ids=("sig-a", "sig-a", "sig-b", "sig-c"),
        class_ids=("class-a", "class-b"),
        points=points,
        targets=torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool),
        known=torch.tensor([[1, 1], [1, 1], [1, 1], [0, 1]], dtype=torch.bool),
        applicable=torch.ones((4, 2), dtype=torch.bool),
        scoring_eligibility=torch.tensor([True, False]),
        geometry=geometry,
        recorded_diagnostics=diagnostics,
    )


def test_box_observation_uses_exact_producer_membership_and_clearance():
    observation = _box_observation()
    expected_membership = box_membership(
        observation.points,
        observation.geometry.lower,
        observation.geometry.upper,
        domain="unbounded",
    )
    expected_clearance = box_clearance(
        observation.points,
        observation.geometry.lower,
        observation.geometry.upper,
        domain="unbounded",
    )
    assert torch.equal(observation.predictions, expected_membership)
    assert torch.equal(observation.clearances, expected_clearance)


def test_occurrence_and_input_signature_population_metrics_stay_distinct():
    observation = _box_observation()
    occurrence = measure_predictive(observation, population="occurrence")
    signature = measure_predictive(observation, population="input_signature")

    assert occurrence["scope"]["population"] == "occurrence"
    assert signature["scope"]["population"] == "input_feature_signature"
    occ_a = occurrence["per_class"]["class-a"]
    sig_a = signature["per_class"]["class-a"]
    assert occ_a["occurrence_support"]["known_positive"] == 2
    assert occ_a["occurrence_support"]["known_negative"] == 1
    assert occ_a["signature_support"]["known_positive"] == pytest.approx(1.5)
    assert occ_a["signature_support"]["known_negative"] == pytest.approx(0.5)
    authoritative = membership_metrics(
        observation.predictions,
        observation.targets,
        observation.known,
        observation.applicable,
        observation.class_ids,
        scoring_eligibility=observation.scoring_eligibility,
    )
    assert occ_a["metrics"]["f1"]["value"] == pytest.approx(
        authoritative["per_class"]["class-a"]["f1"]
    )
    assert occurrence["aggregate"]["f1"]["value"] == pytest.approx(authoritative["micro"]["f1"])
    assert sig_a["metrics"]["f1"]["value"] == pytest.approx(0.4)
    assert sig_a["metrics"]["precision"]["value"] == pytest.approx(0.5)
    assert sig_a["metrics"]["recall"]["value"] == pytest.approx(1 / 3)
    assert sig_a["scoring_eligibility"] is True
    assert occurrence["per_class"]["class-b"]["metrics"]["f1"]["status"] == "unavailable"
    assert occurrence["per_class"]["class-a"]["parent_support"]["eligible"] == 3


def test_undefined_scores_and_clearance_groups_are_not_zero_filled():
    observation = _box_observation()
    result = measure_predictive(observation, population="occurrence")
    # class-b has observations but is explicitly outside scoring eligibility.
    assert result["per_class"]["class-b"]["metrics"]["recall"]["value"] is None
    assert result["per_class"]["class-b"]["metrics"]["recall"]["status"] == "unavailable"
    clearance = measure_clearance(observation, population="occurrence")
    positive = clearance["per_class"]["class-a"]["positive"]
    negative = clearance["per_class"]["class-a"]["negative"]
    assert positive["count"] == 2
    assert negative["count"] == 1
    assert positive["mean"]["value"] == pytest.approx(-0.25)
    assert negative["mean"]["value"] == pytest.approx(0.0)
    assert clearance["per_class"]["class-b"]["positive"]["mean"]["status"] == "unavailable"


def test_geometry_keeps_physical_units_and_reports_scale_free_comparisons():
    initial = _box_observation(state_id="u0").geometry
    current = BoxEndpoints(
        run_id="run-1",
        state_id="u1",
        class_ids=("class-a", "class-b"),
        domain="unbounded",
        lower=torch.tensor([[[0.5, 0.0]], [[-1.0, -1.0]]]),
        upper=torch.tensor([[[1.5, 1.0]], [[1.0, 1.0]]]),
    )
    result = measure_geometry(current, initial=initial)
    first = result["classes"]["class-a"]["primitives"][0]
    assert first["widths"]["value"] == [1.0, 1.0]
    assert first["physical_volume"]["value"] == pytest.approx(1.0)
    assert first["log_volume"]["value"] == pytest.approx(0.0)
    assert first["characteristic_width"]["value"] == pytest.approx(1.0)
    movement = first["change_from_reference"]
    assert movement["center_displacement"]["value"] == pytest.approx(0.5)
    assert movement["normalized_center_displacement"]["value"] == pytest.approx(0.5)
    assert movement["width_ratio"]["value"] == [1.0, 1.0]
    assert result["classes"]["class-b"]["represented_region_volume"]["value"] == pytest.approx(4.0)
    assert result["classes"]["class-b"]["represented_region_volume"]["status"] == "defined"


def test_town_h1w1_volume_preserves_boundary_predicate_and_uses_native_functions():
    frame_lower = torch.tensor([[[-1.0, -1.0]]], dtype=torch.float32)
    frame_upper = torch.tensor([[[1.0, 1.0]]], dtype=torch.float32)
    # Matching window and frame leaves the frame boundary: zero volume is not empty.
    window_lower = frame_lower.unsqueeze(2).clone()
    window_upper = frame_upper.unsqueeze(2).clone()
    geometry = TownEndpoints(
        run_id="town-run",
        state_id="t0",
        class_ids=("class-a",),
        domain="unbounded",
        frame_lower=frame_lower,
        frame_upper=frame_upper,
        window_lower=window_lower,
        window_upper=window_upper,
    )
    points = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    observation = NumericObservations(
        run_id="town-run",
        state_id="t0",
        split="validation",
        entity_type="pizza",
        occurrence_ids=("p1", "p2", "p3"),
        parent_ids=("p1", "p2", "p3"),
        signature_ids=("s1", "s2", "s3"),
        class_ids=("class-a",),
        points=points,
        targets=torch.tensor([[1], [0], [1]], dtype=torch.bool),
        known=torch.ones((3, 1), dtype=torch.bool),
        applicable=torch.ones((3, 1), dtype=torch.bool),
        scoring_eligibility=torch.tensor([True]),
        geometry=geometry,
    )
    assert torch.equal(
        observation.predictions,
        town_membership(
            points, frame_lower, frame_upper, window_lower, window_upper, domain="unbounded"
        ),
    )
    assert torch.equal(
        observation.clearances,
        town_clearance(
            points, frame_lower, frame_upper, window_lower, window_upper, domain="unbounded"
        ),
    )
    assert observation.predictions[:, 0].tolist() == [True, False, True]
    represented = measure_geometry(geometry)["classes"]["class-a"]["represented_region"]
    assert represented["physical_volume"]["value"] == 0
    assert represented["is_empty"]["value"] is False


def test_town_empty_set_is_separate_from_zero_volume_boundary_residual():
    geometry = TownEndpoints(
        run_id="empty-town",
        state_id="t0",
        class_ids=("class-a",),
        domain="unbounded",
        frame_lower=torch.tensor([[[-1.0, -1.0]]]),
        frame_upper=torch.tensor([[[1.0, 1.0]]]),
        window_lower=torch.tensor([[[[-2.0, -2.0]]]]),
        window_upper=torch.tensor([[[[2.0, 2.0]]]]),
    )
    represented = measure_geometry(geometry)["classes"]["class-a"]["represented_region"]
    assert represented["physical_volume"]["value"] == 0
    assert represented["is_empty"]["value"] is True


def test_multiple_box_primitive_volumes_are_not_misreported_as_union_volume():
    geometry = BoxEndpoints(
        run_id="multi-box",
        state_id="u0",
        class_ids=("class-a",),
        domain="unbounded",
        lower=torch.tensor([[[0.0, 0.0], [0.5, 0.5]]]),
        upper=torch.tensor([[[1.0, 1.0], [1.5, 1.5]]]),
    )
    result = measure_geometry(geometry)["classes"]["class-a"]
    assert [primitive["physical_volume"]["value"] for primitive in result["primitives"]] == [1, 1]
    assert result["represented_region_volume"]["value"] is None
    assert result["represented_region_volume"]["status"] == "unavailable"
    assert result["represented_region"]["is_empty"]["value"] is False


def test_embedding_spread_and_within_run_entity_movement_are_population_aware():
    first = _box_observation(state_id="u0")
    spread = measure_embedding_spread(first, population="occurrence")
    assert spread["support"]["entity_count"] == 4
    assert spread["effective_dimension"]["status"] == "defined"
    assert 0 <= spread["normalized_effective_dimension"]["value"] <= 1

    second = _box_observation(
        state_id="u1",
        points=first.points + torch.tensor([0.25, 0.0]),
    )
    movement = compare_entity_positions(first, second, population="occurrence")
    assert movement["support"]["matched_occurrences"] == 4
    assert movement["mean_displacement"]["value"] == pytest.approx(0.25)
    assert movement["normalized_mean_displacement"]["status"] == "defined"
    other_run = _box_observation(run_id="run-2", state_id="u1")
    with pytest.raises(ValueError, match="same run"):
        compare_entity_positions(first, other_run)


def test_signature_embedding_spread_is_not_an_arbitrary_representative():
    observation = _box_observation()
    report = measure_embedding_spread(observation, population="input_signature")
    assert report["support"]["entity_count"] == 3
    bad_points = observation.points.clone()
    bad_points[1, 0] += 0.1
    bad = _box_observation(points=bad_points)
    with pytest.raises(ValueError, match="signature has inconsistent coordinates"):
        measure_embedding_spread(bad, population="input_signature")


def test_invalid_population_identity_and_mask_shapes_fail_closed():
    observation = _box_observation()
    with pytest.raises(ValueError, match="occurrence_ids must be unique"):
        NumericObservations(
            run_id="run-1",
            state_id="u0",
            split="validation",
            entity_type="pizza",
            occurrence_ids=("same", "same", "p3", "p4"),
            parent_ids=observation.parent_ids,
            signature_ids=observation.signature_ids,
            class_ids=observation.class_ids,
            points=observation.points,
            targets=observation.targets,
            known=observation.known,
            applicable=observation.applicable,
            scoring_eligibility=observation.scoring_eligibility,
            geometry=observation.geometry,
        )
    with pytest.raises(ValueError, match="split must be train or validation"):
        NumericObservations(
            run_id="run-1",
            state_id="u0",
            split="test",
            entity_type="pizza",
            occurrence_ids=observation.occurrence_ids,
            parent_ids=observation.parent_ids,
            signature_ids=observation.signature_ids,
            class_ids=observation.class_ids,
            points=observation.points,
            targets=observation.targets,
            known=observation.known,
            applicable=observation.applicable,
            scoring_eligibility=observation.scoring_eligibility,
            geometry=observation.geometry,
        )


def test_seen_unseen_signature_metrics_require_explicit_train_signature_identity():
    observation = _box_observation()
    result = measure_predictive(observation, population="occurrence")
    assert result["seen_unseen"]["status"] == "unavailable"
    stratified = measure_predictive(
        observation,
        population="input_signature",
        training_signature_ids={"sig-a", "sig-c"},
    )
    assert stratified["seen_unseen"]["seen"]["support"]["occurrences"] == 3
    assert stratified["seen_unseen"]["unseen"]["support"]["occurrences"] == 1


def test_zero_covariance_and_empty_metric_support_are_undefined_not_zero():
    points = torch.ones((4, 2), dtype=torch.float32)
    observation = _box_observation(points=points)
    spread = measure_embedding_spread(observation)
    assert spread["effective_dimension"]["value"] is None
    assert spread["effective_dimension"]["status"] == "undefined"
    assert spread["effective_dimension"]["reason"] == "zero covariance"

    empty_points = torch.empty((0, 2), dtype=torch.float32)
    empty = NumericObservations(
        run_id="empty-run",
        state_id="u0",
        split="validation",
        entity_type="pizza",
        occurrence_ids=(),
        parent_ids=(),
        signature_ids=(),
        class_ids=("class-a", "class-b"),
        points=empty_points,
        targets=torch.empty((0, 2), dtype=torch.bool),
        known=torch.empty((0, 2), dtype=torch.bool),
        applicable=torch.empty((0, 2), dtype=torch.bool),
        scoring_eligibility=torch.tensor([True, False]),
        geometry=BoxEndpoints(
            run_id="empty-run",
            state_id="u0",
            class_ids=("class-a", "class-b"),
            domain="unbounded",
            lower=torch.tensor([[[0.0, 0.0]], [[-1.0, -1.0]]], dtype=torch.float32),
            upper=torch.tensor([[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float32),
        ),
    )
    clearance = measure_clearance(empty, population="occurrence")
    assert clearance["per_class"]["class-a"]["positive"]["mean"]["value"] is None
    assert clearance["per_class"]["class-a"]["positive"]["mean"]["status"] == "undefined"


def test_producer_diagnostics_keep_their_provenance_scope_units_and_denominators():
    producer_metric = RecordedDiagnostic(
        name="validation_native_loss",
        value=0.125,
        unit="producer loss units",
        scope="validation pizza task, run run-1, state u0",
        denominator=8,
        support=8,
        provenance="producer",
    )
    observation = _box_observation(diagnostics=(producer_metric,))
    predictive = measure_predictive(observation, population="occurrence")
    assert predictive["recorded_diagnostics"] == [
        {
            "name": "validation_native_loss",
            "value": 0.125,
            "unit": "producer loss units",
            "scope": "validation pizza task, run run-1, state u0",
            "provenance": "producer",
            "denominator": 8,
            "support": 8,
            "status": "defined",
            "reason": None,
        }
    ]
    assert (
        measure_geometry(observation.geometry)["recorded_diagnostics"][0]["provenance"]
        == "producer"
    )


def test_geometry_state_comparison_rejects_cross_run_and_keeps_overflow_explicit():
    base = _box_observation().geometry
    other = BoxEndpoints(
        run_id="run-2",
        state_id="u1",
        class_ids=base.class_ids,
        domain="unbounded",
        lower=base.lower,
        upper=base.upper,
    )
    with pytest.raises(ValueError, match="same run"):
        measure_geometry(other, initial=base)

    # Endpoints are valid float32; their 16-dimensional product is not representable
    # as a finite Python float, while its logarithm remains a valid measurement.
    lower = torch.zeros((1, 1, 16), dtype=torch.float32)
    upper = torch.full((1, 1, 16), 1e30, dtype=torch.float32)
    huge = BoxEndpoints(
        run_id="huge",
        state_id="u0",
        class_ids=("huge-class",),
        domain="unbounded",
        lower=lower,
        upper=upper,
    )
    huge_metric = measure_geometry(huge)["classes"]["huge-class"]["primitives"][0]
    assert huge_metric["physical_volume"]["value"] is None
    assert huge_metric["physical_volume"]["status"] == "undefined"
    assert math.isfinite(huge_metric["log_volume"]["value"])
