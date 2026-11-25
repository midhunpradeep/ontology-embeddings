import numpy as np

from ontology_embeddings.observed_relations import (
    evaluate_observed_population,
    split_observed_records,
)

CLASSES = [
    {"class_id": 7, "name": "A", "iri": "urn:A"},
    {"class_id": 12, "name": "B", "iri": "urn:B"},
]


def fixture():
    # Rows 0 and 1 are the same exact embedding, but only row 0 is jointly eligible.
    points = np.array([[0.0], [0.0], [1.0], [2.0]])
    membership = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
    eligibility = np.array([[1, 1], [1, 0], [1, 1], [0, 1]], dtype=bool)
    return points, membership, eligibility


def test_restricted_and_unrestricted_contingencies_and_cross_task_violations():
    result = evaluate_observed_population(*fixture(), CLASSES)
    pair = result["pairs"][0]
    restricted = pair["supervision_restricted"]["occurrence_weighted"]
    assert restricted["eligible_entity_count"] == 2
    assert restricted["excluded_entity_count"] == 2
    assert restricted["contingency"] == {
        "a_positive_b_positive": 0,
        "a_positive_b_negative": 1,
        "a_negative_b_positive": 1,
        "a_negative_b_negative": 0,
    }
    assert restricted["subsumption_a_to_b"]["antecedent_support"] == 1
    assert restricted["subsumption_a_to_b"]["violation_count"] == 1
    assert restricted["subsumption_a_to_b"]["conditional_violation_rate"] == 1
    unrestricted = pair["unrestricted"]["occurrence_weighted"]
    assert unrestricted["contingency"] == {
        "a_positive_b_positive": 1,
        "a_positive_b_negative": 2,
        "a_negative_b_positive": 1,
        "a_negative_b_negative": 0,
    }
    assert pair["unrestricted_violations_outside_joint_supervision"]["occurrence_weighted"] == {
        "a_to_b": 1,
        "b_to_a": 0,
        "disjointness": 1,
        "eligible_entity_count": 2,
    }


def test_zero_support_is_vacuous_but_rate_undefined():
    points = np.array([[0.0], [1.0]])
    membership = np.array([[0, 1], [0, 0]], dtype=bool)
    eligibility = np.ones_like(membership)
    pair = evaluate_observed_population(points, membership, eligibility, CLASSES)["pairs"][0]
    implication = pair["supervision_restricted"]["occurrence_weighted"]["subsumption_a_to_b"]
    assert implication["antecedent_support"] == 0
    assert implication["conditional_violation_rate"] is None
    assert implication["zero_antecedent_support"]
    assert implication["vacuously_satisfied"]
    assert implication["undefined_reason"] == "zero_antecedent_support"


def test_distinct_embeddings_deduplicate_after_relation_eligibility():
    result = evaluate_observed_population(*fixture(), CLASSES)
    pair = result["pairs"][0]
    restricted = pair["supervision_restricted"]["distinct_embedding"]
    assert restricted["eligible_count"] == 2
    assert restricted["excluded_count"] == 1
    assert restricted["eligible_entity_count"] == 2
    assert restricted["excluded_entity_count"] == 2
    assert restricted["contingency"]["a_positive_b_negative"] == 1
    unrestricted = pair["unrestricted"]["distinct_embedding"]
    assert unrestricted["eligible_count"] == 3
    assert unrestricted["contingency"]["a_positive_b_negative"] == 1
    assert result["entity_occurrence_count"] == 4
    assert result["distinct_embedding_count"] == 3


def test_per_class_occupancy_reports_occurrences_and_unique_points():
    result = evaluate_observed_population(*fixture(), CLASSES)
    occupancy = result["class_occupancy"]["7"]
    assert occupancy["positive_occurrences"] == 3
    assert occupancy["occurrence_count"] == 4
    assert occupancy["positive_distinct_embeddings"] == 2
    assert occupancy["distinct_embedding_count"] == 3


def test_sealed_test_records_never_enter_observed_split_population():
    records = [
        type("R", (), {"sample_id": "train", "split": "train"})(),
        type("R", (), {"sample_id": "validation", "split": "validation"})(),
        type("R", (), {"sample_id": "sealed", "split": "test"})(),
    ]
    selected = split_observed_records(records)
    assert [r.sample_id for r in selected["train"]] == ["train"]
    assert [r.sample_id for r in selected["validation"]] == ["validation"]
    assert all(r.split != "test" for group in selected.values() for r in group)
