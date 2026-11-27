"""Occurrence- and exact-embedding-level consistency for observed BOX memberships."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .features import FEATURE_SCHEMA

EXPECTED_DATASET_ID = "pizza-6a33710d13eb90fdde3e"
EXPECTED_MANIFEST_ID = "c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd"
EXPECTED_MANIFEST_SHA256 = "94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844"
EXPECTED_CLASS_MANIFEST_SHA256 = "e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54"
EXPECTED_TRAIN_SELECTION_SHA256 = "d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219"
EXPECTED_VALIDATION_SELECTION_SHA256 = (
    "a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c"
)


def _ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def verify_historical_dataset(dataset: Any) -> dict[str, Any]:
    """Fail closed unless dataset/vocabulary and historical split identities match."""
    if (
        dataset.dataset_id != EXPECTED_DATASET_ID
        or dataset.manifest_id != EXPECTED_MANIFEST_ID
        or dataset.dataset_manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or dataset.class_manifest_sha256 != EXPECTED_CLASS_MANIFEST_SHA256
    ):
        raise ValueError("dataset manifest/vocabulary identity does not match REL-01 inputs")
    splits = split_observed_records(dataset.records)
    result = {}
    for split, expected_count, expected_hash in (
        ("train", 720, EXPECTED_TRAIN_SELECTION_SHA256),
        ("validation", 240, EXPECTED_VALIDATION_SELECTION_SHA256),
    ):
        ids = [record.sample_id for record in splits[split]]
        identity = _ids_sha256(ids)
        if len(ids) != expected_count or identity != expected_hash:
            raise ValueError(f"historical {split} selection identity mismatch")
        result[split] = {"entity_count": len(ids), "selection_sha256": identity}
    return {
        "dataset_id": dataset.dataset_id,
        "declared_manifest_id": dataset.manifest_id,
        "manifest_file_sha256": dataset.dataset_manifest_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "feature_schema_sha256": FEATURE_SCHEMA["sha256"],
        "encoder_device": "cpu",
        "splits": result,
        "test_records_loaded_for_integrity_only": sum(
            record.split == "test" for record in dataset.records
        ),
        "test_records_evaluated": 0,
    }


def split_observed_records(records: Sequence[Any]) -> dict[str, list[Any]]:
    """Return only evaluation splits; test records are never returned to feature encoding."""
    return {
        split: [record for record in records if record.split == split]
        for split in ("train", "validation")
    }


def _table(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    pp = int(np.count_nonzero(a & b))
    pn = int(np.count_nonzero(a & ~b))
    np_ = int(np.count_nonzero(~a & b))
    nn = int(len(a) - pp - pn - np_)
    return {
        "a_positive_b_positive": pp,
        "a_positive_b_negative": pn,
        "a_negative_b_positive": np_,
        "a_negative_b_negative": nn,
    }


def _rates(table: dict[str, int], eligible: int) -> dict[str, Any]:
    pp = table["a_positive_b_positive"]
    pn = table["a_positive_b_negative"]
    np_ = table["a_negative_b_positive"]
    support_a, support_b = pp + pn, pp + np_
    return {
        "subsumption_a_to_b": {
            "antecedent_support": support_a,
            "violation_count": pn,
            "denominator": support_a,
            "conditional_violation_rate": pn / support_a if support_a else None,
            "zero_antecedent_support": support_a == 0,
            "vacuously_satisfied": support_a == 0,
            "satisfied_on_observed_set": pn == 0,
            "undefined_reason": "zero_antecedent_support" if support_a == 0 else None,
        },
        "subsumption_b_to_a": {
            "antecedent_support": support_b,
            "violation_count": np_,
            "denominator": support_b,
            "conditional_violation_rate": np_ / support_b if support_b else None,
            "zero_antecedent_support": support_b == 0,
            "vacuously_satisfied": support_b == 0,
            "satisfied_on_observed_set": np_ == 0,
            "undefined_reason": "zero_antecedent_support" if support_b == 0 else None,
        },
        "disjointness": {
            "violation_count": pp,
            "denominator": eligible,
            "violation_rate": pp / eligible if eligible else None,
            "undefined_reason": "no_eligible_entities" if eligible == 0 else None,
        },
    }


def _view(
    points: np.ndarray,
    membership: np.ndarray,
    eligible_mask: np.ndarray,
    pair_a: int,
    pair_b: int,
    inverse: np.ndarray,
    first: np.ndarray,
    *,
    distinct: bool,
) -> dict[str, Any]:
    occurrence_count = int(np.count_nonzero(eligible_mask))
    if distinct:
        selected_groups = np.unique(inverse[eligible_mask])
        selected_membership = membership[first[selected_groups]]
        total_unit_count = len(first)
        unit = "distinct_embedding"
    else:
        selected_membership = membership[eligible_mask]
        total_unit_count = len(points)
        unit = "entity"
    table = _table(selected_membership[:, pair_a], selected_membership[:, pair_b])
    count = len(selected_membership)
    return {
        "statistical_unit": unit,
        "eligible_count": count,
        "excluded_count": int(total_unit_count - count),
        "eligible_entity_count": occurrence_count,
        "excluded_entity_count": int(len(points) - occurrence_count),
        "eligible_distinct_embedding_count": int(len(np.unique(inverse[eligible_mask]))),
        "excluded_distinct_embedding_count": int(
            len(first) - len(np.unique(inverse[eligible_mask]))
        ),
        "contingency": table,
        **_rates(table, count),
    }


def _aggregate_pair_records(
    pair_records: list[dict[str, Any]],
    subclass_relations: set[tuple[int, int]] | None = None,
    disjoint_relations: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Aggregate pair measurements, optionally filtered to reference axioms."""
    selected = []
    for row in pair_records:
        pair = (row["a_id"], row["b_id"])
        if subclass_relations is None and disjoint_relations is None:
            selected.append((row, True, True))
            continue
        selected.append(
            (
                row,
                any((a, b) in (subclass_relations or set()) for a, b in (pair, pair[::-1])),
                tuple(sorted(pair)) in (disjoint_relations or set()),
            )
        )

    result = {}
    for view_name in ("supervision_restricted", "unrestricted"):
        result[view_name] = {}
        for unit_name in ("occurrence_weighted", "distinct_embedding"):
            sub_violations = sub_support = dis_violations = dis_opportunities = 0
            for row, has_sub, has_dis in selected:
                view = row[view_name][unit_name]
                if has_sub:
                    key = (
                        "subsumption_a_to_b"
                        if (row["a_id"], row["b_id"]) in (subclass_relations or set())
                        else "subsumption_b_to_a"
                    )
                    stat = view[key]
                    sub_violations += stat["violation_count"]
                    sub_support += stat["antecedent_support"]
                if has_dis:
                    stat = view["disjointness"]
                    dis_violations += stat["violation_count"]
                    dis_opportunities += stat["denominator"]
            sub_rate = sub_violations / sub_support if sub_support else None
            dis_rate = dis_violations / dis_opportunities if dis_opportunities else None
            result[view_name][unit_name] = {
                "reference_subclass_relation_count": sum(has_sub for _, has_sub, _ in selected),
                "subsumption_violation_count": sub_violations,
                "subsumption_antecedent_support": sub_support,
                "subsumption_eligible_population_denominator": sub_support,
                "subsumption_conditional_violation_rate": sub_rate,
                "subsumption_zero_antecedent_support": sub_support == 0,
                "subsumption_vacuously_satisfied": sub_support == 0 and sub_violations == 0,
                "subsumption_supported_satisfied": sub_support > 0 and sub_violations == 0,
                "subsumption_undefined_rate_reason": "zero_antecedent_support"
                if sub_support == 0
                else None,
                "reference_disjoint_pair_count": sum(has_dis for _, _, has_dis in selected),
                "disjointness_violation_count": dis_violations,
                "disjointness_eligible_population_denominator": dis_opportunities,
                "disjointness_conditional_violation_rate": dis_rate,
                "disjointness_vacuously_satisfied": dis_opportunities == 0 and dis_violations == 0,
                "disjointness_supported_satisfied": dis_opportunities > 0 and dis_violations == 0,
                "disjointness_undefined_rate_reason": "no_eligible_pair_entity_opportunities"
                if dis_opportunities == 0
                else None,
            }
    cross = {}
    for unit_name in ("occurrence_weighted", "distinct_embedding"):
        sub_v = dis_v = eligible_count = 0
        for row, has_sub, has_dis in selected:
            counts = row["unrestricted_violations_outside_joint_supervision"][unit_name]
            if has_sub:
                # Directed reference orientation determines the applicable counter.
                key = (
                    "a_to_b"
                    if (row["a_id"], row["b_id"]) in (subclass_relations or set())
                    else "b_to_a"
                )
                sub_v += counts[key]
            if has_dis:
                dis_v += counts["disjointness"]
                eligible_count += counts["eligible_entity_count"]
        cross[unit_name] = {
            "reference_subsumption_violation_count": sub_v,
            "reference_disjointness_violation_count": dis_v,
            "disjointness_cross_supervision_eligible_opportunities": eligible_count,
        }
    return {
        "population_identity": "named_fragment_reference_axioms"
        if subclass_relations is not None
        else "all_candidate_class_pairs",
        "measurements": result,
        "violations_outside_joint_supervision": cross,
    }


def evaluate_observed_population(
    points: np.ndarray,
    memberships: np.ndarray,
    known_applicable: np.ndarray,
    classes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate pair statistics; eligibility is applied before restricted deduplication.

    Inputs are encoded points [entities,d], exact raw geometric memberships [entities,C],
    and jointly-known/applicable memberships [entities,C]. The membership matrix is never
    altered by supervision. ``classes`` is in the same canonical column order.
    """
    points = np.asarray(points)
    membership = np.asarray(memberships, dtype=bool)
    eligible = np.asarray(known_applicable, dtype=bool)
    if points.ndim != 2 or membership.ndim != 2 or eligible.shape != membership.shape:
        raise ValueError("expected points [N,d] and matching membership/eligibility [N,C]")
    if len(points) != len(membership) or membership.shape[1] != len(classes):
        raise ValueError("population dimensions do not match canonical classes")
    if not np.isfinite(points).all():
        raise ValueError("observed points must be finite")
    if len(points):
        _, all_first, inverse = np.unique(points, axis=0, return_inverse=True, return_index=True)
        unique_membership = membership[all_first]
    else:
        inverse = np.empty(0, dtype=np.int64)
        all_first = np.empty(0, dtype=np.int64)
        unique_membership = membership
    occupancy = {}
    for index, item in enumerate(classes):
        values = membership[:, index]
        unique_values = unique_membership[:, index]
        occupancy[str(item["class_id"])] = {
            "class_id": item["class_id"],
            "name": item["name"],
            "iri": item["iri"],
            "positive_occurrences": int(values.sum()),
            "occurrence_count": int(len(points)),
            "occurrence_fraction": float(values.mean()) if len(values) else None,
            "positive_distinct_embeddings": int(unique_values.sum()),
            "distinct_embedding_count": int(len(unique_values)),
            "distinct_embedding_fraction": float(unique_values.mean())
            if len(unique_values)
            else None,
        }
    pair_records = []
    for a, b in combinations(range(len(classes)), 2):
        pair_mask = eligible[:, a] & eligible[:, b]
        all_mask = np.ones(len(points), dtype=bool)
        unrestricted_occ = _view(
            points, membership, all_mask, a, b, inverse, all_first, distinct=False
        )
        unrestricted_unique = _view(
            points, membership, all_mask, a, b, inverse, all_first, distinct=True
        )
        restricted_occ = _view(
            points, membership, pair_mask, a, b, inverse, all_first, distinct=False
        )
        restricted_unique = _view(
            points, membership, pair_mask, a, b, inverse, all_first, distinct=True
        )
        cross_occ = _view(points, membership, ~pair_mask, a, b, inverse, all_first, distinct=False)
        cross_unique = _view(
            points, membership, ~pair_mask, a, b, inverse, all_first, distinct=True
        )
        pair_records.append(
            {
                "a_id": classes[a]["class_id"],
                "a_name": classes[a]["name"],
                "a_iri": classes[a]["iri"],
                "b_id": classes[b]["class_id"],
                "b_name": classes[b]["name"],
                "b_iri": classes[b]["iri"],
                "supervision_restricted": {
                    "occurrence_weighted": restricted_occ,
                    "distinct_embedding": restricted_unique,
                },
                "unrestricted": {
                    "occurrence_weighted": unrestricted_occ,
                    "distinct_embedding": unrestricted_unique,
                },
                "unrestricted_violations_outside_joint_supervision": {
                    "occurrence_weighted": {
                        "a_to_b": cross_occ["subsumption_a_to_b"]["violation_count"],
                        "b_to_a": cross_occ["subsumption_b_to_a"]["violation_count"],
                        "disjointness": cross_occ["disjointness"]["violation_count"],
                        "eligible_entity_count": cross_occ["eligible_entity_count"],
                    },
                    "distinct_embedding": {
                        "a_to_b": cross_unique["subsumption_a_to_b"]["violation_count"],
                        "b_to_a": cross_unique["subsumption_b_to_a"]["violation_count"],
                        "disjointness": cross_unique["disjointness"]["violation_count"],
                        "eligible_entity_count": cross_unique["eligible_entity_count"],
                    },
                },
            }
        )
    aggregate = {}
    for view_name in ("supervision_restricted", "unrestricted"):
        aggregate[view_name] = {}
        for unit_name in ("occurrence_weighted", "distinct_embedding"):
            entries = [row[view_name][unit_name] for row in pair_records]
            a_violations = sum(x["subsumption_a_to_b"]["violation_count"] for x in entries)
            b_violations = sum(x["subsumption_b_to_a"]["violation_count"] for x in entries)
            a_support = sum(x["subsumption_a_to_b"]["antecedent_support"] for x in entries)
            b_support = sum(x["subsumption_b_to_a"]["antecedent_support"] for x in entries)
            disjoint_violations = sum(x["disjointness"]["violation_count"] for x in entries)
            disjoint_denominator = sum(x["disjointness"]["denominator"] for x in entries)
            aggregate[view_name][unit_name] = {
                "directed_subsumption_violation_count": a_violations + b_violations,
                "directed_subsumption_antecedent_support": a_support + b_support,
                "conditional_violation_rate": (
                    (a_violations + b_violations) / (a_support + b_support)
                    if a_support + b_support
                    else None
                ),
                "zero_antecedent_support": a_support + b_support == 0,
                "disjointness_violation_count": disjoint_violations,
                "disjointness_denominator": disjoint_denominator,
                "disjointness_violation_rate": (
                    disjoint_violations / disjoint_denominator if disjoint_denominator else None
                ),
            }
    cross_summary = {}
    for unit_name in ("occurrence_weighted", "distinct_embedding"):
        cross = [
            row["unrestricted_violations_outside_joint_supervision"][unit_name]
            for row in pair_records
        ]
        cross_summary[unit_name] = {
            "directed_subsumption_violation_count": sum(x["a_to_b"] + x["b_to_a"] for x in cross),
            "disjointness_violation_count": sum(x["disjointness"] for x in cross),
        }
    return {
        "entity_occurrence_count": int(len(points)),
        "distinct_embedding_count": int(len(unique_membership)),
        "class_occupancy": occupancy,
        "summary_population_identity": "all_candidate_class_pairs",
        "summary_interpretation": (
            "descriptive statistics over all candidate class pairs; not all pairs are "
            "ontology requirements"
        ),
        "summary": aggregate,
        "cross_task_violations_outside_joint_supervision_population_identity": (
            "all_candidate_class_pairs"
        ),
        "cross_task_violations_outside_joint_supervision": cross_summary,
        "pairs": pair_records,
    }
