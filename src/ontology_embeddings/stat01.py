"""Offline, descriptive dataset information audit (STAT-01)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .affine_smoke import load_train_validation
from .dataset.constants import BASE_CATEGORIES, FLAG_INDIVIDUALS, TOPPING_COLUMNS
from .dataset.ontology import OntologySnapshot
from .features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
    component_features,
    pizza_features,
)

SCHEMA = "stat-01-evidence-v1"
IMPLEMENTATION = "0.1.0"
MANIFEST_ID = "c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd"
DATASET_ID = "pizza-6a33710d13eb90fdde3e"
STATE_NAMES = ("known_negative", "known_positive", "unknown", "inapplicable")


@dataclass(frozen=True)
class Observation:
    ref: str
    parent: str
    features: tuple[float, ...]
    states: tuple[int, ...]
    topping_combo: tuple[str, ...]


def entropy(counts: Iterable[float]) -> tuple[float | None, float | None]:
    vals = [float(x) for x in counts if x > 0]
    total = sum(vals)
    if total <= 0:
        return None, None
    h = -sum((x / total) * math.log(x / total) for x in vals)
    return h, math.exp(h)


def js_divergence(left: Counter, right: Counter) -> float | None:
    keys = set(left) | set(right)
    a, b = sum(left.values()), sum(right.values())
    if not a or not b:
        return None
    p = {k: left[k] / a for k in keys}
    q = {k: right[k] / b for k in keys}
    m = {k: (p[k] + q[k]) / 2 for k in keys}
    return (
        sum(p[k] * math.log(p[k] / m[k]) for k in keys if p[k]) / 2
        + sum(q[k] * math.log(q[k] / m[k]) for k in keys if q[k]) / 2
    )


def states_for(
    targets: Sequence[int], known: Sequence[int], ids: Sequence[int], applicable: set[int]
) -> tuple[int, ...]:
    """Encode canonical class order: negative, positive, unknown, inapplicable."""
    result = []
    for cid in ids:
        if cid not in applicable:
            result.append(3)
        elif not known[cid]:
            result.append(2)
        else:
            result.append(1 if targets[cid] else 0)
    return tuple(result)


def observations(dataset, records, level: str) -> list[Observation]:
    ids = dataset.canonical_class_ids
    applicable = set(
        dataset.pizza_supervision_class_ids
        if level == "pizza"
        else dataset.component_supervision_class_ids
    )
    out = []
    for record in records:
        raw = record.audit.get("raw_generator", {})
        toppings = tuple(sorted(k for k, v in raw.get("topping_counts", {}).items() if v))
        if level == "pizza":
            out.append(
                Observation(
                    record.sample_id,
                    record.sample_id,
                    pizza_features(record),
                    states_for(record.pizza_targets, record.pizza_known_mask, ids, applicable),
                    toppings,
                )
            )
        else:
            for component in record.components:
                out.append(
                    Observation(
                        f"{record.sample_id}/{component.component_id}",
                        record.sample_id,
                        component_features(component),
                        states_for(
                            component.ontology_targets,
                            component.ontology_known_mask,
                            ids,
                            applicable,
                        ),
                        toppings,
                    )
                )
    return out


def _weighted(observations_: list[Observation], mode: str) -> list[float]:
    if mode == "occurrence":
        return [1.0] * len(observations_)
    if mode == "parent":
        groups = Counter(o.parent for o in observations_)
        return [1 / groups[o.parent] for o in observations_]
    signatures = Counter(o.features for o in observations_)
    return [1 / signatures[o.features] for o in observations_]


def _eligible_weights(obs: list[Observation], eligible: list[bool], mode: str) -> list[float]:
    if mode == "occurrence":
        return [float(ok) for ok in eligible]
    if mode == "signature":
        n = Counter(o.features for o, ok in zip(obs, eligible) if ok)
        return [1 / n[o.features] if ok else 0.0 for o, ok in zip(obs, eligible)]
    n = Counter(o.parent for o, ok in zip(obs, eligible) if ok)
    return [1 / n[o.parent] if ok else 0.0 for o, ok in zip(obs, eligible)]


def _binary_stats(n11: float, n10: float, n01: float, n00: float) -> dict[str, Any]:
    n = n11 + n10 + n01 + n00
    h_a, _ = entropy((n11 + n10, n01 + n00))
    h_b, _ = entropy((n11 + n01, n10 + n00))
    h_joint, _ = entropy((n11, n10, n01, n00))

    def div(a, b):
        return a / b if b else None

    den = math.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
    pa = div(n11 + n10, n)
    pb = div(n11 + n01, n)
    pab = div(n11, n)
    return {
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "jointly_known": n,
        "p_b_given_a": div(n11, n11 + n10),
        "p_a_given_b": div(n11, n11 + n01),
        "jaccard": div(n11, n11 + n10 + n01),
        "disagreement": div(n10 + n01, n),
        "phi": (n11 * n00 - n10 * n01) / den if den else None,
        "mutual_information_nats": h_a + h_b - h_joint if n else None,
        "entropy_a_nats": h_a,
        "entropy_b_nats": h_b,
        "joint_entropy_nats": h_joint,
        "positive_lift": div(pab, pa * pb)
        if pa is not None and pb is not None and pa * pb
        else None,
    }


def class_rows(
    obs: list[Observation], ids: Sequence[int], names: Sequence[str], level: str
) -> list[dict[str, Any]]:
    rows = []
    for j, (cid, name) in enumerate(zip(ids, names)):
        vals = [o.states[j] for o in obs]
        sig_by_state = defaultdict(set)
        for o, s in zip(obs, vals):
            if s in (0, 1):
                sig_by_state[s].add(o.features)
        conflict_sigs = {
            s
            for s in set(o.features for o in obs)
            if any(o.features == s and o.states[j] == 0 for o in obs)
            and any(o.features == s and o.states[j] == 1 for o in obs)
        }
        for mode in ("occurrence", "parent", "signature"):
            weights = _weighted(obs, mode)
            masses = {
                state: sum(w for w, s in zip(weights, vals) if s == state) for state in range(4)
            }
            known = [s in (0, 1) for s in vals]
            eligible_w = _eligible_weights(obs, known, mode)
            eligible = sum(eligible_w)
            positive = sum(w for w, s in zip(eligible_w, vals) if s == 1)
            applicable_mass = masses[0] + masses[1] + masses[2]
            rows.append(
                {
                    "class_id": cid,
                    "class_name": name,
                    "entity_type": level,
                    "view": mode,
                    "occurrences": len(obs),
                    "known_positive_support": sum(s == 1 for s in vals),
                    "known_negative_support": sum(s == 0 for s in vals),
                    "unknown_support": sum(s == 2 for s in vals),
                    "inapplicable_support": sum(s == 3 for s in vals),
                    "applicable_mass": applicable_mass,
                    "known_applicable_mass": masses[0] + masses[1],
                    "known_applicable_coverage": (masses[0] + masses[1]) / applicable_mass
                    if applicable_mass
                    else None,
                    "known_positive_mass": masses[1],
                    "known_negative_mass": masses[0],
                    "unknown_mass": masses[2],
                    "inapplicable_mass": masses[3],
                    "eligible_mass": eligible,
                    "positive_prevalence": positive / eligible if eligible else None,
                    "distinct_positive_signatures": len(sig_by_state[1]),
                    "distinct_negative_signatures": len(sig_by_state[0]),
                    "conflicting_signatures": len(conflict_sigs),
                    "known_denominator": sum(known),
                    "applicable_denominator": sum(s != 3 for s in vals),
                    "known_applicable_coverage_occurrence": (sum(known) / sum(s != 3 for s in vals))
                    if sum(s != 3 for s in vals)
                    else None,
                    "eligible_parents": len({o.parent for o, k in zip(obs, known) if k}),
                    "excluded_observations": sum(not k for k in known),
                }
            )
    return rows


def ontology_relations(
    ontology: OntologySnapshot,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    closure = {(a, b) for a in ontology.class_names for b in ontology.ancestors(a) if a != b}
    direct = set()
    for name, element in ontology.class_elements.items():
        for child in element.findall(
            "owl:disjointWith",
            {
                "owl": "http://www.w3.org/2002/07/owl#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            },
        ):
            iri = child.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            other = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if iri else ""
            if other in ontology.class_names:
                direct.add(tuple(sorted((name, other))))
    disjoint = set()
    for a in ontology.class_names:
        for b in ontology.class_names:
            if a >= b:
                continue
            if any(
                tuple(sorted((x, y))) in direct
                for x in ontology.ancestors(a)
                for y in ontology.ancestors(b)
            ):
                disjoint.add((a, b))
    return closure, disjoint


def pair_rows(
    obs: list[Observation], ids: Sequence[int], names: Sequence[str], level: str, relations=None
) -> list[dict[str, Any]]:
    rows = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        state_counts = [[0] * 4 for _ in range(4)]
        signatures = defaultdict(set)
        for o in obs:
            a, b = o.states[i], o.states[j]
            state_counts[a][b] += 1
            if a in (0, 1) and b in (0, 1):
                signatures[(a, b)].add(o.features)
        closure, disjoint = relations if relations else (set(), set())
        row = {
            "class_id_a": ids[i],
            "class_id_b": ids[j],
            "class_name_a": names[i],
            "class_name_b": names[j],
            "entity_type": level,
            "view": "occurrence",
            "reference_a_subclass_b": (names[i], names[j]) in closure,
            "reference_b_subclass_a": (names[j], names[i]) in closure,
            "reference_disjoint": tuple(sorted((names[i], names[j]))) in disjoint,
            "state_contingency": json.dumps(state_counts, separators=(",", ":")),
            "contrasting_signature_support": json.dumps(
                {f"{a}{b}": len(signatures[(a, b)]) for a, b in signatures}, sort_keys=True
            ),
        }
        for mode in ("occurrence", "parent", "signature"):
            complete_weights = _weighted(obs, mode)
            weighted_states = [[0.0] * 4 for _ in range(4)]
            for o, weight in zip(obs, complete_weights):
                weighted_states[o.states[i]][o.states[j]] += weight
            row[f"{mode}_state_contingency"] = json.dumps(weighted_states, separators=(",", ":"))
            eligible = [o.states[i] in (0, 1) and o.states[j] in (0, 1) for o in obs]
            w = _eligible_weights(obs, eligible, mode)
            ct = [[0.0] * 2 for _ in range(2)]
            for o, weight, ok in zip(obs, w, eligible):
                if ok:
                    ct[o.states[i]][o.states[j]] += weight
            # state index 1 is positive
            stats = _binary_stats(ct[1][1], ct[1][0], ct[0][1], ct[0][0])
            for key, val in stats.items():
                row[f"{mode}_{key}"] = val
            row[f"{mode}_excluded_observations"] = sum(not x for x in eligible)
            row[f"{mode}_eligible_parents"] = len({o.parent for o, k in zip(obs, eligible) if k})
            population = (
                len(obs)
                if mode == "occurrence"
                else len({o.parent for o in obs})
                if mode == "parent"
                else len({o.features for o in obs})
            )
            row[f"{mode}_population_units_denominator"] = population
            row[f"{mode}_jointly_known_coverage"] = (
                stats["jointly_known"] / population if population else None
            )
        rows.append(row)
    return rows


def diversity(obs: list[Observation], feature_names: Sequence[str]) -> dict[str, Any]:
    sig = Counter(o.features for o in obs)
    patterns = Counter(o.states for o in obs)
    combos = Counter((o.features, o.states) for o in obs)
    value_distributions = []
    for i, name in enumerate(feature_names):
        c = Counter(o.features[i] for o in obs)
        h, d = entropy(c.values())
        value_distributions.append(
            {
                "feature": name,
                "counts": dict(sorted(c.items(), key=lambda kv: str(kv[0]))),
                "entropy_nats": h,
                "effective_diversity": d,
            }
        )

    def distribution(c):
        h, d = entropy(c.values())
        return {
            "distinct": len(c),
            "entropy_nats": h,
            "effective_diversity": d,
            "multiplicity": {
                "min": min(c.values()) if c else None,
                "max": max(c.values()) if c else None,
                "mean": sum(c.values()) / len(c) if c else None,
                "counts": dict(sorted(Counter(c.values()).items())),
            },
        }

    weighted_patterns = {}
    for mode in ("occurrence", "parent", "signature"):
        masses = Counter()
        for o, w in zip(obs, _weighted(obs, mode)):
            masses[o.states] += w
        h, d = entropy(masses.values())
        weighted_patterns[mode] = {
            "pattern_masses": {
                json.dumps(k, separators=(",", ":")): v for k, v in sorted(masses.items())
            },
            "total_mass": sum(masses.values()),
            "entropy_nats": h,
            "effective_diversity": d,
        }
    return {
        "entity_occurrences": len(obs),
        "parent_pizzas": len({o.parent for o in obs}),
        "feature_signatures": distribution(sig),
        "feature_signature_max_occurrence_share": max(sig.values()) / len(obs) if obs else None,
        "supervision_patterns": distribution(patterns),
        "weighted_supervision_patterns": weighted_patterns,
        "signature_pattern_combinations": len(combos),
        "feature_values": value_distributions,
        "topping_combinations": {
            "distinct": len(Counter(o.topping_combo for o in obs)),
            "counts": [
                {"combination": list(k), "count": v}
                for k, v in sorted(
                    Counter(o.topping_combo for o in obs).items(), key=lambda x: x[0]
                )
            ],
        },
    }


def _mutual_information(pairs: Iterable[tuple[Any, Any]]) -> float | None:
    counts = Counter(pairs)
    total = sum(counts.values())
    if not total:
        return None
    left = Counter()
    right = Counter()
    for (a, b), n in counts.items():
        left[a] += n
        right[b] += n
    return sum(
        (n / total) * math.log((n * total) / (left[a] * right[b]))
        for (a, b), n in counts.items()
        if n
    )


def feature_associations(
    obs: list[Observation],
    feature_names: Sequence[str],
    ids: Sequence[int],
    class_names: Sequence[str],
    level: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    pair_rows_ = []
    for k, feature in enumerate(feature_names):
        vals = Counter(o.features[k] for o in obs)
        for cid, name in zip(range(len(ids)), class_names):
            supports = defaultdict(Counter)
            known_pairs = []
            for o in obs:
                value = o.features[k]
                state = o.states[cid]
                supports[value][state] += 1
                if state in (0, 1):
                    known_pairs.append((value, state))
            for value, count in sorted(vals.items(), key=lambda x: x[0]):
                s = supports[value]
                eligible = s[0] + s[1]
                rows.append(
                    {
                        "entity_type": level,
                        "feature": feature,
                        "feature_value": value,
                        "class_id": ids[cid],
                        "class_name": name,
                        "value_occurrences": count,
                        "known_positive": s[1],
                        "known_negative": s[0],
                        "unknown": s[2],
                        "inapplicable": s[3],
                        "known_applicable": eligible,
                        "positive_prevalence": s[1] / eligible if eligible else None,
                        "conditional_mutual_information_nats": _mutual_information(known_pairs),
                        "feature_entropy_nats": entropy(vals.values())[0],
                    }
                )
    for i, j in itertools.combinations(range(len(feature_names)), 2):
        joint = Counter((o.features[i], o.features[j]) for o in obs)
        left_to_right = defaultdict(set)
        right_to_left = defaultdict(set)
        for a, b in joint:
            left_to_right[a].add(b)
            right_to_left[b].add(a)
        pair_rows_.append(
            {
                "entity_type": level,
                "feature_a": feature_names[i],
                "feature_b": feature_names[j],
                "mutual_information_nats": _mutual_information(
                    (o.features[i], o.features[j]) for o in obs
                ),
                "joint_support": len(joint),
                "joint_counts": json.dumps(
                    [[a, b, n] for (a, b), n in sorted(joint.items())], separators=(",", ":")
                ),
                "constant_a": len({o.features[i] for o in obs}) <= 1,
                "constant_b": len({o.features[j] for o in obs}) <= 1,
                "a_determines_b": all(len(v) == 1 for v in left_to_right.values()),
                "b_determines_a": all(len(v) == 1 for v in right_to_left.values()),
            }
        )
    return rows, pair_rows_


def empirical_relations(
    obs: list[Observation], ids: Sequence[int], names: Sequence[str], relations, level: str
) -> dict[str, Any]:
    closure, disjoint = relations
    implications = []
    equivalences = []
    exclusions = []
    contradictions = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        ct = Counter(
            (o.states[i], o.states[j])
            for o in obs
            if o.states[i] in (0, 1) and o.states[j] in (0, 1)
        )
        n11 = ct[1, 1]
        n10 = ct[1, 0]
        n01 = ct[0, 1]
        n00 = ct[0, 0]
        n = n11 + n10 + n01 + n00
        if n:
            if n11 + n10 and n10 == 0:
                implications.append(
                    {
                        "from_class_id": ids[i],
                        "to_class_id": ids[j],
                        "from_class": names[i],
                        "to_class": names[j],
                        "n11": n11,
                        "n10": n10,
                        "jointly_known": n,
                        "counterexample_refs": [],
                    }
                )
            if n11 + n01 and n01 == 0:
                implications.append(
                    {
                        "from_class_id": ids[j],
                        "to_class_id": ids[i],
                        "from_class": names[j],
                        "to_class": names[i],
                        "n11": n11,
                        "n10": n01,
                        "jointly_known": n,
                        "counterexample_refs": [],
                    }
                )
            if n11 > 0 and n10 == 0 and n01 == 0:
                equivalences.append(
                    {
                        "class_ids": [ids[i], ids[j]],
                        "classes": [names[i], names[j]],
                        "n11": n11,
                        "jointly_known": n,
                    }
                )
            if n11 == 0 and (n11 + n10) > 0 and (n11 + n01) > 0:
                exclusions.append(
                    {
                        "class_ids": [ids[i], ids[j]],
                        "classes": [names[i], names[j]],
                        "positive_a": n10,
                        "positive_b": n01,
                        "jointly_known": n,
                    }
                )
        if (names[i], names[j]) in closure:
            refs = [o.ref for o in obs if o.states[i] == 1 and o.states[j] == 0]
            if refs:
                contradictions.append(
                    {
                        "kind": "subclass_positive_negative",
                        "subclass": names[i],
                        "superclass": names[j],
                        "count": len(refs),
                        "references": refs[:20],
                    }
                )
        if (names[j], names[i]) in closure:
            refs = [o.ref for o in obs if o.states[j] == 1 and o.states[i] == 0]
            if refs:
                contradictions.append(
                    {
                        "kind": "subclass_positive_negative",
                        "subclass": names[j],
                        "superclass": names[i],
                        "count": len(refs),
                        "references": refs[:20],
                    }
                )
        if tuple(sorted((names[i], names[j]))) in disjoint:
            refs = [o.ref for o in obs if o.states[i] == 1 and o.states[j] == 1]
            if refs:
                contradictions.append(
                    {
                        "kind": "disjoint_joint_positive",
                        "classes": [names[i], names[j]],
                        "count": len(refs),
                        "references": refs[:20],
                    }
                )
    return {
        "observed_empirical_implications": implications,
        "empirical_equivalences": equivalences,
        "empirical_exclusions": exclusions,
        "supported_axiom_contradictions": contradictions,
    }


def train_to_validation_implication_reversals(
    train_obs: list[Observation],
    val_obs: list[Observation],
    ids: Sequence[int],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    found = []
    for i, j in itertools.permutations(range(len(ids)), 2):
        train_ct = Counter(
            (o.states[i], o.states[j])
            for o in train_obs
            if o.states[i] in (0, 1) and o.states[j] in (0, 1)
        )
        val_ct = Counter(
            (o.states[i], o.states[j])
            for o in val_obs
            if o.states[i] in (0, 1) and o.states[j] in (0, 1)
        )
        if train_ct[1, 1] > 0 and train_ct[1, 0] == 0 and val_ct[1, 0] > 0:
            refs = [o.ref for o in val_obs if o.states[i] == 1 and o.states[j] == 0]
            found.append(
                {
                    "from_class_id": ids[i],
                    "to_class_id": ids[j],
                    "from_class": names[i],
                    "to_class": names[j],
                    "train_n11": train_ct[1, 1],
                    "train_jointly_known": sum(train_ct.values()),
                    "validation_n10": val_ct[1, 0],
                    "validation_jointly_known": sum(val_ct.values()),
                    "validation_counterexample_refs": refs[:20],
                }
            )
    return found


def reconcile_accounting(
    obs: list[Observation], class_rows_: list[dict[str, Any]], pair_rows_: list[dict[str, Any]]
) -> dict[str, Any]:
    marginals = {}
    for row in class_rows_:
        if row["view"] != "occurrence":
            continue
        counts = [
            row["known_negative_support"],
            row["known_positive_support"],
            row["unknown_support"],
            row["inapplicable_support"],
        ]
        if sum(counts) != len(obs):
            raise AssertionError("class supervision states do not reconcile with entity count")
        marginals[row["class_id"]] = counts
    for row in pair_rows_:
        matrix = json.loads(row["state_contingency"])
        if sum(map(sum, matrix)) != len(obs):
            raise AssertionError("pair state contingency does not reconcile with entity count")
        for state in range(4):
            if sum(matrix[state]) != marginals[row["class_id_a"]][state]:
                raise AssertionError("pair row marginal differs from class-state support")
            if (
                sum(matrix[source][state] for source in range(4))
                != marginals[row["class_id_b"]][state]
            ):
                raise AssertionError("pair column marginal differs from class-state support")
    return {
        "entity_occurrences": len(obs),
        "class_count": len(marginals),
        "class_state_totals_reconciled": True,
        "complete_pair_contingencies_reconciled": len(pair_rows_),
        "four_state_order": list(STATE_NAMES),
    }


def conflict_evidence(
    obs: list[Observation], ids: Sequence[int], names: Sequence[str]
) -> dict[str, Any]:
    signatures = defaultdict(list)
    for o in obs:
        signatures[o.features].append(o)
    conflicts = []
    errors = 0
    known = 0
    for sig, items in signatures.items():
        for j, (cid, name) in enumerate(zip(ids, names)):
            pos = [o.ref for o in items if o.states[j] == 1]
            neg = [o.ref for o in items if o.states[j] == 0]
            errors += min(len(pos), len(neg))
            known += len(pos) + len(neg)
            if pos and neg:
                conflicts.append(
                    {
                        "class_id": cid,
                        "class_name": name,
                        "signature": sig,
                        "positive_count": len(pos),
                        "negative_count": len(neg),
                        "positive_refs": pos[:10],
                        "negative_refs": neg[:10],
                    }
                )
    return {
        "conflicting_signature_class_pairs": len(conflicts),
        "signatures_with_conflict": len({tuple(x["signature"]) for x in conflicts}),
        "affected_classes": sorted({x["class_name"] for x in conflicts}),
        "known_label_support": known,
        "minimum_empirical_error_count": errors,
        "known_label_error_fraction": errors / known if known else None,
        "conflicts": conflicts,
    }


def accumulation(records, dataset, level: str, seeds=(7, 19, 41), intervals=10):
    rows = []
    for seed in seeds:
        order = list(records)
        random.Random(seed).shuffle(order)
        steps = (
            sorted(
                set(
                    [
                        1,
                        *(
                            max(1, round(i * len(order) / intervals))
                            for i in range(1, intervals + 1)
                        ),
                    ]
                )
            )
            if order
            else []
        )
        for n in steps:
            subset = order[:n]
            obs = observations(dataset, subset, level)
            rows.append(
                {
                    "entity_type": level,
                    "seed": seed,
                    "parent_pizzas": n,
                    "entity_occurrences": len(obs),
                    "distinct_signatures": len({o.features for o in obs}),
                    "distinct_supervision_patterns": len({o.states for o in obs}),
                }
            )
    return rows


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


def _write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_audit(data: Path, out: Path, *, allow_noninitial: bool = False) -> dict[str, Any]:
    data = data.resolve()
    out = out.resolve()
    manifest_bytes = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("manifest_id") != MANIFEST_ID:
        if not allow_noninitial:
            raise ValueError(
                "STAT-01 actual invocation requires the specified dataset ID and manifest identity"
            )
    # Existing loader skips test lines before JSON decoding. Do not call the full validator,
    # whose historical implementation inspects all samples.
    dataset, train, validation = load_train_validation(data)
    if dataset.dataset_id != manifest["dataset_id"]:
        raise ValueError("loader/manifest identity mismatch")
    vocab_bytes = (data / "vocabularies" / "ontology_classes.json").read_bytes()
    vocab_sha = hashlib.sha256(vocab_bytes).hexdigest()
    declared_vocab = manifest.get("vocabularies", {}).get("ontology_classes_sha256")
    if vocab_sha != dataset.class_manifest_sha256 or (
        declared_vocab and vocab_sha != declared_vocab
    ):
        raise ValueError("canonical vocabulary checksum mismatch")
    if dataset.manifest_id != MANIFEST_ID and not allow_noninitial:
        raise ValueError("expected manifest identity is absent")
    if len(dataset.canonical_class_ids) != 60:
        raise ValueError("unexpected canonical trainable class ordering")
    splits = {"train": train, "validation": validation}
    split_identity = {}
    for split, records in splits.items():
        split_path = data / "splits" / f"{split}.json"
        split_bytes = split_path.read_bytes()
        split_data = json.loads(split_bytes)
        expected_ids = split_data.get("sample_ids")
        actual_ids = [r.sample_id for r in records]
        if (
            not isinstance(expected_ids, list)
            or set(expected_ids) != set(actual_ids)
            or len(expected_ids) != len(actual_ids)
        ):
            raise ValueError(f"{split} records do not match the original selection manifest")
        split_identity[split] = {
            "selection_manifest_sha256": hashlib.sha256(split_bytes).hexdigest(),
            "ordered_sample_ids_sha256": hashlib.sha256(
                json.dumps(expected_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "pizza_count": len(records),
        }
    if dataset.dataset_id == DATASET_ID and (len(train) != 720 or len(validation) != 240):
        raise ValueError("required dataset partition sizes differ from 720/240")
    out.mkdir(parents=True, exist_ok=True)
    names = dataset.class_names
    ids = dataset.canonical_class_ids
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    relations = ontology_relations(ontology)
    summary = {
        "schema": SCHEMA,
        "ontology_reference_fragment": {
            "subclass": "Explicit named rdfs:subClassOf edges and their transitive named closure.",
            "disjointness": (
                "Explicit named owl:disjointWith edges inherited through the named "
                "subclass closure."
            ),
            "unsupported": (
                "Not a complete OWL reasoner; equivalent classes, restrictions, property "
                "axioms, and owl:AllDisjointClasses are not extracted as reference relations."
            ),
            "named_subclass_pair_count": len(relations[0]),
            "inherited_or_explicit_disjoint_pair_count": len(relations[1]),
        },
        "dataset_id": dataset.dataset_id,
        "manifest_id": dataset.manifest_id,
        "partitions": {},
        "pooled_train_validation": {},
        "limitations": [
            "Generator structural constraints are not established from the upstream source.",
            "Feature-association tables currently report occurrence-weighted distributions only.",
            "Fixed-positive and unsupported classes are separately tabulated, not mixed "
            "into trainable-class statistics.",
        ],
    }
    all_obs = {}
    class_table = []
    special_class_table = []
    pair_table = []
    accum = []
    feature_table = []
    feature_pairs = []
    for split, records in splits.items():
        summary["partitions"][split] = dict(split_identity[split])
        for level, feature_names in (
            ("pizza", PIZZA_FEATURE_NAMES),
            ("component", COMPONENT_FEATURE_NAMES),
        ):
            obs = observations(dataset, records, level)
            all_obs[(split, level)] = obs
            special_entries = [
                item
                for item in json.loads(vocab_bytes)["classes"]
                if item.get("fixed_positive") or item.get("unsupported")
            ]
            entities = []
            for record in records:
                entities.append((record.sample_id, record.pizza_targets, record.pizza_known_mask))
                if level == "component":
                    entities.extend(
                        (
                            f"{record.sample_id}/{component.component_id}",
                            component.ontology_targets,
                            component.ontology_known_mask,
                        )
                        for component in record.components
                    )
            for item in special_entries:
                cid = item["class_id"]
                targets = [target[cid] for _, target, _ in entities]
                masks = [mask[cid] for _, _, mask in entities]
                special_class_table.append(
                    {
                        "partition": split,
                        "entity_type": level,
                        "class_id": cid,
                        "class_name": item["name"],
                        "status": item["supervision_status"],
                        "entity_occurrences": len(entities),
                        "target_positive_count": sum(targets),
                        "target_negative_count": len(targets) - sum(targets),
                        "known_mask_count": sum(masks),
                        "unknown_mask_count": len(masks) - sum(masks),
                        "interpretation": (
                            "metadata only; excluded from trainable-class supervision statistics"
                        ),
                    }
                )
            summary["partitions"][split][level] = diversity(obs, feature_names)
            summary["partitions"][split][level]["feature_label_conflicts"] = conflict_evidence(
                obs, ids, names
            )
            summary["partitions"][split][level]["ontology_empirical_relations"] = (
                empirical_relations(obs, ids, names, relations, level)
            )
            class_results = class_rows(obs, ids, names, level)
            pair_results = pair_rows(obs, ids, names, level, relations)
            summary["partitions"][split][level]["accounting"] = reconcile_accounting(
                obs, class_results, pair_results
            )
            class_table.extend(dict(partition=split, **r) for r in class_results)
            frows, fpairs = feature_associations(obs, feature_names, ids, names, level)
            feature_table.extend(dict(partition=split, **r) for r in frows)
            feature_pairs.extend(dict(partition=split, **r) for r in fpairs)
            pair_table.extend(dict(partition=split, **r) for r in pair_results)
            accum.extend(accumulation(records, dataset, level))
            sig_groups = defaultdict(list)
            for o in obs:
                sig_groups[o.features].append(o)
            with (out / f"signatures-{split}-{level}.jsonl").open("w", encoding="utf-8") as f:
                for sig, items in sorted(sig_groups.items(), key=lambda kv: kv[0]):
                    f.write(
                        json.dumps(
                            {
                                "feature_signature": sig,
                                "occurrence_count": len(items),
                                "supervision_pattern_counts": {
                                    json.dumps(k): v
                                    for k, v in Counter(x.states for x in items).items()
                                },
                                "observations": [
                                    {"ref": x.ref, "parent": x.parent, "states": x.states}
                                    for x in items
                                ],
                            },
                            default=_json_default,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    for level in ("pizza", "component"):
        for split in ("train", "validation"):
            summary["partitions"][split][level][
                "feature_signature_overlap_with_other_partition"
            ] = len(
                {x.features for x in all_obs[(split, level)]}
                & {
                    x.features
                    for x in all_obs[("validation" if split == "train" else "train", level)]
                }
            )
    summary["train_validation_comparison"] = {}
    for level in ("pizza", "component"):
        a, b = all_obs[("train", level)], all_obs[("validation", level)]
        summary["train_validation_comparison"][level] = {
            "signature_overlap": len({x.features for x in a} & {x.features for x in b}),
            "train_only_signatures": len({x.features for x in a} - {x.features for x in b}),
            "validation_only_signatures": len({x.features for x in b} - {x.features for x in a}),
            "signature_js_nats": js_divergence(
                Counter(x.features for x in a), Counter(x.features for x in b)
            ),
            "pattern_js_nats": js_divergence(
                Counter(x.states for x in a), Counter(x.states for x in b)
            ),
            "supervision_pattern_overlap": len({x.states for x in a} & {x.states for x in b}),
            "train_only_supervision_patterns": len({x.states for x in a} - {x.states for x in b}),
            "validation_only_supervision_patterns": len(
                {x.states for x in b} - {x.states for x in a}
            ),
            "train_implications_with_validation_counterexamples": (
                train_to_validation_implication_reversals(a, b, ids, names)
            ),
        }
    summary["diagnostic_flags"] = {}
    for split in ("train", "validation"):
        summary["diagnostic_flags"][split] = {}
        for level in ("pizza", "component"):
            class_evidence = [
                row
                for row in class_table
                if row["partition"] == split
                and row["entity_type"] == level
                and row["view"] == "occurrence"
            ]
            pair_evidence = [
                row
                for row in pair_table
                if row["partition"] == split and row["entity_type"] == level
            ]
            summary["diagnostic_flags"][split][level] = {
                "exact_feature_label_conflict": {
                    "active": summary["partitions"][split][level]["feature_label_conflicts"][
                        "conflicting_signature_class_pairs"
                    ]
                    > 0,
                    "evidence": "feature_label_conflicts in summary.json and signature JSONL",
                },
                "zero_known_support_classes": [
                    {
                        "class_id": row["class_id"],
                        "known_denominator": 0,
                        "evidence_row": "classes.csv",
                    }
                    for row in class_evidence
                    if row["known_denominator"] == 0
                ],
                "no_known_positive_observed": [
                    {
                        "class_id": row["class_id"],
                        "known_positive_support": 0,
                        "known_denominator": row["known_denominator"],
                        "interpretation": "does not establish logical emptiness",
                        "evidence_row": "classes.csv",
                    }
                    for row in class_evidence
                    if row["known_denominator"] > 0 and row["known_positive_support"] == 0
                ],
                "missing_binary_contrast_cells": [
                    {
                        "class_id_a": row["class_id_a"],
                        "class_id_b": row["class_id_b"],
                        "missing_n10": row["occurrence_n10"] == 0,
                        "missing_n01": row["occurrence_n01"] == 0,
                        "jointly_known": row["occurrence_jointly_known"],
                        "evidence_row": "class_pairs.csv",
                    }
                    for row in pair_evidence
                    if row["occurrence_jointly_known"] > 0
                    and (row["occurrence_n10"] == 0 or row["occurrence_n01"] == 0)
                ],
                "empirical_relation_counts": {
                    key: len(
                        summary["partitions"][split][level]["ontology_empirical_relations"][key]
                    )
                    for key in (
                        "observed_empirical_implications",
                        "empirical_equivalences",
                        "empirical_exclusions",
                        "supported_axiom_contradictions",
                    )
                },
                "signature_concentration": {
                    "max_occurrence_share": summary["partitions"][split][level][
                        "feature_signature_max_occurrence_share"
                    ],
                    "distinct_signatures": summary["partitions"][split][level][
                        "feature_signatures"
                    ]["distinct"],
                    "occurrences": summary["partitions"][split][level]["entity_occurrences"],
                    "heuristic_threshold": None,
                },
            }
    for level in ("pizza", "component"):
        pooled = all_obs[("train", level)] + all_obs[("validation", level)]
        summary["pooled_train_validation"][level] = conflict_evidence(pooled, ids, names)
    _write_csv(out / "classes.csv", class_table)
    _write_csv(out / "special_classes.csv", special_class_table)
    _write_csv(out / "class_pairs.csv", pair_table)
    _write_csv(out / "accumulation.csv", accum)
    _write_csv(out / "feature_associations.csv", feature_table)
    _write_csv(out / "feature_pairs.csv", feature_pairs)
    comparison_rows = []
    class_map = {
        (r["partition"], r["entity_type"], r["class_id"], r["view"]): r for r in class_table
    }
    for level in ("pizza", "component"):
        for cid, name in zip(ids, names):
            for view in ("occurrence", "parent", "signature"):
                a = class_map[("train", level, cid, view)]
                b = class_map[("validation", level, cid, view)]
                comparison_rows.append(
                    {
                        "entity_type": level,
                        "measurement": "class_prevalence",
                        "identity_a": cid,
                        "identity_b": name,
                        "view": view,
                        "train_value": a["positive_prevalence"],
                        "validation_value": b["positive_prevalence"],
                        "difference_validation_minus_train": (
                            b["positive_prevalence"] - a["positive_prevalence"]
                        )
                        if a["positive_prevalence"] is not None
                        and b["positive_prevalence"] is not None
                        else None,
                        "train_known_denominator": a["known_denominator"],
                        "validation_known_denominator": b["known_denominator"],
                    }
                )
        for feature in PIZZA_FEATURE_NAMES if level == "pizza" else COMPONENT_FEATURE_NAMES:
            index = (PIZZA_FEATURE_NAMES if level == "pizza" else COMPONENT_FEATURE_NAMES).index(
                feature
            )
            ca = Counter(o.features[index] for o in all_obs[("train", level)])
            cb = Counter(o.features[index] for o in all_obs[("validation", level)])
            comparison_rows.append(
                {
                    "entity_type": level,
                    "measurement": "feature_value_distribution_js",
                    "identity_a": feature,
                    "identity_b": "",
                    "view": "occurrence",
                    "train_value": None,
                    "validation_value": None,
                    "difference_validation_minus_train": js_divergence(ca, cb),
                    "train_known_denominator": sum(ca.values()),
                    "validation_known_denominator": sum(cb.values()),
                }
            )
    feature_pair_map = {
        (r["partition"], r["entity_type"], r["feature_a"], r["feature_b"]): r for r in feature_pairs
    }
    for level in ("pizza", "component"):
        names_for_features = PIZZA_FEATURE_NAMES if level == "pizza" else COMPONENT_FEATURE_NAMES
        for feature_a, feature_b in itertools.combinations(names_for_features, 2):
            a = feature_pair_map[("train", level, feature_a, feature_b)]
            b = feature_pair_map[("validation", level, feature_a, feature_b)]
            av = a["mutual_information_nats"]
            bv = b["mutual_information_nats"]
            comparison_rows.append(
                {
                    "entity_type": level,
                    "measurement": "feature_pair_mutual_information_nats",
                    "identity_a": feature_a,
                    "identity_b": feature_b,
                    "view": "occurrence",
                    "train_value": av,
                    "validation_value": bv,
                    "difference_validation_minus_train": bv - av
                    if av is not None and bv is not None
                    else None,
                    "train_known_denominator": len(all_obs[("train", level)]),
                    "validation_known_denominator": len(all_obs[("validation", level)]),
                }
            )
    pair_map = {
        (r["partition"], r["entity_type"], r["class_id_a"], r["class_id_b"]): r for r in pair_table
    }
    for level in ("pizza", "component"):
        for i, j in itertools.combinations(range(len(ids)), 2):
            a = pair_map[("train", level, ids[i], ids[j])]
            b = pair_map[("validation", level, ids[i], ids[j])]
            for metric in (
                "p_b_given_a",
                "p_a_given_b",
                "jaccard",
                "disagreement",
                "phi",
                "mutual_information_nats",
                "positive_lift",
            ):
                av = a.get("occurrence_" + metric)
                bv = b.get("occurrence_" + metric)
                comparison_rows.append(
                    {
                        "entity_type": level,
                        "measurement": "pair_" + metric,
                        "identity_a": ids[i],
                        "identity_b": ids[j],
                        "view": "occurrence",
                        "train_value": av,
                        "validation_value": bv,
                        "difference_validation_minus_train": bv - av
                        if av is not None and bv is not None
                        else None,
                        "train_known_denominator": a["occurrence_jointly_known"],
                        "validation_known_denominator": b["occurrence_jointly_known"],
                    }
                )
    _write_csv(out / "partition_differences.csv", comparison_rows)
    pizza_obs = all_obs[("train", "pizza")] + all_obs[("validation", "pizza")]
    observed_combos = Counter(o.topping_combo for o in pizza_obs)
    recipe_combinations = defaultdict(Counter)
    for split_records in splits.values():
        for record in split_records:
            raw = record.audit.get("raw_generator", {})
            combo = tuple(sorted(k for k, v in raw.get("topping_counts", {}).items() if v))
            recipe_combinations[record.generator_type][combo] += 1
    topping_instance_bounds = {
        "AnchovyTopping": [3, 5],
        "GarlicTopping": [3, 6],
        "GreenPepperTopping": [5, 7],
        "HamTopping": [3, 4],
        "JalapenoPepperTopping": [2, 4],
        "MushroomTopping": [6, 10],
        "OliveTopping": [6, 10],
        "OnionTopping": [3, 5],
        "ParmesanTopping": [1, 1],
        "PeperonataTopping": [1, 1],
        "PeperoniSausageTopping": [5, 7],
        "PrawnTopping": [5, 8],
        "RocketTopping": [2, 4],
        "SlicedTomatoTopping": [4, 6],
        "SpinachTopping": [2, 4],
        "SultanaTopping": [6, 10],
    }
    coverage = {
        "generator_source": manifest.get("source"),
        "pinned_generator_revision_inspected": "6c0bcb221a957c184012beaf103bebcde4b1f793",
        "generator_rules": {
            "named_recipe": (
                "pinned preparePizza uses ontology class hasTopping/hasBase/hasCountryOfOrigin"
            ),
            "random_recipe": (
                "pinned preparePizza chooses set(choices(toppings, k=randrange(1,6))); "
                "1-5 unique types; base among PizzaBase descendants; origin one of four "
                "named countries or None"
            ),
            "topping_instance_bounds": topping_instance_bounds,
            "bound_interpretation": (
                "Number of sprite placements sampled per type before clipping; saved "
                "visible box/count can differ after overlap/clipping."
            ),
            "retry_rule": (
                "Pinned generator recursively retries when topping overlap rate exceeds 0.40"
            ),
        },
        "observed_combinations_by_generator_type": {
            key: [
                {"combination": list(combo), "count": count}
                for combo, count in sorted(value.items())
            ]
            for key, value in sorted(recipe_combinations.items())
        },
        "configured_generator_types": manifest.get("resolved_config", {}).get("pizza_types", []),
        "configured_generator_type_counts": manifest.get("resolved_config", {}).get(
            "pizza_type_counts", {}
        ),
        "observed_generator_type_counts": dict(
            Counter(r.generator_type for split in splits.values() for r in split)
        ),
        "available_topping_categories": list(TOPPING_COLUMNS),
        "available_base_categories": list(BASE_CATEGORIES),
        "available_origin_categories": [*FLAG_INDIVIDUALS, "<missing>"],
        "observed_topping_count_distributions": {
            name: dict(Counter(o.features[i] for o in pizza_obs))
            for i, name in enumerate(TOPPING_COLUMNS)
        },
        "observed_topping_combinations": [
            {"combination": list(k), "count": v} for k, v in sorted(observed_combos.items())
        ],
        "observed_bases": {
            name: sum(o.features[len(TOPPING_COLUMNS) + i] for o in pizza_obs)
            for i, name in enumerate(BASE_CATEGORIES)
        },
        "observed_origins": {
            name: sum(
                o.features[len(TOPPING_COLUMNS) + len(BASE_CATEGORIES) + i] for o in pizza_obs
            )
            for i, name in enumerate((*FLAG_INDIVIDUALS, "<missing>"))
        },
        "coverage_gaps": {
            "topping_categories_supported_but_unobserved": [
                name
                for i, name in enumerate(TOPPING_COLUMNS)
                if not any(o.features[i] > 0 for o in pizza_obs)
            ],
            "base_categories_supported_but_unobserved": [
                name
                for i, name in enumerate(BASE_CATEGORIES)
                if not any(o.features[len(TOPPING_COLUMNS) + i] > 0 for o in pizza_obs)
            ],
            "origin_categories_supported_but_unobserved": [
                name
                for i, name in enumerate((*FLAG_INDIVIDUALS, "<missing>"))
                if not any(
                    o.features[len(TOPPING_COLUMNS) + len(BASE_CATEGORIES) + i] > 0
                    for o in pizza_obs
                )
            ],
            "evidence": "Pinned generator categories; no train/validation observation; "
            "not a claim about all output possibilities.",
        },
        "structural_constraints": (
            "Named recipes use ontology-declared ingredients. Random recipe topping sets "
            "contain 1-5 unique types and use the listed possible categories; exact joint "
            "generator output is not exhaustively established."
        ),
        "unobserved_combinations": (
            "Not exhaustively enumerated; unsupported possibilities are classified as "
            "possibility not established."
        ),
        "coverage_limit": (
            "Finite train/validation descriptions only; no generated samples or sealed "
            "test records used."
        ),
    }
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2, default=_json_default) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")

    report_lines = [
        "# STAT-01 evidence summary",
        f"Dataset: {dataset.dataset_id}; manifest: {dataset.manifest_id}.",
        "Only training and validation were analyzed; test lines were skipped before JSON decoding.",
        "",
    ]
    for split in ("train", "validation"):
        for level in ("pizza", "component"):
            result = summary["partitions"][split][level]
            err = result["feature_label_conflicts"]
            rel = result["ontology_empirical_relations"]
            report_lines.append(
                f"{split}/{level}: entities={result['entity_occurrences']}, "
                f"signatures={result['feature_signatures']['distinct']}, "
                f"patterns={result['supervision_patterns']['distinct']}, "
                f"E_min={err['minimum_empirical_error_count']}/{err['known_label_support']}, "
                f"implications/equivalences/exclusions="
                f"{len(rel['observed_empirical_implications'])}/"
                f"{len(rel['empirical_equivalences'])}/{len(rel['empirical_exclusions'])}, "
                f"axiom contradictions={len(rel['supported_axiom_contradictions'])}"
            )
    for level, result in summary["train_validation_comparison"].items():
        report_lines.append(
            f"{level} train/validation: signature overlap={result['signature_overlap']}, "
            f"train-only={result['train_only_signatures']}, "
            f"validation-only={result['validation_only_signatures']}, "
            f"JS={result['signature_js_nats']} nats"
        )
    report_lines.extend(
        [
            "",
            "See summary.json, classes.csv, class_pairs.csv, feature_associations.csv, "
            "feature_pairs.csv, coverage.json, and signature JSONL for primary evidence.",
            "Empirical implications are finite-sample patterns, not global ontology claims. "
            "Feature-association evidence is occurrence-weighted; validation is development data.",
        ]
    )
    (out / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "tests"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    artifacts = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file() and p.name != "manifest.json"
    }
    info = {
        "schema": SCHEMA,
        "implementation_version": IMPLEMENTATION,
        "source_sha": source,
        "source_dirty": bool(dirty),
        "dataset": {
            "id": dataset.dataset_id,
            "manifest_id": dataset.manifest_id,
            "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "vocabulary_sha256": dataset.class_manifest_sha256,
        "ontology_sha256": manifest["ontology"]["sha256"],
        "feature_schema": FEATURE_SCHEMA,
        "included_partitions": {
            "train": summary["partitions"]["train"],
            "validation": summary["partitions"]["validation"],
        },
        "test_policy": (
            "Sealed: input lines are discarded by split string before JSON decode; no test "
            "observations/statistics/signatures used."
        ),
        "weights": {
            "occurrence": "one per observation",
            "parent": (
                "components receive 1/count per parent; conditional views renormalize "
                "eligible observations per eligible parent"
            ),
            "signature": (
                "Each signature total mass 1; state conflicts fractionally retained; "
                "conditional views normalize eligible observations within each eligible signature."
            ),
            "parent_balanced_is_not_training_loss": True,
        },
        "logarithm": "natural; entropy and JS in nats",
        "undefined": "JSON null; denominator/reason retained in associated fields",
        "accumulation": {
            "seeds": [7, 19, 41],
            "permutations_per_partition_entity_type": 3,
            "intervals": 10,
        },
        "artifact_sha256": artifacts,
    }
    (out / "manifest.json").write_text(json.dumps(info, indent=2, default=_json_default) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser(description="Offline STAT-01 train/validation dataset audit")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-compatible-dataset",
        action="store_true",
        help="Permit a compatible non-initial dataset (not for the STAT-01 initial invocation).",
    )
    a = p.parse_args()
    run_audit(a.dataset, a.output, allow_noninitial=a.allow_compatible_dataset)


if __name__ == "__main__":
    main()
