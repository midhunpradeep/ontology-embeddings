"""Independent recomputation and preservation checks for REL-01 observed-v2."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def recompute(evaluation: dict[str, Any], population: dict[str, Any]) -> dict[str, Any]:
    """Sum selected per-pair fields directly, without the production aggregator."""
    names: dict[str, int] = {}
    for row in population["pairs"]:
        names[row["a_name"]] = row["a_id"]
        names[row["b_name"]] = row["b_id"]
    references = evaluation["reference"]
    directed = {(names[a], names[b]) for a, b in references["subclass"]}
    unordered = {frozenset((names[a], names[b])) for a, b in references["disjoint"]}
    records = {(row["a_id"], row["b_id"]): row for row in population["pairs"]}
    cross_supervision = {}
    for unit in ("occurrence_weighted", "distinct_embedding"):
        s_cross = d_cross = d_opportunities = 0
        for child, parent in directed:
            key = (child, parent) if child < parent else (parent, child)
            row = records[key]
            counter = "a_to_b" if row["a_id"] == child else "b_to_a"
            s_cross += row["unrestricted_violations_outside_joint_supervision"][unit][counter]
        for pair in unordered:
            row = records[tuple(sorted(pair))]
            counts = row["unrestricted_violations_outside_joint_supervision"][unit]
            d_cross += counts["disjointness"]
            d_opportunities += counts["eligible_entity_count"]
        cross_supervision[unit] = {
            "reference_subsumption_violation_count": s_cross,
            "reference_disjointness_violation_count": d_cross,
            "disjointness_cross_supervision_eligible_opportunities": d_opportunities,
        }
    answer: dict[str, Any] = {}
    for view in ("supervision_restricted", "unrestricted"):
        answer[view] = {}
        for unit in ("occurrence_weighted", "distinct_embedding"):
            s_violation = s_support = d_violation = d_denominator = 0
            for child, parent in directed:
                key = (child, parent) if child < parent else (parent, child)
                row = records[key]
                stat_key = "subsumption_a_to_b" if row["a_id"] == child else "subsumption_b_to_a"
                stat = row[view][unit][stat_key]
                s_violation += stat["violation_count"]
                s_support += stat["antecedent_support"]
            for pair in unordered:
                key = tuple(sorted(pair))
                stat = records[key][view][unit]["disjointness"]
                d_violation += stat["violation_count"]
                d_denominator += stat["denominator"]
            answer[view][unit] = {
                "subsumption_violations": s_violation,
                "subsumption_antecedent_support": s_support,
                "subsumption_rate": s_violation / s_support if s_support else None,
                "disjointness_violations": d_violation,
                "disjointness_eligible_opportunities": d_denominator,
                "disjointness_rate": d_violation / d_denominator if d_denominator else None,
            }
    answer["violations_outside_joint_supervision"] = cross_supervision
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, corrected = load(args.source), load(args.corrected)
    source_evals = source["evaluations"]
    corrected_evals = corrected["evaluations"]
    if len(source_evals) != 6 or len(corrected_evals) != 6:
        raise AssertionError("expected six checkpoint evaluations")
    recomputed = []
    for before, after in zip(source_evals, corrected_evals, strict=True):
        if before["metadata"]["checkpoint_sha256"] != after["metadata"]["checkpoint_sha256"]:
            raise AssertionError("checkpoint identity changed")
        if before["summary"]["relations"] != after["summary"]["relations"]:
            raise AssertionError("published global relation counts changed")
        if before["pairs"] != after["pairs"]:
            raise AssertionError("global per-pair geometry/witness evidence changed")
        for split in ("train", "validation"):
            for entity in ("pizza", "component"):
                old = before["observed"]["populations"][split][entity]
                new = after["observed"]["populations"][split][entity]
                if old["pairs"] != new["pairs"] or old["summary"] != new["summary"]:
                    raise AssertionError("original observed per-pair/all-candidate results changed")
                independent = recompute(before, old)
                for unit, values in independent["violations_outside_joint_supervision"].items():
                    published_cross = new["reference_only_summary"][
                        "violations_outside_joint_supervision"
                    ][unit]
                    if published_cross != values:
                        raise AssertionError("reference cross-supervision filter mismatch")
                for view, units in independent.items():
                    if view == "violations_outside_joint_supervision":
                        continue
                    for unit, values in units.items():
                        published = new["reference_only_summary"]["measurements"][view][unit]
                        checks = {
                            "subsumption_violation_count": values["subsumption_violations"],
                            "subsumption_antecedent_support": values[
                                "subsumption_antecedent_support"
                            ],
                            "subsumption_conditional_violation_rate": values["subsumption_rate"],
                            "disjointness_violation_count": values["disjointness_violations"],
                            "disjointness_eligible_population_denominator": values[
                                "disjointness_eligible_opportunities"
                            ],
                            "disjointness_conditional_violation_rate": values["disjointness_rate"],
                        }
                        if any(published[key] != value for key, value in checks.items()):
                            raise AssertionError(
                                "independent reference aggregate mismatch: "
                                f"{after['metadata']['run_id']} "
                                f"{after['metadata']['label']} {split}/{entity} {view}/{unit}"
                            )
                recomputed.append(
                    {
                        "run_id": after["metadata"]["run_id"],
                        "checkpoint": after["metadata"]["label"],
                        "split": split,
                        "entity": entity,
                        "measurements": independent,
                    }
                )
    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    result = {
        "status": "PASS",
        "verification_provenance": {
            "entry_point": "python scripts/verify_rel01_reference_observed.py",
            "arguments": {
                "--source": args.source.name,
                "--corrected": args.corrected.name,
                "--output": args.output.name,
            },
            "producing_source_revision": source_revision,
        },
        "independent_method": (
            "directly sum reference-filtered per-pair records; no production aggregation helper"
        ),
        "checkpoint_populations_recomputed": len(recomputed),
        "all_candidate_summaries_unchanged": True,
        "global_counts_pairs_and_witness_records_unchanged": True,
        "measurements": recomputed,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "PASS", "populations": len(recomputed)}))


if __name__ == "__main__":
    main()
