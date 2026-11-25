"""Command-line entry point for REL-01 checkpoint-only geometry evaluation."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .dataset.image_data import load_prepared_dataset
from .observed_relations import verify_historical_dataset
from .relation_evaluation import evaluate_checkpoint

RUNS = {
    "20261125T205910-8eac8a31db0a": [("initial", 0), ("best", 45), ("latest", 60)],
    "20261125T212135-1c0e34f5174a": [("initial", 0), ("best", 54), ("latest", 60)],
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--dataset", type=Path, help="verified original dataset; enables observed mode")
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    root = a.evidence.resolve()
    dataset = None
    dataset_identity = None
    if a.dataset is not None:
        dataset = load_prepared_dataset(a.dataset, validate=True)
        dataset_identity = verify_historical_dataset(dataset)
    results = []
    for run, files in RUNS.items():
        for name, epoch in files:
            path = root / "runs" / run / "checkpoints" / f"{name}.pt"
            item = evaluate_checkpoint(
                path,
                evidence_root=root,
                run_id=run,
                label=name,
                observed_dataset=dataset,
                observed_dataset_identity=dataset_identity,
            )
            if item["metadata"]["epoch"] != epoch:
                raise ValueError(f"expected epoch {epoch}, got {item['metadata']['epoch']}")
            results.append(item)
    transitions = []
    for run in RUNS:
        group = [r for r in results if r["metadata"]["run_id"] == run]
        for left, right in ((group[0], group[1]), (group[1], group[2]), (group[0], group[2])):
            changed = {}
            for kind in ("directed", "unordered"):

                def predicate(row):
                    return (
                        row["contained"]
                        if kind == "directed"
                        else row["intersection_status"] == "empty"
                    )

                def reference(row):
                    return (
                        row["reference_subclass"]
                        if kind == "directed"
                        else row["reference_disjoint"]
                    )

                previous = {(x["a_id"], x["b_id"]): x for x in left["pairs"] if x["kind"] == kind}
                current = {(x["a_id"], x["b_id"]): x for x in right["pairs"] if x["kind"] == kind}
                lk = {key for key, row in previous.items() if predicate(row)}
                rk = {key for key, row in current.items() if predicate(row)}
                changed[kind] = {
                    "gained": len(rk - lk),
                    "lost": len(lk - rk),
                    "retained": len(lk & rk),
                    "reference_satisfaction_gained": sum(reference(current[k]) for k in rk - lk),
                    "reference_satisfaction_lost": sum(reference(previous[k]) for k in lk - rk),
                }
            transition = {
                "run_id": run,
                "from": left["metadata"]["label"],
                "to": right["metadata"]["label"],
                "relations": changed,
            }
            if left["observed"] is not None and right["observed"] is not None:
                deltas = {}
                for split in ("train", "validation"):
                    deltas[split] = {}
                    for entity in ("pizza", "component"):
                        deltas[split][entity] = {}
                        old = left["observed"]["populations"][split][entity]["summary"]
                        new = right["observed"]["populations"][split][entity]["summary"]
                        for view in ("supervision_restricted", "unrestricted"):
                            deltas[split][entity][view] = {}
                            for unit in ("occurrence_weighted", "distinct_embedding"):
                                old_stats = old[view][unit]
                                new_stats = new[view][unit]
                                deltas[split][entity][view][unit] = {
                                    key: (new_stats[key] - old_stats[key])
                                    if old_stats[key] is not None and new_stats[key] is not None
                                    else None
                                    for key in (
                                        "directed_subsumption_violation_count",
                                        "directed_subsumption_antecedent_support",
                                        "conditional_violation_rate",
                                        "disjointness_violation_count",
                                        "disjointness_denominator",
                                        "disjointness_violation_rate",
                                    )
                                }
                transition["observed_consistency_delta"] = deltas
            else:
                transition["observed_consistency_delta"] = None
            transitions.append(transition)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "schema": "rel-01-bundle-v2" if dataset is not None else "rel-01-bundle-v1",
        "transitions": transitions,
        "evaluations": results,
    }
    encoded = json.dumps(bundle, indent=2, allow_nan=False) + "\n"
    if a.output.suffix == ".gz":
        with gzip.open(a.output, "wt", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        a.output.write_text(encoded)
    print(
        json.dumps(
            [
                {
                    "run": r["metadata"]["run_id"],
                    "checkpoint": r["metadata"]["label"],
                    "epoch": r["metadata"]["epoch"],
                    "relations": r["summary"]["relations"],
                    "subsumption": r["summary"]["subsumption"],
                    "disjointness": r["summary"]["disjointness"],
                    "observed": (
                        {
                            split: {
                                entity: population["summary"]
                                for entity, population in entities.items()
                            }
                            for split, entities in r["observed"]["populations"].items()
                        }
                        if r["observed"] is not None
                        else None
                    ),
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
