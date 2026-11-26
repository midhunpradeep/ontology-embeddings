"""Create a non-overwriting corrected REL-01 evidence bundle from stored pair records."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from ontology_embeddings.observed_relations import _aggregate_pair_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    with gzip.open(args.input, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)

    for evaluation in bundle["evaluations"]:
        reference = evaluation["reference"]
        name_to_id: dict[str, int] = {}
        for row in evaluation["observed"]["populations"]["train"]["pizza"]["pairs"]:
            name_to_id[row["a_name"]] = row["a_id"]
            name_to_id[row["b_name"]] = row["b_id"]
        subclass_ids = {(name_to_id[a], name_to_id[b]) for a, b in reference["subclass"]}
        disjoint_ids = {
            tuple(sorted((name_to_id[a], name_to_id[b]))) for a, b in reference["disjoint"]
        }
        for split, populations in evaluation["observed"]["populations"].items():
            for entity, population in populations.items():
                population["summary_population_identity"] = "all_candidate_class_pairs"
                population["summary_measurement_interpretation"] = (
                    "descriptive observed relation statistics; not all candidate relations "
                    "are ontology requirements"
                )
                population["reference_only_summary"] = _aggregate_pair_records(
                    population["pairs"], subclass_ids, disjoint_ids
                )
                population["all_candidate_cross_supervision_population_identity"] = (
                    "all_candidate_class_pairs"
                )
                population["reference_only_cross_supervision_population_identity"] = (
                    "named_fragment_reference_axioms"
                )
                population["reference_only_summary"]["population_coordinates"] = {
                    "split": split,
                    "entity_type": entity,
                    "unit_separation": ["occurrence_weighted", "distinct_embedding"],
                    "supervision_views": ["supervision_restricted", "unrestricted"],
                }
        meta = evaluation["metadata"]
        meta.pop("invocation_template", None)
        meta["original_evaluation_invocation"] = {
            "status": "exact_historical_argv_not_recoverable_from_retained_logs_or_run_records",
            "entry_point": "python -m ontology_embeddings.rel01_cli",
            "evaluation_mode": "global_and_observed",
            "arguments": {
                "evidence": "verified BOX-R01 checkpoint bundle (private path omitted)",
                "output": "evidence/rel-01-observed-v1/results.json.gz",
                "--dataset": {
                    "provided": True,
                    "dataset_id": evaluation["metadata"]["dataset"]["dataset_id"],
                    "local_path": "omitted",
                },
            },
            "checkpoint_evidence_identity": {
                "run_id": meta["run_id"],
                "label": meta["label"],
                "epoch": meta["epoch"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
            },
            "dataset_identity": meta["dataset"],
            "producing_source_revision": meta["evaluator_revision"],
            "note": (
                "Values describe the producing operation from retained evidence; "
                "exact argv was not reconstructed or invented."
            ),
        }

    # Add the matching reference-only transition summaries while retaining the
    # existing all-candidate deltas byte-for-byte as field values.
    evaluations = {
        (x["metadata"]["run_id"], x["metadata"]["label"]): x for x in bundle["evaluations"]
    }
    for transition in bundle["transitions"]:
        left = evaluations[(transition["run_id"], transition["from"])]
        right = evaluations[(transition["run_id"], transition["to"])]
        deltas = {}
        for split in ("train", "validation"):
            deltas[split] = {}
            for entity in ("pizza", "component"):
                deltas[split][entity] = {}
                for view in ("supervision_restricted", "unrestricted"):
                    deltas[split][entity][view] = {}
                    for unit in ("occurrence_weighted", "distinct_embedding"):
                        old = left["observed"]["populations"][split][entity][
                            "reference_only_summary"
                        ]["measurements"][view][unit]
                        new = right["observed"]["populations"][split][entity][
                            "reference_only_summary"
                        ]["measurements"][view][unit]
                        deltas[split][entity][view][unit] = {
                            key: (new[key] - old[key])
                            if isinstance(old[key], (int, float))
                            and isinstance(new[key], (int, float))
                            else None
                            for key in (
                                "subsumption_violation_count",
                                "subsumption_antecedent_support",
                                "subsumption_conditional_violation_rate",
                                "disjointness_violation_count",
                                "disjointness_eligible_population_denominator",
                                "disjointness_conditional_violation_rate",
                            )
                        }
        transition["reference_only_observed_consistency_delta"] = {
            "population_identity": "named_fragment_reference_axioms",
            "populations": deltas,
        }

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    bundle["schema"] = "rel-01-observed-bundle-v2"
    bundle["correction_provenance"] = {
        "operation": "reference-only aggregation from existing complete per-pair records",
        "entry_point": "python scripts/rel01_reference_observed_correction.py",
        "arguments": {
            "--input": "evidence/rel-01-observed-v1/results.json.gz",
            "--output": "evidence/rel-01-observed-v2/results.json.gz",
        },
        "checkpoint_evaluation_rerun": False,
        "dataset_access": False,
        "model_inference": False,
        "producing_source_revision": source_revision,
        "source_evidence_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, allow_nan=False)
        stream.write("\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    (args.output.parent / "SHA256SUMS").write_text(f"{digest}  results.json.gz\n")
    print(
        json.dumps(
            {
                "output": args.output.name,
                "sha256": digest,
                "evaluations": len(bundle["evaluations"]),
            }
        )
    )


if __name__ == "__main__":
    main()
