"""Small offline evaluator for stored single-box BOX checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch

from .box import box_membership, validate_endpoints
from .box_training import BoxMultiLabelModel
from .dataset.constants import FIXED_POSITIVE_CLASS
from .dataset.ontology import OntologySnapshot, build_class_vocabulary
from .features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from .relation_geometry import directed, unordered


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference(
    ontology: OntologySnapshot, names: set[str]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    sub = set()
    for child in names:
        sub.update(
            (child, parent)
            for parent in ontology.ancestors(child, include_self=False)
            if parent in names
        )
    asserted = set()
    for name, elem in ontology.class_elements.items():
        if name not in names:
            continue
        for e in elem.findall("{http://www.w3.org/2002/07/owl#}disjointWith"):
            ref = e.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource")
            if ref:
                target = ref.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
                if target in ontology.class_iris and target != name:
                    asserted.add(tuple(sorted((name, target))))
    rdf = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    owl = "http://www.w3.org/2002/07/owl#"
    root = ET.parse(ontology.path).getroot()
    for group in root.findall(f"{{{owl}}}AllDisjointClasses/{{{owl}}}members"):
        members = []
        for node in list(group):
            iri = node.get(f"{{{rdf}}}about")
            local = iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1] if iri else ""
            if local in ontology.class_iris:
                members.append(local)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                asserted.add(tuple(sorted((a, b))))
    # Named subclass closure transports disjointness to all descendants.
    dis = set()
    for a, b in asserted:
        descendants_a = {x for x in ontology.class_iris if a in ontology.ancestors(x)}
        descendants_b = {x for x in ontology.class_iris if b in ontology.ancestors(x)}
        for x in descendants_a & names:
            for y in descendants_b & names:
                if x != y:
                    dis.add(tuple(sorted((x, y))))
    return sub, dis


def evaluate_checkpoint(
    checkpoint: Path, *, evidence_root: Path, run_id: str, label: str
) -> dict[str, Any]:
    digest = _sha(checkpoint)
    manifest = evidence_root / "SHA256SUMS"
    expected = {
        line.split()[1]: line.split()[0]
        for line in manifest.read_text().splitlines()
        if len(line.split()) == 2
    }
    relative = checkpoint.relative_to(evidence_root).as_posix()
    if expected.get(relative) != digest:
        raise ValueError(f"checkpoint evidence SHA mismatch: {checkpoint}")
    run_metadata_path = evidence_root / "runs" / run_id / "run.json"
    run_metadata = json.loads(run_metadata_path.read_text())
    if run_metadata.get("run_id") != run_id:
        raise ValueError("evidence run identity mismatch")
    source = run_metadata.get("source", {})
    if source.get("commit_sha") != "0e8a996794b788e45b96f108e79a03dca2843e4e":
        raise ValueError("evidence model-source revision mismatch")
    dataset_identity = run_metadata.get("dataset", {})
    if (
        dataset_identity.get("dataset_id") != "pizza-6a33710d13eb90fdde3e"
        or dataset_identity.get("declared_manifest_id")
        != "c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd"
        or dataset_identity.get("manifest_identity_sha256")
        != "94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844"
        or dataset_identity.get("train_count") != 720
        or dataset_identity.get("validation_count") != 240
    ):
        raise ValueError("evidence dataset identity mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    vocab = build_class_vocabulary(OntologySnapshot.load(require_pinned_hash=True))
    canonical_vocabulary_sha = hashlib.sha256(
        (
            json.dumps(vocab, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    ).hexdigest()
    if metadata["vocabulary_sha256"] != canonical_vocabulary_sha:
        raise ValueError("checkpoint vocabulary hash differs from canonical vocabulary")
    all_classes = {c["class_id"]: c for c in vocab["classes"]}
    ids = metadata["class_ids"]
    classes = [all_classes[i] for i in ids]
    if [c["name"] for c in classes] != metadata["class_names"] or [
        c["iri"] for c in classes
    ] != metadata["class_iris"]:
        raise ValueError("checkpoint class identity/order differs from canonical vocabulary")
    if metadata["vocabulary_sha256"] != payload["metadata"]["vocabulary_sha256"]:
        raise ValueError("missing vocabulary identity")
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        ids,
        metadata["vocabulary_sha256"],
        latent_dim=metadata["latent_dim"],
        boxes=metadata["boxes"],
        domain=metadata["domain"],
        epsilon=metadata["epsilon"],
        class_names=metadata["class_names"],
        class_iris=metadata["class_iris"],
        fixed_positive_class_ids=metadata["fixed_positive_class_ids"],
    )
    expected_metadata = {
        **model.checkpoint_metadata(
            objective=metadata["objective"]["kind"],
            tau=metadata["objective"]["tau"],
            margin=metadata["objective"]["margin"],
        ),
        "training_config": metadata["training_config"],
    }
    if metadata != expected_metadata:
        raise ValueError("checkpoint/model metadata contract mismatch")
    if model.box_count != 1:
        raise ValueError(f"only K=1 supported, got {model.box_count}")
    state = payload.get("model_state")
    if not isinstance(state, dict) or set(state) != set(model.state_dict()):
        raise ValueError("checkpoint state keys mismatch")
    for key, current in model.state_dict().items():
        candidate = state[key]
        if (
            not isinstance(candidate, torch.Tensor)
            or candidate.shape != current.shape
            or candidate.dtype != current.dtype
            or not torch.isfinite(candidate).all()
        ):
            raise ValueError(f"invalid checkpoint tensor {key}")
    # Stored endpoints are authoritative; CPU decode may differ by one float32 ULP.
    stored = payload.get("geometry")
    if not isinstance(stored, dict) or set(stored) != {"lower", "upper"}:
        raise ValueError("checkpoint geometry snapshot missing")
    lo_t, hi_t = stored["lower"], stored["upper"]
    if (
        lo_t.shape != (len(ids), model.box_count, model.latent_dim)
        or hi_t.shape != lo_t.shape
        or lo_t.dtype != torch.float32
        or hi_t.dtype != torch.float32
        or not torch.isfinite(lo_t).all()
        or not torch.isfinite(hi_t).all()
    ):
        raise ValueError("invalid physical endpoint snapshot")
    validate_endpoints(lo_t, hi_t, domain=model.domain, epsilon=model.epsilon)
    decoded = model._decode_state(state)
    if any(
        not torch.allclose(x.cpu(), y, rtol=0, atol=5e-7)
        for x, y in zip(decoded, (lo_t, hi_t), strict=True)
    ):
        raise ValueError("stored endpoints disagree materially with CPU-decoded geometry")
    model.load_state_dict(state, strict=True)
    lo, hi = (x.detach().cpu().numpy()[:, 0, :].astype("float64") for x in (lo_t, hi_t))
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    trainable = [c for c in classes if c["name"] != FIXED_POSITIVE_CLASS]
    train_names = {c["name"] for c in trainable}
    sub, dis = _reference(ontology, train_names)
    canonical_names = {c["name"] for c in vocab["classes"]}
    all_sub, all_dis = _reference(ontology, canonical_names)
    unsupported = canonical_names - train_names - {FIXED_POSITIVE_CLASS}
    forced_empty = []
    for class_name in canonical_names:
        ancestors = ontology.ancestors(class_name)
        if any(a in ancestors and b in ancestors for a, b in all_dis):
            forced_empty.append(class_name)
    pairs = []
    predsub = set()
    preddis = set()
    directed_viols = []
    overlap_dims = []
    counts = {"containment": 0, "disjoint": 0, "overlap": 0, "equivalence": 0}
    for i, a in enumerate(trainable):
        ai = ids.index(a["class_id"])
        for b in trainable:
            if a["class_id"] == b["class_id"]:
                continue
            bi = ids.index(b["class_id"])
            r = directed(lo[ai], hi[ai], lo[bi], hi[bi])
            if r["contained"]:
                predsub.add((a["name"], b["name"]))
            else:
                directed_viols.append(r["max_violation"])
                point = torch.tensor([r["witness"]["coordinates"]], dtype=torch.float32)
                if not bool(
                    box_membership(
                        point,
                        lo_t[ai : ai + 1],
                        hi_t[ai : ai + 1],
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )[0, 0]
                ) or bool(
                    box_membership(
                        point,
                        lo_t[bi : bi + 1],
                        hi_t[bi : bi + 1],
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )[0, 0]
                ):
                    raise AssertionError("failed containment witness membership verification")
            pairs.append(
                {
                    "kind": "directed",
                    "a_id": a["class_id"],
                    "a_iri": a["iri"],
                    "b_id": b["class_id"],
                    "b_iri": b["iri"],
                    **r,
                    "reference_subclass": (a["name"], b["name"]) in sub,
                    "reference_disjoint": tuple(sorted((a["name"], b["name"]))) in dis,
                }
            )
        for b in trainable[i + 1 :]:
            bi = ids.index(b["class_id"])
            r = unordered(lo[ai], hi[ai], lo[bi], hi[bi])
            namespair = tuple(sorted((a["name"], b["name"])))
            if r["intersection_status"] == "empty":
                preddis.add(namespair)
                counts["disjoint"] += 1
            else:
                counts["overlap"] += 1
                overlap_dims.append(r["intersection_dimensionality"])
            if (
                directed(lo[ai], hi[ai], lo[bi], hi[bi])["contained"]
                and directed(lo[bi], hi[bi], lo[ai], hi[ai])["contained"]
            ):
                counts["equivalence"] += 1
            if r["intersection_status"] != "empty":
                witness = r["intersection_lower"]
                point = torch.tensor([witness], dtype=torch.float32)
                if not bool(
                    box_membership(
                        point,
                        lo_t[ai : ai + 1],
                        hi_t[ai : ai + 1],
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )[0, 0]
                ) or not bool(
                    box_membership(
                        point,
                        lo_t[bi : bi + 1],
                        hi_t[bi : bi + 1],
                        domain=model.domain,
                        epsilon=model.epsilon,
                    )[0, 0]
                ):
                    raise AssertionError("failed intersection witness membership verification")
                r["witness"] = {"type": "intersection", "coordinates": witness}
            pairs.append(
                {
                    "kind": "unordered",
                    "a_id": a["class_id"],
                    "a_iri": a["iri"],
                    "b_id": b["class_id"],
                    "b_iri": b["iri"],
                    **r,
                    "reference_disjoint": namespair in dis,
                }
            )
    counts["containment"] = len(predsub)
    incompatible_sub = {edge for edge in predsub if tuple(sorted(edge)) in dis}
    incompatible_dis = {
        edge for edge in preddis if (edge[0], edge[1]) in sub or (edge[1], edge[0]) in sub
    }

    def metrics(ref, pred):
        inter = len(ref & pred)
        precision = inter / len(pred) if pred else None
        recall = inter / len(ref) if ref else None
        return {
            "reference_count": len(ref),
            "predicted_count": len(pred),
            "intersection_count": inter,
            "precision": precision,
            "precision_undefined_reason": "no_predicted_relations" if not pred else None,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision is not None and recall is not None and precision + recall
                else None
            ),
        }

    per_class = []
    for c in trainable:
        k = ids.index(c["class_id"])
        widths = hi[k] - lo[k]
        logvol = float(__import__("numpy").log(widths).sum())
        outgoing = {
            (row["a_id"], row["b_id"]): row["contained"]
            for row in pairs
            if row["kind"] == "directed"
        }
        contained = [
            other
            for other in trainable
            if other["class_id"] != c["class_id"] and outgoing[(c["class_id"], other["class_id"])]
        ]
        containing = [
            other
            for other in trainable
            if other["class_id"] != c["class_id"] and outgoing[(other["class_id"], c["class_id"])]
        ]
        equivalent = [other for other in contained if outgoing[(other["class_id"], c["class_id"])]]
        unordered_rows = [
            row
            for row in pairs
            if row["kind"] == "unordered" and c["class_id"] in (row["a_id"], row["b_id"])
        ]
        per_class.append(
            {
                "class_id": c["class_id"],
                "name": c["name"],
                "iri": c["iri"],
                "lower": lo[k].tolist(),
                "upper": hi[k].tolist(),
                "center": ((hi[k] + lo[k]) / 2).tolist(),
                "widths": widths.tolist(),
                "min_width": float(widths.min()),
                "max_width": float(widths.max()),
                "mean_width": float(widths.mean()),
                "aspect_ratio": float(widths.max() / widths.min()),
                "log_volume": logvol,
                "volume": math.exp(logvol) if logvol < 709 else None,
                "bounded_domain_volume_fraction": (
                    math.exp(logvol - model.latent_dim * math.log(2))
                    if model.domain == "bounded"
                    else None
                ),
                "bounded_domain_volume_denominator": (
                    2**model.latent_dim if model.domain == "bounded" else None
                ),
                "observed_point_occupancy": None,
                "strictly_contained_classes": len(contained) - len(equivalent),
                "strict_containing_classes": len(containing) - len(equivalent),
                "equivalent_classes": len(equivalent),
                "disjoint_classes": sum(
                    row["intersection_status"] == "empty" for row in unordered_rows
                ),
                "overlapping_classes": sum(
                    row["intersection_status"] != "empty" for row in unordered_rows
                ),
            }
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "schema": "rel-01-v1",
        "metadata": {
            "label": label,
            "run_id": run_id,
            "epoch": payload["epoch"],
            "checkpoint_sha256": digest,
            "model_source_revision": source["commit_sha"],
            "model_source_dirty": source.get("dirty"),
            "evaluator_revision": revision,
            "dataset": {
                "dataset_id": dataset_identity["dataset_id"],
                "declared_manifest_id": dataset_identity["declared_manifest_id"],
                "manifest_file_sha256": dataset_identity["manifest_identity_sha256"],
                "train_count": dataset_identity["train_count"],
                "validation_count": dataset_identity["validation_count"],
                "observed_data_loaded": False,
            },
            "domain": model.domain,
            "latent_dim": model.latent_dim,
            "boxes": model.box_count,
            "class_ids": ids,
            "vocabulary_sha256": model.vocabulary_sha256,
            "ontology_sha256": ontology.sha256,
            "invocation_template": (
                "python -m ontology_embeddings.rel01_cli <verified-evidence-dir> "
                "<new-output.json.gz>"
            ),
            "reference_fragment": (
                "named-subclass-closure+explicit-disjointWith+AllDisjointClasses+"
                "inherited-disjointness-v1"
            ),
            "numerical_policy": (
                "stored float32 endpoints promoted to float64; exact comparisons; "
                "closed boundaries; ratios via log volumes"
            ),
            "status": "global_complete_observed_blocked_dataset_unavailable",
        },
        "summary": {
            "relations": counts,
            "subsumption": metrics(sub, predsub),
            "disjointness": metrics(dis, preddis),
            "reference_subsumption_count": len(sub),
            "reference_disjoint_count": len(dis),
            "reference_coverage": {
                "trainable_class_count": len(trainable),
                "canonical_class_count": len(canonical_names),
                "unsupported_class_count": len(unsupported),
                "unsupported_subsumption_relations": sum(
                    a in unsupported or b in unsupported for a, b in all_sub
                ),
                "unsupported_disjoint_relations": sum(
                    a in unsupported or b in unsupported for a, b in all_dis
                ),
                "domainthing_subsumption_relations": sum(
                    a == FIXED_POSITIVE_CLASS or b == FIXED_POSITIVE_CLASS for a, b in all_sub
                ),
                "domainthing_disjoint_relations": sum(
                    FIXED_POSITIVE_CLASS in pair for pair in all_dis
                ),
                "forced_empty_trainable_classes": sorted(set(forced_empty) & train_names),
            },
            "containment_violation_max": max(directed_viols) if directed_viols else None,
            "overlap_dimensionality_counts": {
                str(d): overlap_dims.count(d) for d in set(overlap_dims)
            },
            "undetermined_subsumption": len(predsub - sub),
            "undetermined_disjointness": len(preddis - dis),
            "evidence_categories": {
                "recovered_subsumption": len(predsub & sub),
                "missed_subsumption": len(sub - predsub),
                "additional_subsumption_not_established": len(predsub - sub),
                "incompatible_subsumption_vs_reference_disjointness": len(incompatible_sub),
                "recovered_disjointness": len(preddis & dis),
                "missed_disjointness": len(dis - preddis),
                "additional_disjointness_not_established": len(preddis - dis),
                "incompatible_disjointness_vs_reference_subsumption": len(incompatible_dis),
                "incompatibility_nonemptiness_condition": True,
            },
        },
        "classes": per_class,
        "pairs": pairs,
        "reference": {
            "subclass": [list(x) for x in sorted(sub)],
            "disjoint": [list(x) for x in sorted(dis)],
        },
    }
