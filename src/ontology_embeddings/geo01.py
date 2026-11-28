"""Thin, fixed-contract launcher for the first GEO 2D unbounded BOX trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from .affine_smoke import load_train_validation
from .box_training import BoxMultiLabelModel, _geometry_pair_sha256, train_box
from .features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from .run_store import RunRecord, _atomic_json

GEO01_DATASET_ID = "pizza-6a33710d13eb90fdde3e"
GEO01_MANIFEST_ID = "c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd"
GEO01_MANIFEST_RAW_SHA256 = "94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844"
GEO01_TRAIN_ORDERED_IDS_SHA256 = "d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219"
GEO01_VALIDATION_ORDERED_IDS_SHA256 = (
    "a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c"
)
GEO01_DECLARED_TEST_COUNT = 240
GEO01_EPOCHS = 60
GEO01_BATCH_SIZE = 64
GEO01_SEED = 23
GEO01_LATENT_DIM = 2
GEO01_BOXES = 1
GEO01_LEARNING_RATE = 0.02
GEO01_EPSILON = 1e-4
GEO01_TAU = 0.1
GEO01_MARGIN = 0.1
GEO03_CONFIG_ID = "GEO03-01-BOX-D2-UNBOUNDED"
GEO03_ATTEMPT_ID = "GEO03-01-BOX-D2-UNBOUNDED-A01"
GEO03_PAIR_ID = "GEO03-BOX-D2"
GEO03_PREPARATION_ID = "GEO03-PREP-BOX-D2"
GEO03_CONFIG_SOURCE_COMMIT = "45bce3e5161b3053fdd59e39bdffa10a6bf20bcc"
GEO03_CONFIG_SHA256 = "b3fb71491a4e06a270934c790177c45bbc680e47732f2bcc60d0007c481215af"


def geo_campaign_identity(config_sha256: str) -> dict[str, str]:
    """Build the RunRecord marker from the coordinator's per-config GEO-03 hash."""
    if (
        not isinstance(config_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None
        or config_sha256 != GEO03_CONFIG_SHA256
    ):
        raise ValueError(
            "GEO-03 per-configuration SHA-256 differs from the verified frozen candidate"
        )
    return {
        "config_id": GEO03_CONFIG_ID,
        "config_sha256": config_sha256,
        "attempt_id": GEO03_ATTEMPT_ID,
    }


def geo_preparation_identity() -> dict[str, str]:
    """Keep pair/preparation IDs separate from GEO-03's exact three-key campaign marker."""
    return {"pair_id": GEO03_PAIR_ID, "preparation_id": GEO03_PREPARATION_ID}


def ordered_sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    """Hash the exact ordered list identity used by the approved train/validation splits."""
    if not isinstance(sample_ids, (list, tuple)) or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids
    ):
        raise ValueError("ordered sample IDs must be nonempty strings")
    encoded = json.dumps(list(sample_ids), separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def prepare_box_reference_initialization(
    dataset: Any, *, seed: int = GEO01_SEED, device: torch.device | str = "cuda:0"
) -> dict[str, Any]:
    """Generate bounded-native BOX targets and raw adapters for a matched carrier pair."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != GEO01_SEED:
        raise ValueError("the approved GEO-01 initialization seed is 23")
    device_obj = torch.device(device)
    if device_obj.type not in ("cpu", "cuda"):
        raise ValueError("initialization device must be CPU or CUDA")
    cuda_devices = [device_obj.index or 0] if device_obj.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        reference = BoxMultiLabelModel(
            len(PIZZA_FEATURE_NAMES),
            len(COMPONENT_FEATURE_NAMES),
            dataset.canonical_class_ids,
            dataset.class_manifest_sha256,
            latent_dim=GEO01_LATENT_DIM,
            boxes=GEO01_BOXES,
            domain="bounded",
            epsilon=GEO01_EPSILON,
            seed=seed,
            class_names=dataset.class_names,
            class_iris=dataset.class_iris,
            fixed_positive_class_ids=getattr(dataset, "fixed_positive_class_ids", ()),
            device=device_obj,
        ).to(device_obj)
    adapter_keys = (
        "pizza_adapter.weight",
        "pizza_adapter.bias",
        "component_adapter.weight",
        "component_adapter.bias",
    )
    reference_state = reference.state_dict()
    adapter_state = {key: reference_state[key].detach().clone() for key in adapter_keys}
    lower, upper = (endpoint.detach().clone() for endpoint in reference.endpoints())
    adapter_hashes = {key: _tensor_sha256(value) for key, value in adapter_state.items()}
    target_sha256 = _geometry_pair_sha256(lower, upper)
    reference_volume_fractions = ((upper - lower) / 2.0).prod(dim=-1).flatten()
    pairing_identity = hashlib.sha256(
        json.dumps(
            {
                "family": reference.family,
                "seed": seed,
                "latent_dim": GEO01_LATENT_DIM,
                "boxes": GEO01_BOXES,
                "class_ids": list(dataset.canonical_class_ids),
                "class_manifest_sha256": dataset.class_manifest_sha256,
                "target_sha256": target_sha256,
                "adapter_sha256": adapter_hashes,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "adapter_state": adapter_state,
        "bounds": (lower, upper),
        "provenance": {
            "schema_version": "geo-box-paired-initialization-v1",
            "pairing_id": pairing_identity,
            "initializer": "native bounded BOX initializer",
            "family": reference.family,
            "reference_carrier": "bounded",
            "target_carrier": "unbounded",
            "seed": seed,
            "latent_dim": GEO01_LATENT_DIM,
            "boxes": GEO01_BOXES,
            "class_ids": list(dataset.canonical_class_ids),
            "class_manifest_sha256": dataset.class_manifest_sha256,
            "target_sha256": target_sha256,
            "adapter_state_sha256": adapter_hashes,
            "reference_volume_fraction_min": float(reference_volume_fractions.min()),
            "reference_volume_fraction_mean": float(reference_volume_fractions.mean()),
            "reference_volume_fraction_max": float(reference_volume_fractions.max()),
            "initialization_limitations": (
                "unbounded targets inherit bounded-native feasible centers and widths; "
                "family-specific starts do not imply latent-axis alignment"
            ),
            "random_stream_note": "BOX family seed-23 native initialization; no TOWN state reused",
        },
    }


def load_approved_train_validation(
    data_root: Path | str,
) -> tuple[Any, list[Any], list[Any], dict[str, Any]]:
    """Load only the existing split-safe train/validation selection and bind its identity."""
    root = Path(data_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest_raw = manifest_path.read_bytes()
    dataset, train_records, validation_records = load_train_validation(root)
    train_ids = [record.sample_id for record in train_records]
    validation_ids = [record.sample_id for record in validation_records]
    manifest_raw_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if dataset.dataset_id != GEO01_DATASET_ID or dataset.manifest_id != GEO01_MANIFEST_ID:
        raise ValueError("dataset ID or declared manifest identity differs from GEO-01")
    if manifest_raw_sha256 != GEO01_MANIFEST_RAW_SHA256:
        raise ValueError("raw dataset manifest checksum differs from GEO-01")
    if dataset.dataset_manifest_sha256 != manifest_raw_sha256:
        raise ValueError("safe loader manifest checksum disagrees with the original manifest")
    if len(train_records) != 720 or len(validation_records) != 240:
        raise ValueError("GEO-01 requires the original ordered 720/240 train/validation splits")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("train or validation sample identities are duplicated")
    if set(train_ids) & set(validation_ids):
        raise ValueError("train and validation sample identities overlap")
    if any(record.split != "train" for record in train_records) or any(
        record.split != "validation" for record in validation_records
    ):
        raise ValueError("safe loader returned a record from outside the approved splits")
    train_hash = ordered_sample_ids_sha256(train_ids)
    validation_hash = ordered_sample_ids_sha256(validation_ids)
    if train_hash != GEO01_TRAIN_ORDERED_IDS_SHA256:
        raise ValueError("ordered train selection identity differs from GEO-01")
    if validation_hash != GEO01_VALIDATION_ORDERED_IDS_SHA256:
        raise ValueError("ordered validation selection identity differs from GEO-01")
    identity = {
        "dataset_id": dataset.dataset_id,
        "manifest_id": dataset.manifest_id,
        "dataset_manifest_sha256": manifest_raw_sha256,
        "class_manifest_sha256": dataset.class_manifest_sha256,
        "class_ids": list(dataset.canonical_class_ids),
        "class_names": list(dataset.class_names),
        "class_iris": list(dataset.class_iris),
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "train_sample_ids_sha256": train_hash,
        "validation_sample_ids_sha256": validation_hash,
        "declared_test_count": GEO01_DECLARED_TEST_COUNT,
        "test_split_decoded": False,
        "test_statistics_used": False,
    }
    return dataset, train_records, validation_records, identity


def approved_box_config(*, device: str = "cuda:0") -> dict[str, Any]:
    return {
        "model": "box-union-two-adapters",
        "domain": "unbounded",
        "latent_dim": GEO01_LATENT_DIM,
        "boxes": GEO01_BOXES,
        "epsilon": GEO01_EPSILON,
        "objective": "logistic",
        "tau": GEO01_TAU,
        "margin": GEO01_MARGIN,
        "learning_rate": GEO01_LEARNING_RATE,
        "epochs": GEO01_EPOCHS,
        "batch_size": GEO01_BATCH_SIZE,
        "device": device,
        "clip_norm": 1.0,
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "weight_decay": 0,
        "amsgrad": False,
        "seed": GEO01_SEED,
    }


def create_geo_run_record(
    artifact_root: Path | str,
    *,
    config_sha256: str,
    job_id: str | None = None,
) -> RunRecord:
    """Register the GEO-03 identity at RunRecord top level before any training work."""
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()):
        raise ValueError("job_id must be a nonempty string when supplied")
    record = RunRecord(
        artifact_root,
        purpose="geo-01-2d-unbounded-box",
        invocation="python -m ontology_embeddings.geo01 --dimension 2 --carrier unbounded",
        config=approved_box_config(device="cuda:0"),
        dataset={
            "dataset_id_expected": GEO01_DATASET_ID,
            "manifest_id_expected": GEO01_MANIFEST_ID,
            "declared_test_count": GEO01_DECLARED_TEST_COUNT,
            "test_split_decoded": False,
        },
        seed=GEO01_SEED,
        issue="30",
    )
    try:
        record.update(
            geo_campaign=geo_campaign_identity(config_sha256),
            geo_preparation=geo_preparation_identity(),
            geo_campaign_source_commit=GEO03_CONFIG_SOURCE_COMMIT,
        )
        if job_id is not None:
            record.update(job_id=job_id)
    except BaseException as error:
        record.fail(error)
        raise
    return record


def run_geo01_box(
    data_root: Path | str,
    *,
    artifact_root: Path | str = "artifacts/geo-01-runs",
    device: str = "cuda:0",
    job_id: str | None = None,
    config_sha256: str,
) -> dict[str, Any]:
    """Launch the fixed 2D unbounded BOX run with its exact GEO-03 config identity."""
    if not isinstance(config_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None:
        raise ValueError("the coordinator-supplied GEO-03 config SHA-256 is required")
    if device != "cuda:0":
        raise ValueError("the authorized GEO-01 substantial run requires device cuda:0")
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()):
        raise ValueError("job_id must be a nonempty string when supplied")
    if not isinstance(config_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", config_sha256) is None:
        raise ValueError("the coordinator-supplied GEO-03 config SHA-256 is required")
    rec = create_geo_run_record(artifact_root, config_sha256=config_sha256, job_id=job_id)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("authorized CUDA device is unavailable; CPU fallback is forbidden")
        if torch.cuda.current_device() != 0:
            torch.cuda.set_device(0)
        if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2060":
            raise RuntimeError(
                "GEO-01 preflight device identity differs from the approved RTX 2060"
            )
        dataset, train_records, validation_records, data_identity = load_approved_train_validation(
            data_root
        )
        preparation = prepare_box_reference_initialization(dataset, seed=GEO01_SEED, device=device)
        rec.update(
            dataset={
                **data_identity,
                "train_sample_ids": [record.sample_id for record in train_records],
                "validation_sample_ids": [record.sample_id for record in validation_records],
            },
            environment={
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": device,
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
                "python": __import__("sys").version,
            },
        )
        started = time.perf_counter()
        result = train_box(
            dataset,
            train_records,
            validation_records,
            artifact_root=artifact_root,
            purpose="geo-01-2d-unbounded-box",
            invocation=rec.manifest["invocation"],
            seed=GEO01_SEED,
            epochs=GEO01_EPOCHS,
            batch_size=GEO01_BATCH_SIZE,
            learning_rate=GEO01_LEARNING_RATE,
            latent_dim=GEO01_LATENT_DIM,
            boxes=GEO01_BOXES,
            domain="unbounded",
            epsilon=GEO01_EPSILON,
            objective="logistic",
            tau=GEO01_TAU,
            margin=GEO01_MARGIN,
            device=device,
            run_record=rec,
            shared_adapter_initial_state=preparation["adapter_state"],
            initial_box_bounds=preparation["bounds"],
            initial_box_class_ids=dataset.canonical_class_ids,
            paired_initialization_provenance=preparation["provenance"],
            dataset_provenance={
                **data_identity,
                "declared_test_count": GEO01_DECLARED_TEST_COUNT,
                "test_split_decoded": False,
                "test_statistics_used": False,
                "job_id": job_id,
            },
            capture_temporal=True,
            portable_artifact_paths=True,
        )
        result["geo01_execution"] = {
            "task": "first-qualified-2d-unbounded-box-trajectory",
            "completed_epochs": len(result["history"]),
            "runtime_seconds": time.perf_counter() - started,
            "handoff": "provisional GEO-02 BOX consumer handoff",
            "predictive_quality_gate": False,
        }
        _atomic_json(rec.path / "summary.json", result)
        rec.update(summary=result, geo01_execution=result["geo01_execution"])
        return result
    except KeyboardInterrupt as error:
        if rec.manifest.get("status") == "running":
            rec.interrupt(error)
        raise
    except BaseException as error:
        if rec.manifest.get("status") == "running":
            rec.fail(error)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/geo-01-runs"))
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--job-id")
    parser.add_argument(
        "--config-sha256",
        required=True,
        help="per-configuration GEO-03 _config_hash(CONFIGURATIONS[index]) (not manifest SHA)",
    )
    args = parser.parse_args(argv)
    result = run_geo01_box(
        args.data_root,
        artifact_root=args.artifact_root,
        device=args.device,
        job_id=args.job_id,
        config_sha256=args.config_sha256,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
