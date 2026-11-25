import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ontology_embeddings.affine_smoke import load_train_validation
from ontology_embeddings.features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from ontology_embeddings.pr3_acceptance import (
    _checkpoint_parameter_updates,
    _read_run_result,
    _validate_main_run_provenance,
    assess_run,
    export_evidence,
    finalize,
    preflight,
    require_ignored_artifact_path,
    run_domain,
    run_geometry_controls,
    validate_data_identity,
    validate_paired_results,
    verify_cuda_runtime,
)
from ontology_embeddings.run_store import find_runs


def test_controls_register_recoverable_geometry_only_runs(tmp_path):
    results = run_geometry_controls(artifact_root=tmp_path, epochs=350)
    assert len(results) == 4
    assert all(row["status"] == "completed" and row["target_recovered"] for row in results)
    assert all(
        row["geometry_updated"]
        and row["geometry_gradient_finite"]
        and row["geometry_gradient_norm_max"] > 0
        for row in results
    )
    assert all("initial_corners" in row and "final_corners" in row for row in results)
    assert len({row["run_id"] for row in results}) == 4
    for row in results:
        folder = tmp_path / row["run_id"]
        assert {"initial.pt", "latest.pt", "checkpoint.pt"} <= {
            p.name for p in (folder / "checkpoints").iterdir()
        }
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert {
            "dimension",
            "boxes",
            "epsilon",
            "dtype",
            "optimizer",
            "learning_rate",
            "clip_norm",
            "seed",
            "device",
            "points",
            "targets",
            "optimizer_epsilon",
            "points_sha256",
            "targets_sha256",
        } <= set(manifest["config"])
        assert manifest["environment"]["torch"] == str(torch.__version__)
        assert row["config"] == manifest["config"]
        assert find_runs(tmp_path, run_id=row["run_id"])[0]["status"] == "completed"
        assert (folder / "metrics.jsonl").is_file()


def test_control_failures_still_leave_index_and_recovery_receipts(monkeypatch, tmp_path):
    import ontology_embeddings.box as box

    def fail_inverse(*args, **kwargs):
        raise ValueError("synthetic injected initialization failure")

    monkeypatch.setattr(box, "inverse_geometry", fail_inverse)
    results = run_geometry_controls(artifact_root=tmp_path, epochs=1)
    assert len(results) == 4 and all(item["status"] == "failed" for item in results)
    for item in results:
        folder = tmp_path / item["run_id"]
        assert json.loads((folder / "manifest.json").read_text())["status"] == "failed"
        assert {"initial.pt", "latest.pt", "checkpoint.pt"} <= {
            p.name for p in (folder / "checkpoints").iterdir()
        }
        assert find_runs(tmp_path, run_id=item["run_id"])[0]["status"] == "failed"


def test_validate_data_identity_requires_historical_exact_ids_and_clean_source():
    dataset = SimpleNamespace(
        dataset_id="pizza-6a33710d13eb90fdde3e",
        dataset_manifest_sha256="manifest",
        class_manifest_sha256="classes",
    )
    train = [SimpleNamespace(sample_id=f"t{i}", split="train") for i in range(720)]
    val = [SimpleNamespace(sample_id=f"v{i}", split="validation") for i in range(240)]
    manifest = {
        "run_id": "20261123T105917-c00ea4941c34",
        "source": {"commit_sha": "522c3f30240e65a431abdfc7694d0006866b0874", "dirty": False},
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "manifest_sha256": "manifest",
            "class_manifest_sha256": "classes",
            "train_sample_ids": [r.sample_id for r in train],
            "validation_sample_ids": [r.sample_id for r in val],
        },
    }
    assert validate_data_identity(dataset, train, val, manifest)["train_ids_sha256"]
    with pytest.raises(ValueError, match="train IDs"):
        validate_data_identity(
            dataset, train[1:] + [SimpleNamespace(sample_id="other", split="train")], val, manifest
        )
    manifest["source"]["dirty"] = True
    with pytest.raises(ValueError, match="source"):
        validate_data_identity(dataset, train, val, manifest)


def test_local_run_artifacts_must_be_gitignored():
    root = Path(__file__).resolve().parents[1]
    require_ignored_artifact_path(root / "artifacts/runs/pr3-preparation")
    with pytest.raises(ValueError, match="not ignored"):
        require_ignored_artifact_path(root / "README.md")


def test_preflight_rejects_existing_export_before_loading_or_cuda(monkeypatch, tmp_path):
    import ontology_embeddings.pr3_acceptance as acceptance

    destination = tmp_path / "already-there"
    destination.mkdir()
    monkeypatch.setattr(acceptance, "load_train_validation", lambda _: pytest.fail("loader called"))
    with pytest.raises(FileExistsError):
        preflight(
            "unused",
            historical_manifest_path="unused",
            device="cpu",
            export_destination=destination,
        )


def test_preflight_checks_evidence_storage_before_cuda_or_loading(monkeypatch, tmp_path):
    import ontology_embeddings.pr3_acceptance as acceptance

    monkeypatch.setattr(acceptance.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="free space"):
        preflight(
            "unused",
            historical_manifest_path="unused",
            device="cpu",
            export_destination=tmp_path / "evidence",
        )


def test_assessment_uses_earliest_best_loss_and_strict_per_task_baseline(tmp_path):
    def row(epoch, loss, pizza, component):
        return {
            "epoch": epoch,
            "validation_loss": loss,
            "diagnostics": {
                "validation": {
                    task: {"micro": {"f1": value}}
                    for task, value in (("pizza", pizza), ("component", component))
                },
                "train": {},
                "feature_signatures": {},
            },
        }

    baselines = {task: {"metrics": {"micro": {"f1": 0.5}}} for task in ("pizza", "component")}
    run_path = tmp_path / "run"
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    torch.save(
        {
            "metadata": {
                "domain": "bounded",
                "latent_dim": 16,
                "boxes": 1,
                "epsilon": 1e-4,
                "class_ids": [1],
                "class_names": ["A"],
                "class_iris": ["urn:A"],
                "vocabulary_sha256": "a" * 64,
                "encoder": {
                    "pizza_features": len(PIZZA_FEATURE_NAMES),
                    "component_features": len(COMPONENT_FEATURE_NAMES),
                    "bounded_activation": "tanh",
                },
                "objective": {"kind": "logistic", "tau": 0.1, "margin": 0.1},
                "training_config": {
                    "model": "box-union-two-adapters",
                    "domain": "bounded",
                    "objective": "logistic",
                    "latent_dim": 16,
                    "boxes": 1,
                    "epsilon": 1e-4,
                    "epochs": 60,
                    "batch_size": 64,
                    "seed": 23,
                    "learning_rate": 0.02,
                    "tau": 0.1,
                    "clip_norm": 1.0,
                    "weight_decay": 0,
                    "margin": 0.1,
                    "optimizer": "Adam",
                    "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
                    "betas": [0.9, 0.999],
                    "optimizer_epsilon": 1e-8,
                    "amsgrad": False,
                    "device": "cuda:0",
                    "paired_initialization": {},
                },
            },
            "model_state": {"geometry.raw_a": torch.zeros(1)},
            "epoch": 2,
            "validation_loss": 0.5,
        },
        best_checkpoint,
    )
    result = {
        "run_id": "assessment-run",
        "run_path": str(run_path),
        "status": "completed",
        "history": [row(1, 1.0, 0.6, 0.6), row(2, 0.5, 0.5, 0.8), row(3, 0.5, 0.8, 0.8)],
        "best_mean_available_task_micro_f1_epoch": 3,
        "best_mean_available_task_micro_f1": 0.8,
        "best_checkpoint_reload_predictions_equivalent": True,
        "checkpoint_paths": {"best": str(best_checkpoint)},
    }
    result["history"] += [row(epoch, 2.0, 0.8, 0.8) for epoch in range(4, 61)]

    def persist_run_evidence():
        summary = json.loads(json.dumps(result))
        (run_path / "summary.json").write_text(json.dumps(summary))
        (run_path / "manifest.json").write_text(
            json.dumps({"run_id": result["run_id"], "status": result["status"], "summary": summary})
        )
        (run_path / "metrics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in result["history"]) + "\n"
        )

    persist_run_evidence()
    assessed = assess_run(result, baselines)
    assert assessed["selected_epoch"] == 2
    assert assessed["accepted_C"] is False
    assert assessed["best_mean_available_task_micro_f1_epoch"] == 3
    result["history"][1]["diagnostics"]["validation"]["pizza"]["micro"]["f1"] = 0.50001
    altered_summary = json.loads(json.dumps(result))
    (run_path / "summary.json").write_text(json.dumps(altered_summary))
    (run_path / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": result["run_id"],
                "status": result["status"],
                "summary": altered_summary,
            }
        )
    )
    assert assess_run(result, baselines)["accepted_C"] is False
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is True
    checkpoint = torch.load(best_checkpoint, weights_only=True)
    checkpoint["epoch"] = 3
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is False
    checkpoint["epoch"] = 2
    checkpoint["metadata"]["latent_dim"] = 15
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is False
    checkpoint["metadata"]["latent_dim"] = 16
    checkpoint["metadata"]["boxes"] = 2
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is False
    checkpoint["metadata"]["boxes"] = 1
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is True
    result["history"] = [row(1, 1.0, 0.6, 0.6)] + [
        row(epoch, 2.0, 0.6, 0.6) for epoch in range(2, 61)
    ]
    result["best_mean_available_task_micro_f1_epoch"] = 1
    result["best_mean_available_task_micro_f1"] = 0.6
    persist_run_evidence()
    checkpoint["epoch"] = 1
    checkpoint["validation_loss"] = 1.0
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is True
    checkpoint["epoch"] = True
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is False
    checkpoint["epoch"] = 1
    checkpoint["validation_loss"] = True
    torch.save(checkpoint, best_checkpoint)
    assert assess_run(result, baselines)["accepted_C"] is False
    checkpoint["validation_loss"] = 1.0
    torch.save(checkpoint, best_checkpoint)
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is True
    result["history"][0]["diagnostics"]["validation"]["pizza"]["micro"]["f1"] = "invalid"
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is False
    result["history"][0]["diagnostics"]["validation"]["pizza"]["micro"]["f1"] = 0.6
    persist_run_evidence()
    assert assess_run(result, {"pizza": baselines["pizza"]})["accepted_C"] is False
    checkpoint["validation_loss"] = 1.0
    torch.save(checkpoint, best_checkpoint)
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is True

    import copy

    for task in ("pizza", "component"):
        original_task_metrics = copy.deepcopy(
            result["history"][0]["diagnostics"]["validation"][task]
        )
        result["history"][0]["diagnostics"]["validation"][task] = None
        persist_run_evidence()
        assert assess_run(result, baselines)["accepted_C"] is False
        result["history"][0]["diagnostics"]["validation"][task] = copy.deepcopy(
            original_task_metrics
        )
        persist_run_evidence()

        for malformed in ("missing-f1", "invalid-f1"):
            changed_metrics = copy.deepcopy(original_task_metrics)
            if malformed == "missing-f1":
                del changed_metrics["micro"]["f1"]
            else:
                changed_metrics["micro"]["f1"] = "invalid"
            result["history"][0]["diagnostics"]["validation"][task] = changed_metrics
            persist_run_evidence()
            assert assess_run(result, baselines)["accepted_C"] is False
            result["history"][0]["diagnostics"]["validation"][task] = copy.deepcopy(
                original_task_metrics
            )
            persist_run_evidence()

        missing_baseline = copy.deepcopy(baselines)
        del missing_baseline[task]
        assert assess_run(result, missing_baseline)["accepted_C"] is False
        malformed_baseline = copy.deepcopy(baselines)
        malformed_baseline[task]["metrics"]["micro"]["f1"] = "invalid"
        assert assess_run(result, malformed_baseline)["accepted_C"] is False

    original_train_metrics = result["history"][0]["diagnostics"]["train"]
    result["history"][0]["diagnostics"]["train"] = None
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is False
    result["history"][0]["diagnostics"]["train"] = original_train_metrics
    persist_run_evidence()
    assert assess_run(result, baselines)["accepted_C"] is True


def test_paired_initialization_verifies_adapter_identity_and_physical_bounds(tmp_path):
    import hashlib

    from ontology_embeddings.pr3_acceptance import DATASET_ID

    lower = [[[-0.2, -0.2]]]
    upper = [[[0.2, 0.2]]]
    results = {}
    for index, domain in enumerate(("bounded", "unbounded"), 1):
        run_path = tmp_path / domain
        checkpoint_dir = run_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        path = checkpoint_dir / "initial.pt"
        run_id = f"20261125T00000{index}-00000000000{index}"
        adapter = {
            "pizza_adapter.weight": torch.ones((16, len(PIZZA_FEATURE_NAMES))),
            "pizza_adapter.bias": torch.ones(16),
            "component_adapter.weight": torch.ones((16, len(COMPONENT_FEATURE_NAMES))),
            "component_adapter.bias": torch.ones(16),
        }
        hashes = {
            key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
            for key, value in adapter.items()
        }
        paired_initialization = {
            "adapter_sha256": hashes,
            "box_bounds": {
                "requested_sha256": "bounds",
                "requested_lower": lower,
                "requested_upper": upper,
            },
        }
        config = {
            "model": "box-union-two-adapters",
            "domain": domain,
            "latent_dim": 16,
            "boxes": 1,
            "epsilon": 1e-4,
            "objective": "logistic",
            "tau": 0.1,
            "margin": 0.1,
            "learning_rate": 0.02,
            "epochs": 60,
            "batch_size": 64,
            "device": "cuda:0",
            "clip_norm": 1.0,
            "optimizer": "Adam",
            "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
            "betas": [0.9, 0.999],
            "optimizer_epsilon": 1e-8,
            "weight_decay": 0,
            "amsgrad": False,
            "seed": 23,
            "paired_initialization": paired_initialization,
        }
        metadata = {
            "domain": domain,
            "latent_dim": 16,
            "boxes": 1,
            "epsilon": 1e-4,
            "class_ids": [1],
            "class_names": ["A"],
            "class_iris": ["urn:A"],
            "vocabulary_sha256": "b" * 64,
            "objective": {"kind": "logistic", "tau": 0.1, "margin": 0.1},
            "encoder": {
                "pizza_features": len(PIZZA_FEATURE_NAMES),
                "component_features": len(COMPONENT_FEATURE_NAMES),
                "bounded_activation": "tanh" if domain == "bounded" else "identity",
            },
            "training_config": config,
        }
        torch.save(
            {
                "metadata": metadata,
                "model_state": adapter,
                "geometry": {"lower": torch.tensor(lower), "upper": torch.tensor(upper)},
                "epoch": 0,
                "validation_loss": None,
            },
            path,
        )
        checkpoint_paths = {"initial": str(path)}
        summary = {
            "run_id": run_id,
            "run_path": str(run_path),
            "status": "completed",
            "checkpoint_paths": checkpoint_paths,
        }
        dataset = {
            "dataset_id": DATASET_ID,
            "dataset_manifest_sha256": "a" * 64,
            "class_manifest_sha256": "b" * 64,
            "class_ids": [1],
            "class_names": ["A"],
            "class_iris": ["urn:A"],
            "fixed_positive_class_ids": [],
            "unsupported_class_ids": [],
            "train_sample_ids": [],
            "validation_sample_ids": [],
        }
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "purpose": "pr3-required-metadata-run",
            "issue": "14",
            "seed": 23,
            "config": config,
            "dataset": dataset,
            "summary": summary,
        }
        (run_path / "summary.json").write_text(json.dumps(summary))
        (run_path / "manifest.json").write_text(json.dumps(manifest))
        results[domain] = {**summary, "checkpoint_paths": checkpoint_paths}

    assert validate_paired_results(results)["max_absolute_error_by_domain"] == {
        "bounded": 0.0,
        "unbounded": 0.0,
    }
    bounded_path = results["bounded"]["checkpoint_paths"]["initial"]
    unbounded_path = results["unbounded"]["checkpoint_paths"]["initial"]
    results["unbounded"]["checkpoint_paths"]["initial"] = bounded_path
    with pytest.raises(ValueError, match="initial checkpoint|domain"):
        validate_paired_results(results)
    unbounded_run = Path(results["unbounded"]["run_path"])
    unbounded_summary_path = unbounded_run / "summary.json"
    unbounded_manifest_path = unbounded_run / "manifest.json"
    original_summary = json.loads(unbounded_summary_path.read_text())
    original_manifest = json.loads(unbounded_manifest_path.read_text())
    altered_summary = {
        **original_summary,
        "checkpoint_paths": {"initial": bounded_path},
    }
    unbounded_summary_path.write_text(json.dumps(altered_summary))
    unbounded_manifest_path.write_text(
        json.dumps({**original_manifest, "summary": altered_summary})
    )
    results["unbounded"]["checkpoint_paths"]["initial"] = bounded_path
    with pytest.raises(ValueError, match="initial checkpoint|domain"):
        validate_paired_results(results)
    unbounded_summary_path.write_text(json.dumps(original_summary))
    unbounded_manifest_path.write_text(json.dumps(original_manifest))
    results["unbounded"]["checkpoint_paths"]["initial"] = unbounded_path

    payload = torch.load(bounded_path, weights_only=True)
    payload["geometry"]["lower"][0, 0, 0] += 1e-5
    torch.save(payload, bounded_path)
    assert validate_paired_results(results)["max_absolute_error_by_domain"][
        "bounded"
    ] == pytest.approx(1e-5, abs=1e-7)
    payload["geometry"]["lower"][0, 0, 0] = -0.2
    torch.save(payload, bounded_path)
    payload = torch.load(unbounded_path, weights_only=True)
    payload["geometry"]["lower"][0, 0, 0] += 3e-5
    torch.save(payload, unbounded_path)
    with pytest.raises(ValueError, match="physical bounds"):
        validate_paired_results(results)
    payload["geometry"]["lower"][0, 0, 0] = -0.2
    torch.save(payload, unbounded_path)
    payload = torch.load(unbounded_path, weights_only=True)
    payload["model_state"]["pizza_adapter.weight"][0, 0] = 9.0
    torch.save(payload, unbounded_path)
    with pytest.raises(ValueError, match="hash|adapter"):
        validate_paired_results(results)


def test_run_domain_rejects_modified_prepared_train_validation_identity_before_training(tmp_path):
    import hashlib

    work = tmp_path / "work"
    work.mkdir()
    artifact_root = tmp_path / "runs"
    artifact_root.mkdir()
    (work / "paired_initial.pt").write_bytes(b"paired")
    dataset = SimpleNamespace(
        dataset_id="fixture",
        dataset_manifest_sha256="manifest",
        class_manifest_sha256="classes",
        canonical_class_ids=(1,),
        class_names=("A",),
        class_iris=("urn:A",),
        fixed_positive_class_ids=(),
        unsupported_class_ids=(),
    )
    train = [SimpleNamespace(sample_id="train-id", split="train")]
    validation = [SimpleNamespace(sample_id="validation-id", split="validation")]
    data_path = work / "train_validation.pt"
    torch.save(
        {"dataset": dataset, "train_records": train, "validation_records": validation}, data_path
    )
    prep = {
        "artifact_root": str(artifact_root),
        "dataset_id": "fixture",
        "device": "cpu",
        "paired_initial_sha256": hashlib.sha256(b"paired").hexdigest(),
        "train_validation_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_identity": {
            "dataset_id": "fixture",
            "manifest_sha256": "manifest",
            "class_manifest_sha256": "classes",
            "class_ids": [1],
            "class_names": ["A"],
            "class_iris": ["urn:A"],
            "fixed_positive_class_ids": [],
            "unsupported_class_ids": [],
            "train_ids_sha256": "wrong-order-or-data",
            "validation_ids_sha256": "wrong-validation",
            "train_count": 1,
            "validation_count": 1,
        },
    }
    (work / "prepare.json").write_text(json.dumps(prep))
    with pytest.raises(ValueError, match="data identity"):
        run_domain(work, "bounded", device="cpu")


def test_cuda_preflight_refuses_unverified_device():
    with pytest.raises(RuntimeError, match="CUDA"):
        verify_cuda_runtime("cpu")


def test_cuda_runtime_records_actual_generic_device_metadata(monkeypatch):
    import ontology_embeddings.pr3_acceptance as acceptance

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Generic accelerator")
    result = acceptance.verify_cuda_runtime("cuda:1")
    assert result["device"] == "cuda:1"
    assert result["device_name"] == "Generic accelerator"
    assert result["torch"] == str(torch.__version__)
    assert result["cuda"] == torch.version.cuda


def test_evidence_export_is_checksummed_and_never_overwrites(tmp_path):
    target = tmp_path / "evidence"
    checkpoint = tmp_path / "selected.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    result = export_evidence(
        target,
        {"dataset_id": "synthetic", "runs": []},
        {"bounded": {"path": str(checkpoint), "run_id": "run-1"}},
    )
    record = json.loads((target / "checksums.json").read_text())
    evidence = json.loads((target / "evidence.json").read_text())
    assert result["bundle_sha256"] == record["bundle_sha256"]
    packaged = evidence["packaged_best_checkpoints"]["bounded"]
    assert packaged["filename"] == "checkpoints/bounded-best.pt"
    assert packaged["run_id"] == "run-1"
    assert packaged["sha256"] == record["files"][packaged["filename"]]
    assert packaged["reference"].startswith("checksummed evidence bundle")
    for name, digest in record["files"].items():
        import hashlib

        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_evidence(target, {"replacement": True})
    assert "replacement" not in (target / "evidence.json").read_text()


def test_checkpoint_group_updates_are_verified_from_persisted_tensors(tmp_path):
    initial = {
        "pizza_adapter.weight": torch.zeros(2),
        "component_adapter.weight": torch.zeros(2),
        "geometry.raw_a": torch.zeros(2),
    }
    latest = {key: value + 1 for key, value in initial.items()}
    paths = {}
    for label, state in (("initial", initial), ("latest", latest)):
        path = tmp_path / f"{label}.pt"
        torch.save({"model_state": state}, path)
        paths[label] = str(path)
    assert all(_checkpoint_parameter_updates({"checkpoint_paths": paths}).values())
    latest["geometry.raw_a"].zero_()
    torch.save({"model_state": latest}, paths["latest"])
    result = _checkpoint_parameter_updates({"checkpoint_paths": paths})
    assert result == {"pizza_adapter": True, "component_adapter": True, "geometry": False}


def test_assessment_fails_closed_without_status_reload_or_selectable_loss():
    baselines = {task: {"metrics": {"micro": {"f1": 0.5}}} for task in ("pizza", "component")}
    base = {
        "status": "completed",
        "best_checkpoint_reload_predictions_equivalent": True,
        "history": [],
    }
    assert assess_run({**base, "status": None}, baselines)["accepted_C"] is False
    assert (
        assess_run({**base, "best_checkpoint_reload_predictions_equivalent": False}, baselines)[
            "accepted_C"
        ]
        is False
    )
    assert assess_run(base, baselines)["accepted_C"] is False


def test_finalize_maps_A_B_C_and_rejects_duplicate_control_combinations(tmp_path, monkeypatch):
    import ontology_embeddings.pr3_acceptance as acceptance
    import ontology_embeddings.run_store as run_store

    work = tmp_path / "work"
    work.mkdir()
    artifact_root = tmp_path / "control-runs"
    source = {
        "commit_sha": "source-sha",
        "dirty": False,
        "relevant_source_sha256": {},
        "source_fingerprint_sha256": None,
        "availability": "available",
        "fingerprint_note": "working-tree source paths hashed; non-source diff details omitted",
    }
    monkeypatch.setattr(run_store, "source_provenance", lambda _: source)
    monkeypatch.setattr(acceptance, "source_provenance", lambda _: source)
    controls = run_geometry_controls(artifact_root=artifact_root)
    prep = {
        "destination": str(tmp_path / "evidence"),
        "artifact_root": str(artifact_root),
        "source": source,
        "dataset_id": "fixture",
        "baselines": {task: {"metrics": {"micro": {"f1": 0.5}}} for task in ("pizza", "component")},
        "limitations": [],
    }
    (work / "prepare.json").write_text(json.dumps(prep))
    verification = {
        "review_references": ["review-a", "review-b"],
        "mathematical_numerical_tests": {"status": "passed", "command": "FAKE", "test_count": 999},
    }
    passed_tests = {
        "status": "passed",
        "command": ["python", "-m", "pytest"],
        "test_count": 10,
        "returncode": 0,
        "output_sha256": "a" * 64,
        "source": source,
    }
    monkeypatch.setattr(acceptance, "_run_mathematical_numerical_tests", lambda _: passed_tests)
    result = finalize(work, {}, controls, verification)
    report = result["report"]
    assert report["acceptance_A"]["status"] == "passed"
    assert report["acceptance_B"]["status"] == "not-established"
    assert report["acceptance_B"]["controls_status"] == "passed"
    assert all(item["status"] == "passed" for item in report["control_verification"])
    assert report["acceptance_C"]["status"] == "not-established"
    assert report["scientific_acceptance_status"] == "not-established"
    assert report["acceptance_A"]["ci_status_is_separate"] is True
    assert report["verification_evidence"]["mathematical_numerical_tests"] == passed_tests
    fabricated_controls = [dict(item, run_path="/tmp/fabricated") for item in controls]
    prep["destination"] = str(tmp_path / "fabricated-controls-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    fabricated = finalize(work, {}, fabricated_controls, verification)["report"]
    assert fabricated["acceptance_B"]["controls_status"] != "passed"
    assert fabricated["acceptance_B"]["status"] != "passed"
    control_manifest_path = Path(controls[0]["run_path"]) / "manifest.json"
    valid_manifest_text = control_manifest_path.read_text()
    altered_manifest = json.loads(valid_manifest_text)
    altered_manifest["config"]["seed"] = 24
    control_manifest_path.write_text(json.dumps(altered_manifest))
    prep["destination"] = str(tmp_path / "invalid-control-manifest-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    invalid_manifest = finalize(work, {}, controls, verification)["report"]
    assert invalid_manifest["acceptance_B"]["controls_status"] != "passed"
    control_manifest_path.write_text(valid_manifest_text)
    metrics_path = Path(controls[0]["run_path"]) / "metrics.jsonl"
    valid_metrics_text = metrics_path.read_text()
    metric_row = json.loads(valid_metrics_text.splitlines()[0])
    metric_row["geometry_gradient_norm"] = 1.0
    metrics_path.write_text(
        "\\n".join([json.dumps(metric_row), *valid_metrics_text.splitlines()[1:]]) + "\\n"
    )
    prep["destination"] = str(tmp_path / "invalid-control-metrics-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    invalid_metrics = finalize(work, {}, controls, verification)["report"]
    assert invalid_metrics["acceptance_B"]["controls_status"] != "passed"
    assert "checksum" in " ".join(invalid_metrics["control_verification"][0]["errors"])
    metrics_path.write_text(valid_metrics_text)
    prep["destination"] = str(tmp_path / "failed-math-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    monkeypatch.setattr(
        acceptance,
        "_run_mathematical_numerical_tests",
        lambda _: {**passed_tests, "status": "failed", "returncode": 1, "test_count": 0},
    )
    failed_math = finalize(work, {}, controls, verification)["report"]
    assert failed_math["acceptance_A"]["status"] == "failed"
    assert failed_math["verification_evidence"]["mathematical_numerical_tests"]["test_count"] == 0
    monkeypatch.setattr(acceptance, "_run_mathematical_numerical_tests", lambda _: passed_tests)
    prep["destination"] = str(tmp_path / "second-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    controls[0]["objective"] = "hinge"
    duplicated = finalize(work, {}, controls, verification)["report"]
    assert duplicated["acceptance_B"]["control_combinations_exact"] is False
    assert duplicated["acceptance_B"]["status"] == "not-established"
    prep["destination"] = str(tmp_path / "third-evidence")
    (work / "prepare.json").write_text(json.dumps(prep))
    controls[0]["objective"] = "logistic"
    controls[0]["geometry_gradient_norm_max"] = 0.0
    weak = finalize(work, {}, controls, verification)["report"]
    assert weak["acceptance_B"]["status"] == "failed"


def test_mathematical_evidence_runs_fixed_tests_and_binds_output_to_source(monkeypatch):
    import hashlib

    import ontology_embeddings.pr3_acceptance as acceptance

    source = {"availability": "available", "dirty": False, "commit_sha": "source-sha"}
    monkeypatch.setattr(acceptance, "source_provenance", lambda _: source)
    output = "12 passed in 1.25s\\n"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(acceptance.subprocess, "run", fake_run)
    evidence = acceptance._run_mathematical_numerical_tests("source-sha")
    assert evidence["status"] == "passed"
    assert evidence["test_count"] == 12
    assert evidence["output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
    assert evidence["source_stable"] is True
    assert calls[0][0][-2:] == ["tests/test_box.py", "tests/test_box_training.py"]
    source_states = iter((source, {**source, "dirty": True}))
    monkeypatch.setattr(acceptance, "source_provenance", lambda _: next(source_states))
    changed_source = acceptance._run_mathematical_numerical_tests("source-sha")
    assert changed_source["status"] == "failed"
    assert changed_source["source_stable"] is False


def test_main_run_provenance_requires_exact_preparation_config_data_and_checkpoint(tmp_path):
    import copy
    import hashlib

    from ontology_embeddings.pr3_acceptance import COMMON, DATASET_ID

    domain = "bounded"
    ids = {"train": [f"t{i}" for i in range(720)], "validation": [f"v{i}" for i in range(240)]}
    hashes = {
        split: hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
        for split, value in ids.items()
    }
    identity = {
        "dataset_id": DATASET_ID,
        "manifest_sha256": "a" * 64,
        "class_manifest_sha256": "b" * 64,
        "class_ids": [1, 20],
        "class_names": ["A", "B"],
        "class_iris": ["urn:A", "urn:B"],
        "fixed_positive_class_ids": [1],
        "unsupported_class_ids": [],
        "train_ids_sha256": hashes["train"],
        "validation_ids_sha256": hashes["validation"],
        "train_count": 720,
        "validation_count": 240,
    }
    source = {
        "commit_sha": "source-sha",
        "dirty": False,
        "relevant_source_sha256": {},
        "source_fingerprint_sha256": None,
        "availability": "available",
        "fingerprint_note": "working-tree source paths hashed; non-source diff details omitted",
    }
    prep = {
        "dataset_identity": identity,
        "source": source,
        "device": "cuda:0",
    }
    config = {
        **COMMON,
        "domain": domain,
        "margin": 0.1,
        "model": "box-union-two-adapters",
        "optimizer": "Adam",
        "optimizer_groups": ["pizza_adapter", "component_adapter", "geometry"],
        "betas": [0.9, 0.999],
        "optimizer_epsilon": 1e-8,
        "amsgrad": False,
        "device": "cuda:0",
        "paired_initialization": {},
    }
    dataset = {
        "dataset_id": identity["dataset_id"],
        "dataset_manifest_sha256": identity["manifest_sha256"],
        "class_manifest_sha256": identity["class_manifest_sha256"],
        "class_ids": identity["class_ids"],
        "class_names": identity["class_names"],
        "class_iris": identity["class_iris"],
        "fixed_positive_class_ids": identity["fixed_positive_class_ids"],
        "unsupported_class_ids": identity["unsupported_class_ids"],
        "train_sample_ids": ids["train"],
        "validation_sample_ids": ids["validation"],
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "checkpoints").mkdir()
    checkpoint_path = run / "checkpoints" / "best.pt"
    metadata = {
        "domain": domain,
        "latent_dim": 16,
        "boxes": 1,
        "epsilon": 1e-4,
        "class_ids": identity["class_ids"],
        "class_names": identity["class_names"],
        "class_iris": identity["class_iris"],
        "vocabulary_sha256": identity["class_manifest_sha256"],
        "encoder": {
            "pizza_features": len(PIZZA_FEATURE_NAMES),
            "component_features": len(COMPONENT_FEATURE_NAMES),
            "bounded_activation": "tanh",
        },
        "objective": {"kind": "logistic", "tau": 0.1, "margin": 0.1},
        "training_config": config,
    }
    torch.save(
        {
            "metadata": metadata,
            "epoch": 1,
            "validation_loss": 1.0,
            "model_state": {"weight": torch.ones(1)},
        },
        checkpoint_path,
    )
    history = [
        {
            "epoch": epoch,
            "validation_loss": float(epoch),
            "diagnostics": {
                "validation": {
                    "pizza": {"micro": {"f1": 0.6}},
                    "component": {"micro": {"f1": 0.7}},
                },
                "train": {},
            },
        }
        for epoch in range(1, 61)
    ]
    summary = {
        "run_id": "run-1",
        "status": "completed",
        "run_path": str(run),
        "history": history,
        "checkpoint_paths": {"best": str(checkpoint_path)},
        "best_checkpoint_reload_predictions_equivalent": True,
    }
    manifest = {
        "run_id": "run-1",
        "status": "completed",
        "config": config,
        "dataset": dataset,
        "source": source,
        "summary": summary,
    }
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in history) + "\n")
    result = {**summary, "run_path": str(run)}
    assert _validate_main_run_provenance(run, result, domain, prep) == []
    changed_equivalence = {
        **result,
        "best_checkpoint_reload_predictions_equivalent": False,
    }
    assert _validate_main_run_provenance(run, changed_equivalence, domain, prep)
    altered_history = copy.deepcopy(history)
    altered_history[0]["diagnostics"]["validation"]["pizza"]["micro"]["f1"] = 0.99
    altered_summary = {**summary, "history": altered_history}
    altered_manifest = {**manifest, "summary": altered_summary}
    (run / "summary.json").write_text(json.dumps(altered_summary))
    (run / "manifest.json").write_text(json.dumps(altered_manifest))
    altered_result = {**altered_summary, "run_path": str(run)}
    assert _validate_main_run_provenance(run, altered_result, domain, prep)
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "manifest.json").write_text(json.dumps(manifest))

    mutations = [
        ("config", "domain", "unbounded"),
        ("config", "objective", "hinge"),
        ("config", "seed", 24),
        ("config", "boxes", True),
        ("dataset", "dataset_id", "wrong-dataset"),
        ("dataset", "class_ids", [True, 20]),
        ("dataset", "train_sample_ids", list(reversed(ids["train"]))),
        ("dataset", "fixed_positive_class_ids", [20]),
    ]
    for section, key, value in mutations:
        altered = copy.deepcopy(manifest)
        altered[section][key] = value
        (run / "manifest.json").write_text(json.dumps(altered))
        assert _validate_main_run_provenance(run, result, domain, prep)
    (run / "manifest.json").write_text(json.dumps(manifest))

    payload = torch.load(checkpoint_path, weights_only=True)
    payload["metadata"]["domain"] = "unbounded"
    torch.save(payload, checkpoint_path)
    assert _validate_main_run_provenance(run, result, domain, prep)
    payload["metadata"] = {**metadata, "class_ids": [True, 20]}
    torch.save(payload, checkpoint_path)
    assert _validate_main_run_provenance(run, result, domain, prep)
    payload["metadata"] = {**metadata, "boxes": True}
    torch.save(payload, checkpoint_path)
    assert _validate_main_run_provenance(run, result, domain, prep)
    payload["metadata"] = metadata
    torch.save(payload, checkpoint_path)
    payload["epoch"] = True
    torch.save(payload, checkpoint_path)
    assert _validate_main_run_provenance(run, result, domain, prep)
    payload["epoch"] = 1
    payload["validation_loss"] = True
    torch.save(payload, checkpoint_path)
    assert _validate_main_run_provenance(run, result, domain, prep)
    payload["validation_loss"] = 1.0
    torch.save(payload, checkpoint_path)
    (run / "summary.json").write_text(json.dumps({**summary, "status": "failed"}))
    assert _validate_main_run_provenance(run, result, domain, prep)
    prep.update(
        destination=str(tmp_path / "evidence"),
        artifact_root=str(tmp_path),
        baselines={task: {"metrics": {"micro": {"f1": 0.0}}} for task in ("pizza", "component")},
        limitations=[],
    )
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "prepare.json").write_text(json.dumps(prep))
    final = finalize(tmp_path / "work", {"bounded": result})
    assert final["report"]["acceptance_C"]["status"] != "passed"
    assert final["report"]["scientific_acceptance_status"] != "passed"


def test_read_run_result_requires_consistent_manifest_and_summary(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    summary = {"run_id": "run-1", "status": "completed", "history": []}
    manifest = {"run_id": "run-1", "status": "completed", "summary": summary}
    (run / "summary.json").write_text(json.dumps(summary))
    (run / "manifest.json").write_text(json.dumps(manifest))
    assert _read_run_result(run)["status"] == "completed"
    manifest["run_id"] = "different"
    (run / "manifest.json").write_text(json.dumps(manifest))
    rejected = _read_run_result(run)
    assert rejected["status"] == "invalid-provenance"
    baselines = {task: {"metrics": {"micro": {"f1": 0.0}}} for task in ("pizza", "component")}
    assert assess_run(rejected, baselines)["accepted_C"] is False
    (run / "manifest.json").unlink()
    assert _read_run_result(run)["status"] == "invalid-provenance"
    (run / "summary.json").unlink()
    (run / "manifest.json").write_text(json.dumps({"run_id": "run-1", "status": "completed"}))
    assert _read_run_result(run)["status"] == "invalid-provenance"


def test_final_report_keeps_compact_geometry_history_and_historical_reference(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    artifact_root = tmp_path / "runs"
    prep = {
        "destination": str(tmp_path / "evidence"),
        "artifact_root": str(artifact_root),
        "dataset_id": "fixture",
        "dataset_hashes": {"manifest": "m", "samples": "s"},
        "selection_hashes": {"train_ids_sha256": "t", "validation_ids_sha256": "v"},
        "baselines": {task: {"metrics": {"micro": {"f1": 0.5}}} for task in ("pizza", "component")},
        "limitations": [],
    }
    (work / "prepare.json").write_text(json.dumps(prep))
    history = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "validation_loss": 1.0,
            "diagnostics": {
                "validation": {task: {"micro": {"f1": 0.6}} for task in ("pizza", "component")}
            },
            "encoder_gradient_norm_pre_clip_mean": 0.2,
            "geometry_gradient_norm_pre_clip_mean": 0.3,
            "geometry": {
                "width_min": 0.1,
                "width_max": 0.4,
                "training_point_coordinate_std_mean": 0.5,
            },
        }
    ]
    result = {
        domain: {
            "run_id": f"run-{domain}",
            "run_path": str(artifact_root / f"run-{domain}"),
            "status": "completed",
            "history": history,
            "geometry_snapshots": {
                label: {"lower": [[0.0]], "upper": [[1.0]]}
                for label in ("initial", "best", "final")
            },
        }
        for domain in ("bounded", "unbounded")
    }
    report = finalize(work, result)["report"]
    assert report["runs"]["bounded"]["physical_geometry_snapshots"].keys() == {
        "initial",
        "best",
        "final",
    }
    assert report["runs"]["bounded"]["epoch_trajectory"][0]["validation_task_micro_f1"] == {
        "pizza": 0.6,
        "component": 0.6,
    }
    assert report["runs"]["bounded"]["epoch_trajectory"][0]["geometry_scale"]["width_max"] == 0.4
    assert report["runs"]["bounded"]["raw_run_reference"] == {
        "run_id": "run-bounded",
        "relative_path": "run-bounded",
        "local_only": True,
    }
    historical = report["historical_affine_comparison"]
    assert historical["run_id"] == "20261123T105917-c00ea4941c34"
    assert historical["validation_micro_f1"] == {"pizza": 0.9894, "component": 1.0}
    assert historical["limitations"]


def test_safe_loader_skips_malformed_test_lines_before_decoding(tmp_path):
    source = Path("examples/pizza-48")
    for name in ("manifest.json", "samples.jsonl"):
        shutil.copy2(source / name, tmp_path / name)
    shutil.copytree(source / "vocabularies", tmp_path / "vocabularies")
    lines = (tmp_path / "samples.jsonl").read_text().splitlines()
    replaced = []
    for line in lines:
        if '"split":"test"' in line or '"split": "test"' in line:
            replaced.append('{"split":"test", BROKEN JSON MUST NOT DECODE')
        else:
            replaced.append(line)
    (tmp_path / "samples.jsonl").write_text("\n".join(replaced) + "\n")
    _, train, validation = load_train_validation(tmp_path)
    assert train and validation
    assert all(r.split == "train" for r in train)
    assert all(r.split == "validation" for r in validation)
