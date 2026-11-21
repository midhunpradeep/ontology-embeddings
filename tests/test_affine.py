from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ontology_embeddings.affine_smoke import run_smoke
from ontology_embeddings.dataset.image_data import targets_in_trainable_order
from ontology_embeddings.dataset.schema import ComponentRecord, DatasetRecord
from ontology_embeddings.features import (
    COMPONENT_FEATURE_NAMES,
    FEATURE_SCHEMA,
    PIZZA_FEATURE_NAMES,
    component_features,
    pizza_features,
)
from ontology_embeddings.model import AffineMultiLabelModel, masked_task_loss
from ontology_embeddings.run_store import RunRecord, find_runs
from ontology_embeddings.training import _loss, _pack, train


def record(*, topping_count: int = 1, category: str = "AnchovyTopping", sample: str = "train"):
    n = 66
    component = ComponentRecord(
        component_id=f"{sample}:0",
        detector_category_id=1,
        detector_category=category,
        bbox_xyxy_original=(0, 0, 1, 1),
        ontology_targets=(0,) * n,
        ontology_known_mask=(0,) * n,
        annotation_path="labels/a.json",
    )
    raw = {
        "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
        "has_base": "ThinAndCrispyBase",
        "country_of_origin": None,
    }
    raw["topping_counts"]["AnchovyTopping"] = topping_count
    return DatasetRecord(
        sample_id=sample,
        split="train",
        generator_type="excluded",
        image_path="x.png",
        image_width=1,
        image_height=1,
        pizza_targets=(0,) * n,
        pizza_known_mask=(0,) * n,
        pizza_source_labels=(),
        components=(component,),
        audit={"raw_generator": raw},
        source_annotations={},
    )


def test_features_are_versioned_deterministic_and_target_independent():
    first = record(topping_count=2)
    changed_labels = replace(first, pizza_targets=(1,) * 66)
    assert pizza_features(first) == pizza_features(changed_labels)
    assert pizza_features(first) == pizza_features(first)
    assert len(pizza_features(first)) == len(PIZZA_FEATURE_NAMES)
    assert len(component_features(first.components[0])) == len(COMPONENT_FEATURE_NAMES)
    assert FEATURE_SCHEMA["version"] == "metadata-oracle-v1"
    assert sum(component_features(first.components[0])) == 1
    assert pizza_features(first)[0] == 2
    assert pizza_features(first)[-1] == 1  # explicit missing-origin one-hot


def test_shared_affine_model_has_separate_adapters_and_exact_zero_boundary():
    model = AffineMultiLabelModel(3, 4, classes=2, latent_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[0] = 0
        model.head.bias[1] = 0.1
    predictions = model.predict(torch.zeros((1, 3)), "pizza")
    assert predictions.tolist() == [[False, True]]
    assert model.pizza_adapter is not model.component_adapter
    assert model.head.out_features == 2


def test_masked_loss_normalizes_known_applicable_and_per_pizza_component_means():
    # Unknown entities are omitted; the component reduction is by parent pizza.
    p_logits = torch.tensor([[-2.0], [2.0]])
    p_targets = torch.tensor([[0.0], [1.0]])
    p_known = torch.tensor([[True], [True]])
    p_app = torch.ones_like(p_known)
    c_logits = torch.tensor([[-2.0], [2.0], [2.0]])
    c_targets = torch.tensor([[0.0], [1.0], [1.0]])
    c_known = torch.ones_like(c_targets, dtype=torch.bool)
    c_app = torch.ones_like(c_known)
    total, tasks = masked_task_loss(
        p_logits, p_targets, p_known, p_app, c_logits, c_targets, c_known, c_app, [0, 0, 1]
    )
    assert total is not None
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([2.0]), torch.tensor([1.0])
    )
    assert tasks["pizza"] == pytest.approx(expected, abs=1e-5)
    assert tasks["component"] == pytest.approx(expected, abs=1e-5)
    assert total == pytest.approx(expected, abs=1e-5)


def test_class_targets_project_in_nontrivial_canonical_id_order():
    base = record()
    pizza_targets = [0] * 66
    pizza_known = [0] * 66
    component_targets = [0] * 66
    component_known = [0] * 66
    pizza_targets[4], pizza_targets[1], pizza_targets[3] = 1, 0, 1
    pizza_known[4], pizza_known[1], pizza_known[3] = 1, 0, 1
    component_targets[4], component_targets[1], component_targets[3] = 0, 1, 1
    component_known[4], component_known[1], component_known[3] = 1, 1, 0
    item = replace(
        base,
        pizza_targets=tuple(pizza_targets),
        pizza_known_mask=tuple(pizza_known),
        components=(replace(base.components[0], ontology_targets=tuple(component_targets),
                            ontology_known_mask=tuple(component_known)),),
    )
    dataset = SimpleNamespace(canonical_class_ids=(4, 1, 3))
    assert targets_in_trainable_order(dataset, item) == (
        (1, 0, 1), (1, 0, 1), ((0, 1, 1),), ((1, 1, 0),)
    )


def test_repeated_component_occurrences_are_all_parent_indexed():
    first = record(sample="first")
    second = record(sample="second")
    batch = [replace(first, components=(first.components[0], first.components[0])),
             replace(second, components=(second.components[0],))]
    dataset = SimpleNamespace(canonical_class_ids=tuple(range(60)))
    packed = _pack(batch, dataset, torch.device("cpu"))
    assert packed[3].shape[0] == 3
    assert packed[6] == [0, 0, 1]
    assert packed[3][0].tolist() == packed[3][1].tolist()


def test_known_but_inapplicable_nonfinite_logits_are_excluded_for_both_tasks():
    pizza = torch.tensor([[0.4, float("nan")]], requires_grad=True)
    pizza_targets = torch.tensor([[1.0, 0.0]])
    pizza_known = torch.tensor([[True, True]])
    pizza_applicable = torch.tensor([[True, False]])
    component = torch.tensor([[-0.7, float("inf")]], requires_grad=True)
    component_targets = torch.tensor([[0.0, 1.0]])
    component_known = torch.tensor([[True, True]])
    component_applicable = torch.tensor([[True, False]])
    total, tasks = masked_task_loss(
        pizza,
        pizza_targets,
        pizza_known,
        pizza_applicable,
        component,
        component_targets,
        component_known,
        component_applicable,
        [0],
    )
    assert total is not None and torch.isfinite(total)
    assert tasks["pizza"].item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([0.4]), torch.tensor([1.0])
        ).item()
    )
    assert tasks["component"].item() == pytest.approx(
        torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor([-0.7]), torch.tensor([0.0])
        ).item()
    )
    total.backward()
    assert torch.isfinite(pizza.grad[0, 0]) and pizza.grad[0, 1] == 0
    assert torch.isfinite(component.grad[0, 0]) and component.grad[0, 1] == 0


def test_masked_nonfinite_logits_do_not_contaminate_eligible_loss_or_gradients():
    pizza = torch.tensor([[0.2, float("nan")]], requires_grad=True)
    pizza_targets = torch.tensor([[1.0, 0.0]])
    pizza_known = torch.tensor([[True, False]])
    pizza_applicable = torch.tensor([[True, True]])
    component = torch.tensor([[0.3, float("inf")]], requires_grad=True)
    component_targets = torch.tensor([[0.0, 1.0]])
    component_known = torch.tensor([[True, False]])
    component_applicable = torch.tensor([[True, True]])
    total, tasks = masked_task_loss(
        pizza,
        pizza_targets,
        pizza_known,
        pizza_applicable,
        component,
        component_targets,
        component_known,
        component_applicable,
        [0],
    )
    assert total is not None and torch.isfinite(total)
    assert torch.isfinite(tasks["pizza"]) and torch.isfinite(tasks["component"])
    total.backward()
    assert torch.isfinite(pizza.grad[0, 0])
    assert pizza.grad[0, 1] == 0
    assert torch.isfinite(component.grad[0, 0])
    assert component.grad[0, 1] == 0


def test_initial_run_index_failure_is_recorded_failed_and_original_error_reraised(
    tmp_path, monkeypatch
):
    dataset = SimpleNamespace(
        dataset_id="initial-index-failure",
        dataset_manifest_sha256="m",
        class_manifest_sha256="c",
        schema_version="v",
        canonical_class_ids=(),
        pizza_supervision_class_ids=(),
        component_supervision_class_ids=(),
    )
    original_index = RunRecord._index
    calls = 0

    def fail_initial_index(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected initial index failure")
        return original_index(self)

    monkeypatch.setattr(RunRecord, "_index", fail_initial_index)
    with pytest.raises(OSError, match="injected initial index failure"):
        train(dataset, [], artifact_root=tmp_path, invocation="initial index failure")
    entries = find_runs(tmp_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    manifest = json.loads((tmp_path / entries[0]["run_id"] / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "injected initial index failure" in manifest["error"]


def test_initial_run_update_failure_is_recorded_failed_and_original_error_reraised(
    tmp_path, monkeypatch
):
    dataset = SimpleNamespace(
        dataset_id="update-failure",
        dataset_manifest_sha256="m",
        class_manifest_sha256="c",
        schema_version="v",
        canonical_class_ids=(),
        pizza_supervision_class_ids=(),
        component_supervision_class_ids=(),
    )
    original_index = RunRecord._index
    calls = 0

    def fail_once_after_registration(self):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected update-index failure")
        return original_index(self)

    monkeypatch.setattr(RunRecord, "_index", fail_once_after_registration)
    with pytest.raises(OSError, match="injected update-index failure"):
        train(dataset, [], artifact_root=tmp_path, invocation="index failure")
    entries = find_runs(tmp_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    manifest = json.loads((tmp_path / entries[0]["run_id"] / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "injected update-index failure" in manifest["error"]


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    from ontology_embeddings.training import _atomic_torch_save

    checkpoint = tmp_path / "latest.pt"
    torch.save({"version": "previous"}, checkpoint)

    def fail_temporary_save(value, path):
        assert Path(path) != checkpoint
        Path(path).write_bytes(b"partial replacement")
        raise OSError("injected checkpoint write failure")

    monkeypatch.setattr(torch, "save", fail_temporary_save)
    with pytest.raises(OSError, match="injected checkpoint write failure"):
        _atomic_torch_save({"version": "new"}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"version": "previous"}


def test_training_failure_is_registered_and_indexed(tmp_path):
    dataset = SimpleNamespace(
        dataset_id="bad",
        dataset_manifest_sha256="m",
        class_manifest_sha256="c",
        schema_version="v",
        canonical_class_ids=(),
        pizza_supervision_class_ids=(),
        component_supervision_class_ids=(),
    )
    with pytest.raises(ValueError, match="at least one pizza"):
        train(dataset, [], artifact_root=tmp_path, invocation="bad invocation")
    entries = find_runs(tmp_path)
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert find_runs(tmp_path, dataset_id="bad")[0]["run_id"] == entries[0]["run_id"]
    assert (
        find_runs(tmp_path, source_sha=entries[0]["source_sha"])[0]["run_id"]
        == entries[0]["run_id"]
    )
    run_id = entries[0]["run_id"]
    manifest = json.loads((tmp_path / run_id / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "at least one pizza" in manifest["error"]
    assert manifest["source"]["commit_sha"]


def test_run_provenance_does_not_attribute_an_unrelated_cwd_git_repo(tmp_path, monkeypatch):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=foreign, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=foreign, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=foreign, check=True)
    (foreign / "unrelated.txt").write_text("unrelated")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=foreign, check=True)
    subprocess.run(["git", "commit", "-qm", "foreign"], cwd=foreign, check=True)
    foreign_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=foreign, check=True, text=True, capture_output=True
    ).stdout.strip()
    monkeypatch.chdir(foreign)
    model_root = Path(__file__).resolve().parents[1]
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=model_root, check=True, text=True, capture_output=True
    ).stdout.strip()
    rec = RunRecord(tmp_path / "runs", purpose="debug", invocation="test", config={})
    assert rec.manifest["source"]["commit_sha"] == expected_sha
    assert rec.manifest["source"]["commit_sha"] != foreign_sha


def test_smoke_loader_failure_creates_one_failed_indexed_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_smoke(tmp_path / "missing", artifact_root=tmp_path / "runs")
    entries = find_runs(tmp_path / "runs")
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    manifest = json.loads((tmp_path / "runs" / entries[0]["run_id"] / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "FileNotFoundError" in manifest["error"]


def test_tiny_linear_overfit_updates_both_adapters_and_shared_head(tmp_path):
    class_ids = tuple(range(60))
    data = SimpleNamespace(
        dataset_id="synthetic-linear-control",
        dataset_manifest_sha256="synthetic",
        class_manifest_sha256="classes",
        schema_version="synthetic-v1",
        canonical_class_ids=class_ids,
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(1,),
    )
    examples = []
    for i, positive in enumerate((False, True, False, True)):
        comps = []
        for j in range(2):
            category = "AnchovyTopping" if positive else "GarlicTopping"
            targets = [0] * 66
            known = [0] * 66
            targets[1] = int(positive)
            known[1] = 1
            comps.append(
                ComponentRecord(
                    component_id=f"{i}:{j}",
                    detector_category_id=j + 1,
                    detector_category=category,
                    bbox_xyxy_original=(0, 0, 1, 1),
                    ontology_targets=tuple(targets),
                    ontology_known_mask=tuple(known),
                    annotation_path="synthetic",
                )
            )
        targets = [0] * 66
        known = [0] * 66
        targets[0] = int(positive)
        known[0] = 1
        raw = {
            "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
            "has_base": "ThinAndCrispyBase",
            "country_of_origin": None,
        }
        raw["topping_counts"]["AnchovyTopping"] = int(positive)
        examples.append(
            DatasetRecord(
                sample_id=f"synthetic-{i}",
                split="train",
                generator_type="do-not-use",
                image_path="",
                image_width=1,
                image_height=1,
                pizza_targets=tuple(targets),
                pizza_known_mask=tuple(known),
                pizza_source_labels=(),
                components=tuple(comps),
                audit={"raw_generator": raw},
                source_annotations={},
            )
        )
    result = train(
        data,
        examples,
        (),
        artifact_root=tmp_path,
        epochs=60,
        learning_rate=0.04,
        seed=11,
        purpose="smoke",
        invocation="tiny-linear-test",
    )
    final = result["history"][-1]
    assert final["train_loss"] < 1e-3
    assert final["train_pizza_loss"] < 1e-3
    assert final["train_component_loss"] < 1e-3
    assert final["validation_loss"] is None
    assert final["validation_pizza_loss"] is None
    assert final["validation_component_loss"] is None
    checkpoints = tmp_path / result["run_id"] / "checkpoints"
    assert (checkpoints / "latest.pt").is_file()
    assert not (checkpoints / "best.pt").exists()
    assert result["parameter_updates"] == {
        "pizza_adapter": True,
        "component_adapter": True,
        "head": True,
    }


def test_epoch_train_loss_reduces_over_full_partition_after_updates(tmp_path):
    class_ids = (0,)
    data = SimpleNamespace(
        dataset_id="uneven-batch-loss",
        dataset_manifest_sha256="synthetic",
        class_manifest_sha256="classes",
        schema_version="synthetic-v1",
        canonical_class_ids=class_ids,
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(),
    )
    examples = []
    for index, positive in enumerate((False, True, False)):
        item = record(topping_count=index, sample=f"uneven-{index}")
        targets = [0] * 66
        known = [0] * 66
        targets[0] = int(positive)
        known[0] = 1
        examples.append(replace(item, pizza_targets=tuple(targets), pizza_known_mask=tuple(known)))

    result = train(
        data,
        examples,
        (),
        artifact_root=tmp_path,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=19,
        purpose="smoke",
        invocation="uneven-batch-loss-test",
    )
    assert len(examples) % 2 == 1
    model = AffineMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), len(class_ids), latent_dim=16
    )
    checkpoint = torch.load(
        Path(result["run_path"]) / "checkpoints" / "latest.pt", weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.eval()
    with torch.no_grad():
        full_loss, full_tasks, _ = _loss(model, examples, data, torch.device("cpu"))
    assert result["history"][0]["train_loss"] == pytest.approx(full_loss.item(), abs=1e-7)
    assert result["history"][0]["train_pizza_loss"] == pytest.approx(
        full_tasks["pizza"].item(), abs=1e-7
    )
    assert result["history"][0]["train_component_loss"] is None


def test_empty_component_collection_is_a_valid_pizza_batch(tmp_path):
    data = SimpleNamespace(
        dataset_id="empty-components",
        dataset_manifest_sha256="synthetic",
        class_manifest_sha256="classes",
        schema_version="synthetic-v1",
        canonical_class_ids=tuple(range(60)),
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(1,),
    )
    targets = [0] * 66
    known = [0] * 66
    known[0] = 1
    raw = {
        "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
        "has_base": "ThinAndCrispyBase",
        "country_of_origin": None,
    }
    only_pizza = replace(
        record(),
        components=(),
        pizza_targets=tuple(targets),
        pizza_known_mask=tuple(known),
        audit={"raw_generator": raw},
    )
    result = train(
        data,
        [only_pizza],
        [only_pizza],
        artifact_root=tmp_path,
        epochs=1,
        purpose="smoke",
        invocation="empty-components-test",
    )
    assert result["status"] == "completed"
    assert result["history"][0]["validation_component_loss"] is None
