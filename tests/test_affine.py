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
from ontology_embeddings.run_store import RunRecord, export_run, find_runs, verify_export
from ontology_embeddings.training import (
    _loss,
    _pack,
    _signature_diagnostics,
    _task_metrics,
    shuffle_training_labels,
    train,
    train_shuffled_label_control,
)


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


def test_task_metrics_count_only_parent_pizzas_with_eligible_component_labels():
    data = SimpleNamespace(
        dataset_id="eligible-component-pizzas",
        dataset_manifest_sha256="synthetic",
        class_manifest_sha256="classes",
        schema_version="synthetic-v1",
        canonical_class_ids=(0,),
        pizza_supervision_class_ids=(),
        component_supervision_class_ids=(0,),
    )
    eligible = record(sample="eligible")
    component_known = [0] * 66
    component_known[0] = 1
    eligible_component = replace(
        eligible.components[0], ontology_known_mask=tuple(component_known)
    )
    eligible = replace(eligible, components=(eligible_component,))
    no_eligible_labels = record(sample="no-eligible-labels")
    model = AffineMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), classes=1, latent_dim=2
    )

    _, _, eligible_batch = _loss(model, [eligible], data, torch.device("cpu"))
    _, _, ineligible_batch = _loss(model, [no_eligible_labels], data, torch.device("cpu"))

    assert _task_metrics(eligible_batch, ["class"])["component"]["eligible_pizza_count"] == 1
    assert _task_metrics(ineligible_batch, ["class"])["component"]["eligible_pizza_count"] == 0
    assert _task_metrics(eligible_batch, ["class"])["pizza"]["eligible_pizza_count"] == 0

    _, _, mixed_validation_batch = _loss(
        model, [eligible, no_eligible_labels], data, torch.device("cpu")
    )
    signatures = _signature_diagnostics(
        [eligible], [eligible, no_eligible_labels], mixed_validation_batch, ["class"]
    )["component"]
    assert signatures["validation_seen_metrics"]["eligible_pizza_count"] == 1
    assert signatures["validation_unseen_metrics"]["eligible_pizza_count"] == 0


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


def test_shuffled_training_control_is_seeded_and_preserves_validation_records(tmp_path):
    first_targets, second_targets, known = [0] * 66, [0] * 66, [0] * 66
    second_targets[0] = 1
    known[0] = 1
    first = replace(
        record(sample="first"), pizza_targets=tuple(first_targets), pizza_known_mask=tuple(known)
    )
    second = replace(
        record(sample="second"), pizza_targets=tuple(second_targets), pizza_known_mask=tuple(known)
    )
    shuffled_a = shuffle_training_labels([first, second], seed=31)
    shuffled_b = shuffle_training_labels([first, second], seed=31)
    assert [item.pizza_targets for item in shuffled_a] == [
        item.pizza_targets for item in shuffled_b
    ]
    assert sorted(item.pizza_targets for item in shuffled_a) == sorted(
        [first.pizza_targets, second.pizza_targets]
    )
    assert [item.sample_id for item in shuffled_a] == ["first", "second"]
    assert [item.pizza_known_mask for item in shuffled_a] == [
        first.pizza_known_mask,
        second.pizza_known_mask,
    ]
    validation = record(sample="validation")
    before = validation.pizza_targets
    data = SimpleNamespace(
        dataset_id="shuffle",
        dataset_manifest_sha256="m",
        class_manifest_sha256="c",
        schema_version="v",
        canonical_class_ids=(0,),
        class_names=("class-0",),
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(),
    )
    result = train_shuffled_label_control(
        data,
        [first, second],
        [validation],
        artifact_root=tmp_path,
        seed=31,
        epochs=1,
    )
    assert result["status"] == "completed"
    manifest = json.loads((Path(result["run_path"]) / "manifest.json").read_text())
    assert manifest["purpose"] == "shuffled-label-control"
    assert manifest["dataset"]["validation_sample_ids"] == ["validation"]
    assert validation.pizza_targets == before


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
        components=(
            replace(
                base.components[0],
                ontology_targets=tuple(component_targets),
                ontology_known_mask=tuple(component_known),
            ),
        ),
    )
    dataset = SimpleNamespace(canonical_class_ids=(4, 1, 3))
    assert targets_in_trainable_order(dataset, item) == (
        (1, 0, 1),
        (1, 0, 1),
        ((0, 1, 1),),
        ((1, 1, 0),),
    )


def test_repeated_component_occurrences_are_all_parent_indexed():
    first = record(sample="first")
    second = record(sample="second")
    batch = [
        replace(first, components=(first.components[0], first.components[0])),
        replace(second, components=(second.components[0],)),
    ]
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


def test_component_parent_map_rejects_length_and_range_errors():
    p_logits = torch.zeros((2, 1))
    p_targets = torch.zeros_like(p_logits)
    p_mask = torch.ones_like(p_logits, dtype=torch.bool)
    c_logits = torch.zeros((1, 1))
    c_targets = torch.zeros_like(c_logits)
    c_mask = torch.ones_like(c_logits, dtype=torch.bool)
    args = (p_logits, p_targets, p_mask, p_mask, c_logits, c_targets, c_mask, c_mask)
    with pytest.raises(ValueError, match="component_parent length"):
        masked_task_loss(*args, [])
    with pytest.raises(ValueError, match="component_parent.*range"):
        masked_task_loss(*args, [2])


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


def test_run_export_checksum_and_interruption_status(tmp_path):
    run = RunRecord(tmp_path / "runs", purpose="debug", invocation="export", config={}, issue="8")
    run.metric({"epoch": 1})
    run.interrupt(KeyboardInterrupt())
    assert find_runs(tmp_path / "runs", status="interrupted", issue="8")[0]["run_id"] == run.run_id
    exported = export_run(tmp_path / "runs", run.run_id, tmp_path / "exported" / run.run_id)
    assert verify_export(exported["path"])
    (Path(exported["path"]) / "metrics.jsonl").write_text("changed")
    assert not verify_export(exported["path"])


def test_find_runs_ignores_torn_trailing_index_line_and_keeps_search(tmp_path):
    record_store = RunRecord(
        tmp_path,
        purpose="debug",
        invocation="index test",
        config={"model": "affine"},
        dataset={"dataset_id": "fixture"},
    )
    with (tmp_path / "index.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"run_id":')
    entries = find_runs(tmp_path, dataset_id="fixture")
    assert [item["run_id"] for item in entries] == [record_store.run_id]


@pytest.mark.parametrize("terminal_status", ["failed", "interrupted"])
def test_find_runs_reconciles_manifest_after_torn_terminal_index_append(
    tmp_path, terminal_status
):
    store = tmp_path / "runs"
    run = RunRecord(
        store,
        purpose="debug",
        invocation="terminal reconciliation",
        config={"model": "affine"},
        dataset={"dataset_id": "fixture"},
    )
    if terminal_status == "failed":
        run.fail(RuntimeError("synthetic failure"))
    else:
        run.interrupt(KeyboardInterrupt("synthetic interruption"))

    index = store / "index.jsonl"
    lines = index.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    index.write_bytes(lines[0] + b'{"run_id":')

    matches = find_runs(store, status=terminal_status)
    assert [item["run_id"] for item in matches] == [run.run_id]
    assert matches[0]["dataset_id"] == "fixture"


def test_find_runs_discovers_manifest_without_index_row(tmp_path):
    store = tmp_path / "runs"
    run = RunRecord(
        store,
        purpose="debug",
        invocation="manifest recovery",
        config={"model": "affine"},
        dataset={"dataset_id": "manifest-only"},
    )
    (store / "index.jsonl").write_bytes(b"")

    matches = find_runs(store, dataset_id="manifest-only", status="running")
    assert [item["run_id"] for item in matches] == [run.run_id]


def test_trainer_rejects_sealed_test_records_before_evaluation(tmp_path):
    data = SimpleNamespace(
        dataset_id="sealed",
        dataset_manifest_sha256="m",
        class_manifest_sha256="c",
        schema_version="v",
        canonical_class_ids=(0,),
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(),
    )
    sealed = replace(record(sample="sealed"), split="test")
    with pytest.raises(ValueError, match="sealed test"):
        train(data, [record()], [sealed], artifact_root=tmp_path)
    assert find_runs(tmp_path, status="failed")[0]["dataset_id"] == "sealed"


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
    unseen = replace(
        only_pizza,
        sample_id="unseen-feature",
        audit={
            "raw_generator": {
                **raw,
                "topping_counts": {**raw["topping_counts"], "AnchovyTopping": 2},
            }
        },
    )
    result = train(
        data,
        [only_pizza],
        [only_pizza, unseen],
        artifact_root=tmp_path,
        epochs=1,
        purpose="smoke",
        invocation="empty-components-test",
    )
    assert result["status"] == "completed"
    assert result["best_checkpoint_reload_metrics_equivalent"] is True
    assert result["baseline"]["pizza"]["training_known"][0] == 1
    assert (
        result["history"][0]["diagnostics"]["validation"]["pizza"]["per_class"]["0"][
            "known_negative"
        ]
        == 2
    )
    signatures = result["history"][0]["diagnostics"]["feature_signatures"]["pizza"]
    assert signatures["validation_seen_count"] == 1
    assert signatures["validation_unseen_count"] == 1
    assert signatures["overlap_distinct_signatures"] == 1
    assert result["history"][0]["validation_component_loss"] is None
