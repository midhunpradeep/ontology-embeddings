from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ontology_embeddings.box import box_clearance, box_membership
from ontology_embeddings.box_training import BoxMultiLabelModel, train_box
from ontology_embeddings.dataset.schema import ComponentRecord, DatasetRecord
from ontology_embeddings.features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from ontology_embeddings.shared_training import masked_geometric_task_loss
from ontology_embeddings.town_training import (
    TownMultiLabelModel,
    load_box_initialization,
    train_town,
)


def _dataset():
    return SimpleNamespace(
        dataset_id="town-test",
        dataset_manifest_sha256="dataset-sha",
        schema_version="v1",
        class_manifest_sha256="vocab-sha",
        canonical_class_ids=(0, 1),
        class_names=("A", "B"),
        class_iris=("urn:A", "urn:B"),
        pizza_supervision_class_ids=(0, 1),
        component_supervision_class_ids=(0, 1),
        fixed_positive_class_ids=(),
        unsupported_class_ids=(),
    )


def _record(sample, split, positive):
    targets = [0] * 66
    known = [0] * 66
    targets[0] = int(positive)
    known[0] = 1
    targets[1] = int(not positive)
    known[1] = 1
    component_targets = list(targets)
    component_known = list(known)
    component = ComponentRecord(
        f"{sample}:0", 1, "AnchovyTopping", (0, 0, 1, 1),
        tuple(component_targets), tuple(component_known), "town-test",
    )
    raw = {
        "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
        "has_base": "ThinAndCrispyBase",
        "country_of_origin": None,
    }
    raw["topping_counts"]["AnchovyTopping"] = int(positive)
    return DatasetRecord(
        sample, split, "synthetic", "", 1, 1, tuple(targets), tuple(known), (),
        (component,), {"raw_generator": raw}, {},
    )


def _town(dataset, *, domain="unbounded", houses=1, windows=0, seed=19, latent_dim=3):
    return TownMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), dataset.canonical_class_ids,
        dataset.class_manifest_sha256, latent_dim=latent_dim, houses=houses, windows=windows,
        domain=domain, seed=seed, class_names=dataset.class_names, class_iris=dataset.class_iris,
    )


def test_h1w0_is_box_equivalent_for_membership_scores_gradients_loss_and_update():
    torch.manual_seed(19)
    dataset = _dataset()
    box = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), dataset.canonical_class_ids,
        dataset.class_manifest_sha256, latent_dim=3, boxes=1, domain="unbounded", seed=19,
        class_names=dataset.class_names, class_iris=dataset.class_iris,
    )
    town = _town(dataset, domain="unbounded", houses=1, windows=0, seed=19)
    with torch.no_grad():
        town.pizza_adapter.load_state_dict(copy.deepcopy(box.pizza_adapter.state_dict()))
        town.component_adapter.load_state_dict(copy.deepcopy(box.component_adapter.state_dict()))
        town.geometry.raw_frame_a.copy_(box.geometry.raw_a)
        town.geometry.raw_frame_b.copy_(box.geometry.raw_b)
    assert town.geometry.raw_window_a is None
    assert not any("raw_window" in name for name, _ in town.named_parameters())
    assert [group["name"] for group in town.optimizer_parameter_groups()] == [
        "pizza_adapter", "component_adapter", "frame_geometry"
    ]

    features = torch.randn((5, len(PIZZA_FEATURE_NAMES)))
    component_features = torch.randn((3, len(COMPONENT_FEATURE_NAMES)))
    targets = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.], [1., 0.]])
    component_targets = torch.tensor([[0., 1.], [1., 0.], [0., 1.]])
    known = torch.ones_like(targets, dtype=torch.bool)
    component_known = torch.ones_like(component_targets, dtype=torch.bool)
    applicable = torch.ones_like(known)
    component_applicable = torch.ones_like(component_known)
    parents = [0, 1, 1]
    box_lo, box_hi = box.endpoints()
    frame_lo, frame_hi, window_lo, window_hi = town.endpoints()
    assert window_lo.shape == window_hi.shape == (2, 1, 0, 3)
    for entity_features, entity in ((features, "pizza"), (component_features, "component")):
        box_points = box.encode(entity_features, entity)
        town_points = town.encode(entity_features, entity)
        assert torch.equal(box_clearance(box_points, box_lo, box_hi, domain="unbounded"),
                           town.geometry.clearance(town_points))
        assert torch.equal(box_membership(box_points, box_lo, box_hi, domain="unbounded"),
                           town.geometry.membership(town_points))

    def objective(model, score_points):
        pizza_scores = score_points(model.encode(features, "pizza"))
        component_scores = score_points(model.encode(component_features, "component"))
        return masked_geometric_task_loss(
            pizza_scores, targets, known, applicable,
            component_scores, component_targets, component_known, component_applicable, parents,
            kind="logistic",
        )[0]

    def box_score(points):
        lower, upper = box.endpoints()
        return box_clearance(points, lower, upper, domain="unbounded")

    box_loss = objective(box, box_score)
    town_loss = objective(town, town.geometry.clearance)
    assert torch.equal(box_loss, town_loss)
    box_loss.backward()
    town_loss.backward()
    for box_name, town_name in (
        ("pizza_adapter.weight", "pizza_adapter.weight"),
        ("pizza_adapter.bias", "pizza_adapter.bias"),
        ("component_adapter.weight", "component_adapter.weight"),
        ("component_adapter.bias", "component_adapter.bias"),
        ("geometry.raw_a", "geometry.raw_frame_a"),
        ("geometry.raw_b", "geometry.raw_frame_b"),
    ):
        box_grad = dict(box.named_parameters())[box_name].grad
        town_grad = dict(town.named_parameters())[town_name].grad
        assert torch.allclose(box_grad, town_grad, atol=1e-7, rtol=0), (box_name, town_name)

    box_opt = torch.optim.Adam(box.parameters(), lr=0.01)
    town_opt = torch.optim.Adam(town.optimizer_parameter_groups(), lr=0.01)
    box_opt.zero_grad(set_to_none=True)
    town_opt.zero_grad(set_to_none=True)
    box_loss = objective(box, box_score)
    town_loss = objective(town, town.geometry.clearance)
    box_loss.backward()
    town_loss.backward()
    torch.nn.utils.clip_grad_norm_(box.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(town.parameters(), 1.0)
    box_opt.step()
    town_opt.step()
    for box_name, town_name in (
        ("pizza_adapter.weight", "pizza_adapter.weight"),
        ("pizza_adapter.bias", "pizza_adapter.bias"),
        ("component_adapter.weight", "component_adapter.weight"),
        ("component_adapter.bias", "component_adapter.bias"),
        ("geometry.raw_a", "geometry.raw_frame_a"),
        ("geometry.raw_b", "geometry.raw_frame_b"),
    ):
        assert torch.allclose(
            dict(box.named_parameters())[box_name], dict(town.named_parameters())[town_name],
            atol=2e-6, rtol=0,
        )


@pytest.mark.parametrize("objective", ["logistic", "hinge"])
def test_trainer_produced_checkpoint_reloads_and_reports_frame_window_updates(tmp_path, objective):
    dataset = _dataset()
    train = (_record("train-0", "train", True), _record("train-1", "train", False))
    validation = (_record("valid-0", "validation", True), _record("valid-1", "validation", False))
    result = train_town(
        dataset, train, validation, artifact_root=tmp_path / "runs", seed=2, epochs=1,
        batch_size=2, latent_dim=3, houses=1, windows=1, domain="unbounded",
        learning_rate=0.01, objective=objective,
    )
    assert result["status"] == "completed"
    assert result["parameter_updates"]["frame_geometry"]
    assert result["parameter_updates"]["window_geometry"]
    assert result["best_checkpoint_reload_predictions_equivalent"] is True
    assert result["history"][0]["gradient_norm_pre_clip_mean"]["frame_geometry"] > 0
    assert result["history"][0]["gradient_norm_pre_clip_mean"]["window_geometry"] > 0
    assert "window_geometry" in result["history"][0]["movement_norms"]
    assert result["runtime_seconds"] > 0
    assert result["peak_rss_kib_process_cumulative"] > 0
    assert result["geometry_snapshots"]["initial"]["windows"]["count"] == 1

    consumer = _town(dataset, domain="unbounded", houses=1, windows=1, seed=2)
    consumer.load_checkpoint(
        result["checkpoint_paths"]["best"], objective=objective, tau=0.1, margin=0.1,
        training_config=result["training_config"],
    )
    assert torch.equal(
        consumer.predict(torch.randn((0, len(PIZZA_FEATURE_NAMES))), "pizza"),
        torch.empty((0, 2), dtype=torch.bool),
    )


def test_trainer_rejects_invalid_optimizer_update_without_persisting_invalid_state(
    tmp_path, monkeypatch
):
    dataset = _dataset()
    train = (_record("train-0", "train", True), _record("train-1", "train", False))
    real_step = torch.optim.Adam.step

    def poison_step(optimizer, *args, **kwargs):
        result = real_step(optimizer, *args, **kwargs)
        frame_group = next(group for group in optimizer.param_groups
                           if group.get("name") == "frame_geometry")
        frame_group["params"][0].data[0, 0, 0] = float("nan")
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", poison_step)
    with pytest.raises(FloatingPointError, match="invalid TOWN update"):
        train_town(
            dataset, train, artifact_root=tmp_path / "runs", seed=3, epochs=1, batch_size=2,
            latent_dim=3, houses=1, windows=0, domain="unbounded",
        )
    run = next(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    initial = torch.load(run / "checkpoints" / "initial.pt", weights_only=True)
    latest = torch.load(run / "checkpoints" / "latest.pt", weights_only=True)
    assert all(torch.equal(initial["model_state"][key], latest["model_state"][key])
               for key in initial["model_state"])
    assert (run / "manifest.json").exists()


def test_town_checkpoint_rejects_geometry_tamper_before_state_mutation(tmp_path):
    dataset = _dataset()
    model = _town(dataset, windows=0, seed=31)
    checkpoint = tmp_path / "town.pt"
    config = {"seed": 31, "purpose": "checkpoint-validation"}
    model.save_checkpoint(
        checkpoint, objective="hinge", tau=0.1, margin=0.2, epoch=0,
        validation_loss=None, training_config=config,
    )
    payload = torch.load(checkpoint, weights_only=True)
    payload["geometry"]["frame_lower"][0, 0, 0] += 0.25
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    consumer = _town(dataset, windows=0, seed=31)
    before = {key: value.detach().clone() for key, value in consumer.state_dict().items()}
    with pytest.raises(ValueError, match="snapshot"):
        consumer.load_checkpoint(
            tampered, objective="hinge", tau=0.1, margin=0.2, training_config=config,
        )
    assert all(torch.equal(before[key], value) for key, value in consumer.state_dict().items())


def test_historical_box_trainer_checkpoint_is_loaded_and_validated(tmp_path):
    dataset = _dataset()
    train = (_record("train-0", "train", True), _record("train-1", "train", False))
    result = train_box(
        dataset, train, artifact_root=tmp_path / "box-runs", seed=21, epochs=1, batch_size=2,
        latent_dim=3, boxes=1, domain="unbounded", learning_rate=0.01,
    )
    checkpoint = result["checkpoint_paths"]["initial"]
    latest_checkpoint = result["checkpoint_paths"]["latest"]
    loaded = load_box_initialization(
        checkpoint, dataset, domain="unbounded", houses=1, latent_dim=3,
    )
    assert loaded["frame_lower"].shape == (2, 1, 3)
    assert set(loaded["adapter_state"]) == {
        "pizza_adapter.weight", "pizza_adapter.bias", "component_adapter.weight",
        "component_adapter.bias",
    }
    assert loaded["provenance"]["producer_family"] == "box-union-classifier"
    assert loaded["provenance"]["source_epoch"] == 0
    with pytest.raises(ValueError, match="initial.*epoch 0"):
        load_box_initialization(
            latest_checkpoint, dataset, domain="unbounded", houses=1, latent_dim=3,
        )
    integrated = train_town(
        dataset, train, (_record("valid-0", "validation", True),),
        artifact_root=tmp_path / "town-runs", seed=22, epochs=1, batch_size=2,
        latent_dim=3, houses=1, windows=1, domain="unbounded",
        initial_box_checkpoint=checkpoint,
    )
    assert integrated["training_config"]["historical_box_initialization"]["artifact_sha256"] == loaded["provenance"]["artifact_sha256"]
    with pytest.raises(ValueError, match="class|vocabulary|identity"):
        wrong = SimpleNamespace(**{**vars(dataset), "class_manifest_sha256": "wrong-vocab"})
        load_box_initialization(checkpoint, wrong, domain="unbounded", houses=1, latent_dim=3)
