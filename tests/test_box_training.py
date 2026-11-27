from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ontology_embeddings.box_training import (
    BoxMultiLabelModel,
    _geometry_diagnostics,
    masked_box_task_loss,
    train_box,
)
from ontology_embeddings.dataset.schema import ComponentRecord, DatasetRecord
from ontology_embeddings.features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES


def _record(sample, split, positive):
    n = 66
    targets = [0] * n
    known = [0] * n
    targets[0] = int(positive)
    known[0] = 1
    ct = [0] * n
    ck = [0] * n
    ct[1] = int(positive)
    ck[1] = 1
    comp = ComponentRecord(
        f"{sample}:0", 1, "AnchovyTopping", (0, 0, 1, 1), tuple(ct), tuple(ck), "synthetic"
    )
    raw = {
        "topping_counts": {name.split(":", 1)[1]: 0 for name in PIZZA_FEATURE_NAMES[:16]},
        "has_base": "ThinAndCrispyBase",
        "country_of_origin": None,
    }
    raw["topping_counts"]["AnchovyTopping"] = int(positive)
    return DatasetRecord(
        sample,
        split,
        "excluded",
        "",
        1,
        1,
        tuple(targets),
        tuple(known),
        (),
        (comp,),
        {"raw_generator": raw},
        {},
    )


def _dataset():
    return SimpleNamespace(
        dataset_id="tiny",
        dataset_manifest_sha256="data",
        schema_version="v1",
        class_manifest_sha256="vocab",
        canonical_class_ids=(0, 1),
        class_names=("A", "B"),
        class_iris=("iri:A", "iri:B"),
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(1,),
    )


def test_box_masked_loss_matches_hierarchical_pr2_reduction_and_ignores_unknowns():
    ps = torch.tensor([[1.0, float("nan")], [3.0, 2.0]])
    pt = torch.tensor([[1.0, float("nan")], [1.0, 0.0]])
    pk = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    pa = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    cs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
    ct = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    ck = torch.ones_like(ct, dtype=torch.bool)
    ca = torch.ones_like(ck)
    total, tasks = masked_box_task_loss(ps, pt, pk, pa, cs, ct, ck, ca, [0, 0, 1])
    expected_p = (
        torch.nn.functional.softplus(torch.tensor(-10.0))
        + (
            torch.nn.functional.softplus(torch.tensor(-30.0))
            + torch.nn.functional.softplus(torch.tensor(20.0))
        )
        / 2
    ) / 2
    # component entities are averaged within each parent before the parent mean.
    component_rows = torch.stack(
        [
            torch.nn.functional.softplus(torch.tensor([-10.0, 20.0])).mean(),
            torch.nn.functional.softplus(torch.tensor([30.0, -40.0])).mean(),
            torch.nn.functional.softplus(torch.tensor([0.0, 10.0])).mean(),
        ]
    )
    expected_c = ((component_rows[:2].mean() + component_rows[2]) / 2).item()
    assert tasks["pizza"].item() == pytest.approx(expected_p)
    assert tasks["component"].item() == pytest.approx(expected_c)
    assert total.item() == pytest.approx((expected_p + expected_c) / 2)


def test_bounded_encoder_rejects_overflow_before_tanh_can_hide_it():
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (4,),
        "vocab",
        latent_dim=2,
        domain="bounded",
    )
    with torch.no_grad():
        model.pizza_adapter.weight.fill_(torch.finfo(torch.float32).max)
        model.pizza_adapter.bias.zero_()
    features = torch.full((1, len(PIZZA_FEATURE_NAMES)), 2.0)
    with pytest.raises(ValueError, match="pre-activation"):
        model.encode(features, "pizza")


def test_box_forward_chunk_and_checkpoint_class_order_validation(tmp_path):
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (4, 9),
        "vocab",
        latent_dim=3,
        domain="bounded",
        seed=3,
    )
    x = torch.randn(7, len(PIZZA_FEATURE_NAMES))
    empty = torch.empty((0, len(PIZZA_FEATURE_NAMES)))
    assert model.predict(empty, "pizza").shape == (0, 2)
    assert model.clearances(empty, "pizza").shape == (0, 2)
    assert torch.allclose(
        model.clearances(x, "pizza"),
        torch.cat([model.clearances(x[i : i + 2], "pizza") for i in range(0, 7, 2)]),
    )
    assert torch.equal(model.predict(x, "pizza"), model.predict(x, "pizza", point_chunk_size=2))
    model.zero_grad(set_to_none=True)
    full_input = x.clone().requires_grad_()
    model.clearances(full_input, "pizza").sum().backward()
    full_input_grad = full_input.grad.clone()
    full_parameter_grads = {
        name: None if parameter.grad is None else parameter.grad.clone()
        for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    chunked_input = x.clone().requires_grad_()
    model.clearances(chunked_input, "pizza", point_chunk_size=2).sum().backward()
    assert torch.allclose(full_input_grad, chunked_input.grad, atol=1e-6, rtol=0)
    assert all(
        (expected is None and parameter.grad is None)
        or (
            expected is not None
            and parameter.grad is not None
            and torch.allclose(expected, parameter.grad, atol=1e-6, rtol=0)
        )
        for name, parameter in model.named_parameters()
        for expected in (full_parameter_grads[name],)
    )
    checkpoint = tmp_path / "box.pt"
    model.save_checkpoint(checkpoint)
    reversed_model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (9, 4),
        "vocab",
        latent_dim=3,
        domain="bounded",
        seed=3,
    )
    before = {key: value.clone() for key, value in reversed_model.state_dict().items()}
    with pytest.raises(ValueError, match="metadata"):
        reversed_model.load_checkpoint(checkpoint)
    assert all(
        torch.equal(before[key], value) for key, value in reversed_model.state_dict().items()
    )
    other_vocabulary = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (4, 9),
        "different-vocabulary",
        latent_dim=3,
        domain="bounded",
        seed=3,
    )
    with pytest.raises(ValueError, match="metadata"):
        other_vocabulary.load_checkpoint(checkpoint)
    model.load_checkpoint(checkpoint)
    points = torch.tensor([[1.0, -1.0, 0.0], [1.01, 0.0, 0.0]])
    assert model.domain_thing_membership(points).tolist() == [True, False]
    bad_payload = torch.load(checkpoint, weights_only=True)
    bad_payload["model_state"]["geometry.raw_a"][0, 0, 0] = float("nan")
    corrupt = tmp_path / "bad.pt"
    torch.save(bad_payload, corrupt)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="non-finite"):
        model.load_checkpoint(corrupt)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    bad_snapshot = torch.load(checkpoint, weights_only=True)
    bad_snapshot["geometry"]["lower"][0, 0, 0] += 0.01
    corrupt_snapshot = tmp_path / "bad-snapshot.pt"
    torch.save(bad_snapshot, corrupt_snapshot)
    with pytest.raises(ValueError, match="snapshot"):
        model.load_checkpoint(corrupt_snapshot)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_train_box_pairs_adapters_and_physical_bounds_across_domains(tmp_path):
    base = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (0, 1),
        "vocab",
        latent_dim=16,
        domain="bounded",
        seed=19,
    )
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    lower, upper = base.geometry.endpoints()
    kwargs = dict(
        train_records=[_record("train", "train", True)],
        validation_records=[_record("val", "validation", False)],
        epochs=1,
        latent_dim=16,
        boxes=1,
        shared_adapter_initial_state=shared,
        initial_box_bounds=(lower.detach().clone(), upper.detach().clone()),
        initial_box_class_ids=(0, 1),
        purpose="paired-smoke",
    )
    bounded = train_box(_dataset(), artifact_root=tmp_path / "bounded", domain="bounded", **kwargs)
    unbounded = train_box(
        _dataset(), artifact_root=tmp_path / "unbounded", domain="unbounded", **kwargs
    )
    bp = torch.load(bounded["checkpoint_paths"]["initial"], weights_only=True)
    up = torch.load(unbounded["checkpoint_paths"]["initial"], weights_only=True)
    for key in shared:
        assert torch.equal(bp["model_state"][key], up["model_state"][key])
    tolerance = 2e-5
    assert torch.max((bp["geometry"]["lower"] - up["geometry"]["lower"]).abs()).item() <= tolerance
    assert torch.max((bp["geometry"]["upper"] - up["geometry"]["upper"]).abs()).item() <= tolerance
    assert torch.max((bp["geometry"]["lower"] - lower).abs()).item() <= tolerance
    assert torch.max((bp["geometry"]["upper"] - upper).abs()).item() <= tolerance
    assert torch.max((up["geometry"]["lower"] - lower).abs()).item() <= tolerance
    assert torch.max((up["geometry"]["upper"] - upper).abs()).item() <= tolerance
    assert "_lower_offset" not in bp["model_state"]
    paired = bp["metadata"]["training_config"]["paired_initialization"]
    paired_unbounded = up["metadata"]["training_config"]["paired_initialization"]
    paired_bounds = paired["box_bounds"]
    paired_unbounded_bounds = paired_unbounded["box_bounds"]
    assert paired_bounds["reconstruction_tolerance_abs"] == tolerance
    assert paired_bounds["max_absolute_reconstruction_error"] <= tolerance
    assert paired_bounds["decoded_actual_sha256"]
    assert paired_bounds["requested_sha256"] == paired_unbounded_bounds["requested_sha256"]
    assert paired["adapter_sha256"] == paired_unbounded["adapter_sha256"]
    row = bounded["history"][0]
    assert row["encoder_gradient_norm_pre_clip_mean"] >= 0
    assert row["geometry_gradient_norm_post_clip_mean"] >= 0
    assert "positive_clearance" in row["geometry"]
    assert "negative_clearance" in row["geometry"]
    assert "winning_box_exact_tie_count" in row["geometry"]
    assert "winning_face_counts" in row["geometry"]
    assert all(key in bounded["geometry_snapshots"] for key in ("initial", "best", "final"))


def test_geometry_diagnostics_split_exact_box_and_face_ties_symmetrically():
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (0, 1),
        "vocab",
        latent_dim=1,
        boxes=2,
        domain="unbounded",
    )
    with torch.no_grad():
        model.geometry.raw_a.fill_(0)
        model.geometry.raw_b.zero_()
    model.encode = lambda features, entity: torch.zeros((len(features), 1))
    result = _geometry_diagnostics(
        model, [_record("diagnostic", "train", True)], _dataset(), torch.device("cpu")
    )
    assert result["winning_box_counts"] == [2.0, 2.0]
    assert result["winning_box_exact_tie_count"] == 4
    assert result["winning_face_exact_tie_count"] == 8
    assert result["winning_face_counts"] == [[2.0, 2.0]]
    assert result["inactive_box_fraction"] == 0
    assert "pizza" in result["entity_clearances"]
    assert set(result["target_clearances"]) == {"0", "1"}
    assert "width_floor_fraction" in result
    assert result["transform_saturation"]["width_softplus_derivative_near_zero_fraction"] == 0
    assert result["transform_saturation"]["center_identity_saturation_fraction"] is None


def test_diagnostics_distinguish_box_ties_from_nested_face_ties():
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (0, 1),
        "vocab",
        latent_dim=1,
        boxes=2,
        domain="unbounded",
    )
    model.encode = lambda features, entity: torch.zeros((len(features), 1))

    # Each tied box has one distinct limiting face: this is a box tie, not a face tie.
    # Construct ties directly in raw coordinates so they remain exact across
    # supported PyTorch softplus implementations (inverse/decode can round by 1 ulp).
    with torch.no_grad():
        model.geometry.raw_a.zero_()
        model.geometry.raw_b[0, :, 0] = torch.tensor([0.125, -0.125])
        model.geometry.raw_b[1, :, 0] = torch.tensor([0.125, -0.125])
    result = _geometry_diagnostics(
        model, [_record("diagnostic", "train", True)], _dataset(), torch.device("cpu")
    )
    assert result["winning_box_exact_tie_count"] == 4
    assert result["winning_face_exact_tie_count"] == 0
    assert result["winning_face_counts"] == [[2.0, 2.0]]

    # One winning box has two tied faces and the other has one. Activity weights
    # must follow max-over-box then min-over-face symmetric gradients (3:1).
    with torch.no_grad():
        model.geometry.raw_a[:, 0, :].zero_()
        model.geometry.raw_a[:, 1, :].fill_(1.0)
        model.geometry.raw_b.zero_()
        half_widths = model.epsilon / 2.0 + torch.nn.functional.softplus(model.geometry.raw_a)
        # Offset the wider box by exactly the representable half-width difference;
        # its lower-face clearance then equals the centered box's two-face tie.
        model.geometry.raw_b[:, 1, :] = half_widths[:, 1, :] - half_widths[:, 0, :]
    result = _geometry_diagnostics(
        model, [_record("diagnostic", "train", True)], _dataset(), torch.device("cpu")
    )
    assert result["winning_box_exact_tie_count"] == 4
    assert result["winning_face_exact_tie_count"] == 4
    assert result["winning_face_counts"] == [[3.0, 1.0]]


def test_geometry_diagnostics_report_domain_specific_transform_saturation():
    record = [_record("saturation", "train", True)]
    bounded = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (0, 1),
        "vocab",
        latent_dim=1,
        domain="bounded",
    )
    with torch.no_grad():
        bounded.geometry.raw_a.fill_(100)
        bounded.geometry.raw_b.fill_(100)
    bounded.encode = lambda features, entity: torch.zeros((len(features), 1))
    bounded_result = _geometry_diagnostics(bounded, record, _dataset(), torch.device("cpu"))[
        "transform_saturation"
    ]
    assert bounded_result["width_sigmoid_saturated_fraction"] == 1
    assert bounded_result["center_tanh_saturated_fraction"] == 1
    assert bounded_result["encoder_tanh_saturation_fraction"] == 0

    unbounded = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        (0, 1),
        "vocab",
        latent_dim=1,
        domain="unbounded",
    )
    with torch.no_grad():
        unbounded.geometry.raw_a.fill_(-15)
        unbounded.geometry.raw_b.fill_(100)
    unbounded.encode = lambda features, entity: torch.zeros((len(features), 1))
    unbounded_result = _geometry_diagnostics(unbounded, record, _dataset(), torch.device("cpu"))[
        "transform_saturation"
    ]
    assert unbounded_result["width_softplus_derivative_near_zero_fraction"] == 1
    assert unbounded_result["center_identity_saturation_fraction"] is None


def test_box_endpoints_are_pure_f12_decode_without_repair_buffers():
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), (0,), "vocab", latent_dim=2
    )
    decoded = model.geometry.endpoints()
    endpoints = model.endpoints()
    assert all(torch.equal(actual, expected) for actual, expected in zip(endpoints, decoded))
    assert "_lower_offset" not in model.state_dict()
    assert "_upper_offset" not in model.state_dict()


def test_paired_geometry_reconstruction_over_tolerance_fails_without_repair(tmp_path, monkeypatch):
    from ontology_embeddings import box

    base = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), (0, 1), "vocab", latent_dim=2
    )
    lower, upper = (value.detach().clone() for value in base.geometry.endpoints())
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    before = {key: value.clone() for key, value in shared.items()}
    original_inverse = box.inverse_geometry

    def mismatched_inverse(candidate_lower, candidate_upper, **kwargs):
        raw_a, raw_b = original_inverse(candidate_lower, candidate_upper, **kwargs)
        if torch.equal(candidate_lower, lower) and torch.equal(candidate_upper, upper):
            raw_b = raw_b + 0.1
        return raw_a, raw_b

    monkeypatch.setattr(box, "inverse_geometry", mismatched_inverse)
    with pytest.raises(ValueError, match="exceeds absolute tolerance"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            artifact_root=tmp_path,
            epochs=1,
            latent_dim=2,
            shared_adapter_initial_state=shared,
            initial_box_bounds=(lower, upper),
            initial_box_class_ids=(0, 1),
        )
    assert all(torch.equal(before[key], shared[key]) for key in shared)


def test_unrepresentable_paired_geometry_fails_closed(tmp_path):
    base = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), (0, 1), "vocab", latent_dim=2
    )
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    lower = torch.full((2, 1, 2), -1.0)
    upper = torch.full((2, 1, 2), 1.0)
    with pytest.raises(ValueError, match="no finite inverse"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            artifact_root=tmp_path,
            epochs=1,
            latent_dim=2,
            shared_adapter_initial_state=shared,
            initial_box_bounds=(lower, upper),
            initial_box_class_ids=(0, 1),
        )


def test_invalid_pairing_class_order_fails_without_mutating_shared_state(tmp_path):
    base = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), (0, 1), "vocab", latent_dim=2
    )
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    before = {key: value.clone() for key, value in shared.items()}
    lower, upper = base.geometry.endpoints()
    with pytest.raises(ValueError, match="class order"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            artifact_root=tmp_path,
            epochs=1,
            latent_dim=2,
            shared_adapter_initial_state=shared,
            initial_box_bounds=(lower, upper),
            initial_box_class_ids=(1, 0),
        )
    assert all(torch.equal(before[key], shared[key]) for key in shared)


def test_invalid_shared_adapter_state_fails_before_any_training_mutation(tmp_path):
    base = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES), len(COMPONENT_FEATURE_NAMES), (0, 1), "vocab", latent_dim=2
    )
    shared = {
        key: value.detach().clone()
        for key, value in base.state_dict().items()
        if key.startswith(("pizza_adapter.", "component_adapter."))
    }
    shared["pizza_adapter.weight"][0, 0] = float("nan")
    before = {key: value.clone() for key, value in shared.items()}
    with pytest.raises(ValueError, match="invalid shared adapter"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            artifact_root=tmp_path,
            epochs=1,
            latent_dim=2,
            shared_adapter_initial_state=shared,
        )
    assert all(
        torch.equal(before[key].isnan(), shared[key].isnan())
        and torch.equal(torch.nan_to_num(before[key]), torch.nan_to_num(shared[key]))
        for key in shared
    )


def test_group_gradient_norms_distinguish_pre_and_post_clipping(tmp_path):
    result = train_box(
        _dataset(),
        [_record("train-positive", "train", True), _record("train-negative", "train", False)],
        [_record("validation-positive", "validation", True)],
        artifact_root=tmp_path,
        epochs=1,
        batch_size=1,
        latent_dim=2,
        seed=271,
        tau=1e-6,
    )
    row = result["history"][0]
    assert row["gradient_norm_pre_clip_mean"] > 1
    assert row["gradient_norm_post_clip_mean"] <= 1 + 1e-6
    assert row["encoder_gradient_norm_pre_clip_mean"] > row["encoder_gradient_norm_post_clip_mean"]
    assert (
        row["geometry_gradient_norm_pre_clip_mean"] > row["geometry_gradient_norm_post_clip_mean"]
    )


def test_tiny_joint_box_smoke_updates_all_groups_and_persists_valid_states(tmp_path):
    train = [_record(f"train-{i}", "train", bool(i % 2)) for i in range(4)]
    validation = [_record(f"val-{i}", "validation", bool(i % 2)) for i in range(2)]
    result = train_box(
        _dataset(),
        train,
        validation,
        artifact_root=tmp_path,
        epochs=2,
        batch_size=2,
        latent_dim=3,
        seed=13,
        purpose="smoke",
    )
    assert result["status"] == "completed"
    assert all(result["parameter_updates"].values())
    assert result["best_checkpoint_reload_predictions_equivalent"]
    assert result["best_mean_available_task_micro_f1_epoch"] is not None
    for name in ("initial.pt", "best.pt", "latest.pt"):
        assert (tmp_path / result["run_id"] / "checkpoints" / name).is_file()


def test_tiny_training_accepts_pizzas_with_no_component_occurrences(tmp_path):
    train = [replace(_record("train-empty", "train", True), components=())]
    validation = [replace(_record("val-empty", "validation", True), components=())]
    result = train_box(
        _dataset(),
        train,
        validation,
        artifact_root=tmp_path,
        epochs=1,
        latent_dim=2,
        purpose="empty-components-smoke",
    )
    assert result["status"] == "completed"
    assert result["parameter_updates"]["pizza_adapter"]
    assert result["parameter_updates"]["geometry"]
    assert result["history"][0]["train_component_loss"] is None
    assert result["history"][0]["validation_component_loss"] is None


def test_validation_without_known_applicable_targets_fails_before_updates(tmp_path):
    validation = _record("val-unknown", "validation", False)
    component = replace(
        validation.components[0],
        ontology_known_mask=(0,) * len(validation.components[0].ontology_known_mask),
    )
    validation = replace(
        validation,
        pizza_known_mask=(0,) * len(validation.pizza_known_mask),
        components=(component,),
    )
    with pytest.raises(ValueError, match="validation has no eligible"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            [validation],
            artifact_root=tmp_path,
            epochs=2,
            latent_dim=2,
        )
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    checkpoints = run_dirs[0] / "checkpoints"
    initial = torch.load(checkpoints / "initial.pt", weights_only=True)
    latest = torch.load(checkpoints / "latest.pt", weights_only=True)
    assert all(
        torch.equal(initial["model_state"][name], latest["model_state"][name])
        for name in initial["model_state"]
    )


def test_box_trainer_rejects_sealed_test_records_before_training(tmp_path):
    with pytest.raises(ValueError, match="train records and validation records only"):
        train_box(
            _dataset(),
            [_record("sealed", "test", True)],
            artifact_root=tmp_path,
            epochs=1,
        )


def test_invalid_optimizer_update_fails_closed_and_preserves_latest(tmp_path, monkeypatch):
    original_step = torch.optim.Adam.step

    def corrupt_after_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        optimizer.param_groups[0]["params"][0].data.fill_(float("nan"))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", corrupt_after_step)
    with pytest.raises(FloatingPointError, match="invalid BOX update 1"):
        train_box(
            _dataset(),
            [_record("train", "train", True)],
            artifact_root=tmp_path,
            epochs=1,
            latent_dim=2,
            purpose="numerical-fault",
        )
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    checkpoint = run_dirs[0] / "checkpoints" / "latest.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert all(torch.isfinite(value).all() for value in payload["model_state"].values())
