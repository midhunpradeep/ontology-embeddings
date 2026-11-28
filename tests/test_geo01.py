from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

import ontology_embeddings.geo01 as geo01
from ontology_embeddings.box_training import BoxMultiLabelModel
from ontology_embeddings.dataset.schema import ComponentRecord, DatasetRecord
from ontology_embeddings.features import COMPONENT_FEATURE_NAMES, PIZZA_FEATURE_NAMES
from ontology_embeddings.geo01 import (
    GEO03_ATTEMPT_ID,
    GEO03_CONFIG_ID,
    GEO03_CONFIG_SHA256,
    GEO03_CONFIG_SOURCE_COMMIT,
    create_geo_run_record,
    geo_campaign_identity,
    geo_preparation_identity,
    load_approved_train_validation,
    ordered_sample_ids_sha256,
    prepare_box_reference_initialization,
)
from ontology_embeddings.geo_reconstruction import reconstruct_box_checkpoint


def _dataset():
    return SimpleNamespace(
        dataset_id="tiny-geo",
        manifest_id="manifest-id",
        dataset_manifest_sha256="manifest-bytes",
        class_manifest_sha256="vocab-hash",
        canonical_class_ids=(0, 1),
        class_names=("A", "B"),
        class_iris=("iri:A", "iri:B"),
        pizza_supervision_class_ids=(0,),
        component_supervision_class_ids=(1,),
        fixed_positive_class_ids=(),
        unsupported_class_ids=(),
    )


def _record(sample_id: str, split: str, *, positive: bool) -> DatasetRecord:
    pizza_targets = [0, 0]
    pizza_known = [0, 0]
    pizza_targets[0] = int(positive)
    pizza_known[0] = 1
    component_targets = [0, 0]
    component_known = [0, 0]
    component_targets[1] = int(positive)
    component_known[1] = 1
    component = ComponentRecord(
        f"{sample_id}:component-0",
        1,
        "AnchovyTopping",
        (0, 0, 1, 1),
        tuple(component_targets),
        tuple(component_known),
        "fixture",
    )
    return DatasetRecord(
        sample_id,
        split,
        "excluded",
        "",
        1,
        1,
        tuple(pizza_targets),
        tuple(pizza_known),
        (),
        (component,),
        {
            "raw_generator": {
                "topping_counts": {
                    name.split(":", 1)[1]: int(positive) if name.endswith("AnchovyTopping") else 0
                    for name in PIZZA_FEATURE_NAMES[:16]
                },
                "has_base": "ThinAndCrispyBase",
                "country_of_origin": None,
            }
        },
        {},
    )


def test_campaign_identity_is_top_level_run_record_data_and_requires_a_supplied_hash(
    tmp_path,
):
    supplied_config_hash = GEO03_CONFIG_SHA256
    identity = {
        "config_id": GEO03_CONFIG_ID,
        "config_sha256": supplied_config_hash,
        "attempt_id": GEO03_ATTEMPT_ID,
    }

    assert geo_campaign_identity(supplied_config_hash) == identity
    with pytest.raises(ValueError, match="per-configuration"):
        geo_campaign_identity("not-a-config-hash")

    record = create_geo_run_record(tmp_path / "runs", config_sha256=supplied_config_hash)
    persisted = json.loads((record.path / "manifest.json").read_text())
    assert persisted["run_id"] == record.run_id
    assert persisted["geo_campaign"] == identity
    assert persisted["geo_preparation"] == geo_preparation_identity()
    assert persisted["geo_campaign_source_commit"] == GEO03_CONFIG_SOURCE_COMMIT
    assert set(persisted["geo_campaign"]) == {"config_id", "config_sha256", "attempt_id"}
    assert "geo_campaign" not in persisted["config"]


def test_ordered_sample_identity_is_json_compact_and_order_sensitive():
    assert ordered_sample_ids_sha256(["a", "b"]) == hashlib.sha256(b'["a","b"]').hexdigest()
    assert ordered_sample_ids_sha256(["a", "b"]) != ordered_sample_ids_sha256(["b", "a"])


def test_geo_loader_binds_only_approved_manifest_and_ordered_train_validation_identities(
    tmp_path, monkeypatch
):
    manifest_raw = b"approved manifest identity bytes"
    (tmp_path / "manifest.json").write_bytes(manifest_raw)
    train_ids = [f"train-{index}" for index in range(720)]
    validation_ids = [f"validation-{index}" for index in range(240)]
    dataset = _dataset()
    dataset.dataset_id = "tiny-geo"
    dataset.manifest_id = "manifest-id"
    dataset.dataset_manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    train = [SimpleNamespace(sample_id=sample_id, split="train") for sample_id in train_ids]
    validation = [
        SimpleNamespace(sample_id=sample_id, split="validation") for sample_id in validation_ids
    ]
    monkeypatch.setattr(geo01, "GEO01_DATASET_ID", dataset.dataset_id)
    monkeypatch.setattr(geo01, "GEO01_MANIFEST_ID", dataset.manifest_id)
    monkeypatch.setattr(geo01, "GEO01_MANIFEST_RAW_SHA256", dataset.dataset_manifest_sha256)
    monkeypatch.setattr(
        geo01, "GEO01_TRAIN_ORDERED_IDS_SHA256", ordered_sample_ids_sha256(train_ids)
    )
    monkeypatch.setattr(
        geo01,
        "GEO01_VALIDATION_ORDERED_IDS_SHA256",
        ordered_sample_ids_sha256(validation_ids),
    )
    monkeypatch.setattr(geo01, "load_train_validation", lambda _root: (dataset, train, validation))

    _, loaded_train, loaded_validation, identity = load_approved_train_validation(tmp_path)

    assert loaded_train == train
    assert loaded_validation == validation
    assert identity["train_sample_ids_sha256"] == ordered_sample_ids_sha256(train_ids)
    assert identity["validation_sample_ids_sha256"] == ordered_sample_ids_sha256(validation_ids)
    assert identity["declared_test_count"] == 240
    assert identity["test_split_decoded"] is False
    assert identity["test_statistics_used"] is False


def test_native_bounded_box_initialization_emits_pairable_physical_target():
    dataset = _dataset()
    prepared = prepare_box_reference_initialization(dataset, seed=23, device="cpu")

    assert set(prepared["adapter_state"]) == {
        "pizza_adapter.weight",
        "pizza_adapter.bias",
        "component_adapter.weight",
        "component_adapter.bias",
    }
    lower, upper = prepared["bounds"]
    assert lower.shape == upper.shape == (2, 1, 2)
    assert torch.all(lower >= -1) and torch.all(upper <= 1)
    assert prepared["provenance"]["reference_carrier"] == "bounded"
    assert prepared["provenance"]["target_carrier"] == "unbounded"
    assert prepared["provenance"]["seed"] == 23
    assert prepared["provenance"]["target_sha256"]
    assert prepared["provenance"]["adapter_state_sha256"]
    assert 0.05 <= prepared["provenance"]["reference_volume_fraction_min"]
    assert prepared["provenance"]["reference_volume_fraction_max"] <= 0.25
    assert prepared["provenance"]["initialization_limitations"]
    for key, tensor in prepared["adapter_state"].items():
        assert (
            prepared["provenance"]["adapter_state_sha256"][key]
            == hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()
        )


def test_standalone_reconstruction_preserves_occurrence_signature_and_exact_membership(tmp_path):
    dataset = _dataset()
    train = [
        _record("train-a", "train", positive=True),
        _record("train-b", "train", positive=True),
    ]
    validation = [_record("validation-a", "validation", positive=False)]
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        dataset.canonical_class_ids,
        dataset.class_manifest_sha256,
        latent_dim=2,
        boxes=1,
        domain="unbounded",
        seed=23,
        class_names=dataset.class_names,
        class_iris=dataset.class_iris,
    )
    training_config = {
        "model": "box-union-two-adapters",
        "domain": "unbounded",
        "latent_dim": 2,
        "boxes": 1,
        "epsilon": 1e-4,
        "objective": "logistic",
        "tau": 0.1,
        "margin": 0.1,
    }
    checkpoint = tmp_path / "state.pt"
    model.save_checkpoint(
        checkpoint,
        objective="logistic",
        tau=0.1,
        margin=0.1,
        epoch=0,
        training_config=training_config,
    )

    observation = reconstruct_box_checkpoint(
        checkpoint,
        dataset,
        train,
        validation,
        training_config=training_config,
        device="cpu",
    )

    train_view = observation["populations"]["train"]
    assert observation["class_ids"] == [0, 1]
    assert observation["geometry"]["lower"] == model.endpoints()[0].tolist()
    assert len(train_view["pizza_occurrences"]) == 2
    assert len(train_view["component_occurrences"]) == 2
    assert all(len(row["coordinates"]) == 2 for row in train_view["pizza_occurrences"])
    assert all(
        len(row["predictions"]) == 2 and len(row["clearance"]) == 2
        for row in train_view["pizza_occurrences"]
    )
    assert train_view["pizza_occurrences"][0]["known_mask"] == [True, False]
    assert train_view["pizza_occurrences"][0]["applicable_mask"] == [True, False]
    assert train_view["component_occurrences"][0]["parent_sample_id"] == "train-a"
    assert train_view["component_occurrences"][0]["occurrence_id"] == "train-a:component-0"
    signature_groups = train_view["distinct_feature_signatures"]["pizza"]
    assert len(signature_groups) == 1
    assert next(iter(signature_groups.values()))["occurrence_ids"] == ["train-a", "train-b"]


def test_standalone_reconstruction_rejects_test_records_before_model_evaluation(tmp_path):
    dataset = _dataset()
    model = BoxMultiLabelModel(
        len(PIZZA_FEATURE_NAMES),
        len(COMPONENT_FEATURE_NAMES),
        dataset.canonical_class_ids,
        dataset.class_manifest_sha256,
        latent_dim=2,
        boxes=1,
        domain="unbounded",
        seed=23,
        class_names=dataset.class_names,
        class_iris=dataset.class_iris,
    )
    config = {"objective": "logistic", "tau": 0.1, "margin": 0.1}
    checkpoint = tmp_path / "state.pt"
    model.save_checkpoint(
        checkpoint,
        objective="logistic",
        tau=0.1,
        margin=0.1,
        epoch=0,
        training_config=config,
    )

    with pytest.raises(ValueError, match="train and validation"):
        reconstruct_box_checkpoint(
            checkpoint,
            dataset,
            [_record("sealed", "test", positive=True)],
            [],
            training_config=config,
            device="cpu",
        )
