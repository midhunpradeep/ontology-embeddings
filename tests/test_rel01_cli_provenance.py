from types import SimpleNamespace

from ontology_embeddings.rel01_cli import _invocation_provenance


def test_actual_observed_cli_invocation_is_sanitized_and_identifies_inputs():
    dataset = SimpleNamespace(
        dataset_id="pizza-6a33710d13eb90fdde3e",
        manifest_id="c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd",
    )
    actual = _invocation_provenance("box-r01", "results.json.gz", dataset)
    assert actual == {
        "status": "recorded_actual_cli_arguments",
        "entry_point": "python -m ontology_embeddings.rel01_cli",
        "evaluation_mode": "global_and_observed",
        "arguments": {
            "evidence": "box-r01",
            "output": "results.json.gz",
            "dataset": {
                "provided": True,
                "dataset_id": dataset.dataset_id,
                "declared_manifest_id": dataset.manifest_id,
            },
        },
    }
    assert "/" not in str(actual)


def test_global_only_invocation_records_dataset_absence():
    actual = _invocation_provenance("box-r01", "geometry.json.gz", None)
    assert actual["evaluation_mode"] == "global_only"
    assert actual["arguments"]["dataset"] == {"provided": False}
