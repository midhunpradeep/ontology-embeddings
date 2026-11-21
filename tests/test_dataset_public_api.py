from __future__ import annotations

import json
from pathlib import Path

from ontology_embeddings.cli import main
from ontology_embeddings.dataset import load_prepared_dataset, validate_dataset

PIZZA_48 = Path(__file__).parents[1] / "examples" / "pizza-48"


def test_pizza_48_loads_with_canonical_vocabulary_and_validates() -> None:
    dataset = load_prepared_dataset(PIZZA_48)
    report = validate_dataset(PIZZA_48, write_report=False)

    assert len(dataset.records) == 48
    assert len(dataset.canonical_class_ids) == 60
    assert dataset.class_names[:3] == ("American", "AmericanHot", "AnchovyTopping")
    assert report.valid, [issue.to_dict() for issue in report.errors]


def test_dataset_validate_cli_reports_fixture_integrity(capsys) -> None:
    result = main(["validate", str(PIZZA_48), "--no-write-report", "--json"])

    output = json.loads(capsys.readouterr().out)
    assert result == 0
    assert output["valid"] is True
    assert output["dataset_id"] == "pizza-dcc148789aecf8af86a3"
