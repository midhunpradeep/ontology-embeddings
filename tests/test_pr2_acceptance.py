from pathlib import Path

from ontology_embeddings.affine_smoke import load_train_validation
from ontology_embeddings.pr2_acceptance import support_report


def test_support_report_counts_only_applicable_train_validation_rows():
    dataset, train, validation = load_train_validation(Path("examples/pizza-48"))

    train_report = support_report(dataset, train)
    validation_report = support_report(dataset, validation)

    assert train_report["pizza"]["eligible_entity_count"] == 16
    assert train_report["pizza"]["eligible_pizza_count"] == 16
    assert train_report["component"]["entity_count"] == 289
    assert train_report["component"]["eligible_pizza_count"] == 16
    assert train_report["pizza"]["per_class"]["American"]["known_positive"] == 1
    assert train_report["component"]["per_class"]["American"]["applicable"] is False
    assert train_report["component"]["per_class"]["American"]["not_applicable"] == 289
    assert validation_report["pizza"]["per_class"]["American"]["known_positive"] == 1
    assert validation_report["component"]["per_class"]["American"]["not_applicable"] == 253
