import pytest
import torch

from ontology_embeddings.evaluation import class_frequency_baseline, membership_metrics


def test_membership_metrics_report_support_and_undefined_denominators():
    result = membership_metrics(
        torch.tensor([[False, False, True], [True, False, False]]),
        torch.tensor([[False, True, False], [False, False, False]]),
        torch.tensor([[True, True, False], [True, False, False]]),
        torch.tensor([[True, True, True], [True, False, True]]),
        ["known-negative", "unsupported-positive", "unknown"],
    )
    assert result["per_class"]["known-negative"]["known_negative"] == 2
    assert result["per_class"]["known-negative"]["precision"] == 0
    assert result["per_class"]["known-negative"]["recall"] is None
    assert result["per_class"]["unsupported-positive"]["known_positive"] == 1
    assert result["per_class"]["unsupported-positive"]["known_negative"] == 0
    assert result["per_class"]["unknown"]["unknown"] == 2
    assert result["per_class"]["unknown"]["f1"] is None
    assert result["per_class"]["known-negative"]["not_applicable"] == 0


def test_f1_is_defined_from_confusion_counts_when_precision_or_recall_is_undefined():
    # First class has a known positive but no predictions; second has a false positive
    # despite no known positives. Both confusion-count F1 scores are defined as zero.
    result = membership_metrics(
        torch.tensor([[False, True, False], [False, False, False]]),
        torch.tensor([[True, False, False], [False, False, False]]),
        torch.tensor([[True, True, True], [False, False, True]]),
        torch.ones((2, 3), dtype=torch.bool),
        ["missed-positive", "false-positive-only", "no-decisions"],
    )

    missed = result["per_class"]["missed-positive"]
    assert missed["precision"] is None
    assert missed["recall"] == 0
    assert missed["f1"] == 0
    false_positive = result["per_class"]["false-positive-only"]
    assert false_positive["precision"] == 0
    assert false_positive["recall"] is None
    assert false_positive["f1"] == 0
    assert result["per_class"]["no-decisions"]["f1"] is None
    assert result["macro_f1"] == 0
    assert result["weighted_f1"] == 0


def test_micro_f1_is_defined_from_confusion_counts_without_predicted_positives():
    result = membership_metrics(
        torch.tensor([[False], [False]]),
        torch.tensor([[True], [False]]),
        torch.ones((2, 1), dtype=torch.bool),
        torch.ones((2, 1), dtype=torch.bool),
    )
    assert result["micro"] == {"precision": None, "recall": 0, "f1": 0}

    false_positive_only = membership_metrics(
        torch.tensor([[True]]),
        torch.tensor([[False]]),
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones((1, 1), dtype=torch.bool),
    )
    assert false_positive_only["micro"] == {"precision": 0, "recall": None, "f1": 0}


def test_frequency_baseline_is_fitted_only_to_train_and_scores_original_validation():
    train_y = torch.tensor([[1, 0], [1, 0], [0, 1]])
    train_known = torch.ones_like(train_y, dtype=torch.bool)
    app = torch.ones_like(train_known)
    validation_y = torch.tensor([[0, 1]])
    result = class_frequency_baseline(
        train_y,
        train_known,
        app,
        validation_y,
        torch.ones_like(validation_y, dtype=torch.bool),
        torch.ones_like(validation_y, dtype=torch.bool),
        ["a", "b"],
    )
    assert result["training_positive"] == [2, 1]
    assert result["training_frequency"] == pytest.approx([2 / 3, 1 / 3])
    assert result["metrics"]["per_class"]["a"]["recall"] is None
    assert result["metrics"]["per_class"]["b"]["recall"] == 0
