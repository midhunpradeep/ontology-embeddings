import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from ontology_embeddings.affine_smoke import load_train_validation
from ontology_embeddings.dataset.ontology import OntologySnapshot
from ontology_embeddings.stat01 import (
    Observation,
    _binary_stats,
    _eligible_weights,
    _special_entities,
    _weighted,
    accumulation,
    class_rows,
    conflict_evidence,
    diversity,
    entropy,
    js_divergence,
    ontology_relations,
    pair_rows,
    reconcile_accounting,
    states_for,
)


def obs(ref, parent, signature, states):
    return Observation(ref, parent, (signature,), tuple(states), ())


def test_four_state_representation_and_canonical_ids():
    assert states_for([1, 0, 0, 0], [1, 1, 0, 1], [2, 3, 1, 0], {0, 2, 3}) == (2, 0, 3, 1)


def test_occurrence_parent_and_signature_weights():
    values = [obs("p1/a", "p1", 0, (1,)), obs("p1/b", "p1", 1, (0,)), obs("p2/a", "p2", 2, (1,))]
    assert _weighted(values, "occurrence") == [1, 1, 1]
    assert _weighted(values, "parent") == [0.5, 0.5, 1]
    assert _weighted(values, "signature") == [1, 1, 1]
    assert _eligible_weights(values, [True, False, True], "parent") == [1, 0, 1]


def test_signature_balancing_preserves_conflicting_label_frequencies():
    values = [obs("a", "p", 0, (0,)), obs("b", "q", 0, (1,)), obs("c", "r", 1, (1,))]
    weights = _weighted(values, "signature")
    assert weights == [0.5, 0.5, 1]
    assert sum(w for w, x in zip(weights, values) if x.states[0] == 0) == 0.5
    assert sum(w for w, x in zip(weights, values) if x.states[0] == 1) == 1.5
    report = conflict_evidence(values, [9], ["canonical-class"])
    assert report["minimum_empirical_error_count"] == 1
    assert report["conflicting_signature_class_pairs"] == 1
    assert report["known_label_support"] == 3


def test_complete_four_state_contingency_and_jointly_known_binary_stats():
    values = [
        obs("11", "a", 0, (1, 1)),
        obs("10", "b", 1, (1, 0)),
        obs("01", "c", 2, (0, 1)),
        obs("00", "d", 3, (0, 0)),
        obs("unknown", "e", 4, (2, 1)),
        obs("inapplicable", "f", 5, (3, 0)),
    ]
    row = pair_rows(values, [3, 8], ["A", "B"], "pizza")[0]
    assert row["state_contingency"] == "[[1,1,0,0],[1,1,0,0],[0,1,0,0],[1,0,0,0]]"
    assert row["occurrence_n11"] == 1
    assert row["occurrence_n10"] == 1
    assert row["occurrence_n01"] == 1
    assert row["occurrence_n00"] == 1
    assert row["occurrence_jointly_known"] == 4
    assert row["occurrence_excluded_observations"] == 2
    check = reconcile_accounting(values, class_rows(values, [3, 8], ["A", "B"], "pizza"), [row])
    assert check["complete_pair_contingencies_reconciled"] == 1


def test_binary_formulas_constant_labels_and_zero_denominators():
    values = _binary_stats(2, 1, 1, 2)
    assert values["p_b_given_a"] == 2 / 3
    assert values["p_a_given_b"] == 2 / 3
    assert values["jaccard"] == 0.5
    assert values["disagreement"] == 1 / 3
    assert values["phi"] == 1 / 3
    assert values["mutual_information_nats"] > 0
    constant = _binary_stats(0, 0, 2, 3)
    assert constant["phi"] is None
    assert constant["p_b_given_a"] is None
    assert constant["entropy_a_nats"] == 0
    assert _binary_stats(0, 0, 0, 0)["jointly_known"] == 0


def test_entropy_js_and_empty_distributions():
    h, diversity = entropy([1, 1, 0])
    assert h == __import__("math").log(2)
    assert diversity == 2
    assert entropy([]) == (None, None)
    assert js_divergence(Counter(a=1), Counter(b=1)) == __import__("math").log(2)
    assert js_divergence(Counter(), Counter(a=1)) is None


def test_empty_distributions_and_contingency_views():
    assert diversity([], ["x"])["feature_signatures"]["entropy_nats"] is None
    empty_pair = pair_rows([], [0, 1], ["A", "B"], "pizza")[0]
    assert empty_pair["occurrence_jointly_known"] == 0
    assert empty_pair["occurrence_phi"] is None
    values = [obs("a", "p", 0, (1, 1)), obs("b", "p", 1, (1, 0)), obs("c", "q", 0, (0, 1))]
    row = pair_rows(values, [0, 1], ["A", "B"], "component")[0]
    assert row["parent_n11"] == 0.5
    assert row["signature_n11"] == 0.5
    assert row["signature_state_contingency"]


def test_accumulation_is_deterministic_and_keeps_components_with_parents():
    root = Path(__file__).parents[1] / "examples" / "pizza-48"
    dataset, train, _ = load_train_validation(root)
    first = accumulation(train[:8], dataset, "component", seeds=(7,), intervals=4)
    second = accumulation(train[:8], dataset, "component", seeds=(7,), intervals=4)
    assert first == second
    assert all(row["entity_occurrences"] >= row["parent_pizzas"] for row in first)


def test_safe_loader_skips_sealed_test_before_json_decode(tmp_path):
    root = Path(__file__).parents[1] / "examples" / "pizza-48"
    for relative in (
        "manifest.json",
        "vocabularies/ontology_classes.json",
        "vocabularies/detector_categories.json",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    source = (root / "samples.jsonl").read_text(encoding="utf-8")
    (tmp_path / "samples.jsonl").write_text(
        source + '{"split":"test" malformed\n', encoding="utf-8"
    )
    _, train, validation = load_train_validation(tmp_path)
    assert train and validation


def test_supported_named_hierarchy_and_disjointness():
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    closure, disjoint = ontology_relations(ontology)
    assert ("American", "Pizza") in closure
    assert ("AnchovyTopping", "PrawnTopping") in disjoint
    # The all-disjoint group includes these named members; duplicates from any
    # binary source remain a single canonical unordered relation.
    assert ("GreenPepperTopping", "JalapenoPepperTopping") in disjoint
    assert ("OliveTopping", "SpinachTopping") in disjoint
    assert ("American", "Country") in disjoint  # inherited from Pizza / Country
    assert len(disjoint) == len(set(disjoint))
    assert all(a < b for a, b in disjoint)

    # A binary source overlapping a group-expanded pair does not duplicate it.
    binary = ET.SubElement(
        ontology.class_elements["GreenPepperTopping"],
        "{http://www.w3.org/2002/07/owl#}disjointWith",
    )
    binary.set(
        "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource",
        "#JalapenoPepperTopping",
    )
    assert ontology_relations(ontology)[1] == disjoint


def test_special_class_entities_use_level_specific_denominators():
    record = SimpleNamespace(
        sample_id="p1",
        pizza_targets=[1],
        pizza_known_mask=[1],
        components=[
            SimpleNamespace(component_id="c1", ontology_targets=[1], ontology_known_mask=[1]),
            SimpleNamespace(component_id="c2", ontology_targets=[0], ontology_known_mask=[1]),
        ],
    )
    assert len(_special_entities([record], "pizza")) == 1
    components = _special_entities([record], "component")
    assert len(components) == 2
    assert all("/" in ref for ref, _, _ in components)
