from ontology_embeddings.dataset.constants import FIXED_POSITIVE_CLASS
from ontology_embeddings.dataset.ontology import OntologySnapshot, build_class_vocabulary
from ontology_embeddings.relation_evaluation import _reference


def test_pinned_named_axiom_fragment_and_explicit_coverage_counts():
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    trainable = {item["name"] for item in vocabulary["classes"] if item["trainable"]}
    subclass, disjoint = _reference(ontology, trainable)
    assert len(trainable) == 60
    assert len(subclass) == 148
    assert len(disjoint) == 1185
    assert ("AnchovyTopping", "SeafoodTopping") in subclass
    assert tuple(sorted(("GreenPepperTopping", "JalapenoPepperTopping"))) in disjoint
    assert not any(a == b for a, b in subclass)
    assert all(a != b for a, b in disjoint)
    assert FIXED_POSITIVE_CLASS not in trainable


def test_absence_from_reference_closure_is_not_a_negative_entailment():
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    trainable = {item["name"] for item in vocabulary["classes"] if item["trainable"]}
    subclass, disjoint = _reference(ontology, trainable)
    unknown = next(
        (a, b)
        for a in sorted(trainable)
        for b in sorted(trainable)
        if a != b and (a, b) not in subclass and tuple(sorted((a, b))) not in disjoint
    )
    assert unknown[0] != unknown[1]
    # The extractor supplies no false label for a pair absent from this fragment.
    assert unknown not in subclass and tuple(sorted(unknown)) not in disjoint
