from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from ontology_embeddings.dataset.constants import DETECTOR_CATEGORIES, TOPPING_COLUMNS
from ontology_embeddings.dataset.ontology import (
    ASSET_ONTOLOGY_PATH,
    OWL,
    RDF,
    RDFS,
    OntologySnapshot,
    build_class_vocabulary,
)

XML = "http://www.w3.org/XML/1998/namespace"


def _ontology_with_mushroom_parent(
    tmp_path,
    reference: str | None,
    *,
    parent_base: str | None = None,
) -> OntologySnapshot:
    tree = ET.parse(ASSET_ONTOLOGY_PATH)
    mushroom = next(
        element
        for element in tree.getroot().findall("owl:Class", {"owl": OWL})
        if element.get(f"{{{RDF}}}about") == "#MushroomTopping"
    )
    parent = mushroom.find("rdfs:subClassOf", {"rdfs": RDFS})
    assert parent is not None
    if reference is None:
        mushroom.remove(parent)
    else:
        parent.set(f"{{{RDF}}}resource", reference)
    if parent_base is not None:
        parent.set(f"{{{XML}}}base", parent_base)
    path = tmp_path / "mushroom-parent-fixture.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return OntologySnapshot.load(path)


@pytest.mark.parametrize(
    "source",
    [
        "MushroomTopping",
        "ParmesanTopping",
        "SultanaTopping",
    ],
)
def test_supported_equivalent_class_expansion_adds_component_targets(source: str) -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert "VegetarianTopping" in ontology.component_class_targets(source)


def test_legitimate_relative_named_parent_expands_component_targets() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)

    assert "VegetableTopping" in ontology.component_class_targets("MushroomTopping")


def test_legitimate_absolute_named_parent_matching_declared_iri_preserves_targets(tmp_path) -> None:
    pinned = OntologySnapshot.load(require_pinned_hash=True)
    ontology = _ontology_with_mushroom_parent(tmp_path, pinned.class_iris["VegetableTopping"])

    assert ontology.component_class_targets("MushroomTopping") == pinned.component_class_targets(
        "MushroomTopping"
    )


@pytest.mark.parametrize(
    ("reference", "parent_base"),
    [
        ("https://foreign.example/ontology#VegetableTopping", None),
        ("#VegetableTopping", "https://foreign.example/ontology/"),
    ],
)
def test_foreign_named_parent_with_colliding_local_name_is_unsupported(
    tmp_path, reference: str, parent_base: str | None
) -> None:
    ontology = _ontology_with_mushroom_parent(tmp_path, reference, parent_base=parent_base)

    targets = ontology.component_class_targets("MushroomTopping")
    assert "VegetableTopping" not in targets
    assert "VegetarianTopping" not in targets


@pytest.mark.parametrize("reference", [None, "#UnknownNamedSuperclass"])
def test_missing_or_unknown_named_superclass_is_not_inferred(tmp_path, reference) -> None:
    ontology = _ontology_with_mushroom_parent(tmp_path, reference)

    targets = ontology.component_class_targets("MushroomTopping")
    assert "VegetableTopping" not in targets
    assert "VegetarianTopping" not in targets


def test_pinned_class_ids_and_all_component_targets_remain_unchanged() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    ids = {entry["name"]: entry["class_id"] for entry in vocabulary["classes"]}

    assert ids["MushroomTopping"] == 26
    assert ids["VegetarianTopping"] == 63
    assert len(vocabulary["classes"]) == 66
    assert len(vocabulary["trainable_class_ids"]) == 60
    assert {
        category: ontology.component_class_targets(category) for category in DETECTOR_CATEGORIES
    } == {
        "America": frozenset({"Country", "DomainThing"}),
        "AnchovyTopping": frozenset(
            {"AnchovyTopping", "DomainThing", "Food", "PizzaTopping", "SeafoodTopping"}
        ),
        "DeepPanBase": frozenset({"DeepPanBase", "DomainThing", "Food", "PizzaBase"}),
        "England": frozenset({"Country", "DomainThing"}),
        "France": frozenset({"Country", "DomainThing"}),
        "GarlicTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "GarlicTopping",
                "PizzaTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "GreenPepperTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "GreenPepperTopping",
                "PepperTopping",
                "PizzaTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "HamTopping": frozenset(
            {"DomainThing", "Food", "HamTopping", "MeatTopping", "PizzaTopping"}
        ),
        "Italy": frozenset({"Country", "DomainThing"}),
        "JalapenoPepperTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "JalapenoPepperTopping",
                "PepperTopping",
                "PizzaTopping",
                "SpicyTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "MushroomTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "MushroomTopping",
                "PizzaTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "OliveTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "OliveTopping",
                "PizzaTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "OnionTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "OnionTopping",
                "PizzaTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "ParmesanTopping": frozenset(
            {
                "CheeseTopping",
                "DomainThing",
                "Food",
                "ParmesanTopping",
                "PizzaTopping",
                "VegetarianTopping",
            }
        ),
        "PeperonataTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "PeperonataTopping",
                "PepperTopping",
                "PizzaTopping",
                "SpicyTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "PeperoniSausageTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "MeatTopping",
                "PeperoniSausageTopping",
                "PizzaTopping",
                "SpicyTopping",
            }
        ),
        "PrawnTopping": frozenset(
            {"DomainThing", "Food", "PizzaTopping", "PrawnTopping", "SeafoodTopping"}
        ),
        "RocketTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "PizzaTopping",
                "RocketTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "SlicedTomatoTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "PizzaTopping",
                "SlicedTomatoTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "SpinachTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "PizzaTopping",
                "SpinachTopping",
                "VegetableTopping",
                "VegetarianTopping",
            }
        ),
        "SultanaTopping": frozenset(
            {
                "DomainThing",
                "Food",
                "FruitTopping",
                "PizzaTopping",
                "SultanaTopping",
                "VegetarianTopping",
            }
        ),
        "ThinAndCrispyBase": frozenset({"DomainThing", "Food", "PizzaBase", "ThinAndCrispyBase"}),
    }


def test_equivalent_class_targets_keep_canonical_vocabulary() -> None:
    ontology = OntologySnapshot.load(require_pinned_hash=True)
    vocabulary = build_class_vocabulary(ontology)
    ids = {entry["name"]: entry["class_id"] for entry in vocabulary["classes"]}

    assert ids["MushroomTopping"] == 26
    assert ids["VegetarianTopping"] == 63
    assert len(vocabulary["classes"]) == 66
    assert len(vocabulary["trainable_class_ids"]) == 60
    assert ontology.component_class_targets("MushroomTopping") == {
        "MushroomTopping",
        "VegetableTopping",
        "VegetarianTopping",
        "PizzaTopping",
        "Food",
        "DomainThing",
    }

    vegetarian_targets = {
        "AnchovyTopping": False,
        "GarlicTopping": True,
        "GreenPepperTopping": True,
        "HamTopping": False,
        "JalapenoPepperTopping": True,
        "MushroomTopping": True,
        "OliveTopping": True,
        "OnionTopping": True,
        "ParmesanTopping": True,
        "PeperonataTopping": True,
        "PeperoniSausageTopping": False,
        "PrawnTopping": False,
        "RocketTopping": True,
        "SlicedTomatoTopping": True,
        "SpinachTopping": True,
        "SultanaTopping": True,
    }
    for source in TOPPING_COLUMNS:
        assert (
            "VegetarianTopping" in ontology.component_class_targets(source)
        ) is vegetarian_targets[source]


def _ontology_with_nested_vegetable_disjunct(tmp_path) -> OntologySnapshot:
    """Create an unsupported nested union operand that only mentions VegetableTopping."""
    tree = ET.parse(ASSET_ONTOLOGY_PATH)
    namespaces = {"owl": OWL, "rdf": RDF}
    vegetarian = next(
        element
        for element in tree.getroot().findall("owl:Class", namespaces)
        if element.get(f"{{{RDF}}}about") == "#VegetarianTopping"
    )
    equivalence = vegetarian.find("owl:equivalentClass", namespaces)
    assert equivalence is not None
    intersection = equivalence.find("owl:Class/owl:intersectionOf", namespaces)
    assert intersection is not None
    union_class = next(child for child in intersection if child.tag == f"{{{OWL}}}Class")
    union = union_class.find("owl:unionOf", namespaces)
    assert union is not None
    vegetable = next(
        child for child in union if child.get(f"{{{RDF}}}about") == "#VegetableTopping"
    )
    union.remove(vegetable)

    nested_intersection_class = ET.SubElement(union, f"{{{OWL}}}Class")
    nested_intersection = ET.SubElement(
        nested_intersection_class,
        f"{{{OWL}}}intersectionOf",
        {"{http://www.w3.org/1999/02/22-rdf-syntax-ns#}parseType": "Collection"},
    )
    ET.SubElement(
        nested_intersection, f"{{{RDF}}}Description", {f"{{{RDF}}}about": "#VegetableTopping"}
    )
    ET.SubElement(
        nested_intersection, f"{{{RDF}}}Description", {f"{{{RDF}}}about": "#FruitTopping"}
    )

    path = tmp_path / "nested-vegetarian-equivalence.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return OntologySnapshot.load(path)


def test_unsupported_nested_equivalent_expression_does_not_expand_targets(tmp_path) -> None:
    ontology = _ontology_with_nested_vegetable_disjunct(tmp_path)

    assert "VegetarianTopping" not in ontology.component_class_targets("MushroomTopping")
