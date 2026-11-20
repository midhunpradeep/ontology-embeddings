from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ontology_embeddings.dataset import (
    load_prepared_dataset,
    targets_in_trainable_order,
    validate_dataset,
)
from ontology_embeddings.dataset.ontology import build_class_vocabulary, build_detector_vocabulary

FIXTURE = Path(__file__).parents[1] / "examples" / "pizza-48"
SOURCE_ONTOLOGY_IRI = "https://www.univ-littoral.fr/lisic/sysreic/pizzaiolo.xml"

# Independent SHA-256 references measured from the unchanged fixture at town_model commit
# 1788acfc65eb1038fe9e4c595cab6ef0f25407fd.
SOURCE_FILE_SHA256 = {
    "manifest.json": "61c9926de9c2ec501e48ef6a081e71452de101e870eca6dd2d58441be85fff6d",
    "samples.jsonl": "970582e121cdb58249c2f9ea469c2d0172d85b8d7f284d8c06eba954193b74fb",
    "vocabularies/ontology_classes.json": (
        "e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54"
    ),
    "vocabularies/detector_categories.json": (
        "48c980a33269b740352464c5c2d5f3a6fd9ed93f84ed58e5367f74664d806cea"
    ),
    "splits/train.json": "d21401e3d0185a448e236a4a3e507af3dbece2227dc4ef64e66f18e4fe82c4d9",
    "splits/validation.json": "88062f4d8b17363ede06803d8898471b5a2c62fd51e16b8f246390e7889bf1c2",
    "splits/test.json": "8846960a889cd2471f87ce390ed88d9d49eaabafd133023197e9cc186bbfa606",
}

# Full canonical source class order (IDs are zero-based positions in this literal).
SOURCE_CLASS_NAMES = (
    "American",
    "AmericanHot",
    "AnchovyTopping",
    "Cajun",
    "Capricciosa",
    "CheeseTopping",
    "CheesyPizza",
    "Country",
    "DeepPanBase",
    "DomainThing",
    "Fiorentina",
    "Food",
    "FourSeasons",
    "FruitTopping",
    "GarlicTopping",
    "GenerousPizza",
    "Giardiniera",
    "GreenPepperTopping",
    "HamTopping",
    "Hot",
    "JalapenoPepperTopping",
    "LaReine",
    "MeatTopping",
    "Medium",
    "Mild",
    "Mushroom",
    "MushroomTopping",
    "NamedPizza",
    "Napoletana",
    "NonVegetarianPizza",
    "OliveTopping",
    "OnionTopping",
    "Parmense",
    "ParmesanTopping",
    "PeperonataTopping",
    "PeperoniSausageTopping",
    "PepperTopping",
    "Pizza",
    "PizzaBase",
    "PizzaTopping",
    "PrawnTopping",
    "ProteinPizza",
    "RealAmericanPizza",
    "RealEnglishPizza",
    "RealFrenchPizza",
    "RealItalianPizza",
    "RocketTopping",
    "SeafoodTopping",
    "Siciliana",
    "SlicedTomatoTopping",
    "SloppyGiuseppe",
    "Soho",
    "Spiciness",
    "SpicyPizza",
    "SpicyTopping",
    "SpicyVegePizza",
    "SpinachTopping",
    "SultanaTopping",
    "ThickPizza",
    "ThinAndCrispyBase",
    "ValuePartition",
    "VegetableTopping",
    "VegetarianPizza",
    "VegetarianTopping",
    "Veneziana",
    "VerySpicyPizza",
)
SOURCE_FIXED_POSITIVE = frozenset({"DomainThing"})
SOURCE_UNSUPPORTED = frozenset({"Hot", "Medium", "Mild", "Spiciness", "ValuePartition"})
SOURCE_DETECTOR_CATEGORIES = (
    "America",
    "AnchovyTopping",
    "DeepPanBase",
    "England",
    "France",
    "GarlicTopping",
    "GreenPepperTopping",
    "HamTopping",
    "Italy",
    "JalapenoPepperTopping",
    "MushroomTopping",
    "OliveTopping",
    "OnionTopping",
    "ParmesanTopping",
    "PeperonataTopping",
    "PeperoniSausageTopping",
    "PrawnTopping",
    "RocketTopping",
    "SlicedTomatoTopping",
    "SpinachTopping",
    "SultanaTopping",
    "ThinAndCrispyBase",
)
SOURCE_FLAG_INDIVIDUALS = frozenset({"America", "England", "France", "Italy"})
SOURCE_SPLIT_IDS = {
    "train": [
        "img_00003",
        "img_00006",
        "img_00009",
        "img_00012",
        "img_00013",
        "img_00016",
        "img_00024",
        "img_00025",
        "img_00026",
        "img_00028",
        "img_00032",
        "img_00033",
        "img_00039",
        "img_00044",
        "img_00046",
        "img_00047",
    ],
    "validation": [
        "img_00002",
        "img_00004",
        "img_00005",
        "img_00007",
        "img_00015",
        "img_00018",
        "img_00019",
        "img_00020",
        "img_00027",
        "img_00035",
        "img_00036",
        "img_00037",
        "img_00038",
        "img_00041",
        "img_00042",
        "img_00043",
    ],
    "test": [
        "img_00000",
        "img_00001",
        "img_00008",
        "img_00010",
        "img_00011",
        "img_00014",
        "img_00017",
        "img_00021",
        "img_00022",
        "img_00023",
        "img_00029",
        "img_00030",
        "img_00031",
        "img_00034",
        "img_00040",
        "img_00045",
    ],
}


def _class_manifest() -> dict:
    return json.loads((FIXTURE / "vocabularies" / "ontology_classes.json").read_text())


def _class_id(name: str) -> int:
    # These IDs are explicit source-vocabulary positions, not model output indices.
    return SOURCE_CLASS_NAMES.index(name)


def test_fixture_files_match_independent_source_sha256_references() -> None:
    for relative_path, expected_sha256 in SOURCE_FILE_SHA256.items():
        actual_sha256 = hashlib.sha256((FIXTURE / relative_path).read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256, relative_path

    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    assert manifest["source"]["generator_commit"] == "6c0bcb221a957c184012beaf103bebcde4b1f793"
    assert manifest["ontology"]["class_count"] == 66
    assert manifest["dataset_id"] == "pizza-dcc148789aecf8af86a3"


def test_class_and_detector_vocabularies_preserve_source_identities_and_order() -> None:
    dataset = load_prepared_dataset(FIXTURE)
    source_classes = _class_manifest()["classes"]
    generated_classes = build_class_vocabulary()["classes"]

    assert source_classes == generated_classes
    assert [item["class_id"] for item in source_classes] == list(range(66))
    assert [item["name"] for item in source_classes] == list(SOURCE_CLASS_NAMES)
    assert [item["iri"] for item in source_classes] == [
        f"{SOURCE_ONTOLOGY_IRI}#{name}" for name in SOURCE_CLASS_NAMES
    ]
    assert [item["supervision_status"] for item in source_classes] == [
        "fixed_positive"
        if item["name"] in SOURCE_FIXED_POSITIVE
        else "unsupported"
        if item["name"] in SOURCE_UNSUPPORTED
        else "trainable"
        for item in source_classes
    ]
    assert sum(item["trainable"] for item in source_classes) == 60
    assert [item["name"] for item in source_classes if item["fixed_positive"]] == ["DomainThing"]
    assert [item["name"] for item in source_classes if item["unsupported"]] == [
        "Hot",
        "Medium",
        "Mild",
        "Spiciness",
        "ValuePartition",
    ]
    expected_trainable_ids = tuple(
        index
        for index, name in enumerate(SOURCE_CLASS_NAMES)
        if name not in SOURCE_FIXED_POSITIVE and name not in SOURCE_UNSUPPORTED
    )
    assert dataset.canonical_class_ids == expected_trainable_ids
    assert dataset.class_names == tuple(
        SOURCE_CLASS_NAMES[index] for index in expected_trainable_ids
    )
    assert dataset.class_iris == tuple(
        f"{SOURCE_ONTOLOGY_IRI}#{SOURCE_CLASS_NAMES[index]}" for index in expected_trainable_ids
    )
    assert dataset.fixed_positive_class_ids == (9,)
    assert dataset.unsupported_class_ids == (19, 23, 24, 52, 60)

    source_detectors = json.loads(
        (FIXTURE / "vocabularies" / "detector_categories.json").read_text()
    )
    generated_detectors = build_detector_vocabulary()
    assert source_detectors == generated_detectors
    categories = source_detectors["categories"]
    assert [item["category_id"] for item in categories] == list(range(1, 23))
    assert [item["name"] for item in categories] == list(SOURCE_DETECTOR_CATEGORIES)
    assert [item["identity"]["kind"] for item in categories] == [
        "owl_individual" if item["name"] in SOURCE_FLAG_INDIVIDUALS else "owl_class"
        for item in categories
    ]
    assert "America" not in SOURCE_CLASS_NAMES
    assert categories[0]["identity"] == {
        "kind": "owl_individual",
        "iri": f"{SOURCE_ONTOLOGY_IRI}#America",
    }
    assert categories[1]["identity"] == {
        "kind": "owl_class",
        "iri": f"{SOURCE_ONTOLOGY_IRI}#AnchovyTopping",
    }


def test_fixture_load_validation_and_exact_source_split_assignments() -> None:
    dataset = load_prepared_dataset(FIXTURE)
    report = validate_dataset(FIXTURE, write_report=False)

    assert report.valid, [issue.to_dict() for issue in report.errors]
    assert dataset.dataset_id == "pizza-dcc148789aecf8af86a3"
    assert len(dataset.records) == 48
    assert sum(len(record.components) for record in dataset.records) == 803
    for split, expected_ids in SOURCE_SPLIT_IDS.items():
        split_manifest = json.loads((FIXTURE / "splits" / f"{split}.json").read_text())
        assert split_manifest["sample_ids"] == expected_ids
        assert [record.sample_id for record in dataset.split_records(split)] == expected_ids
    assert {record.sample_id: record.split for record in dataset.records} == {
        sample_id: split
        for split, sample_ids in SOURCE_SPLIT_IDS.items()
        for sample_id in sample_ids
    }


def test_pizza_and_component_targets_distinguish_positive_negative_and_unknown() -> None:
    dataset = load_prepared_dataset(FIXTURE)
    records = {record.sample_id: record for record in dataset.records}

    no_origin = records["img_00003"]
    assert no_origin.audit["raw_generator"]["country_of_origin"] is None
    assert no_origin.pizza_source_labels == (
        "DomainThing",
        "Food",
        "Pizza",
        "Capricciosa",
        "SpicyPizza",
        "NamedPizza",
        "NonVegetarianPizza",
    )
    assert [
        name for index, name in enumerate(SOURCE_CLASS_NAMES) if no_origin.pizza_targets[index]
    ] == [
        "Capricciosa",
        "DomainThing",
        "Food",
        "NamedPizza",
        "NonVegetarianPizza",
        "Pizza",
        "SpicyPizza",
    ]
    assert (
        no_origin.pizza_targets[_class_id("RealAmericanPizza")],
        no_origin.pizza_known_mask[_class_id("RealAmericanPizza")],
    ) == (0, 1)
    assert (
        no_origin.pizza_targets[_class_id("DomainThing")],
        no_origin.pizza_known_mask[_class_id("DomainThing")],
    ) == (1, 1)
    assert (
        no_origin.pizza_targets[_class_id("Hot")],
        no_origin.pizza_known_mask[_class_id("Hot")],
    ) == (0, 0)  # A zero mask means unknown, not a known negative.

    american_origin = records["img_00009"]
    assert american_origin.audit["raw_generator"]["country_of_origin"] == "America"
    assert (
        american_origin.pizza_targets[_class_id("RealAmericanPizza")],
        american_origin.pizza_known_mask[_class_id("RealAmericanPizza")],
    ) == (1, 1)

    anchovy = no_origin.components[0]
    assert anchovy.detector_category == "AnchovyTopping"
    assert [
        name for index, name in enumerate(SOURCE_CLASS_NAMES) if anchovy.ontology_targets[index]
    ] == ["AnchovyTopping", "DomainThing", "Food", "PizzaTopping", "SeafoodTopping"]
    assert (
        anchovy.ontology_targets[_class_id("AnchovyTopping")],
        anchovy.ontology_known_mask[_class_id("AnchovyTopping")],
    ) == (1, 1)
    assert (
        anchovy.ontology_targets[_class_id("OliveTopping")],
        anchovy.ontology_known_mask[_class_id("OliveTopping")],
    ) == (0, 1)
    assert (
        anchovy.ontology_targets[_class_id("DomainThing")],
        anchovy.ontology_known_mask[_class_id("DomainThing")],
    ) == (1, 1)
    assert (
        anchovy.ontology_targets[_class_id("Hot")],
        anchovy.ontology_known_mask[_class_id("Hot")],
    ) == (0, 0)

    pizza_targets, pizza_known, component_targets, component_known = targets_in_trainable_order(
        dataset, no_origin
    )
    real_american_column = dataset.class_names.index("RealAmericanPizza")
    assert (pizza_targets[real_american_column], pizza_known[real_american_column]) == (0, 1)
    assert len(component_targets) == len(no_origin.components)
    assert len(component_known) == len(no_origin.components)
    assert len(component_targets[0]) == len(dataset.canonical_class_ids) == 60
    anchovy_column = dataset.class_names.index("AnchovyTopping")
    assert (component_targets[0][anchovy_column], component_known[0][anchovy_column]) == (1, 1)


def test_repeated_source_components_keep_occurrence_ids_and_original_pixel_coordinates() -> None:
    dataset = load_prepared_dataset(FIXTURE)
    record = next(record for record in dataset.records if record.sample_id == "img_00003")
    repeated_anchovies = [
        component
        for component in record.components
        if component.detector_category == "AnchovyTopping"
    ]

    assert [
        (item.component_id, item.detector_category_id, item.bbox_xyxy_original)
        for item in repeated_anchovies
    ] == [
        (
            "img_00003/AnchovyTopping/occurrence_000",
            2,
            (124.0, 155.0, 173.0, 178.0),
        ),
        (
            "img_00003/AnchovyTopping/occurrence_001",
            2,
            (23.0, 102.0, 70.0, 130.0),
        ),
        (
            "img_00003/AnchovyTopping/occurrence_002",
            2,
            (120.0, 34.0, 160.0, 67.0),
        ),
    ]
    assert all(
        item.annotation_path == "labels/img_00003_bboxes.json" for item in repeated_anchovies
    )
