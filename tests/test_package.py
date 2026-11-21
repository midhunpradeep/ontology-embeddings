from importlib.metadata import version

import ontology_embeddings


def test_importable_package_version_matches_distribution_metadata() -> None:
    assert ontology_embeddings.__version__ == version("ontology-embeddings")
