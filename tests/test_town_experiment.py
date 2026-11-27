from __future__ import annotations

import pytest

from ontology_embeddings.town_experiment import TownExperimentConfig


def test_town_experiment_config_validates_fixed_capacity_and_objective():
    config = TownExperimentConfig.from_mapping({
        "domain": "unbounded", "houses": 1, "windows": 1,
        "objective": "logistic", "epochs": 2, "batch_size": 4,
    })
    assert config.resolved_dict()["houses"] == 1
    assert config.resolved_dict()["windows"] == 1
    with pytest.raises(ValueError, match="windows"):
        TownExperimentConfig.from_mapping({"windows": -1})
    with pytest.raises(ValueError, match="unknown"):
        TownExperimentConfig.from_mapping({"adaptive_windows": True})
    with pytest.raises(ValueError, match="objective"):
        TownExperimentConfig.from_mapping({"objective": "bce"})


def test_town_config_file_requires_exact_table_and_rejects_bad_values(tmp_path):
    good = tmp_path / "town.toml"
    good.write_text('[town]\nhouses = 1\nwindows = 0\ndomain = "bounded"\n')
    config = TownExperimentConfig.from_toml(good)
    assert (config.houses, config.windows, config.domain) == (1, 0, "bounded")
    bad = tmp_path / "bad.toml"
    bad.write_text('[town]\nhouses = 1\n[extra]\nvalue = 1\n')
    with pytest.raises(ValueError, match="exactly one"):
        TownExperimentConfig.from_toml(bad)
