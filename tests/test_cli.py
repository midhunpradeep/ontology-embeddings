from __future__ import annotations

from pathlib import Path

from ontology_embeddings import cli


def test_generate_uses_invoking_project_directory(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "dataset.toml").write_text(
        'total_instances = 6\n'
        'pizza_types = ["American", "random"]\n'
        '[pizza_type_counts]\n'
        'American = 3\n'
        'random = 3\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    captured: dict[str, object] = {}

    def prepare_dataset(config, output_path, **kwargs):
        captured["output_path"] = output_path
        captured["project_root"] = kwargs["project_root"]
        return Path(output_path)

    monkeypatch.setattr(cli, "prepare_dataset", prepare_dataset)

    result = cli.main(
        ["generate", "--config", "dataset.toml", "--output", "artifacts/datasets/test"]
    )

    assert result == 0
    assert captured["project_root"] == project.resolve()
    assert captured["output_path"] == Path("artifacts/datasets/test")
