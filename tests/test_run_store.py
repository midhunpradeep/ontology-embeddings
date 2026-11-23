from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ontology_embeddings import run_store
from ontology_embeddings.run_store import RunRecord, find_runs, source_provenance


def _git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, text=True, capture_output=True
    ).stdout.strip()


def _source_tree(path: Path) -> Path:
    (path / "src/ontology_embeddings").mkdir(parents=True)
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (path / "src/ontology_embeddings/run_store.py").write_text("# source\n", encoding="utf-8")
    return path


def _git_repo(path: Path) -> str:
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Provenance test")
    _git(path, "config", "user.email", "provenance@example.invalid")
    _git(path, "add", ".")
    _git(path, "commit", "-qm", "initial")
    return _git(path, "rev-parse", "HEAD")


def test_editable_source_tree_records_its_own_checkout(tmp_path, monkeypatch):
    root = _source_tree(tmp_path / "editable-source")
    sha = _git_repo(root)
    monkeypatch.setattr(run_store, "__file__", str(root / "src/ontology_embeddings/run_store.py"))

    record = RunRecord(tmp_path / "runs", purpose="test", invocation="editable", config={})

    assert record.manifest["source"]["commit_sha"] == sha
    assert record.manifest["source"]["dirty"] is False
    assert record.manifest["source"]["availability"] == "available"


def test_non_editable_package_does_not_infer_checkout_from_cwd(tmp_path, monkeypatch):
    foreign = _source_tree(tmp_path / "foreign")
    foreign_sha = _git_repo(foreign)
    installed = tmp_path / "site-packages/ontology_embeddings"
    installed.mkdir(parents=True)
    fake_module = installed / "run_store.py"
    fake_module.write_text("# installed package\n", encoding="utf-8")
    monkeypatch.setattr(run_store, "__file__", str(fake_module))
    monkeypatch.chdir(foreign)

    record = RunRecord(tmp_path / "runs", purpose="test", invocation="installed", config={})
    assert record.manifest["source"]["commit_sha"] is None
    assert record.manifest["source"]["dirty"] is None
    assert record.manifest["source"]["availability"] == "unavailable"
    assert record.manifest["source"]["commit_sha"] != foreign_sha


def test_git_repository_environment_cannot_redirect_source_provenance(tmp_path, monkeypatch):
    module_root = _source_tree(tmp_path / "module-source")
    module_sha = _git_repo(module_root)
    foreign_root = _source_tree(tmp_path / "foreign-source")
    (foreign_root / "foreign.txt").write_text("different checkout\n", encoding="utf-8")
    foreign_sha = _git_repo(foreign_root)
    assert module_sha != foreign_sha

    (module_root / "src/ontology_embeddings/untracked.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    excludes_file = tmp_path / "git-excludes"
    excludes_file.write_text("untracked.py\n", encoding="utf-8")
    monkeypatch.setenv("GIT_DIR", str(foreign_root / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(module_root))
    monkeypatch.setenv("GIT_INDEX_FILE", str(foreign_root / ".git/index"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(foreign_root / ".git/objects"))
    monkeypatch.setenv("GIT_ALTERNATE_OBJECT_DIRECTORIES", str(foreign_root / ".git/objects"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(foreign_root / ".git"))
    monkeypatch.setenv("GIT_NAMESPACE", "redirected")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.excludesFile")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(excludes_file))

    provenance = source_provenance(module_root)

    assert provenance["availability"] == "available"
    assert provenance["commit_sha"] == module_sha
    assert provenance["commit_sha"] != foreign_sha
    assert provenance["dirty"] is True
    assert "src/ontology_embeddings/untracked.py" in provenance["relevant_source_sha256"]


def test_git_unavailable_is_explicit_and_dirty_state_unknown(tmp_path):
    root = _source_tree(tmp_path / "source")

    provenance = source_provenance(root)

    assert provenance["commit_sha"] is None
    assert provenance["dirty"] is None
    assert provenance["availability"] == "unavailable"
    assert "git provenance unavailable" in provenance["fingerprint_note"]


def test_source_dirty_state_is_recorded_and_index_reconciles_sha(tmp_path):
    root = _source_tree(tmp_path / "source")
    sha = _git_repo(root)
    clean = source_provenance(root)
    assert clean["commit_sha"] == sha
    assert clean["dirty"] is False

    (root / "src/ontology_embeddings/new.py").write_text("VALUE = 1\n", encoding="utf-8")
    dirty = source_provenance(root)
    assert dirty["commit_sha"] == sha
    assert dirty["dirty"] is True
    assert "src/ontology_embeddings/new.py" in dirty["relevant_source_sha256"]

    store = tmp_path / "runs"
    manifest = {
        "run_id": "known-sha",
        "status": "failed",
        "source": dirty,
        "dataset": {},
        "config": {},
    }
    run = RunRecord(store, purpose="test", invocation="index", config={})
    run.manifest.update(manifest)
    run._persist()
    run._index()

    rows = find_runs(store, source_sha=sha, status="failed")
    assert len(rows) == 1
    assert rows[0]["source_sha"] == sha
    assert find_runs(store, source_sha="None") == []
    assert find_runs(store, source_sha="") == []


def test_legacy_index_sha_remains_searchable(tmp_path):
    store = tmp_path / "runs"
    store.mkdir()
    sha = "0123456789abcdef0123456789abcdef01234567"
    manifest = {
        "run_id": "legacy-run",
        "status": "completed",
        "source": {"commit_sha": sha, "dirty": False},
        "dataset": {},
        "config": {},
    }
    run_dir = store / "legacy-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    rows = find_runs(store, source_sha=sha)

    assert len(rows) == 1
    assert rows[0]["run_id"] == "legacy-run"
    assert rows[0]["source_sha"] == sha


def test_unavailable_sha_does_not_match_unknown_run(tmp_path, monkeypatch):
    installed_module = tmp_path / "site-packages/ontology_embeddings/run_store.py"
    installed_module.parent.mkdir(parents=True)
    installed_module.write_text("# installed package\n", encoding="utf-8")
    monkeypatch.setattr(run_store, "__file__", str(installed_module))
    record = RunRecord(tmp_path / "runs", purpose="test", invocation="no git", config={})
    assert record.manifest["source"]["commit_sha"] is None
    assert find_runs(tmp_path / "runs", source_sha="None") == []
    indexed = json.loads((tmp_path / "runs/index.jsonl").read_text().splitlines()[0])
    assert indexed["source_sha"] is None
    assert find_runs(tmp_path / "runs", status="running")[0]["run_id"] == record.run_id
