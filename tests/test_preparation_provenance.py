from __future__ import annotations

import subprocess
from pathlib import Path

from ontology_embeddings.dataset.preparation import _git_provenance


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_provenance_marks_untracked_sources_dirty_but_ignores_generated_artifacts(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "project"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Dataset provenance test")
    _git(repository, "config", "user.email", "provenance-test@example.invalid")
    (repository / ".gitignore").write_text("artifacts/cache/\n", encoding="utf-8")
    tracked_source = repository / "tracked.py"
    tracked_source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "tracked.py")
    _git(repository, "commit", "-q", "-m", "baseline")
    revision = _git(repository, "rev-parse", "HEAD")

    cache_file = repository / "artifacts" / "cache" / "generated.bin"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"ignored generated data")

    assert _git_provenance(repository) == (revision, False)

    untracked_source = repository / "src" / "new_module.py"
    untracked_source.parent.mkdir()
    untracked_source.write_text("VALUE = 2\n", encoding="utf-8")
    assert _git_provenance(repository) == (revision, True)

    untracked_source.unlink()
    assert _git_provenance(repository) == (revision, False)
