# ontology-embeddings

A small, installable research package for ontology-aware multi-label learning.

## Current scope

This repository is being rebuilt in staged, reviewed increments. The current bootstrap
provides Python packaging, contributor tooling, a minimal importable package, tests, and CI.
It does **not yet contain the migrated dataset subsystem or a learning model**. Dataset
migration is a separate deliverable; model implementation is explicitly deferred to a later
milestone and authorization.

The eventual research compares ontology-aware representations. Planned evidence is staged:
metadata-oracle learning is infrastructure evidence, controlled geometric tasks address
representational behavior, and image-derived inputs are needed for claims about visual learning.
This bootstrap supports none of those claims. In particular, it contains no affine, Box, Town,
CNN, or visualization implementation and reports no model results.

## Install

Requires Python 3.11 or 3.12.

```bash
python -m pip install .
```

For editable development with tests and linting, see [Contributing](CONTRIBUTING.md).

## Generated outputs

Use `artifacts/datasets/`, `artifacts/runs/`, `artifacts/checkpoints/`,
`artifacts/cache/`, and `artifacts/tmp/` for local generated outputs; these locations are
ignored by Git. Keep source files and small test fixtures in tracked source/test locations,
not inside these generated-output directories.

## Provenance and licensing

No dataset or third-party source files have been migrated in this bootstrap. The pinned source
and the required provenance and license review process are recorded in
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) and [`docs/licenses/README.md`](docs/licenses/README.md).
No repository-wide license has been selected; do not infer one for future migrated material.
