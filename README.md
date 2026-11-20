# ontology-embeddings

A small, installable research package for ontology-aware multi-label learning.

## Current scope

This repository contains the migrated Pizzaïolo dataset subsystem: pinned generator adapter,
preparation, ontology/vocabulary and source-label inference, prepared records, deterministic
type-stratified splits, integrity validation, statistics, and dataset loading/subsets. It does
not contain or implement any affine, Box, Town, CNN, training, experiment-logging, dashboard,
or visualization subsystem.

The prepared schema preserves unknown-label masks, source audit metadata, component occurrence
IDs, and original-pixel annotation geometry. Synthetic absences are known negatives only under
the pinned generator's prepared-data contract; they are not a general closed-world interpretation
of OWL.

## Install and validate the fixture

Requires Python 3.11 or 3.12.

```bash
python -m pip install ".[dev]"
oe-dataset validate examples/pizza-48 --no-write-report
```

The committed `examples/pizza-48` dataset is an unchanged 48-image prepared fixture for loading,
validation, and smoke checks; it is not intended for learning claims. See
[`docs/DATASET_SCHEMA.md`](docs/DATASET_SCHEMA.md) for the wire contract and
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for migration and asset provenance.

## Generate a dataset

`dataset.toml` provides the default generation configuration. Generation uses the pinned
Pizzaïolo commit and writes a new immutable dataset directory. The existing output path must not
exist. The generator source is cloned into the ignored `artifacts/sources/pizzaiolo` cache unless
an already-pinned, clean checkout is passed explicitly.

```bash
oe-dataset generate --config dataset.toml --output artifacts/datasets/pizza-run-01
```

Other data commands are `oe-dataset validate <dataset>` and `oe-dataset inspect <dataset>`.

## Generated outputs

Use `artifacts/datasets/`, `artifacts/sources/`, `artifacts/runs/`, `artifacts/checkpoints/`,
`artifacts/cache/`, and `artifacts/tmp/` for local generated outputs and source caches; these
locations are ignored by Git. Keep source files and the prepared fixture in tracked package and
example directories, not in generated-output locations.

## Provenance and licensing

The pinned `town_model` source checkout has no top-level project license. No repository-wide
license is selected or inferred here. The migrated upstream Pizzaïolo ontology/data assets have
their separate CC BY-NC 4.0 attribution notice; that license does not cover the source project's
own code. See [`docs/licenses/README.md`](docs/licenses/README.md) and
[`docs/licenses/UPSTREAM_PIZZAIOLO.md`](docs/licenses/UPSTREAM_PIZZAIOLO.md).
