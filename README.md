# ontology-embeddings

A small, installable research package for ontology-aware multi-label learning.

## Current scope

The repository contains the migrated Pizzaïolo dataset subsystem, an affine metadata-oracle
baseline, and a trainable axis-aligned BOX model with a shared classifier/training pipeline for
pizza and component labels. BOX and affine evidence concerns ontology-aware multi-label learning;
it is not image recognition or a full OWL reasoner. Available research utilities include the
STAT-01 offline dataset audit, REL-01 checkpoint geometry and observed-relation evaluation, and a
standalone Lean 4 + Mathlib formal foundation. TOWN, CNN, detector, dashboard, and visualization
implementations remain out of scope.

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

## Dataset audit — STAT-01

Run the offline descriptive audit on a compatible prepared dataset:

```bash
python -m ontology_embeddings.stat01 \
  --dataset /path/to/prepared-dataset \
  --output /path/to/new-audit-directory
```

The audit analyzes train and validation only; it does not decode or analyze sealed test records.
It reports class/pair coverage, feature and signature diversity, observed relations, and related
descriptive evidence. The committed [STAT-01 report and evidence](docs/experiments/stat-01/report.md)
are tied to their recorded dataset and source identities. The ontology extraction is a named-class
subclass/disjointness fragment, not complete OWL reasoning; observed associations and finite-sample
implications are not causal claims or global axioms. Feature-association tables are
occurrence-weighted, and generator coverage is not exhaustive.

## Learned geometry and relation evaluation

The `src/ontology_embeddings/box.py` and `box_training.py` modules provide trainable bounded and
unbounded axis-aligned BOX geometry and the pizza/component classifier training pipeline. The
historical [affine baseline evidence](docs/experiments/pr2-affine-evidence/README.md) describes the
earlier metadata-oracle baseline.

REL-01 evaluates stored checkpoint geometry from existing BOX run evidence, without retraining.
For example, global relation evaluation uses:

```bash
oe-rel01 /path/to/box-evidence /path/to/new-rel01-output
```

Pass `--dataset /path/to/original-prepared-dataset` to additionally run observed train/validation
consistency evaluation. This mode validates the historical dataset identity and does not evaluate
test examples. The [corrected REL-01 evidence](evidence/rel-01-observed-v2/README.md) separates
exact global geometric relations, reference-only observed consistency for supported named-fragment
axioms, and descriptive all-candidate statistics. It also reports supervision-restricted and
unrestricted populations, occurrence-weighted and distinct-embedding units, and undefined rates
for zero denominators. The historical observed-v1 all-candidate rates are retained but are not
ontology-axiom violation rates.

## Formal verification

A standalone Lean 4 + Mathlib project lives in [`formal/`](formal/). It records
representation-independent set semantics, a theorem ledger, axiom audit, and a reusable
verification command. From the `formal/` directory, run:

```bash
./scripts/verify.sh
```

This checks the existing Lean project and its audited theorem dependencies; the formal statements
do not establish that the Python implementation satisfies those semantics. See
[`formal/README.md`](formal/README.md) for the pinned toolchain, setup, proof workflow, and limits.

## Generate a dataset

`dataset.toml` provides the default generation configuration. Run generation from the project
directory: configuration and output paths are relative to the current working directory, and the
pinned-source cache is stored under that project root. Generation uses the pinned Pizzaïolo commit
and writes a new immutable dataset directory. The existing output path must not exist. The
generator source is cloned into the ignored `artifacts/sources/pizzaiolo` cache unless an
already-pinned, clean checkout is passed explicitly.

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
