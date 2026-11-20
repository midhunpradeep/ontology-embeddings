# Provenance and migration manifest

## Source project

The dataset implementation and prepared fixture were migrated from the read-only source
repository at <https://github.com/midhunpradeep/town_model>, branch `main`, exact commit
`1788acfc65eb1038fe9e4c595cab6ef0f25407fd`. The migration was performed from the local archived
checkout at that revision. The archived checkout remains unchanged.

The source repository has **no top-level project license file**. No blanket project license is
created or implied by this migration. The copied source project's own code is not covered by the
separate upstream Pizzaïolo license notice.

## Migrated-file manifest

The following paths were copied from the pinned source commit. Python files and tests were
adapted only for the new package namespace and repository layout, as detailed below.

| Source path at `town_model` commit | Destination | Contents |
| --- | --- | --- |
| `src/town_model/config.py` | `src/ontology_embeddings/config.py` | Dataset generation/split configuration and TOML parsing |
| `src/town_model/dataset/__init__.py` | `src/ontology_embeddings/dataset/__init__.py` | Dataset public exports (expanded for loading and validation) |
| `src/town_model/dataset/constants.py` | `src/ontology_embeddings/dataset/constants.py` | Schema, pinned source, vocabulary, and split constants |
| `src/town_model/dataset/generator.py` | `src/ontology_embeddings/dataset/generator.py` | Pinned upstream source adapter and raw generation entry |
| `src/town_model/dataset/generator_worker.py` | `src/ontology_embeddings/dataset/generator_worker.py` | Process-isolated deterministic generator worker |
| `src/town_model/dataset/image_data.py` | `src/ontology_embeddings/dataset/image_data.py` | Validated dataset loading, subsets, target projection |
| `src/town_model/dataset/labels.py` | `src/ontology_embeddings/dataset/labels.py` | Audited 32-label source inference predicates |
| `src/town_model/dataset/ontology.py` | `src/ontology_embeddings/dataset/ontology.py` | Pinned ontology inventory, narrow inference, vocabularies |
| `src/town_model/dataset/preparation.py` | `src/ontology_embeddings/dataset/preparation.py` | Atomic generation, target preparation, manifest publication |
| `src/town_model/dataset/schema.py` | `src/ontology_embeddings/dataset/schema.py` | Prepared record schema and JSONL serialization |
| `src/town_model/dataset/splitting.py` | `src/ontology_embeddings/dataset/splitting.py` | Deterministic generator-type-stratified split policy |
| `src/town_model/dataset/statistics.py` | `src/ontology_embeddings/dataset/statistics.py` | Split and class-level known/positive/negative statistics |
| `src/town_model/dataset/validation.py` | `src/ontology_embeddings/dataset/validation.py` | Dataset integrity, semantic, and annotation validator |
| `src/town_model/dataset/assets/pizzaiolo.xml` | `src/ontology_embeddings/dataset/assets/pizzaiolo.xml` | Pinned ontology snapshot, byte-identical |
| `dataset.toml` | `dataset.toml` | Default generation settings |
| `docs/DATASET_SCHEMA.md` | `docs/DATASET_SCHEMA.md` | Prepared wire contract and annotation semantics; dashboard-only section removed |
| `docs/licenses/UPSTREAM_PIZZAIOLO.md` | `docs/licenses/UPSTREAM_PIZZAIOLO.md` | Upstream attribution/license notice; package path updated |
| `schemas/sample-v1.schema.json` | `schemas/sample-v1.schema.json` | Machine-readable version 1 sample schema |
| `examples/pizza-48/**` | `examples/pizza-48/**` | Complete prepared 48-image fixture, copied as-is |
| `tests/conftest.py` | `tests/conftest.py` | Deterministic raw-data test fixture for prep tests |
| `tests/test_config.py` | `tests/test_config.py` | Dataset configuration tests |
| `tests/test_dataset.py` | `tests/test_dataset.py` | Preparation, schema, masks, annotations, and validation tests |
| `tests/test_generator.py` | `tests/test_generator.py` | Pinned generator cache tests |
| `tests/test_generator_worker.py` | `tests/test_generator_worker.py` | Stable random-candidate worker tests |
| `tests/test_labels.py` | `tests/test_labels.py` | Source-label predicate tests |
| `tests/test_ontology.py` | `tests/test_ontology.py` | Vocabulary and component target tests |
| `tests/test_phase3_dataset.py` | `tests/test_image_dataset.py` | Loader/subset/target-projection tests (renamed to remove phase history) |
| `tests/test_phase3_d5a_ontology_entailments.py` | `tests/test_ontology_entailments.py` | Narrow entailment and IRI-resolution tests (renamed) |

The test package paths were rewritten from `town_model` to `ontology_embeddings`. The fixture
file set and bytes are checked against the pinned source fixture; the fixture manifest and its
asset hashes remain untouched. `examples/pizza-48` is a smoke/loader-validation fixture, not a
learning dataset.

### Deliberately omitted source paths

- `src/town_model/dataset/watch.py` is dashboard-cache invalidation support and has no core data
  generation, loading, or validation dependency; it is omitted.
- The old application/CLI shell, dashboard, classifier backend, model and experiment code, run
  logging, unrelated reports, and experimental scaffolding were not migrated.
- The upstream Pizzaïolo generator repository is not copied into the package. `generator.py`
  retrieves/checks a clean checkout pinned to commit
  `6c0bcb221a957c184012beaf103bebcde4b1f793` into the ignored local cache on explicit generation.

## Packaging/import adaptations

- Imports and the subprocess module path were changed from `town_model` to
  `ontology_embeddings`; dataset behavior and on-disk record schema were not renamed. Ruff
  formatting was applied to migrated Python sources/tests, and two unused local mapping
  constructions were removed; serialized values and runtime semantics are preserved.
- Dataset configuration was separated into `ontology_embeddings.config` instead of importing
  the old application shell.
- `ontology_embeddings.dataset` exposes the prepared loader, validator, record types, target
  projection, and deterministic subset API.
- `oe-dataset` is a small data-only CLI (`generate`, `validate`, `inspect`) implemented without
  old application, classifier, or dashboard imports.
- The package includes the XML ontology asset. Runtime dependencies are NumPy, OpenCV headless,
  Owlready2, pandas, Pillow, matplotlib, and scikit-learn. The latter two are imported by the
  pinned upstream generator module; they are retained for generator operation, not classifier
  code. Pytest and Ruff remain development dependencies. Torch, torchvision, Streamlit, and the
  old dashboard/model backend dependencies were not carried over.
- The generation cache is under ignored `artifacts/sources/`; generated datasets are under
  ignored `artifacts/datasets/`. The committed `examples/pizza-48` fixture is outside those rules.

## Third-party attribution and terms

The pinned upstream [SysReIC/Pizzaïolo](https://github.com/SysReIC/pizzaiolo) revision
`6c0bcb221a957c184012beaf103bebcde4b1f793` declares **CC BY-NC 4.0**. Authors are Grégory
Bourguin and Arnaud Lewandowski (SysReIC/LISIC/ULCO); the suggested citation is Grégory Bourguin
and Arnaud Lewandowski, *Pizzaïolo Dataset: Des Images Synthétiques Ontologiquement Explicables*,
<https://hal.science/hal-04401953>. See [`licenses/UPSTREAM_PIZZAIOLO.md`](licenses/UPSTREAM_PIZZAIOLO.md)
for the preserved notice. The copied ontology snapshot and prepared fixture contain upstream
ontology/generated images and annotations; review upstream acknowledgements and terms before
redistribution or commercial use. CC BY-NC 4.0 does not grant rights to the source project's
own code.

The new repository remains private. This migration changes neither repository visibility nor
publishes material elsewhere.

## Validation evidence

- Installed the package and test/lint tools with
  `./.venv/bin/python -m pip install -e '.[dev]'`.
- `./.venv/bin/python -m pytest -q` completed with **93 passed**; `./.venv/bin/ruff check .`
  completed with **All checks passed**.
- `./.venv/bin/python -m pip wheel --no-deps . --wheel-dir build/wheels` passed, building
  `build/wheels/ontology_embeddings-0.1.0-py3-none-any.whl` (SHA-256
  `f770c76bb3b4a4036148feca700a991b232b1f2f28775c11fc32ed245e8c5b32`). The wheel was installed
  with `./.venv/bin/python -m pip install --no-deps --upgrade --target build/wheel-install
  build/wheels/ontology_embeddings-0.1.0-py3-none-any.whl`. A process importing from that wheel
  loaded its packaged ontology,
  validated/loaded all 48 fixture records, and had neither `town_model` nor `torch` in
  `sys.modules`.
- `./.venv/bin/oe-dataset validate examples/pizza-48 --no-write-report --json` passed: dataset
  `pizza-dcc148789aecf8af86a3`, 48 records, split sizes 16/16/16, zero errors.
- Real generation used the upstream repository at `6c0bcb221a957c184012beaf103bebcde4b1f793`
  and this exact command twice (distinct output paths):

  ```text
  ./.venv/bin/oe-dataset generate --config dataset.toml --output artifacts/datasets/issue2-real-final-a --total-instances 6 --pizza-types American random --type-counts '{"American": 3, "random": 3}' --generation-seed 31415 --split-seed 27182
  ./.venv/bin/oe-dataset generate --config dataset.toml --output artifacts/datasets/issue2-real-final-b --total-instances 6 --pizza-types American random --type-counts '{"American": 3, "random": 3}' --generation-seed 31415 --split-seed 27182
  ```

  Both runs validated and produced dataset ID `pizza-12f170a153661557c94f`, six samples, and 72
  component occurrences. Record objects were equal across runs, covering source-inferred labels,
  pizza/component targets and masks, split identities, component order/IDs, original-pixel xyxy
  boxes, and annotation references. The full immutable-asset checksum maps were equal. Split IDs
  were train `img_00000,img_00004`, validation `img_00001,img_00005`, and test
  `img_00002,img_00003`. Both manifests record preparation source SHA
  `12bcb929c500124f646c1702537f9bd64a0c85c3` with `preparation_dirty: true`; both share 34
  immutable asset hashes. Validation report IDs are `214818a64e21a3bebe307c2880714daf858c1ac4cbc4ed354e5efdda7ec83eaf`
  and `3f1e52cc15b322bb0242c5863be8928a8395ee83a78e3a7fe8ce589eeef6e5d4` for outputs `issue2-real-final-a`
  and `issue2-real-final-b`. Full artifacts are under
  `artifacts/datasets/issue2-real-final-a/` and `artifacts/datasets/issue2-real-final-b/` and are
  intentionally ignored local generation artifacts, not committed.
- The first real-generation attempt before dependency installation stopped at import with the
  concrete environment error `ModuleNotFoundError: No module named 'matplotlib'`. Inspection of
  the pinned upstream module confirmed its unconditional matplotlib and scikit-learn imports;
  both were added as necessary generator dependencies, after which both real attempts passed.
  An initial wheel command with `--no-build-isolation` also failed because setuptools was absent
  from the venv; the standard build-isolation command above passed.
- The 204 files under `examples/pizza-48/` were compared against the archived source fixture with
  `diff -qr`; the trees matched with no differences. The archived source checkout still has the
  pinned source commit and a clean worktree.

The generated outputs are local artifacts, not committed fixture changes. No model or training
experiment was run, and no test-set predictive feedback was used.
