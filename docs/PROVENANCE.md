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
| `tests/test_phase3_d5a_ontology_entailments.py` | `tests/test_ontology_entailments.py` | Dataset target-expansion and named-IRI resolution tests (renamed) |

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
  ignored `artifacts/datasets/`. The CLI uses the invoking project directory as the root for both
  conventions. The committed `examples/pizza-48` fixture is outside those rules.

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

- The remediation code is commit `891923003984d5635c4868e5907f64094cce0f9c` on
  `thesis/pr1-dataset-migration`. `./.venv/bin/python -m pip install -e '.[dev]'` completed;
  `./.venv/bin/python -m pytest -q` reported **88 passed** (relation-query-only tests were removed
  as out of scope); `./.venv/bin/ruff check .` reported **All checks passed**; and
  `./.venv/bin/python -m pip check` reported **No broken requirements found**.
- `./.venv/bin/python -m pip wheel --no-deps . --wheel-dir build/remediation-wheels` built
  `ontology_embeddings-0.1.0-py3-none-any.whl` (SHA-256
  `8626cb3a99296c430a0b2e0d4f133fcb5c1d6c8715ed4d64fa0706af63482d0e`). The wheel was installed
  non-editably with `./.venv/bin/python -m pip install --no-deps --target
  build/remediation-wheel-install build/remediation-wheels/ontology_embeddings-0.1.0-py3-none-any.whl`;
  an import resolved to `build/remediation-wheel-install/ontology_embeddings/__init__.py`.
- From the repository root, the wheel-installed `oe-dataset` command validated
  `examples/pizza-48`: dataset `pizza-dcc148789aecf8af86a3`, 48 records, 16/16/16 split sizes,
  zero errors and 178 support warnings. The archived fixture remains unchanged; its 204 files previously matched the
  pinned source fixture byte-for-byte.
- Two actual pinned-generator runs used the source at
  `6c0bcb221a957c184012beaf103bebcde4b1f793`, the non-editable wheel above, and the exact code
  commit above. Both manifests record that full `preparation_commit` and
  `preparation_dirty: false`. Commands (different output directories):

  ```text
  PYTHONPATH="$PWD/build/remediation-wheel-install" ./.venv/bin/oe-dataset generate --config dataset.toml --output artifacts/datasets/issue2-remediation-891923-a --source-repo artifacts/sources/pizzaiolo --total-instances 6 --pizza-types American random --type-counts '{"American": 3, "random": 3}' --generation-seed 31415 --split-seed 27182
  PYTHONPATH="$PWD/build/remediation-wheel-install" ./.venv/bin/oe-dataset generate --config dataset.toml --output artifacts/datasets/issue2-remediation-891923-b --source-repo artifacts/sources/pizzaiolo --total-instances 6 --pizza-types American random --type-counts '{"American": 3, "random": 3}' --generation-seed 31415 --split-seed 27182
  ```

  Both generated dataset ID `pizza-12f170a153661557c94f`, six samples, 72 component occurrences,
  and the same 34-entry immutable-asset checksum map. Semantic record JSON and all split manifests
  were equal. Split IDs were train `img_00000,img_00004`, validation `img_00001,img_00005`, and
  test `img_00002,img_00003`. Wheel-based validation passed with zero errors and 281 low-support
  warnings per run. Validation report IDs were
  `870f7d1bb4bbdbaebcb096cb6b93fdd67c6335c2a91810aa02615be59fedf354` (`...-a`) and
  `c438903574b8248c1b617375417650a0854bc39213e28b55dded6523e52d2505` (`...-b`).
- A further wheel-installed CLI smoke from the repository root omitted `--source-repo`, so it used
  the project's existing `artifacts/sources/pizzaiolo` cache and wrote the output under
  `artifacts/datasets/issue2-remediation-891923-wheel-cache`; it produced the same dataset ID,
  72 components, preparation SHA, and `preparation_dirty: false`, validated with zero errors, and
  did not create a source cache inside the wheel installation. Its validation report ID was
  `23243566377c334f565f451151f2380a7418c8d635f8e5d54d933bed58c4db68` (281 low-support warnings).

All generation outputs above are ignored local artifacts, not committed and **not remotely
retrievable**. No model or training experiment was run, and no test-set predictive feedback was
used.

### Direct-API non-editable-wheel follow-up

After committing the cwd-default fix and tests, code commit
`d609cfe887e3e047d98670faf66027e34db27b70` was clean. A non-editable wheel was built from that
commit and installed into an isolated target directory. Wheel SHA-256:
`c60d90cec7f32b41c9925a1fe62d91e24b40e776eaa99c6c9ef6cc4b1d109455`. The following direct
Python API invocation intentionally omitted both `project_root` and `source_repo`; the imported
module came from the installed wheel, while the pinned source cache and outputs resolved under the
calling project directory, not the wheel installation:

```sh
./.venv/bin/python -m pip wheel --no-deps . --wheel-dir build/issue2-cwd-fix-wheels
./.venv/bin/python -m pip install --no-deps --target build/issue2-cwd-fix-wheel-install build/issue2-cwd-fix-wheels/ontology_embeddings-0.1.0-py3-none-any.whl
PYTHONPATH="$PWD/build/issue2-cwd-fix-wheel-install" ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

import ontology_embeddings
from ontology_embeddings.config import DatasetConfig
from ontology_embeddings.dataset.preparation import prepare_dataset
from ontology_embeddings.dataset.schema import load_records
from ontology_embeddings.dataset.validation import validate_dataset

CODE_SHA = "d609cfe887e3e047d98670faf66027e34db27b70"
ROOT = Path.cwd().resolve()
WHEEL_SITE = ROOT / "build/issue2-cwd-fix-wheel-install"
assert WHEEL_SITE in Path(ontology_embeddings.__file__).resolve().parents
source_cache = ROOT / "artifacts/sources/pizzaiolo"
assert source_cache.is_dir()
config = DatasetConfig(
    total_instances=6,
    pizza_types=("American", "random"),
    type_counts={"American": 3, "random": 3},
    generation_seed=31415,
    split_seed=27182,
)

runs = []
for suffix in ("a", "b"):
    output = ROOT / f"artifacts/datasets/issue2-cwd-api-{CODE_SHA[:7]}-{suffix}"
    # Intentionally omit project_root and source_repo to exercise cwd defaults and cwd cache.
    prepared = prepare_dataset(config, output)
    assert prepared == output and ROOT in prepared.parents
    manifest = json.loads((prepared / "manifest.json").read_text())
    records = load_records(prepared / "samples.jsonl")
    report = validate_dataset(prepared, write_report=False)
    assert manifest["source"]["preparation_commit"] == CODE_SHA
    assert manifest["source"]["preparation_dirty"] is False
    assert manifest["source"]["generator_commit"] == "6c0bcb221a957c184012beaf103bebcde4b1f793"
    assert report.valid
    report_json = json.loads((prepared / "reports/validation.json").read_text())
    splits = {
        name: json.loads((prepared / "splits" / f"{name}.json").read_text())["sample_ids"]
        for name in ("train", "validation", "test")
    }
    runs.append(
        {
            "output": prepared,
            "manifest": manifest,
            "records": [record.to_dict() for record in records],
            "report": report_json,
            "splits": splits,
            "component_count": sum(len(record.components) for record in records),
        }
    )

first, second = runs
first_records, second_records = first["records"], second["records"]
checks = {
    "records": first_records == second_records,
    "labels_targets_masks": all(
        a["pizza"] == b["pizza"]
        and [
            (item["ontology_targets"], item["ontology_known_mask"])
            for item in a["components"]
        ]
        == [
            (item["ontology_targets"], item["ontology_known_mask"])
            for item in b["components"]
        ]
        for a, b in zip(first_records, second_records, strict=True)
    ),
    "splits": first["splits"] == second["splits"],
    "component_ids_order_geometry": all(
        [
            (
                item["component_id"],
                item["detector_category_id"],
                item["detector_category"],
                item["bbox_xyxy_original"],
            )
            for item in a["components"]
        ]
        == [
            (
                item["component_id"],
                item["detector_category_id"],
                item["detector_category"],
                item["bbox_xyxy_original"],
            )
            for item in b["components"]
        ]
        for a, b in zip(first_records, second_records, strict=True)
    ),
    "annotation_refs": all(
        a["source_annotations"] == b["source_annotations"]
        and [item["annotation_path"] for item in a["components"]]
        == [item["annotation_path"] for item in b["components"]]
        for a, b in zip(first_records, second_records, strict=True)
    ),
    "asset_checksums": first["manifest"]["asset_checksums"]
    == second["manifest"]["asset_checksums"],
    "dataset_ids": first["manifest"]["dataset_id"] == second["manifest"]["dataset_id"],
}
assert all(checks.values()), checks
assert first["component_count"] == second["component_count"] == 72
assert not (WHEEL_SITE / "artifacts/sources/pizzaiolo").exists()
print(
    json.dumps(
        {
            "module": str(Path(ontology_embeddings.__file__).resolve()),
            "cwd": str(ROOT),
            "source_cache": str(source_cache.resolve()),
            "wheel_site_has_source_cache": False,
            "code_sha": CODE_SHA,
            "preparation_dirty": [run["manifest"]["source"]["preparation_dirty"] for run in runs],
            "dataset_ids": [run["manifest"]["dataset_id"] for run in runs],
            "validation_report_ids": [run["report"].get("report_id") for run in runs],
            "split_ids": first["splits"],
            "component_count": first["component_count"],
            "asset_checksum_count": len(first["manifest"]["asset_checksums"]),
            "validation": [
                {
                    "valid": run["report"]["valid"],
                    "errors": run["report"]["error_count"],
                    "warnings": run["report"]["warning_count"],
                }
                for run in runs
            ],
            "equality": checks,
            "outputs": [str(run["output"].relative_to(ROOT)) for run in runs],
        },
        indent=2,
        sort_keys=True,
    )
)
PY
```

Both runs recorded code SHA `d609cfe887e3e047d98670faf66027e34db27b70`,
`preparation_dirty: false`, and generator SHA `6c0bcb221a957c184012beaf103bebcde4b1f793`.
They produced dataset ID `pizza-12f170a153661557c94f` and **72 component occurrences** each.
Validation passed with zero errors (281 low-support warnings per run). Validation report IDs:
`f65a3dc4ab1f1bcf27bcee1cd7efc1adcd102891e24f7ae3e74353b20f297aca` (`...-a`) and
`dbfde5a22b2897be1e3113246d1d6b145fc6da8f2ce47c4d2b3279748d6fa541` (`...-b`). Splits were
train `img_00000,img_00004`, validation `img_00001,img_00005`, and test
`img_00002,img_00003`. Equality checks passed for full records, pizza/component labels/targets/
known masks, split manifests, component IDs/order/categories/original-pixel geometry, source and
component annotation references, all 34 asset checksums, and dataset IDs. The outputs are
`artifacts/datasets/issue2-cwd-api-d609cfe-a` and `artifacts/datasets/issue2-cwd-api-d609cfe-b`;
the wheel and outputs are ignored local artifacts, not committed and **not remotely retrievable**.
