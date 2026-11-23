# PR 2 affine metadata-oracle acceptance experiment

This package is a reproducible engineering experiment, not an image-recognition result or evidence of learned ontology geometry. Inputs are primitive generator metadata for pizzas and oracle detector-category identities for components. It does not implement or claim CNN, Box, or Town learning.

## Reproduction

Use the repository's `.venv` (Python 3.12 and CPU PyTorch 2.6.0+cpu). Generate the deterministic larger dataset if it is not already present; the ignored dataset is deliberately not committed:

```bash
.venv/bin/oe-dataset generate \
  --output artifacts/datasets/pr2-support-1200 \
  --total-instances 1200 --generation-seed 42 --split-seed 42
.venv/bin/oe-dataset validate artifacts/datasets/pr2-support-1200 --no-write-report --json
PYTHONPATH=src .venv/bin/python -m ontology_embeddings.pr2_acceptance \
  --dataset artifacts/datasets/pr2-support-1200
```

The generator applies the original type-stratified deterministic 60/20/20 split to the 16 configured generator types, with 75 instances of each type. Dataset selection used only train/validation known class support: the 48-image fixture provides just 16 train and 16 validation pizzas and sparse support; the 1,200 candidate was selected as a compact larger candidate with support across the task-applicable classes. Exact per-split and per-class positive/negative/unknown/applicability counts are in `metrics.json`. The sealed test targets are not read by the trainer or used for dataset/model/config selection. The integrity validator may inspect all partitions for structural integrity; no test predictive results are produced.

The runner creates three automatic local run records (main learning, shuffled-training-label control, and persistent linearly learnable tiny overfit control) in ignored `artifacts/runs/`. Main/control histories, checkpoint reload equivalence, class-frequency baseline, class metrics, feature-signature overlap, support, and run IDs are summarized in tracked `metrics.json`. The main run's selected best-by-combined-validation-BCE checkpoint is tracked as `best.pt`; `checksums.json` covers both files. Local paths, complete run histories, datasets, images, and test material are not included in Git and are not remotely durable.
