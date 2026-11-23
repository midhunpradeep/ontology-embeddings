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

The generator applies the original type-stratified deterministic 60/20/20 split to the 16 configured generator types, with 75 instances of each type. Dataset selection used only train/validation known class support: the 48-image fixture provides just 16 train and 16 validation pizzas and sparse support; the 1,200 candidate was selected as a compact larger candidate with support across the task-applicable classes. Exact per-split and per-class positive/negative/unknown/applicability counts are in `metrics.json`. The train/validation loader skips test JSONL rows before parsing. Dataset integrity/semantic validation checks can inspect all partitions; its pass/error status was used only as a data-integrity gate, not as predictive, model-selection, or dataset/config-selection feedback. No test predictions or predictive metrics were produced.

## Selected completed evidence

The three selected completed run bundles were produced from clean source commit `522c3f30240e65a431abdfc7694d0006866b0874`:

- Larger learning: `20261123T105917-c00ea4941c34` (60 epochs, seed 23).
- Shuffled-training-label control: `20261123T110554-8f5338dd36e6` (60 epochs, shuffle/training seed 29).
- Persistent tiny overfit control: `20261123T111217-b7ae01874d21` (60 epochs, seed 11).

The larger run selected epoch 28 by combined validation BCE (0.01401536); best reload reproduced validation metrics/predictions. Validation micro-F1 was 0.9894 for pizza and 1.0000 for components, compared with train-frequency baselines of 0.6476 and 0.7320. The shuffled control scored 0.6677 and 0.7320 respectively; the component control matched the baseline, so the single control is not evidence beyond the baseline there. Pizza signature diagnostics include 143 seen / 97 unseen validation rows; all 4,028 component validation rows have seen one-hot signatures. Pizza classes `Food` and `Pizza` are positive-only in both development partitions; remaining task-applicable classes have both signs. The exact class supports, per-class metrics, baselines, curves, overlap diagnostics, hashes, configurations, checkpoint references, and limitations are in `metrics.json`.

The main run's selected best-by-combined-validation-BCE checkpoint is tracked as `best.pt` (9,836 bytes). `checksums.json` covers the report, README, and checkpoint. A later invocation `20261123T111432-30a6ffe1ea8e` was stopped by the 30-minute timeout at epoch 38: its raw manifest still says `running`, and it is explicitly recorded as incomplete in `metrics.json`, never as a successful result. It used packaging/provenance source `c7644f7a75f44b2843bf853cf16157172449b12a`; it did not replace the selected completed run.

Local paths, complete run histories, datasets, images, and test material are not included in Git and are not remotely durable. Only the compact report and selected best checkpoint are tracked.
