# TOWN final-checkpoint metadata correction

The original epoch-60 `final.pt` artifacts are preserved byte-for-byte. Their `validation_loss` metadata copied the best epoch's loss instead of the loss from epoch 60. The authoritative full-precision values are the final records in each original run's `metrics.jsonl` (line 60); file and record SHA-256 values are in [`correction-record.json`](correction-record.json).

| Run | Original final.pt SHA-256 | Recorded metadata | Correct final-history loss |
|---|---|---:|---:|
| Unbounded `20261127T142303-452045129f33` | `dc476ecd9c59fc8fc37041093c7889af0034d795184a8de3bfb2693ee1a89c48` | 0.012100125662982464 | 0.012150946073234081 |
| Bounded `20261127T142545-2e2b89819b9f` | `6e589eeebc691476b18657f98c70b5c2ec7c00907061002ec64d5070a71e06af` | 0.05192398652434349 | 0.05312584340572357 |

Both runs were produced from clean experimental source revision `3714b8f1e7e1aa94ab252734c6a73b251973077f`. The producer correction is source revision `aff5e888b7206af5be45ac14a26254af5ad26487`. It stores the native final-epoch validation loss in `final.pt`, preserves `None` without a validation partition, and leaves best-checkpoint selection and metadata unchanged.

This is a metadata-only correction. It does not alter model state, physical geometry, training, evaluation, configuration, selected checkpoints/metrics, or scientific conclusions. Neither run was repeated. No derivative checkpoint was necessary: the original bytes remain authoritative artifacts and this record reconciles the inaccurate field against retained producer history. See [`correction-record.json`](correction-record.json) for full provenance and hashes.
