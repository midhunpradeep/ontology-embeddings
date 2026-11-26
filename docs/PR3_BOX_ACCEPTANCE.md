# PR 3 BOX milestone — acceptance closeout

**Scope:** BOX classifier implementation and PR 3.3 acceptance only. This report distinguishes implementation provenance, scientific evidence, and current integration verification. It is not a definitive comparative thesis result.

## Disposition

The original implementation at `0e8a996794b788e45b96f108e79a03dca2843e4e` satisfies acceptance levels **A, B, and C** under issue #11's preliminary single-seed protocol. The two approved 60-epoch models were not retrained or retuned. The four prescribed fixed-coordinate controls were run once on the clean original implementation SHA. The original model-acceptance finalizer reports all three levels passed with run provenance verified.

A packaging-only correction on the integration branch prevents checkpoint export when run provenance is invalid. The correction does not change model or training behavior and does not retroactively change original training provenance.

## Data and provenance reconciliation

- Dataset: `pizza-6a33710d13eb90fdde3e`, 1,200 records; historical train/validation/test counts 720/240/240.
- Declared dataset manifest identity: `c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd`.
- Raw `manifest.json` bytes SHA-256 (distinct from declared identity): `94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844`.
- Vocabulary SHA-256: `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`.
- Historical ordered train/validation ID sequence hashes: `d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219` / `a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c`. Preparation artifacts were checked against their own recorded SHA-256 values. Both durable run manifests contain 720/240 IDs and the same raw manifest/vocabulary hashes.
- The test partition remains sealed for predictive use: no test records were decoded, tuned on, selected against, or evaluated. The preparation record identifies the test count and explicitly records the no-test-use limitation.
- Both successful run manifests identify clean source `0e8a996794b788e45b96f108e79a03dca2843e4e`. They used the same logistic config (d=16, K=1, 60 epochs, batch 64, seed 23, Adam, LR .02, tau .1, clipping 1, epsilon 1e-4, no weight decay) and the paired prepared initialization. Adapter hashes match exactly; reconstructed physical box bounds differed by at most `5.96e-8` (tolerance `2e-5`).
- The failed first unbounded invocation `20261125T205722-f01807b66750` is retained and indexed as a pre-training CUDA alias/device validation failure. It is not counted as a model run. Successful runs and all preparation/checkpoint/metric artifacts were recovered locally and reconciled with the immutable sanitized evidence bundle.

## Acceptance A — mathematical and numerical correctness: PASS

The original clean checkout ran the focused mathematical/training tests: **33 passed** in 5.61 seconds. This covers the existing BOX transforms, membership, objectives, gradients, training/checkpoint contracts and regression checks. The evidence records the exact command, source before/after, and output SHA-256. The full integration regression suite later passed separately (225 tests; see verification below).

## Acceptance B — fixed-coordinate recovery and joint learning: PASS

All four indexed controls ran with the prescribed 2D fixture, fixed points, 350 epochs, CPU float32, and exact source SHA. Each recovered membership `[true, true, false, false, false, false]`, had finite nonzero geometry gradients and actual endpoint updates, and retained valid initial/final checkpoints. Run IDs, configs, verification status and checksums are recorded in the machine-readable evidence.

Both substantial runs also updated the pizza adapter, component adapter and geometry. Encoder and geometry gradient norms were positive and finite in all 60 epochs; selected-checkpoint reload predictions were equivalent. Paired initialization validation passed.

## Acceptance C — predictive improvement and generalization evidence: PASS

Selection is minimum combined native validation loss (earliest tie). The best hard-membership mean-of-available-task micro-F1 epochs are reported separately; neither replaces the native-loss-selected checkpoint.

| Domain | Loss-selected epoch / validation loss | Pizza micro / macro F1 (31 defined classes) | Component micro / macro F1 (30 defined classes) | Best mean-task F1 epoch |
|---|---:|---:|---:|---:|
| Unbounded identity | 45 / 0.01507348 | 0.98698153 / 0.98657935 | 1.00000000 / 1.00000000 | 45 (0.99349077) |
| Bounded tanh | 54 / 0.02292295 | 0.98492159 / 0.98238998 | 0.99952368 / 0.99992260 | 59 (0.99344795) |

Train-only frequency baselines are pizza `0.64755352` and component `0.73198468` micro-F1; each domain strictly exceeds each corresponding baseline at its loss-selected checkpoint. Per-class counts, scores, supports, undefined classes, and seen/unseen signature metrics are retained in BOX-R01 full histories. For validation pizza signatures, 143/240 were seen and 97/240 unseen: unbounded F1 `0.99949622 / 0.96813354`, bounded `0.99595960 / 0.96856287`. All 4,028 component occurrences had seen signatures; there were no unseen component signatures, so unseen component F1 is undefined. Macro F1 and the full class-level support/metric records are included in the retained history, not inferred from micro scores.

This is metadata-oracle prediction on one fixed split and one seed. It is preliminary evidence, not visual recognition, external generalization, or definitive comparative thesis evidence. Against the historical affine validation results (pizza/component micro-F1 `0.98944830 / 1.00000000`), BOX pizza scores are lower; unbounded component ties and bounded is slightly lower. The comparison is descriptive and does not isolate the geometric effect.

## Integration-only correction

The finalizer previously selected checkpoint files for export based only on a selectable loss/checkpoint path, even after marking provenance invalid. A focused regression test reproduces this exact boundary with a selected checkpoint and deliberately invalid provenance. The export eligibility guard now also requires globally valid run provenance. The test verifies that no checkpoint file or misleading selected-checkpoint reference is emitted. This change is packaging-only; original source metadata remains untouched, and no training was repeated to match a packaging SHA.

## Verification and durable evidence

- Full regression suite on the final integration source: **225 passed**. One initial attempt using the CUDA experiment venv lacked this source checkout's distribution metadata (224 passed, one environment-only package metadata test failed); the suite was run once successfully with the existing PR2 PyTorch environment, without installing dependencies.
- Focused post-correction export-guard test: **1 passed**.
- Ruff checks (`ruff check src tests`, changed-file format check) and `git diff --check`: passed.
- Historical affine compatibility: the original PR2 `best.pt` loaded strictly into the affine model; both pizza and component forward outputs had shape `[1, 60]` and finite logits.
- Immutable full BOX-R01 artifact reference: [`thesis/box-r01-evidence` at `5899bb43df4dbbdd56b9d1dc822c5a841994535c`](https://github.com/midhunpradeep/ontology-embeddings/tree/thesis/box-r01-evidence/evidence/box-r01). Its package-level and per-run checksums verified; both compressed metric histories contain all 60 epochs. The package contains no raw dataset or decoded sample-ID lists. The PR includes only compact acceptance evidence and references this immutable bundle rather than copying its checkpoints/history.
- Original model finalization and the compact acceptance summary are checksummed. Current integration verification is recorded separately from historical training provenance.

## Limitations

Single seed and one required configuration per domain; no independent test evaluation; no external/visual generalization; no claim of BOX superiority to affine; no global ontology-relation guarantee; and no definitive thesis conclusion. STAT-01 and REL-01 remain separate investigations and are not PR 3 acceptance gates. CI and the two fresh final-diff independent review axes remain required before this PR is merge-ready. Human approval is required to merge.
