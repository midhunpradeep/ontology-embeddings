# REL-01 observed-instance evaluation

**Completion:** global geometry and observed train/validation evaluation are implemented and executed for all six verified BOX checkpoints. The 240 sealed test records were loaded only as part of dataset integrity validation and were not sent to feature extraction, model encoding, membership evaluation, or metrics.

## Provenance and verification

- Evaluator source commit: `282ccdacffddaa4188d6cf609e3e68ac72e37bf8` on `thesis/rel-01`.
- Original model source: `0e8a996794b788e45b96f108e79a03dca2843e4e`.
- Dataset: `pizza-6a33710d13eb90fdde3e`; declared manifest identity `c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd`; raw manifest SHA-256 `94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844` (separate identities).
- The retained PR2 dataset copy passed the existing `load_prepared_dataset(validate=True)` integrity path. Canonical vocabulary SHA-256: `e803479520664a5de823848c41a29f45a70f6d2a755049a9be14b52312cddd54`.
- Historical train and validation selections were checked against the original run identities: train 720, SHA-256 `d9f7ef8f7ea3d20def144c052c8b9d3ff88c538036798a780e36f55bee30e219`; validation 240, SHA-256 `a9fe9f1b18c635f8ab1067a65d926541b64da7734c59fefc66a00aa780fe9c0c`. Both match the historical affine run manifest and the two BOX-R01 run selections. Test records evaluated: zero.
- All six checkpoint checksums and actual epochs were verified. Observed features use the original metadata feature schema and existing encoders on CPU; exact membership uses the checksummed physical endpoints stored in each checkpoint. The full per-pair/per-class evidence records the evaluator, model, dataset, feature schema, run, checkpoint, split hashes, device and numerical policy.
- Full evidence: `results.json.gz`; SHA-256 is in `SHA256SUMS`. The earlier global-only bundle at `evidence/rel-01/` is unchanged.

## What the measurements mean

Each population is kept separate: train/validation × pizza/component. Raw geometric membership is evaluated against every trainable class without task masks. The supervision-restricted view filters each relation to entities where both labels are known and both classes apply; the unrestricted view includes every observed entity. Each pair has its four predicted membership cells, both directed implication supports/violations, disjointness violations, explicit denominators and excluded counts. Distinct embeddings are grouped by exact stored encoded-coordinate equality, after eligibility is established for each restricted relation. Rates below aggregate pair–entity events over class pairs; they are not independent-entity statistics or significance tests.

### Global relation counts (unchanged from the global-only evidence)

| Domain | Checkpoint (epoch) | Containment | Disjoint pairs | Overlap pairs |
|---|---|---:|---:|---:|
| Unbounded | Initial (0) | 0 | 0 | 1770 |
| Unbounded | Selected (45) | 86 | 329 | 1441 |
| Unbounded | Latest (60) | 90 | 389 | 1381 |
| Bounded | Initial (0) | 0 | 0 | 1770 |
| Bounded | Selected (54) | 0 | 140 | 1630 |
| Bounded | Latest (60) | 0 | 155 | 1615 |

Global exact counts exactly match the previously published global-only measurements. Runtime assertions also verified that no observed point violates a predicted global containment or global disjointness relation under the stored endpoints.

### Selected checkpoint observed consistency

Rates are conditional subsumption violation rate / disjointness violation rate. `R` is supervision-restricted; `U` is unrestricted. Count values are rates across the corresponding population, not model accuracy.

| Domain | Split / entity | Occurrences / distinct points | R subsumption / disjointness | U subsumption / disjointness |
|---|---|---:|---:|---:|
| Unbounded | Train pizza | 720 / 414 | 0.7954 / 0.0459 | 0.8727 / 0.0173 |
| Unbounded | Train component | 12,338 / 22 | 0.8643 / 0.0213 | 0.8650 / 0.0190 |
| Unbounded | Validation pizza | 240 / 174 | 0.7968 / 0.0450 | 0.8743 / 0.0168 |
| Unbounded | Validation component | 4,028 / 22 | 0.8646 / 0.0212 | 0.8654 / 0.0189 |
| Bounded | Train pizza | 720 / 414 | 0.7958 / 0.0458 | 0.8591 / 0.0205 |
| Bounded | Train component | 12,338 / 22 | 0.8644 / 0.0213 | 0.8463 / 0.0242 |
| Bounded | Validation pizza | 240 / 174 | 0.7971 / 0.0453 | 0.8592 / 0.0204 |
| Bounded | Validation component | 4,028 / 22 | 0.8647 / 0.0212 | 0.8471 / 0.0239 |

The result demonstrates why component occurrences must not be treated as independent locations: 12,338 training occurrences collapse to only 22 exact component embeddings (4,028 validation occurrences also collapse to 22). For example, unbounded selected train-component restricted disjointness is 114,060 violations / 5,367,030 pair-entity opportunities (0.02125) occurrence-weighted, versus 181 / 9,570 (0.01890) over distinct embeddings. Both measurements are preserved per pair.

Unrestricted evaluation exposes many violations outside joint supervision. Among unbounded selected train components, the unrestricted-minus-restricted population contains 3,872,157 directed subsumption violation events and 301,435 disjointness events; bounded selected has 4,357,784 and 413,682 respectively. These are pair–entity events, not unique violating entities or label errors. Unsupervised memberships were not treated as prediction mistakes.

### Relation to global axioms and checkpoint changes

For the 148 named-fragment subclass relations, all had some unrestricted observed antecedent support when populations are pooled, but 74 per individual population have zero supervision-restricted support. Across all four restricted populations, 870 of the 1,185 reference disjoint pairs had no jointly eligible observed entity; per-pair evidence identifies each relation. For global geometry misses, observed `A+ / B−` evidence exists for 10 of 75 missed reference subclass pairs at unbounded selected and 35 of 148 at bounded selected (across train/validation and both entity types). The full output also identifies support/violations per view and split. A missing global containment is not itself evidence of an observed violation; reference absence remains open-world.

Initial, selected and latest observed summaries and occurrence/distinct deltas are included in `transitions`. Training changes substantially increase observed conditional containment violations from initial to selected; selected-to-latest changes are smaller and population-dependent. The bounded selected-to-latest component unrestricted rates move from 0.8463 to 0.8433 in train and 0.8471 to 0.8438 in validation, while disjointness rates move from 0.02417 to 0.02535 and 0.02392 to 0.02517. These are descriptive measurements only.

The ontology comparison remains the named-axiom fragment (148 subclass, 1,185 disjoint among 60 trainable classes), not a complete OWL reasoner. The prior CPU-vs-CUDA raw-parameter decoding discrepancy is retained in the global evidence documentation; observed membership uses the checkpoint's validated stored physical endpoints as authoritative. No test predictions, raw records, per-entity IDs, or full per-entity embeddings are included.

## Verification

- Complete regression: **239 passed** (58.45 s).
- Focused REL-01 geometry/reference/observed tests: **15 passed**.
- Ruff and `git diff --check`: passed.
- One focused independent reviewer of the new observed-instance functionality: **PASS**, no concrete findings.
