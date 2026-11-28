# REL-01 offline geometric evaluation

**Status:** global geometry complete for all six verified checkpoints; observed-instance mode blocked because the exact historical dataset is not available locally. No raw records or per-entity embeddings are included.

- Evaluator: commit `ca7f31e123b398f9ab41d0c108fc0304898c4034` on `thesis/rel-01`.
- Model source: `0e8a996794b788e45b96f108e79a03dca2843e4e`.
- Full per-class, per-pair, reference and transition evidence: `results.json.gz` (SHA-256 `f815bbd6855c6c30ca7d0529645a99d4d3a920cfb587d80757af361fc068f823`).
- Run/checkpoint identities are embedded in the JSON; checkpoints were verified against the immutable BOX-R01 evidence checksums. Evaluator rechecks checkpoint hashes, checkpoint metadata, canonical vocabulary hash, class IDs/names/IRIs and expected epochs.

## Principal results

Exact relations are containment/disjoint unordered-pair counts; nonempty intersections include closed-boundary contact.

| Domain | Checkpoint | Epoch | Containments | Disjoint pairs | Overlap pairs | Recovered subclass | Recovered disjoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unbounded | Initial | 0 | 0 | 0 | 1770 | 0/148 | 0/1185 |
| Unbounded | Selected | 45 | 86 | 329 | 1441 | 73/148 | 260/1185 |
| Unbounded | Latest | 60 | 90 | 389 | 1381 | 75/148 | 309/1185 |
| Bounded | Initial | 0 | 0 | 0 | 1770 | 0/148 | 0/1185 |
| Bounded | Selected | 54 | 0 | 140 | 1630 | 0/148 | 117/1185 |
| Bounded | Latest | 60 | 0 | 155 | 1615 | 0/148 | 128/1185 |

The named-axiom reference covers 148 non-reflexive named subclass pairs and 1,185 unordered disjoint pairs among the 60 trainable classes. It supports explicit named subclass closure, explicit `disjointWith`, `AllDisjointClasses` expansion and inherited disjointness; it is not a complete OWL reasoner. No forced-empty trainable classes were detected in this fragment. Five unsupported classes remain outside the primary metrics; relations touching them are separately counted in JSON. `DomainThing` is not represented by an arbitrary finite box and its reference relations are separately accounted for.

The global exact relation and selected recovery counts agree with all supplied preliminary cross-checks. Unbounded selected reference-agreement precision/recall: subclass 0.849/0.493, disjointness 0.790/0.219. Bounded selected: subclass precision/recall undefined/0; disjointness 0.836/0.099. Extra learned relations are not treated as contradictions unless incompatible with an explicit supported condition and the learned class regions are nonempty.

### Limitations

The original `pizza-6a33710d13eb90fdde3e` dataset was not found in local search. Therefore no train/validation observed-embedding evaluation was run; no substitute data was used, and test data was not accessed. Global relations use stored, hash-verified physical float32 endpoints promoted to float64 for diagnostics. CPU re-decoding of some CUDA-trained raw checkpoint parameters differed from the stored endpoints by at most `2.3841858e-7`; the evaluator retains and validates the stored endpoints as authoritative and rejects larger decode discrepancies (5e-7 bound). Exact geometric predicates themselves use no epsilon.

Verification: 234 tests passed (complete suite, 53.67 s); REL-01 analytical/reference tests: 10 passed; Ruff checks passed. One focused independent reviewer found a vocabulary-provenance validation gap; canonical vocabulary hash verification was added before the final full-suite run. Dataset-mode review remains incomplete because its required dataset is unavailable.
