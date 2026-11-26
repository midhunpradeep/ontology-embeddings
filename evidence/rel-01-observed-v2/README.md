# REL-01 corrected observed evidence (v2)

## Disposition and interpretation

This non-overwriting correction bundle addresses the two findings in the consolidated review of REL-01 ([issue comment 7178267291](https://github.com/midhunpradeep/ontology-embeddings/issues/16#issuecomment-7178267291)). It is derived from the complete stored per-pair records in `rel-01-observed-v1`; no dataset access, model inference, checkpoint reevaluation, or retraining was performed. The earlier global-only and observed-v1 bundles remain unchanged.

The observed-v1 README's reported **80–86% rates are explicitly superseded as ontology-axiom violation rates**. They are conditional violation rates over all candidate class-pair implications, most of which are not supported ontology axioms. The arithmetic remains valid as descriptive all-candidate statistics, but those values must not be interpreted as reference-axiom consistency. Correct named-fragment reference-only results are below and in `results.json.gz`.

Keep these separate:

1. **Global geometric relation satisfaction:** exact inclusion/disjointness predicates between stored class boxes, over the global class-pair universe. These are geometric facts about the learned regions, not observed-point statistics.
2. **Reference-only observed axiom consistency:** observed pair–entity measurements filtered to 148 supported directed named-fragment subclass relations (child → parent) and 1,185 supported unordered disjoint pairs.
3. **All-candidate observed descriptive statistics:** the original aggregate over every candidate pair. Retained unchanged in the nested `summary`, identified as `all_candidate_class_pairs`; not every candidate implication is an ontology requirement.

Each observed population remains separate (train/validation × pizza/component), as do supervision-restricted and unrestricted views, and occurrence-weighted and distinct-exact-embedding units. Restricted eligibility requires both labels known and both classes applicable; unrestricted evaluation includes every observed entity. These are pair–entity measurements, not independent-entity samples, significance tests, or classification errors. Reference-supported pair–entity conflicts describe observed geometric inconsistencies, not independent classification errors. Cross-supervision counts apply the same reference-pair filters and remain event counts, not unique entities.

Subsumption is counted in the reference child-to-parent direction only; disjoint reference pairs are counted once. Each result retains violations, antecedent support, eligible opportunities, denominators, conditional rates, and definedness/satisfaction status. A zero denominator has a `null` rate and a reason. Zero support with no violations is marked vacuously satisfied, separately from supported satisfaction.

## Selected-checkpoint reference-only results

`S = subsumption violations / antecedent support`; `D = disjointness violations / eligible pair–entity opportunities`. `R-O` and `R-X` are supervision-restricted occurrence-weighted and distinct-embedding; `U-O` and `U-X` are unrestricted. `0/0` means the rate is **undefined**, not zero. Distinct embeddings are exact stored-coordinate groups, deduplicated only after each relation's eligibility filter.

| Model / population | R-O (S; D) | R-X (S; D) | U-O (S; D) | U-X (S; D) |
|---|---|---|---|---|
| Unbounded train pizza | 0/8523; 0/0 | 0/4895; 0/0 | 78/9876; 5343/853200 | 60/5724; 3222/490590 |
| Unbounded train component | 0/92598; 0/3886470 | 0/144; 0/6930 | 174/175173; 177832/14620530 | 1/254; 244/26070 |
| Unbounded validation pizza | 0/2799; 0/0 | 0/2005; 0/0 | 31/3243; 1687/284400 | 24/2329; 1212/206190 |
| Unbounded validation component | 0/30122; 0/1268820 | 0/144; 0/6930 | 69/56924; 57660/4773180 | 1/254; 244/26070 |
| Bounded train pizza | 1/8527; 0/0 | 1/4921; 0/0 | 810/10903; 8953/853200 | 353/6049; 4276/490590 |
| Bounded train component | 0/92598; 0/3886470 | 0/144; 0/6930 | 945/209765; 224328/14620530 | 6/311; 324/26070 |
| Bounded validation pizza | 0/2827; 0/0 | 0/2045; 0/0 | 263/3638; 3006/284400 | 153/2554; 1900/206190 |
| Bounded validation component | 0/30122; 0/1268820 | 0/144; 0/6930 | 320/68037; 72558/4773180 | 6/311; 324/26070 |

The table agrees with the independent numerical derivation in the review for all eight selected populations and all four measurement views. Results for all six checkpoints and all 24 populations are in the structured bundle. `transitions` retains original all-candidate deltas and adds reference-only consistency deltas with explicit population identities.

The filtered cross-supervision reference event counts are in each population's `reference_only_summary.violations_outside_joint_supervision`. The old all-candidate cross-supervision totals remain in their original field and are labeled `all_candidate_class_pairs`.

## Preserved global and all-candidate evidence

The original global counts, witnesses, geometry records and per-pair measurements are unchanged. Selected-checkpoint global counts remain: unbounded (epoch 45) 86 containments / 329 disjoint pairs / 1,441 overlaps; bounded (epoch 54) 0 / 140 / 1,630. Both historical evidence bundles and their checksums remain unchanged.

The original all-candidate observed aggregate values are retained exactly, not replaced. Their identity is now explicit in this bundle. They describe the all-candidate population and must not be renamed ontology violations. Absence from the supported named fragment is not a negative entailment; the reference remains a limited named-axiom fragment, not complete OWL reasoning.

## Invocation and producing-operation provenance

The original evaluation's exact historical argv was not recoverable from retained logs or run records, and is not invented. Each evaluation records the known entry point, observed mode, sanitized argument names/non-sensitive values, checkpoint SHA-256/run/epoch, dataset identity and historical producing evaluator revision; local paths are omitted. The exact-command status is explicitly `not_recoverable` in `original_evaluation_invocation`.

The subsequent evidence correction is a different operation. `correction_provenance` records the actual aggregation entry point, input/output evidence bundle values, source evidence SHA-256, correction source revision, and confirms no checkpoint evaluation/dataset access/model inference. The production CLI records the actual sanitized evaluation invocation, checkpoint/dataset identity, and producing source revision for future runs. The independent direct-recomputation operation is separately identified in `independent-verification.json`.

## Verification

- Independent direct recomputation from stored pair records: **PASS**, all six checkpoints × four populations (24); separate code path from the reusable aggregation helper.
- Independent cross-check against reviewer-provided selected-checkpoint occurrence and distinct-embedding results: **exact match** (all numerators and denominators above).
- Original global pair/witness records and all-candidate observed per-pair measurements/aggregate summaries: **unchanged**, checked against observed-v1.
- Focused REL-01 tests: **11 passed**; Ruff and `git diff --check`: **passed** at correction preparation. Full regression and targeted independent review are recorded in the issue handoff.

Source implementation commit: `0f25b3d9352781044eaba2684c257d037c0d06f7`. Evidence correction invocation and SHA-256 are recorded in `correction_provenance` and `SHA256SUMS`.
