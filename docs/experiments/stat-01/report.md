# STAT-01 evidence summary
Dataset: pizza-6a33710d13eb90fdde3e; manifest: c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd.
Only training and validation were analyzed; test lines were skipped before JSON decoding.

train/pizza: entities=720, signatures=414, patterns=61, E_min=0/22320, implications/equivalences/exclusions=119/1/269, axiom contradictions=0
train/component: entities=12338, signatures=22, patterns=19, E_min=0/370140, implications/equivalences/exclusions=95/2/338, axiom contradictions=0
validation/pizza: entities=240, signatures=174, patterns=41, E_min=0/7440, implications/equivalences/exclusions=124/2/277, axiom contradictions=0
validation/component: entities=4028, signatures=22, patterns=19, E_min=0/120840, implications/equivalences/exclusions=95/2/338, axiom contradictions=0
pizza train/validation: signature overlap=84, train-only=330, validation-only=90, JS=0.3713703909444379 nats
component train/validation: signature overlap=22, train-only=0, validation-only=0, JS=0.0006721404829537397 nats

See summary.json, classes.csv, class_pairs.csv, feature_associations.csv, feature_pairs.csv, coverage.json, and signature JSONL for primary evidence.
Empirical implications are finite-sample patterns, not global ontology claims. Feature-association evidence is occurrence-weighted; validation is development data.

Correction: the supported ontology fragment now includes named-member `owl:AllDisjointClasses` pair expansion and inherited disjointness. This adds 312 component-applicable disjoint pairs to the 3 previously represented pairs (315 total); no joint-positive contradiction was found in train or validation. Component special-class denominators now count components only: `DomainThing` train support is 12,338 (validation 4,028). The fixed-positive/unsupported special-class semantics are unchanged.

Correction evidence was regenerated from the retained train/validation signature evidence and pinned ontology with implementation source `d000c04cfc81dcb17ede15a19f3a5ade39db45c0`; unaffected audit measurements were preserved. The original raw dataset is not present in this correction environment, so no fresh full data-loader audit was possible. CSV outputs are serialized with LF, and checksums cover the final artifact bytes.
