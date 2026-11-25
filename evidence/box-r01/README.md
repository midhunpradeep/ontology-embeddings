# BOX-R01 sanitized experiment evidence

This package preserves the two completed 60-epoch BOX runs, per-epoch metrics and diagnostics, checkpoint files, configuration/provenance, timing, and paired-initialization verification. `results.json` summarizes selected results and baseline comparisons. It intentionally excludes the raw dataset, decoded train/validation records, sample-ID lists, original run-store indexes, local absolute paths, and unredacted local logs.

Source: `0e8a996794b788e45b96f108e79a03dca2843e4e`. Dataset: `pizza-6a33710d13eb90fdde3e`; declared manifest identity: `c10c5d7966ba12badc644190191f76f94169f1d10c77aee0c971c861582455cd`; loader raw manifest file SHA-256: `94be3c3617349bd6d48ce1eaa5afe53e60b62a3711c8bb366c631306430f0844`. Exact historical train/validation selections were checked, but only their hashes/counts are included. Test data was not decoded or evaluated.

Each run has a sanitized `run.json` with invocation templates, gzip-compressed full `metrics.jsonl` (all epochs, per-class support, seen/unseen signatures, geometry and gradient diagnostics), checksummed initial/best/latest checkpoint files, and a file checksum index. Package-level `paired-verification.json`, `timing.json`, and `failure.json` record pairing, timing, and the failed pre-training invocation.
