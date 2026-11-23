# Local affine run records

Each invocation is recorded under `artifacts/runs/<run-id>` and appended to the JSONL index. `find_runs` skips torn or malformed index lines, then reconciles indexed state from each readable run manifest. This recovers a persisted terminal status after a torn index append and discovers run directories whose manifest exists without an index row. Run records, histories, metrics and checkpoints are local files; this project does **not** establish off-machine or session-independent durability, replication, backup, or remote retention. Do not cite a local path as a durable public evidence link.

`export_run(root, run_id, destination)` copies a bundle into a caller-selected path and writes SHA-256 checksums for each copied file plus an aggregate digest. `verify_export(path)` detects missing/changed copied files. The export is staged and atomically renamed on the destination filesystem. A caller must separately place exports on storage with the required retention/durability guarantees; the helper does not provide those guarantees.

Run manifests distinguish source commit SHA and dirty source fingerprint, resolved training config, seed/environment, dataset/class identities and selected train/validation sample IDs. Training rejects records marked `test`; the metadata-oracle smoke loader skips test JSONL lines before parsing them.
