# Local affine run records

Each invocation is recorded under `artifacts/runs/<run-id>` and appended to the JSONL index. `find_runs` skips torn or malformed index lines, then reconciles indexed state from each readable run manifest. This recovers a persisted terminal status after a torn index append and discovers run directories whose manifest exists without an index row. Run records, histories, metrics and checkpoints are local files; this project does **not** establish off-machine or session-independent durability, replication, backup, or remote retention. Do not cite a local path as a durable public evidence link.

`export_run(root, run_id, destination)` copies a bundle into a caller-selected path and writes SHA-256 checksums for each copied file plus an aggregate digest. `verify_export(path)` detects missing/changed copied files. The export is staged and atomically renamed on the destination filesystem. A caller must separately place exports on storage with the required retention/durability guarantees; the helper does not provide those guarantees.

Run manifests distinguish source commit SHA and dirty source fingerprint, resolved training config, seed/environment, dataset/class identities and selected train/validation sample IDs. Training rejects records marked `test`; the metadata-oracle smoke loader skips test JSONL lines before parsing them.

## Source identity availability

Source discovery is based only on the installed `run_store.py` module location, never the
process working directory. Provenance Git subprocesses discard inherited `GIT_*` environment
variables, which could redirect repository, worktree, index, refs, objects, namespace, or status
behavior away from the implementation checkout. In an editable/source-tree checkout, the module
must be at `<repository>/src/ontology_embeddings/run_store.py`, the root must contain the project
marker
`pyproject.toml`, and Git must report that same directory as the repository root. If these checks
succeed, the manifest records the full commit SHA and Git working-tree dirty state; relevant
changed source paths are hashed where present. A regular (non-editable) installation places the
module under `site-packages` and therefore cannot identify its source checkout. In that case
`source.commit_sha` and `source.dirty` are explicitly `null`, `source.availability` is
`"unavailable"`, and `fingerprint_note` explains the limitation. The implementation deliberately
does not use cwd or accept an unchecked SHA override, which could attribute an unrelated checkout.

The searchable `source_sha` index field is null when provenance is unavailable. Filtering by a
known SHA continues to match historical and current runs with that exact SHA; a missing SHA is
never treated as a match, including when a caller supplies the string `"None"`.
