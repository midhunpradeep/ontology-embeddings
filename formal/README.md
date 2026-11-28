# Formal mathematics environment

`OntologyEmbeddings/Semantics.lean` is a small Mathlib-backed foundation: concepts are sets over an abstract type, and interpretations are restricted to an explicit carrier. This is deliberately not a model of boxes, TOWN, affine regions, the Python implementation, or new ontology mathematics. See [`THEOREM_LEDGER.md`](THEOREM_LEDGER.md) for the LEAN-00 results and the lightweight research process.

## Pinned project and local installation

- Lean: `leanprover/lean4:v4.27.0` (`formal/lean-toolchain`).
- Mathlib: `v4.27.0`, resolved commit `a3a10db0e9d66acbebf76c5e6a135066525ac900`.
- `lake-manifest.json` is the exact dependency lockfile; do not delete it or run `lake update` casually.
- Imports are targeted (`Mathlib.Data.Set.Basic`, `Mathlib.Data.Set.Insert`); the broad `Mathlib` import has no precompiled umbrella artifact in the LEAN-00 setup.

For this host, the LEAN-00 toolchain was copied out of temporary `/tmp/lean00-elan-home` into the durable user-owned installation `$HOME/.local/share/lean-elan-lean01`. No global shell configuration, default elan installation, editor extension, or Pi configuration was changed. Use a clean shell with this explicit environment (do not source the old temporary setup):

```bash
export LEAN_ELAN_HOME="$HOME/.local/share/lean-elan-lean01"
export ELAN_HOME="$LEAN_ELAN_HOME"
export PATH="$ELAN_HOME/bin:$PATH"
cd /home/midhun/Projects/ovgu/sem8/worktrees/ontology-embeddings-lean-01/formal
lake --version
lean --version
lake exe cache get       # fetch compatible Mathlib artifacts when needed
./scripts/verify.sh      # the one verification entry point
```

The verifier honors `LEAN_ELAN_HOME`; if unset it uses that user-owned location when present, otherwise the shell's existing elan. `lake exe cache get` is a one-time/repeatable cache preparation step, not a dependency update. `.lake/` is local, ignored build/cache state. On a new host install elan by its official instructions, keep the committed toolchain and manifest, then run the cache command; do not compile all of Mathlib from source as a fallback.

A fresh-shell check without inherited PATH or elan settings can be run from the repository root:

```bash
env -i HOME="$HOME" PATH=/usr/bin:/bin \
  LEAN_ELAN_HOME="$HOME/.local/share/lean-elan-lean01" \
  bash -c 'cd /home/midhun/Projects/ovgu/sem8/worktrees/ontology-embeddings-lean-01/formal && ./scripts/verify.sh'
```

## Verify and inspect proofs

`./scripts/verify.sh` builds the Lake project, rejects source-level `sorry`, `admit`, and `axiom` tokens, then runs [`AxiomAudit.lean`](AxiomAudit.lean) against every LEAN-00 theorem. It fails on missing audit entries and any dependency outside Lean's foundational `propext`, `Classical.choice`, and `Quot.sound`. The foundations are not user assumptions; in particular `Classical.choice` appears only for excluded middle. All other existing results are axiom-free or rely on `propext` for set extensionality. Extend the audit inventory when adding theorem statements. This is a focused project audit, not a proof of the intended meaning of the statements.

Other useful commands, run in `formal/`:

```bash
lake build
lake env lean OntologyEmbeddings/Semantics.lean
lake env lean AxiomAudit.lean
```

### One proof, step by step

Open `OntologyEmbeddings/Semantics.lean` and locate `subsumes_trans`. In words: if every carrier point in `a` is in `b`, and every carrier point in `b` is in `c`, then every carrier point in `a` is in `c`. Its proof is ordinary function composition:

```lean
theorem subsumes_trans (carrier a b c : Set α)
    (hab : Subsumes carrier a b) (hbc : Subsumes carrier b c) :
    Subsumes carrier a c := by
  intro x hx hxa
  exact hbc x hx (hab x hx hxa)
```

`intro` turns the goal into a point `x`, carrier-membership `hx`, and `a`-membership `hxa`. `hab x hx hxa` gives `x ∈ b`; pass that to `hbc` to get `x ∈ c`. The command `lake env lean OntologyEmbeddings/Semantics.lean` checks the source file. `#check OntologyEmbeddings.Semantics.subsumes_trans` reports its type; `#print axioms ...` displays its dependencies.

### Editor / interactive goals

VS Code is available on this host, but the Lean 4 extension is **not installed**. No extension was installed because that would change the user's editor setup. To opt in, install the established `Lean 4` extension (`leanprover.lean4`) through VS Code's Extensions view, then launch VS Code from a shell carrying the environment above:

```bash
export ELAN_HOME="$HOME/.local/share/lean-elan-lean01"
export PATH="$ELAN_HOME/bin:$PATH"
code /home/midhun/Projects/ovgu/sem8/worktrees/ontology-embeddings-lean-01/formal
```

Open `OntologyEmbeddings/Semantics.lean`. Lean's language server checks the file as you edit. Place the cursor on a tactic line to view the current goal(s) in InfoView; hover a name for its type/documentation; errors appear inline and in Problems. `intro` exposes hypotheses, while `exact` must close the displayed goal. The editor/LSP is the established direct interactive proof-state interface; on this host it requires the user's opt-in extension installation.

Search before reinventing a lemma:

- In the editor, use `#check Set.mem_inter_iff` (or another candidate) and `#find` for search where the relevant search command is available.
- Try Mathlib's `exact?` or `apply?` in a proof; they suggest a term/tactic for the current goal. Confirm the suggestion against the displayed goal and then run the verifier.
- From a terminal, search the checked-out Mathlib sources, e.g. `rg 'mem_inter|inter.*iff' .lake/packages/mathlib/Mathlib/Data/Set`; then inspect a candidate with `#check` or hover. `grep`/`rg` searches source text, so it is a discovery aid, not proof evidence.
- Experiment in a separate scratch `.lean` file under `formal/` (or a temporary copy), and keep the committed mathematical statement unchanged until the author approves it. `lake env lean <scratch-file>` gives CLI diagnostics; run `./scripts/verify.sh` before treating a project change as verified.

## Agent-assisted proof loop (available today)

Pi has shell/file-editing tools, but inspection found no configured Lean MCP server or Pi Lean integration. A `lean-beam-mcp` elan shim was present in the temporary bootstrap `bin/`, but its pinned toolchain has no such executable; it is not a working agent integration. Do not rely on it. The usable agent loop is therefore the normal CLI: inspect the declaration and its assumptions; search Mathlib source / `#check` candidates; edit a proof; run `lake env lean ...` or the full verifier; read compiler diagnostics; iterate. Lean also provides `trace_state` inside a tactic proof to print the current goal in compiler output, so proof states can be exposed to a CLI-using agent without an MCP server.

Demonstrated scratch loop (the `trace_state` output is the live goal; the completed proof then compiles):

```lean
import OntologyEmbeddings.Semantics
open OntologyEmbeddings.Semantics

example {α : Type} (carrier a b c : Set α)
    (hab : Subsumes carrier a b) (hbc : Subsumes carrier b c) :
    Subsumes carrier a c := by
  intro x hx hxa
  trace_state
  exact hbc x hx (hab x hx hxa)
```

Save it as `formal/AgentLoop.lean`, then run `lake env lean AgentLoop.lean` from `formal/`. The agent sees the goal and any errors in the command output; delete the scratch file afterward. This demonstrates the current CLI path, not a direct Lean LSP session in Pi. No Pi configuration or additional agent framework is required.

## Research workflow and limits

The theorem ledger separates (1) author approval of the intended statement and domain, (2) formalization/proof construction, (3) independent review of assumptions, statement relevance, and dependencies, and (4) mechanical Lean verification. The kernel validates the proof relative to the formal statement and its assumptions; it does not determine whether the statement captures the thesis question. No LEAN-00 result has independent review signoff yet.
