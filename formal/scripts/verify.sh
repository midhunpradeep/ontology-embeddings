#!/usr/bin/env bash
# Single local/CI entry point: build project and audit dependencies of every
# named LEAN-00 theorem. Keep the theorem list in AxiomAudit.lean in sync.
set -euo pipefail

FORMAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FORMAL_DIR"

# This is a dedicated user-owned elan home, not global shell configuration.
# Existing system elan installations remain usable via LEAN_ELAN_HOME override.
if [[ -d "${LEAN_ELAN_HOME:-$HOME/.local/share/lean-elan-lean01}" ]]; then
  export ELAN_HOME="${LEAN_ELAN_HOME:-$HOME/.local/share/lean-elan-lean01}"
  export PATH="$ELAN_HOME/bin:$PATH"
fi

# Fail closed on admitted proofs or project-declared axioms, including unused ones.
source_violation=0
while IFS= read -r source; do
  if grep -nE '(^|[^[:alnum:]_])(sorry|admit|axiom)([^[:alnum:]_]|$)' "$source"; then
    source_violation=1
  fi
done < <(find OntologyEmbeddings -type f -name '*.lean' -print; echo AxiomAudit.lean)
if (( source_violation )); then
  echo 'ERROR: unfinished proof or project-declared axiom found in Lean sources.' >&2
  exit 1
fi

lake build

audit_output="$(mktemp)"
trap 'rm -f "$audit_output"' EXIT
lake env lean AxiomAudit.lean >"$audit_output" 2>&1 || {
  cat "$audit_output" >&2
  exit 1
}
cat "$audit_output"
python3 - "$audit_output" <<'PY'
import re
import sys
from pathlib import Path

output = Path(sys.argv[1]).read_text()
expected = {
    "subsumes_refl",
    "subsumes_trans",
    "equivalent_iff_mutually_subsumes",
    "disjoint_symm",
    "disjoint_iff_conjunction_equivalent_bottom",
    "complement_member_iff_not_member",
    "concept_excluded_middle",
    "concept_noncontradiction",
    "example_positive_membership",
    "example_negative_membership",
    "example_unlabeled_point_can_be_a_member",
}
allowed = {"propext", "Classical.choice", "Quot.sound"}
source_theorems = set()
for source in Path("OntologyEmbeddings").rglob("*.lean"):
    source_theorems.update(re.findall(r"(?m)^\s*theorem\s+([A-Za-z0-9_']+)", source.read_text()))
if source_theorems != expected:
    print(
        f"ERROR: every project theorem must be audited; "
        f"missing audit entries={sorted(source_theorems - expected)}, "
        f"stale entries={sorted(expected - source_theorems)}",
        file=sys.stderr,
    )
    raise SystemExit(1)
records = {}
pattern = re.compile(
    r"'OntologyEmbeddings\.Semantics\.([^']+)' "
    r"(does not depend on any axioms|depends on axioms: \[([^]]*)\])"
)
for match in pattern.finditer(output):
    name, status, dependency_text = match.groups()
    dependencies = set() if status.startswith("does not") else {
        item.strip() for item in dependency_text.split(",") if item.strip()
    }
    records[name] = dependencies
missing = expected - records.keys()
extra = records.keys() - expected
if missing or extra:
    print(f"ERROR: audit inventory mismatch; missing={sorted(missing)}, extra={sorted(extra)}", file=sys.stderr)
    raise SystemExit(1)
failed = {name: sorted(deps - allowed) for name, deps in records.items() if deps - allowed}
if failed:
    print(f"ERROR: unauthorized theorem dependencies: {failed}", file=sys.stderr)
    raise SystemExit(1)
print(f"Axiom audit passed: {len(records)} project theorems; allowed Lean foundations only: {sorted(allowed)}")
PY
