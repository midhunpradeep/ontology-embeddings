# Formal semantics bootstrap

This standalone Lean project is a small mathematical starting point for reasoning about ontology concepts. It uses Mathlib's `Set α` directly. It is deliberately independent of coordinates, dimensions, classifiers, and geometric representations.

## Model

In `OntologyEmbeddings/Semantics.lean`, a concept is a set of points `α`, and a carrier is the domain of points under consideration. `Member carrier concept x` means both that `x` is in the carrier and in the concept. Subsumption is pointwise inclusion restricted to the carrier; equivalence means the same membership answers there. Disjointness means no carrier point belongs to both concepts.

Conjunction and disjunction are Mathlib set intersection and union, restricted to the carrier. Complement is `carrier \ concept`; top is the carrier and bottom is the empty set. Thus these operators have the intended meanings even when a set contains points outside the chosen carrier. No assumption says the carrier is nonempty.

`PositivelyAnnotated` is a separate predicate for available positive labels, not a definition of concept membership. The finite `Bool` example proves that `true` is a member even though it has no positive annotation. Missing supervision therefore is not treated as logical negation.

## Checked results

The module proves:

- subsumption is reflexive and transitive;
- equivalence holds exactly when subsumption holds in both directions;
- disjointness is symmetric and holds exactly when the carrier-restricted conjunction is equivalent to bottom;
- within the carrier, complement membership is equivalent to not being a member;
- concept-level excluded middle and noncontradiction;
- concrete positive membership (`true`), negative membership (`false`), and an unlabeled point that remains a member.

The proofs use no project-defined axioms or admitted/unfinished proofs. `#print axioms` is included in the source and run during compilation. Constructive results report no axioms where Lean can prove them constructively. Extensional set/equivalence results use Lean's standard `propext`; classical excluded middle additionally uses standard `Classical.choice` and `Quot.sound`. These are Lean's built-in logical foundations, not assumptions introduced by this project. The explicit `classical` command is confined to `concept_excluded_middle`.

## Toolchain and build

The pinned compiler is Lean `4.27.0` (`leanprover/lean4:v4.27.0`). Mathlib is pinned to tag `v4.27.0`; `lake-manifest.json` records the resolved Mathlib commit (`a3a10db0e9d66acbebf76c5e6a135066525ac900`) and exact commits of its transitive dependencies.

With elan installed and network access for dependencies and cache artifacts:

```sh
cd formal
lake exe cache get
lake build
```

`lake build` compiles the project (including the semantic module); its axiom audit is printed as part of compilation. To check the module directly instead:

```sh
lake env lean OntologyEmbeddings/Semantics.lean
```

The `.lake/` directory contains downloaded packages, compiled artifacts, and caches and is excluded from Git. If provisioning the toolchain manually, install it with `elan toolchain install leanprover/lean4:v4.27.0` before running the commands above.

## Extending this foundation

A subsequent change can add small representation-independent definitions or theorems in `OntologyEmbeddings/Semantics.lean`, then rebuild with `lake build`. Keep semantic membership distinct from observed labels. State clearly whether a theorem is carrier-relative and use Mathlib's set operations and lemmas before introducing new abstractions.

## Limits

This is an abstract set-semantics foundation, not a model of boxes, affine regions, TOWN geometry, relations, full description logic, or dataset/training behavior. It does not establish that any particular embedding implementation satisfies these semantics. Mathlib setup is standalone under `formal/`; no repository CI, Python package, or global Lean configuration was changed.
