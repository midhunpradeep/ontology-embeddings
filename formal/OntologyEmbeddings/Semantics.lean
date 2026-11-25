import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Insert

/-!
Representation-independent semantics for concepts over an abstract carrier.
Concepts are Mathlib sets; all interpretation is restricted to a given carrier.
-/
namespace OntologyEmbeddings.Semantics

variable {α : Type*}

abbrev Concept (α : Type*) := Set α

def Member (carrier concept : Set α) (x : α) : Prop :=
  x ∈ carrier ∧ x ∈ concept

/- Positive labels are recorded separately; lack of a label is not negation. -/
def PositivelyAnnotated (annotations : Set α) (x : α) : Prop :=
  x ∈ annotations

def Subsumes (carrier a b : Set α) : Prop :=
  ∀ x, x ∈ carrier → x ∈ a → x ∈ b

def Equivalent (carrier a b : Set α) : Prop :=
  ∀ x, x ∈ carrier → (x ∈ a ↔ x ∈ b)

def Disjoint (carrier a b : Set α) : Prop :=
  ∀ x, x ∈ carrier → x ∈ a → x ∉ b

def Conjunction (carrier a b : Set α) : Set α := carrier ∩ a ∩ b

def Disjunction (carrier a b : Set α) : Set α := carrier ∩ (a ∪ b)

def Complement (carrier a : Set α) : Set α := carrier \ a

def Top (carrier : Set α) : Set α := carrier

def Bottom (_carrier : Set α) : Set α := ∅

theorem subsumes_refl (carrier a : Set α) : Subsumes carrier a a := by
  intro x _ hx
  exact hx

theorem subsumes_trans (carrier a b c : Set α)
    (hab : Subsumes carrier a b) (hbc : Subsumes carrier b c) :
    Subsumes carrier a c := by
  intro x hx hxa
  exact hbc x hx (hab x hx hxa)

theorem equivalent_iff_mutually_subsumes (carrier a b : Set α) :
    Equivalent carrier a b ↔ Subsumes carrier a b ∧ Subsumes carrier b a := by
  constructor
  · intro h
    constructor
    · intro x hx hxa
      exact (h x hx).mp hxa
    · intro x hx hxb
      exact (h x hx).mpr hxb
  · rintro ⟨hab, hba⟩ x hx
    exact ⟨fun hxa => hab x hx hxa, fun hxb => hba x hx hxb⟩

theorem disjoint_symm (carrier a b : Set α)
    (h : Disjoint carrier a b) : Disjoint carrier b a := by
  intro x hx hxb hxa
  exact (h x hx hxa) hxb

theorem disjoint_iff_conjunction_equivalent_bottom (carrier a b : Set α) :
    Disjoint carrier a b ↔
      Equivalent carrier (Conjunction carrier a b) (Bottom carrier) := by
  constructor
  · intro h x hx
    constructor
    · intro hxab
      exact False.elim (h x hx hxab.1.2 hxab.2)
    · intro hbot
      simp [Bottom] at hbot
  · intro h x hx hxa hxb
    have hconj : x ∈ Conjunction carrier a b := by
      simp [Conjunction, hx, hxa, hxb]
    have hbottom : x ∈ Bottom carrier := (h x hx).mp hconj
    simp [Bottom] at hbottom

theorem complement_member_iff_not_member (carrier a : Set α) (x : α)
    (hx : x ∈ carrier) :
    Member carrier (Complement carrier a) x ↔ ¬ Member carrier a x := by
  simp [Member, Complement, hx]

/- The classical principle is isolated here; the preceding laws are constructive. -/
theorem concept_excluded_middle (carrier a : Set α) (x : α) :
    Member carrier a x ∨ ¬ Member carrier a x := by
  classical
  exact em (Member carrier a x)

theorem concept_noncontradiction (carrier a : Set α) (x : α) :
    ¬ (Member carrier a x ∧ ¬ Member carrier a x) := by
  intro h
  exact h.2 h.1

/- A finite, concrete carrier: true belongs; false does not. -/
def exampleCarrier : Set Bool := {true, false}
def exampleConcept : Set Bool := {true}

theorem example_positive_membership :
    Member exampleCarrier exampleConcept true := by
  simp [Member, exampleCarrier, exampleConcept]

theorem example_negative_membership :
    ¬ Member exampleCarrier exampleConcept false := by
  simp [Member, exampleCarrier, exampleConcept]

def examplePositiveAnnotations : Set Bool := {false}

theorem example_unlabeled_point_can_be_a_member :
    ¬ PositivelyAnnotated examplePositiveAnnotations true ∧
      Member exampleCarrier exampleConcept true := by
  simp [PositivelyAnnotated, examplePositiveAnnotations, Member,
    exampleCarrier, exampleConcept]

#print axioms subsumes_refl
#print axioms subsumes_trans
#print axioms equivalent_iff_mutually_subsumes
#print axioms disjoint_symm
#print axioms disjoint_iff_conjunction_equivalent_bottom
#print axioms complement_member_iff_not_member
#print axioms concept_excluded_middle
#print axioms concept_noncontradiction
#print axioms example_positive_membership
#print axioms example_negative_membership
#print axioms example_unlabeled_point_can_be_a_member

end OntologyEmbeddings.Semantics
