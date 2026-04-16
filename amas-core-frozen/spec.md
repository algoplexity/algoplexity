# 📘 AMAS-core (FROZEN SPEC v1.0)

## *Invariant Substrate for Admissibility-Constrained Systems*

---

# 0. FREEZE DECLARATION

This document defines the **immutable core of AMAS**.

After this point:

> No additional invariants, observers, or projection rules may modify the meaning of AMAS-core without versioned replacement.

AMAS-core is:

* not extensible inline
* not semantically re-interpretable
* not dependent on downstream systems (CIO, Neural BDM, etc.)

---

# 1. ONTOLOGICAL PRIMITIVES

## 1.1 Artifact Space

[
r \in \mathcal{R}
]

where:

[
r = {A_t}_{t=1}^{T}, \quad A_t \in {0,1}^{N \times N}
]

**Constraint:**

* finite
* fully observable
* no latent state

---

## 1.2 State Equivalence Base

AMAS does NOT define structure.

It defines:

[
\sim
]

an equivalence relation over artifacts.

---

# 2. INVARIANT DEFINITION (CORE AXIOM)

## 2.1 Primary Axiom

An invariant is:

[
I := \mathcal{R} / \sim
]

where:

* ( \sim ) is induced solely by admissible transformations
* no observer enters this definition
* no measurement enters this definition

---

## 2.2 Meaning Constraint

> Invariants are not properties.
> Invariants are equivalence classes under constrained transformation.

---

# 3. ADMISSIBLE TRANSFORMATIONS (FINAL FORM)

## 3.1 Definition

A transformation:

[
T : \mathcal{R} \rightarrow \mathcal{R}
]

is admissible iff:

[
T \in \mathcal{A}
]

---

## 3.2 Closure Conditions

The admissible set 𝒜 is:

* closed under composition
* identity-preserving
* structure-preserving (defined below)

---

## 3.3 Structure-Preservation Constraint

A transformation is admissible only if:

[
r_1 \sim r_2 \Rightarrow T(r_1) \sim T(r_2)
]

This is the **core invariance requirement**.

---

## 3.4 Forbidden Transformations

Explicitly disallowed:

* arbitrary relabeling that changes equivalence classes
* transformations that collapse distinguishable invariant classes
* transformations that introduce non-reversible structure

---

# 4. EQUIVALENCE RELATION (INDUCED, NOT DEFINED)

## 4.1 Definition by Induction

[
a \sim b \iff \forall T \in \mathcal{A}, ; T(a) \equiv T(b)
]

Interpretation:

> Two artifacts are equivalent if no admissible transformation can distinguish them.

---

## 4.2 Key Consequence

AMAS equivalence is:

* operational, not semantic
* transformation-based, not feature-based
* observer-free

---

# 5. INVARIANT CLOSURE PRINCIPLE

## 5.1 Closure Condition

The system is valid only if:

[
T(r) \in [r]_\sim \quad \forall T \in \mathcal{A}
]

Meaning:

> admissible transformations never leave an equivalence class

---

## 5.2 No-Collapse Rule

If:

[
r_1 \not\sim r_2
]

then:

[
T(r_1) \not\sim T(r_2)
]

for all admissible T.

---

## 5.3 No-Splitting Rule

If:

[
r_1 \sim r_2
]

then:

[
T(r_1) \sim T(r_2)
]

---

# 6. REFINEMENT IS THE ONLY CONTROLLED OPERATION

## 6.1 Definition

Refinement is:

> partitioning equivalence classes without violating admissibility closure.

Formally:

[
[r]_\sim \rightarrow {[r]_1, [r]_2, \dots}
]

---

## 6.2 Constraint

Refinement must preserve:

* closure consistency
* transformation invariance
* equivalence stability under 𝒜

---

# 7. OBSERVER EXCLUSION PRINCIPLE

## 7.1 Hard Constraint

Observers DO NOT exist in AMAS-core.

No mapping:

[
O : \mathcal{R} \rightarrow \mathbb{R}
]

is part of AMAS-core.

---

## 7.2 Consequence

All measurement systems are:

> external projections, not ontological primitives

This includes:

* CIO observers
* Neural BDM
* entropy measures
* Markov models

---

# 8. PROJECTION EXCLUSION PRINCIPLE

## 8.1 Hard Constraint

No projection function:

[
\phi(r)
]

is defined in AMAS-core.

Reason:

> projection introduces representation bias and is strictly external.

---

## 8.2 Implication

* AMAS-core is representation-free
* all φ live in downstream systems only

---

# 9. INVARIANCE GUARANTEE (CORE RESULT)

If T ∈ 𝒜, then:

[
[r]*\sim = [T(r)]*\sim
]

This is the only guaranteed property.

---

# 10. SYSTEM BOUNDARY DEFINITION

AMAS-core defines ONLY:

### ✔ allowed:

* equivalence classes
* admissible transformations
* invariance conditions

---

### ❌ forbidden:

* measurement
* prediction
* projection
* interpretation
* observer models
* complexity metrics

---

# 11. RELATION TO DOWNSTREAM SYSTEMS

| System      | Role relative to AMAS-core                    |
| ----------- | --------------------------------------------- |
| CIO         | defines observers over projections (external) |
| Neural BDM  | estimator over φ(r) (external)                |
| Experiments | probes equivalence stability                  |
| UI          | visualization only                            |

---

# 12. FREEZE RULE (MOST IMPORTANT SECTION)

> AMAS-core MUST NOT be modified by:

* experimental results
* observer definitions
* estimator design
* implementation constraints

Only replacement via:

[
\text{AMAS-core}^{(v+1)}
]

is permitted.

---

# 13. FINAL STATEMENT

AMAS-core defines:

> the invariant equivalence structure of admissible transformation systems

It does not define:

* what is observed
* how structure is measured
* what is computed
* what is predicted

It defines only:

> what cannot be distinguished under admissible change

---

# 🧭 What this accomplishes (important)

You now have:

### ✔ A frozen ontological substrate

Everything else must respect it.

### ✔ Clean separation from CIO and Neural BDM

No contamination risk if discipline is maintained.

### ✔ A falsifiable foundation

Because now only downstream systems can fail — not the substrate.

---


