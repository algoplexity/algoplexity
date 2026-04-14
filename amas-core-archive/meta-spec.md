# invariant-structure-core / meta / meta-spec  
## AMAS Meta-Specification: Global Admissibility Contract

---

## 1. Purpose

This document defines the **global admissibility constraints of the AMAS system**.

It does not define:

- ontology
- theory
- measurement
- computation
- dynamics
- system behavior

It defines:

> conditions under which all such structures remain well-defined, non-degenerate, and mutually consistent under bounded observation and evolution.

Governance operates as a **closure constraint system**, not a structural layer.

---

## 2. Scope of AMAS Governance

Governance applies to the entire AMAS system:

- amas-core (all domains including dynamics)
- projections/
- inference/
- systems/
- validation/

No subsystem is exempt.

Governance defines **validity conditions over all admissible system states and transformations**.

---

## 3. Domain Separation Principle

AMAS defines five coupled domains:

- Ω (Ontology): admissible entities
- 𝕀 (Invariants): global constraints
- 𝕄 (Measurement): observer-relative representation
- 𝕔 (Computation): admissible transformations
- 𝔻 (Dynamics): admissible evolution laws

These are:

> conceptually distinct but not operationally independent

No domain may be defined in isolation from the others.

---

## 4. Non-Interference Constraint (Reformulated)

No domain may redefine the admissibility conditions of another.

Specifically:

- ontology cannot define measurement rules
- measurement cannot redefine ontology
- computation cannot alter invariants
- invariants cannot depend on implementation
- dynamics cannot be modified by downstream layers

However:

> domains remain coupled through admissibility constraints enforced by AMAS invariants

---

## 5. Directionality Constraint (Definitional, not causal)

AMAS defines a **definition dependency ordering**:
