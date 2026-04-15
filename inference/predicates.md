# AMAS Predicate Basis Specification

**Domain:** inference
**Version:** v1.0 (Orthogonal / Audit-Compliant)

---

# 1. Purpose

This document defines the **canonical predicate basis**:

[
{C_i}
]

used to partition the artifact space:

[
X = {A_{t-W}, \dots, A_t}
]

Predicates:

* operate on projections of (X)
* produce Boolean outputs
* introduce **no semantic interpretation**
* satisfy AMAS audit constraints

---

# 2. Domain Positioning

## 2.1 Primary Domain

`inference`

Predicates are Boolean-valued functionals:

[
C_i : \phi(r) \rightarrow {0,1}
]

---

## 2.2 Cross-Domain Dependencies

### From `projections`

Defines input slices:

[
X \rightarrow {X^{(E)}, X^{(T)}, X^{(\Delta)}, X^{(N)}, X^{(M)}, X^{(TM)}}
]

---

### From `audit-spec`

Enforces:

* orthogonality
* independence
* non-degeneracy

---

### From `validation`

Predicates are later composed into testable hypotheses.

---

# 3. Projection Slice Binding (MANDATORY)

Each predicate is bound to exactly one projection slice.

| Slice          | Symbol         | Description                   |
| -------------- | -------------- | ----------------------------- |
| Edge-local     | (X^{(E)})      | Raw edge values               |
| Transition     | (X^{(T)})      | Pairwise temporal transitions |
| Difference     | (X^{(\Delta)}) | XOR change signal             |
| Node           | (X^{(N)})      | Node adjacency vectors        |
| Motif          | (X^{(M)})      | Local 3-node subgraphs        |
| Temporal Motif | (X^{(TM)})     | Motif transitions             |

---

# 4. Hard Separation Rule

> A predicate MUST NOT access data outside its assigned slice.

This guarantees:

* no shared sufficient statistics
* no hidden coupling
* audit admissibility

---

# 5. Canonical Predicate Set

---

## C₁ — Edge Persistence

**Domain:** (X^{(E)})

[
C_1 = 1 \iff \exists (i,j,t): A_t[i,j] = A_{t+1}[i,j]
]

**Interpretation (informal):**
Some connections persist over time.

---

## C₂ — Transition Non-Degeneracy

**Domain:** (X^{(T)})

[
C_2 = 1 \iff \exists (i,j,t): (A_t, A_{t+1}) \in {(0,1),(1,0)}
]

**Interpretation:**
At least one edge changes state.

---

## C₃ — Change Signal Presence

**Domain:** (X^{(\Delta)})

[
C_3 = 1 \iff \exists (i,j,t): A_{t+1}[i,j] \oplus A_t[i,j] = 1
]

**Interpretation:**
System exhibits detectable change signal.

---

## C₄ — Node Structural Distinction

**Domain:** (X^{(N)})

[
C_4 = 1 \iff \exists i \neq j, t: A_t[i,:] \neq A_t[j,:]
]

**Interpretation:**
Nodes are not structurally identical.

---

## C₅ — Motif Diversity

**Domain:** (X^{(M)})

[
C_5 = 1 \iff \exists (i,j,k,t): \text{motif types differ}
]

**Interpretation:**
Multiple local subgraph patterns exist.

---

## C₆ — Motif Temporal Instability

**Domain:** (X^{(TM)})

[
C_6 = 1 \iff \exists (i,j,k,t): M_t \neq M_{t+1}
]

**Interpretation:**
Local structures evolve over time.

---

# 6. Admissibility Definition

[
A({C_i}) = \bigwedge_{i=1}^{6} C_i
]

An artifact is admissible iff all predicates evaluate to 1.

---

# 7. Independence Guarantee

For all ( i \neq j ):

[
X / C_i \neq X / C_j
]

This holds because:

* each predicate operates on a non-isomorphic projection
* no shared aggregation exists
* no statistic is reused

---

# 8. Role in AMAS Pipeline

| Layer                    | Role                  |
| ------------------------ | --------------------- |
| systems                  | generate (X)          |
| projections              | define slices         |
| **inference (this doc)** | define predicates     |
| audit-spec               | validate independence |
| validation               | test hypotheses       |

---

# 9. Non-Interpretation Constraint

Predicates MUST NOT:

* define coordination
* define stability
* assign labels
* encode objectives

They only:

> partition the artifact space

---

# 10. Regime Construction (Deferred)

Higher-level structures (e.g. “coordination regimes”) are defined as:

[
R_k = f(C_1, \dots, C_6)
]

These belong to:

`validation` or `meta-spec`

---

# 11. Compliance Summary

| Requirement        | Status |
| ------------------ | ------ |
| Domain separation  | ✅      |
| Projection binding | ✅      |
| Orthogonality      | ✅      |
| No leakage         | ✅      |
| Audit-ready        | ✅      |

---

# 12. Final Statement

This predicate basis constitutes:

> a minimal, orthogonal, AMAS-admissible partition of the artifact space

It introduces **no semantics**, only structure.

---
