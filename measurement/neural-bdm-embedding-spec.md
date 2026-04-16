# AMAS / measurement / neural-bdm-embedding-spec.md

---

## 1. Purpose

This document defines the **formal admissible embedding of neural Block Decomposition Method (neural BDM) estimators within AMAS measurement constraints**.

It specifies how neural approximations of Kolmogorov complexity may be constructed, applied, and interpreted over AMAS representations without violating:

* invariants (1-invariants)
* dynamics (2-dynamics)
* morphism constraints
* inference isolation
* validation independence

It does NOT define:

* predicates {C_i}
* system execution logic
* admissibility decisions
* semantic interpretation of complexity
* causal or cognitive claims

It defines:

> a constraint-preserving embedding of learned complexity estimators over φ(r) in AMAS measurement space.

---

# 2. Core Principle

Neural BDM is not a classifier.

Neural BDM is not a validator.

Neural BDM is:

> a learned approximation of description length over structured decompositions of φ(r)

It produces:

```id="nbdm_01"
K̂(X) ∈ ℝ
```

as an estimator of:

```id="k_true"
K(X)
```

(up to observer-bound approximation limits)

---

# 3. Domain Binding Constraint

Neural BDM operates ONLY on:

```id="nbdm_domain"
X = φ(r)
```

It MAY optionally access:

* projection slices X^{(·)}

BUT ONLY as deterministic re-encodings of φ(r)

It MUST NOT access:

* predicates {C_i}
* validation outputs
* system execution states beyond φ(r)
* meta-spec rules
* downstream interpretations

---

# 4. Structural Definition of Neural BDM

Neural BDM is defined as:

```id="nbdm_def"
K̂(X) = f_θ( Decompose(X) )
```

Where:

### 4.1 Decomposition operator

```id="nbdm_dec"
Decompose(X) = {B₁, B₂, ..., B_n}
```

Constraints:

* fixed deterministic decomposition rule
* identical across evaluations
* independent of learned parameters

---

### 4.2 Neural estimator

```id="nbdm_net"
f_θ : {B_i} → ℝ
```

Constraints:

* θ is frozen during evaluation
* no online learning
* no feedback from predicates or validation
* no adaptation based on dataset structure during inference

---

# 5. AMAS Admissibility Constraints

Neural BDM is AMAS-admissible only if all hold:

---

## 5.1 Projection-only constraint

```id="nbdm_c1"
input(f_θ) ⊆ φ(r)
```

No external system state is allowed.

---

## 5.2 Non-decision constraint

```id="nbdm_c2"
K̂(X) ∉ {0,1}
```

Neural BDM MUST NOT output Boolean values.

It is explicitly forbidden from:

* classification
* admissibility decisions
* thresholding for validity

---

## 5.3 Non-predicate coupling constraint

```id="nbdm_c3"
∀ C_i: K̂(X) ⟂ C_i
```

Meaning:

* no dependency on inference layer
* no co-training with predicates
* no indirect feature reuse from C_i outputs

---

## 5.4 Non-validation coupling constraint

```id="nbdm_c4"
K̂(X) ⟂ V(X)
```

Where V is validation.

Neural BDM MUST NOT:

* influence ACCEPT/REJECT decisions
* be tuned using validation labels
* be calibrated using system-level admissibility outcomes

---

## 5.5 Fixed decomposition constraint

Decomposition must satisfy:

```id="nbdm_c5"
∀ X: Decompose(X) is invariant under observer run
```

No stochastic segmentation.

No adaptive tiling.

No learned partitioning.

---

## 5.6 Compression consistency constraint

Neural estimate MUST respect monotonicity under refinement:

If X → X' is a lossless refinement:

```id="nbdm_c6"
K̂(X') ≥ K̂(X)
```

(Up to bounded estimator variance)

---

# 6. Measurement Role Definition

Neural BDM defines:

> an observer-relative estimate of algorithmic description length over AMAS artifacts

It produces:

* scalar complexity estimates
* optional vectorized block contributions

It does NOT produce:

* structure labels
* regime classifications
* causal interpretations

---

# 7. Allowed Outputs

Neural BDM outputs MAY include:

### 7.1 Global estimate

```id="nbdm_out1"
K̂(X)
```

---

### 7.2 Block-level contributions

```id="nbdm_out2"
{K̂(B₁), K̂(B₂), ..., K̂(B_n)}
```

---

### 7.3 Temporal complexity trace

```id="nbdm_out3"
K̂(X_t)
```

---

BUT ALL OUTPUTS MUST REMAIN:

* real-valued or vector-valued
* non-Boolean
* non-decisional

---

# 8. Forbidden Capabilities

Neural BDM MUST NOT:

* classify coordination regimes
* detect “intelligence”
* infer phase transitions
* determine admissibility
* modify AMAS-core constraints
* generate predicates

---

# 9. Separation Guarantees

Neural BDM is isolated from:

| Layer      | Constraint              |
| ---------- | ----------------------- |
| invariants | cannot access           |
| dynamics   | cannot observe directly |
| morphisms  | cannot condition on     |
| predicates | strictly independent    |
| validation | no feedback loop        |

---

# 10. Composition Rule

Neural BDM MAY be composed with other measurement operators:

```id="nbdm_comp"
M_total = g(K̂₁, K̂₂, ...)
```

ONLY IF:

* no predicate leakage occurs
* no thresholding into Boolean space
* composition remains purely numerical

---

# 11. Interpretation Boundary

This module explicitly forbids:

> interpreting K̂(X) as intelligence, coordination, or system quality

Any such interpretation belongs ONLY to:

* meta-analysis external to AMAS
* hypothesis frameworks (not core system)

---

# 12. Relationship to AMAS Stack

| Layer                     | Role                  |
| ------------------------- | --------------------- |
| systems                   | generate trajectories |
| projections               | define φ(r)           |
| measurement (this module) | estimate K̂(X)        |
| inference                 | define C_i            |
| validation                | ACCEPT/REJECT         |

---

# 13. Failure Modes

Neural BDM is invalid if:

* it is trained using validation outputs
* it uses predicate outputs as features
* it produces Boolean decisions
* it adapts decomposition based on downstream tasks
* it encodes semantic labels

---

# 14. Final Statement

Neural BDM defines:

> an AMAS-admissible, observer-bound approximation of Kolmogorov complexity over structured system trajectories, strictly isolated from inference and validation layers

---

# 15. Structural consequence (important)

With this module in place:

* CIO generates trajectories (systems)
* AMAS defines structure (invariants/dynamics/morphisms)
* predicates define Boolean partitions
* neural BDM defines complexity geometry
* validation decides admissibility

---

# 16. Key achievement

You now have:

> a fully stratified algorithmic information pipeline inside a constraint system where no layer can “cheat upward” into semantics

---


