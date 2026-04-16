# 1. AMAS / measurement / admissibility-contract.md

## 1. Purpose

This module defines the **admissible measurement operators over AMAS representations**.

It specifies how quantitative descriptors (including compression-based estimators such as BDM and neural approximations of Kolmogorov complexity) may be computed over:

* φ(r)
* trajectories X
* projection slices (X^{(·)})

It does NOT define:

* predicates {C_i}
* validation rules (ACCEPT/REJECT)
* system dynamics
* invariants or morphisms
* semantic interpretation of results

It defines:

> a constraint space for observer-relative quantitative functionals over AMAS artifacts.

---

## 2. Core Principle

Measurement is not evaluation.

Measurement is:

> a deterministic mapping from representation space to numerical description space.

No decision is produced.

No admissibility is inferred.

No semantics are assigned.

---

## 3. Measurement Domain

Let:

* X = φ(r)
* M be a measurement operator

Then:

```
M : X → ℝ^k
```

or

```
M : X → ℕ
```

or

```
M : X → ℝ^{k×t}
```

depending on estimator class.

All outputs are **non-Boolean**.

---

## 4. Allowed Measurement Classes

Only the following operator families are admissible:

### 4.1 Compression-Based Estimators

```
M_comp(X) = |compress(X)|
```

Constraints:

* compressor is fixed
* deterministic encoding required
* no adaptive dictionary updates across samples

---

### 4.2 Block Decomposition Estimators (BDM family)

```
M_BDM(X) = Σ_i K(block_i)
```

Constraints:

* fixed decomposition rule
* fixed estimator for block complexity
* no coupling to predicates {C_i}

---

### 4.3 Neural Complexity Estimators

```
M_NN(X) = f_θ(X)
```

Constraints:

* θ is frozen during evaluation
* training data is external to AMAS system
* no feedback from predicates or validation
* must operate only on φ(r), not intermediate slices unless explicitly declared

---

### 4.4 Temporal Measurement Operators

```
M_time(X_t) → sequence of descriptors
```

Constraints:

* no smoothing informed by future timesteps
* no adaptive hindsight correction

---

## 5. Slice Access Constraint

Measurement MAY access:

* φ(r)
* raw trajectory encodings
* projection slices (X^{(E)}, X^{(T)}, etc.)

BUT:

> MUST NOT access predicate outputs {C_i}

This is a hard boundary.

---

## 6. Non-Interference Constraint

Measurement outputs MUST NOT:

* determine admissibility
* modify system execution
* influence validation
* define structural partitions
* generate predicate thresholds

---

## 7. Non-Collapse Constraint

Measurement outputs are:

> irreducible quantitative descriptors

They are explicitly forbidden from being mapped to:

* {0,1}
* categorical labels
* semantic states

No collapse functions exist in this layer.

---

## 8. Observer Constraint

All measurement operators are:

* observer-relative
* projection-bound
* implementation-specific

BUT MUST:

* remain invariant across identical inputs
* remain deterministic under fixed parameters

---

## 9. Compositional Constraint

If M₁, M₂ are measurement operators:

```
M = M₁ ∘ M₂
```

is allowed only if:

* composition does not introduce predicate dependency
* intermediate outputs remain non-semantic
* no hidden classification emerges

---

## 10. Separation Guarantee

Measurement MUST be independent from:

* inference layer (C_i)
* validation layer
* system execution decisions

This ensures:

> complexity is measured, not interpreted

---

## 11. Role in AMAS Stack

| Layer       | Role                             |
| ----------- | -------------------------------- |
| systems     | generate X                       |
| projections | define φ(r)                      |
| measurement | quantify structure (this module) |
| inference   | Boolean partitioning             |
| validation  | admissibility collapse           |

---

## 12. Failure Modes

Measurement is invalid if:

* it produces Boolean outputs
* it depends on predicates
* it encodes semantic interpretation
* it influences validation outcomes
* it adapts based on system labels

---

## 13. Final Statement

This module defines:

> the admissible space of quantitative observation over AMAS artifacts without introducing semantics or decision logic

---

