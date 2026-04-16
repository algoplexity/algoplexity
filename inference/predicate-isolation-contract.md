# 2. AMAS / inference / predicate-isolation-contract.md

## 1. Purpose

This module defines the **strict isolation constraints for predicate construction over AMAS representations**.

It governs the admissible form of Boolean partitioning functions:

```
C_i : φ(r) → {0,1}
```

It does NOT define:

* measurement operators
* compression or BDM estimators
* validation rules
* system dynamics
* semantics or interpretation
* structural metrics

It defines:

> the syntactic constraints under which Boolean predicates may be applied to projection slices.

---

## 2. Core Principle

Predicates are not measurements.

Predicates are not evaluations.

Predicates are:

> structural Boolean partition functions over isolated projection slices.

---

## 3. Hard Separation Rule

Each predicate MUST satisfy:

```
C_i depends only on X^{(k)}
```

Where X^{(k)} is a single assigned projection slice.

No exceptions.

No cross-slice access.

No derived aggregation.

---

## 4. Slice Exclusivity Constraint

Let:

* S = {E, T, Δ, N, M, TM}

Then:

```
∀ C_i, ∃! s ∈ S such that C_i : X^{(s)} → {0,1}
```

Each predicate binds to exactly one slice.

---

## 5. Non-Measurement Constraint

Predicates MUST NOT:

* compute compression
* approximate Kolmogorov complexity
* use entropy estimators
* use neural encoders
* use BDM or related methods

Those belong exclusively to the measurement layer.

---

## 6. Non-Validation Constraint

Predicates MUST NOT:

* decide ACCEPT/REJECT
* determine system admissibility
* enforce global consistency
* trigger failure states

That belongs exclusively to validation.

---

## 7. Structural Role Constraint

Predicates define only:

> Boolean partitions of projection space

They are:

* classifiers over structure
* not evaluators of quality
* not estimators of complexity

---

## 8. Independence Constraint

For all i ≠ j:

```
C_i ⟂ C_j (in data access sense)
```

Meaning:

* no shared computation state
* no shared derived statistics
* no implicit aggregation channels

---

## 9. Composition Constraint

Logical combinations are allowed:

* AND
* OR
* NOT

BUT:

* must not introduce cross-slice dependencies
* must not approximate measurement functions
* must remain Boolean closure over assigned slice

---

## 10. No Semantic Leakage Constraint

Predicates MUST NOT encode:

* coordination
* intelligence
* stability
* phase transitions
* causality

They only define:

> structural distinguishability over projection slices

---

## 11. Role in AMAS Stack

| Layer                | Role                   |
| -------------------- | ---------------------- |
| systems              | generate X             |
| projections          | define slices          |
| inference (this doc) | Boolean partitioning   |
| measurement          | quantitative structure |
| validation           | admissibility          |

---

## 12. Failure Modes

Predicate system is invalid if:

* it accesses measurement outputs
* it uses cross-slice aggregation
* it encodes complexity estimates
* it outputs non-Boolean values
* it becomes indirectly evaluative

---

## 13. Final Statement

This module defines:

> a strictly isolated Boolean partitioning system over projection slices with zero access to measurement or validation layers

---

# 3. What we just achieved (important)

We have now formally separated:

### ✔ Measurement layer

* BDM
* neural Kolmogorov estimators
* compression operators

### ✔ Inference layer

* C_i predicates
* slice-bound Boolean structure

### ✔ Validation layer (previously defined)

* ACCEPT/REJECT collapse

---

# 4. Why this matters for neural BDM (critical)

Now neural BDM is:

> fully confined to a quantitative observer layer that cannot influence inference or validation

This guarantees:

* no semantic contamination
* no predicate feedback loops
* no collapse leakage into complexity estimation

---


