## validation/admissibility-contract.md

This module defines the **only thing that is allowed to interpret trajectories**.

It does not simulate.
It does not generate.
It does not adapt.

It enforces a fixed, external collapse from structure → boolean.

---

# ADMISSIBILITY CONTRACT (VALIDATION LAYER)

## 1. Purpose

Given a trajectory tensor:

```id="v1"
X ∈ ℝ^{T × N × D}
```

this module defines a deterministic mapping:

```id="v2"
f: X → {ACCEPT, REJECT}
```

No intermediate semantics are allowed.

No scoring is stored.

No feedback is returned to the system.

---

## 2. Core Principle

Validation is:

> a static filter over representations, not a process over systems.

It does not observe dynamics.

It only evaluates **encoded traces of dynamics**.

---

## 3. Representation Encoding (Fixed Observer Projection)

Before any evaluation, trajectories are converted into a fixed discrete representation:

### 3.1 Quantization

```id="v3"
X_q = round(X / δ)
```

Where:

* δ = fixed resolution constant
* no adaptive scaling allowed

---

### 3.2 Flattening

```id="v4"
B = bytes(X_q)
```

All structure is reduced to a byte stream.

This is the **only allowed representation interface**.

---

## 4. Admissibility Predicate (Core Constraint)

A trajectory is admissible if and only if:

### 4.1 Activity Constraint

System is not degenerate:

```id="v5"
Var(mean_i(X[:, i, :])) > θ₁
```

Ensures non-trivial dynamics.

---

### 4.2 Structural Compressibility Constraint

Define:

```id="v6"
K_actual = len(compress(B))
K_raw = len(B)
```

Admissibility condition:

```id="v7"
K_actual / K_raw < θ₂
```

Interpretation:

* low compressibility ratio = structured coherence
* high entropy noise fails

---

## 5. Decision Rule (Hard Collapse)

```id="v8"
if (activity_pass AND compressibility_pass):
    return ACCEPT
else:
    return REJECT
```

No partial credit exists.

No probabilities exist.

No ranking exists.

---

## 6. Closure Constraint (Non-Negotiable)

The validator MUST NOT:

* modify input trajectories
* feed results back into generators
* store running statistics during execution
* adjust thresholds dynamically
* infer causes of ACCEPT/REJECT

This ensures:

> validation is structurally inert

---

## 7. Experimental Invariance Rule

Across all experiments:

* δ is fixed
* θ₁ is fixed
* θ₂ is fixed
* compression algorithm is fixed
* encoding scheme is fixed

Only generators may vary.

---

## 8. Interpretation Boundary

The validator does NOT detect:

* intelligence
* coordination
* learning
* agency

It only detects:

> compressible structure under a fixed observational encoding

All interpretation is external.

---

## 9. Output Schema

Each evaluation produces:

```id="v9"
{
  "experiment_id": int,
  "generator_type": str,
  "accept_count": int,
  "reject_count": int,
  "accept_rate": float
}
```

This is the only permitted measurement output.

---

## 10. Scientific Meaning (External Only)

Any claims such as:

* “collective intelligence”
* “coordination”
* “phase transition”
* “robust structure”

are **not produced by this module**.

They are derived only from comparing:

> ACCEPT rate distributions across generator families

---

