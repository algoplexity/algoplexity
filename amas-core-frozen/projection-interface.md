# 📘 AMAS / amas-core / 2-projections / projection-interface.md

## *Frozen Measurement Interface over Invariant State Space*

---

# 0. FREEZE DECLARATION

This module defines the **only allowed interface between AMAS-core and all external systems**.

It is:

* strictly directional (AMAS-core → representation)
* non-invertible in general
* non-generative of structure
* forbidden from influencing invariants

Any violation implies:

> AMAS-core contamination and invalid admissibility reasoning.

---

# 1. PURPOSE

The projection layer defines:

[
\phi : \mathcal{R} \rightarrow X
]

where:

* ( \mathcal{R} ) = AMAS artifact space
* ( X ) = representation space used by observers and estimators

---

## Critical Constraint

> φ does NOT preserve structure.
> φ only preserves *accessibility of structure*.

---

# 2. PROJECTION IS NOT AN INVERSE OF INVARIANTS

## 2.1 Key Principle

[
\phi(r_1) = \phi(r_2) \not\Rightarrow r_1 \sim r_2
]

and equally:

[
r_1 \sim r_2 \not\Rightarrow \phi(r_1) = \phi(r_2)
]

---

## Interpretation

Projection is:

> lossy, observer-conditioned, and representation-dependent

BUT:

> it cannot redefine equivalence in AMAS-core

---

# 3. PROJECTION TYPES (ONLY VALID FOR DOWNSTREAM USE)

All φ must be composed of **fixed, declared transforms**.

---

## 3.1 Structural Projection

[
\phi_S(r) = {A_t}_{t=1}^{T}
]

Preserves:

* adjacency structure

Destroys:

* latent equivalence information

---

## 3.2 Temporal Flattening Projection

[
\phi_T(r) = \text{vec}(A_1, ..., A_T)
]

Preserves:

* ordering

Destroys:

* structural locality

---

## 3.3 Edge-Space Projection

[
\phi_E(r) = {A_t[i,j]}
]

Preserves:

* edge existence

Destroys:

* node-level structure

---

## 3.4 Motif Projection

[
\phi_M(r) = {(i,j,k)\text{-subgraphs}}
]

Preserves:

* local topology

Destroys:

* global invariants

---

## 3.5 Perturbation Projection

[
\phi_\delta(r) = \phi(r + \epsilon)
]

Used only for:

* stability testing
* sensitivity analysis

NOT for defining invariants.

---

# 4. PROJECTION CONSTRAINTS

---

## 4.1 Determinism Constraint

[
\phi(r) \text{ must be deterministic}
]

No stochastic projection allowed.

---

## 4.2 Non-Adaptive Constraint

[
\phi \neq \phi(r, \theta_t)
]

No dependency on:

* learning
* observers
* experiments

---

## 4.3 No Feedback Constraint

[
\phi(r) \text{ cannot depend on } O_i(\phi(r))
]

This prevents:

> observer-induced representation collapse

---

## 4.4 No Invariant Leakage Constraint

Projection MUST NOT encode:

* equivalence classes
* invariance labels
* structural identity assumptions

---

# 5. RELATION TO AMAS-CORE

---

## 5.1 Directionality

[
\mathcal{R} \rightarrow X
]

Not reversible.

---

## 5.2 No Back-Influence Principle

[
\phi(r) \not\rightarrow \sim
]

Meaning:

> projections cannot redefine AMAS equivalence

---

## 5.3 Consistency Requirement

If:

[
r_1 \sim r_2
]

then:

[
\phi(r_1), \phi(r_2)
]

may differ arbitrarily.

This is intentional.

---

# 6. RELATION TO OBSERVERS (CIO LAYER)

Observers are defined ONLY over projection space:

[
O_\alpha : X \rightarrow \mathbb{R}
]

---

## Important Constraint

Observers:

* cannot access ( r )
* cannot modify φ
* cannot influence AMAS-core

---

# 7. RELATION TO NEURAL BDM

Neural BDM operates only on:

[
\phi(r)
]

NOT on:

* equivalence classes
* AMAS-core invariants
* transformation definitions

---

## Critical Separation

| Layer      | Operates on |
| ---------- | ----------- |
| AMAS-core  | r, ∼        |
| Projection | φ(r)        |
| CIO        | O(φ(r))     |
| Neural BDM | K(φ(r))     |

No cross-layer mixing allowed.

---

# 8. PROJECTION INVARIANCE LIMIT (IMPORTANT RESULT)

There is NO projection φ such that:

[
\forall r_1 \sim r_2,\quad \phi(r_1)=\phi(r_2)
]

This is forbidden by design.

---

## Interpretation

> No representation fully captures invariance.

This ensures:

* irreducibility
* observer dependence
* No-Free-Resolution consistency

---

# 9. FAILURE MODES (WHAT WOULD BREAK THE SYSTEM)

Projection is INVALID if it:

### ❌ introduces equivalence encoding

### ❌ depends on observers

### ❌ adapts based on data

### ❌ reconstructs AMAS invariants

### ❌ becomes invertible

---

# 10. ROLE IN FULL SYSTEM STACK

| Layer                | Role                      |
| -------------------- | ------------------------- |
| AMAS-core            | defines invariants        |
| projection-interface | exposes structure         |
| CIO                  | interprets projections    |
| Neural BDM           | estimates complexity      |
| experiments          | test invariance stability |

---

# 11. FINAL STATEMENT

Projection is:

> a constrained lossy interface between invariant reality and representational systems

It is NOT:

* a feature extractor
* a learned embedding
* a sufficient statistic
* a semantic encoder

---

# 🧭 What this achieves (critical)

You now have:

### ✔ AMAS-core (ontology frozen)

### ✔ Projection interface (representation frozen)

### ✔ CIO (observer layer externalized)

### ✔ Neural BDM (estimator isolated)

This completes the **non-contamination triangle**:

> invariants → projection → observation → estimation

with strict one-way flow.

---


