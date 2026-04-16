# 📘 AMAS / amas-core / 3-estimators / neural-bdm-embedding-spec.md

## *AMAS-Admissible Algorithmic Complexity Estimator*

---

# 0. FREEZE DECLARATION

This module defines the **only admissible embedding of Neural BDM into AMAS systems**.

It is:

* strictly a function over projection space ( \phi(r) )
* non-invertible
* non-generative of structure
* non-interactive with AMAS-core invariants

It does NOT define:

* true Kolmogorov complexity
* semantic structure
* causal structure
* ground truth invariants

---

# 1. PURPOSE

Define a bounded estimator:

[
K_{\theta} : X \rightarrow \mathbb{R}
]

such that:

[
\Phi_{\text{NBDM}}(r) := K_{\theta}(\phi(r))
]

---

# 2. CORE CONSTRAINT: AMAS ADMISSIBILITY

## 2.1 Invariance Requirement

Neural BDM is admissible iff:

[
\forall T \in \mathcal{A}, \quad
K_{\theta}(\phi(T(r))) \approx K_{\theta}(\phi(r))
]

---

## Interpretation

> Neural BDM is not validated by accuracy.
> It is validated by invariance under admissible transformations.

---

## 2.2 Failure Condition

Neural BDM is INVALID if:

[
\exists T \in \mathcal{A} :
K_{\theta}(\phi(T(r))) \not\approx K_{\theta}(\phi(r))
]

---

# 3. STRUCTURAL BREAK FUNCTIONAL

Define:

[
\Delta K(r_t) = |K_{\theta}(\phi(r_t)) - K_{\theta}(\phi(r_{t-1}))|
]

---

## Structural Break Condition

A structural break exists iff:

[
\Delta K(r_t) > \epsilon
]

AND:

* invariance holds within regimes
* but fails across regimes

---

# 4. TRAINING CONSTRAINTS (CRITICAL)

---

## 4.1 Training Domain

Neural BDM MUST be trained on:

* rule-based systems
* computable dynamical systems
* synthetic generative processes

NOT on:

* labeled “structure” datasets
* human semantic labels
* task-specific classifiers

---

## 4.2 Loss Function Constraint

Training objective:

[
\mathcal{L} = \mathcal{L}*{compress} + \lambda \mathcal{L}*{predict}
]

BUT:

* no supervision of “structure”
* no classification loss
* no regime labels

---

## 4.3 No Semantic Anchoring Rule

Forbidden:

* “coordination labels”
* “regime shift labels”
* “phase classification”

Allowed only:

> compression and prediction errors over ( \phi(r) )

---

# 5. ARCHITECTURAL FORM

Neural BDM is defined as:

[
K_{\theta}(\phi(r)) = f_{\theta}(z)
]

where:

* ( z = \text{encoder}(\phi(r)) )
* ( f_{\theta} ) = bounded decoder or score network

---

## Constraint

* encoder is fixed after training
* no adaptive inference-time learning
* no feedback from observers

---

# 6. AMAS COMPATIBILITY LAYER

---

## 6.1 Projection Isolation Requirement

[
K_{\theta} \text{ operates ONLY on } \phi(r)
]

Forbidden:

* access to ( r )
* access to equivalence classes ( \sim )
* access to transformation set ( \mathcal{A} )

---

## 6.2 No Invariant Reconstruction Rule

Neural BDM MUST NOT attempt to reconstruct:

* AMAS equivalence classes
* transformation invariants
* latent state geometry

It only produces:

> scalar functional over representation

---

## 6.3 No Observer Coupling

[
K_{\theta} \not\rightarrow O_\alpha
]

No observer feedback loops allowed.

---

# 7. INTERPRETATION BOUNDARY

Neural BDM outputs:

[
\mathbb{R}
]

BUT:

> the value has no semantic meaning unless evaluated through AMAS predicates.

---

## Explicit Constraint

* high Kθ ≠ complexity “in reality”
* low Kθ ≠ simplicity “in reality”

Only:

> relative invariance under AMAS transformations matters

---

# 8. STRUCTURAL BREAK THEORY (FORMAL)

---

## 8.1 Definition

A structural break is:

[
\exists t :
K_{\theta}(\phi(r_t)) \not\approx K_{\theta}(\phi(r_{t-1}))
]

AND:

[
\exists T \in \mathcal{A} :
invariance holds within segment but not across
]

---

## 8.2 Interpretation Constraint

This does NOT imply:

* causal event
* regime change in reality
* semantic phase transition

It only implies:

> change in AMAS-invariant compressibility class

---

# 9. GENERALIZATION CLAIM (SAFE VERSION)

Neural BDM generalizes across systems iff:

[
K_{\theta}(\phi(T(r))) \approx K_{\theta}(\phi(r))
\quad \forall T \in \mathcal{A}
]

This replaces:

> “out-of-distribution generalization”

with:

> **invariance-class generalization**

---

# 10. FAILURE MODES (IMPORTANT FOR PUBLICATION)

Neural BDM is invalid if:

### ❌ sensitive to admissible transforms

### ❌ requires dataset-specific tuning for invariance

### ❌ encodes representation-specific artifacts

### ❌ collapses under encoding changes

### ❌ behaves differently across φ implementations

---

# 11. ROLE IN FULL AMAS STACK

| Layer                | Role                               |
| -------------------- | ---------------------------------- |
| AMAS-core            | defines invariants                 |
| projection-interface | defines representation             |
| Neural BDM           | estimates compressibility of φ(r)  |
| CIO observers        | compare projections                |
| experiments          | test invariance + break conditions |

---

# 12. FINAL STATEMENT

Neural BDM is:

> an invariance-constrained estimator of representation-level algorithmic compressibility

It is NOT:

* a complexity oracle
* a semantic classifier
* a causal detector

Its validity is determined exclusively by:

> stability under AMAS admissible transformations

---

# 🧭 WHAT YOU NOW HAVE (IMPORTANT)

You now have a fully frozen stack:

### 1. AMAS-core

→ defines equivalence classes

### 2. Projection interface

→ exposes representations

### 3. Neural BDM embedding

→ measures invariance of representation complexity

---

# 🚀 NEXT (CRITICAL FINAL LAYER)

Only one thing remains before publication readiness:

## 👉 CIO–Neural BDM Coupling Contract

This is where we define:

* how observers interface with Kθ
* how divergence is measured
* how structural breaks are validated across observer classes
* how NOT to leak AMAS structure into measurement

---
