# AMAS / observation-contract / v3 (UNIFIED)

## 1. Purpose

Defines admissible computation of all observables over AMAS systems.

It replaces:

* measurement contract
* estimator interpretation layer
* field-based observer interpretations

---

## 2. Core Principle

> Observables may extract information, but may not introduce structure.

---

## 3. Fundamental restriction (CRITICAL)

Any observable must satisfy:

[
O(r) = f(\phi(r))
]

AND:

> no structure may be assumed over output indices unless explicitly encoded in φ(r)

---

## 4. Structural prohibition (key upgrade)

The following are strictly forbidden unless explicitly encoded in φ:

* gradients
* manifolds
* continuity
* neighborhoods
* spatial adjacency
* field interpretations

This is the **closure axiom for measurement**.

---

## 5. Allowed structure primitives

Only these are permitted:

* finite index sets
* declared relations (R \subseteq (i,j))
* temporal indexing (explicit in φ)
* bounded difference operators

---

# 3. Revised predicate-isolation contract (CIO layer fix)

## Core change:

Predicates no longer operate on “slices”.

They operate on:

> **typed projections of φ(r) with declared relation bindings**

So instead of:

* X^(E), X^(T), etc.

you now have:

[
X_R = (\phi(r), R)
]

and:

[
C_i : X_R \rightarrow {0,1}
]

---

## Key upgrade:

Independence is no longer assumed.

It is enforced via:

> disjoint relation partitions, not disjoint data slices

This removes hidden coupling through shared index semantics.

---

# 4. Revised CIO–AMAS coupling contract (critical fix)

## 4.1 Neural BDM redefinition (final correct form)

Neural BDM is:

[
\mathcal{B}(r) = {b_i = g_i(\phi(r))}
]

BUT now:

* indices have no geometry
* no field interpretation exists

---

## 4.2 Structure is explicit, not emergent

Define:

[
R_{BDM} \subseteq (i,j)
]

Then:

[
\Delta_R(b) = b_i - b_j
]

No gradients.

No fields.

No topology.

Only declared relational structure.

---

## 4.3 CIO predicate coupling (final form)

[
C_i(r) = h(\mathcal{B}(r), \Delta_{R_{BDM}}(\mathcal{B}(r)))
]

This is the **maximum admissible expressive form under AMAS v2**.

---

# 5. What this unified revision achieves

You now eliminate three failure modes:

---

## (1) Hidden geometry collapse

No more implicit manifold assumptions.

---

## (2) Estimator reification

BDM cannot become a “thing in space”.

---

## (3) Predicate leakage

CIO cannot reconstruct structure beyond φ.

---

# 6. Deep structural insight (important)

What you are really doing is this:

> moving AMAS from “constraint system over states”
> to “constraint system over representational admissibility of structure itself”

That is why measurement keeps trying to become geometry — because classical ML always reintroduces structure through observables.

AMAS v2 now explicitly blocks that.

---

# 7. Direct answer to your question

Yes — all three must be revised.

But not separately.

They must be unified into:

> a single measurement-inference coupling layer with strict prohibition on emergent structure outside φ(r)

---

# 8. Final synthesis

You now have a clean 3-layer architecture:

### 1. AMAS-core

* invariants + dynamics

### 2. observation-contract v3 (NEW)

* measurement without structure emergence

### 3. CIO inference layer

* predicates over relationally declared observables

### 4. coupling contract

* neural BDM as bounded observer ensemble, not field

---
