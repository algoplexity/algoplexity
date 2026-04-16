## 📄 AMAS / amas-core / observability / projection-interface.md

### (This is the missing missing piece)

---

## 1. Purpose

This module defines:

> the allowed interface between AMAS-core structures and external observers.

It does NOT define:

* observers
* estimators
* metrics
* inference
* learning

It defines:

> what can be *legally extracted* from AMAS states without becoming part of AMAS.

---

## 2. Core Principle (Separation Axiom)

Let:

* S ∈ AMAS state space
* φ(S) = projection into external space

Then:

> φ is NOT part of AMAS
> φ is an *epistemic interface only*

---

## 3. Projection Irreducibility Constraint (NEW)

A valid projection φ must satisfy:

### 3.1 Non-injectivity constraint

There must exist:

```
S1 ≠ S2  such that  φ(S1) = φ(S2)
```

Meaning:

> projections cannot preserve full invariant identity

---

### 3.2 Non-reconstructability constraint

There must NOT exist:

```
φ⁻¹ that recovers AMAS invariants
```

Meaning:

> observers cannot reconstruct invariant structure

---

### 3.3 Invariant Blindness Constraint

For any invariant class:

```
I = S / ~
```

projection must satisfy:

> φ does NOT distinguish equivalence classes perfectly

---

## 4. Observer Ontological Status Rule

Any observer O is:

* external to AMAS
* not part of state space
* not part of dynamics
* not part of morphisms

Formally:

```
O ∉ AMAS
```

Observers are:

> *epistemic overlays, not ontological entities*

---

## 5. Measurement Constraint

Any measurement M must satisfy:

```
M(S) = f(φ(S))
```

Where:

* M operates ONLY on projection space
* never on AMAS state space directly

---

## 6. No Back-Propagation Rule

No external computation:

* may modify S
* may modify invariants
* may influence morphisms

unless explicitly defined in AMAS-core dynamics (which CIO is NOT allowed to do)

---

## 7. Interface Closure Condition

AMAS-core is closed under:

* invariants
* dynamics
* morphisms

AMAS is NOT closed under:

* observation
* compression
* estimation
* learning

---

## 8. Key Consequence

This formally enforces:

> CIO cannot “enter” AMAS — it can only *observe projections of it*

---


