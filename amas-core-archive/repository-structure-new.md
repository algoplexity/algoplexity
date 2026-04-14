# AMAS / repository-structure Specification (v2.0)

## Algorithmic Mesoscope Admissibility System — Canonical Canvas

---

## 1. Purpose and Scope

This document defines the **canonical repository structure for AMAS (Algorithmic Mesoscope Admissibility System)**.

It specifies:

* structural decomposition of the system
* domain boundaries and directional constraints
* separation between admissibility core and instantiations

It DOES NOT define:

* ontology semantics
* theoretical claims
* measurement models
* computational rules
* dynamics laws
* validity conditions

This document specifies **structure only**, under AMAS admissibility constraints.

---

## 2. Top-Level Domains (Global View)

The repository is partitioned into:

```
amas-core/
projections/
inference/
systems/
validation/
```

These define **strict epistemic and causal boundaries**.

---

## 3. Canonical Map (Global Structure)

```
amas-core/
    1-ontology/
    2-theory/
    3-measurement/
    4-computation/
    5-invariants/
    6-dynamics/
    meta/

projections/
inference/
systems/
validation/
```

### Critical update:

* `6-dynamics/` is a **first-class core layer**
* cybernetics is fully contained in `6-dynamics/`

---

## 4. Domain Definitions and Roles

---

## 4.1 amas-core/

The AMAS core defines the **admissible universe of structure + evolution**.

```
1-ontology/      → admissible entities and relations
2-theory/        → structural laws of the system
3-measurement/   → observer-relative representation
4-computation/   → admissible transformations of representations
5-invariants/    → non-negotiable constraints
6-dynamics/      → admissible state evolution laws
meta/           → governance, audit, constraints
```

### Core principle:

> AMAS-core defines both **what exists AND how it may evolve**

### Constraints:

AMAS-core MUST:

* remain closed under admissibility rules
* define all evolution laws in `6-dynamics/`

AMAS-core MUST NOT:

* contain system implementations
* define execution logic
* embed optimization procedures

---

## 4.2 projections/

Projections are **observer-relative embeddings of AMAS structure and dynamics**.

```
projections/
    <projection-name>/
        1-ontology/
        2-measurement/
        3-computation/
        4-system-model/
```

### Semantics

Each projection:

* maps AMAS admissible states into a domain context
* preserves invariants
* preserves dynamics structure (no modification)

### Constraint upgrade:

Projections MAY NOT:

* define or alter dynamics (`6-dynamics/`)
* introduce new transition rules
* redefine observer class

Projections are:

> coordinate charts over AMAS, not alternative systems

---

## 4.3 inference/

Inference is defined strictly as:

> transformation and analysis over admissible trajectories

```
inference/
    1-detection/
    2-deconvolution/
    3-intervention/
```

### Role update:

* Detection → identify admissible transitions
* Deconvolution → extract structure from trajectories
* Intervention → selection over admissible action set ONLY

### Constraint:

Inference MUST NOT:

* optimize system behavior
* learn or modify dynamics
* construct control policies

Inference operates on:

> AMAS trajectories, not system states directly

---

## 4.4 systems/

Systems are **execution substrates of AMAS trajectories**.

```
systems/
    <system-name>/
```

### Role:

* execute admissible trajectories
* instantiate observers
* simulate AMAS dynamics

### Constraint:

Systems MUST NOT:

* define control laws
* introduce optimization loops
* modify admissibility rules

Systems are:

> execution-only realizations of AMAS trajectories

---

## 4.5 validation/

Validation is the **external falsification layer of AMAS trajectories**.

```
validation/
    experiments/
```

### Role:

* test invariance preservation
* verify trajectory admissibility
* validate observer consistency

### Constraint:

Validation MUST NOT:

* redefine core semantics
* alter measurement models
* introduce alternative dynamics

Validation is:

> empirical probing of AMAS-consistent realizations

---

## 5. Global Directionality Law

```
amas-core → projections → inference → systems → validation
```

### Hard constraint:

No reverse causal influence is permitted.

Specifically:

* systems cannot modify inference
* inference cannot modify projections
* projections cannot modify core

---

## X. AMAS Inter-Domain Admissibility Contract

All interactions between AMAS domains MUST satisfy a global admissibility contract.

---

### X.1 Contract definition

Let D_i, D_j be any AMAS domains.

Any operation:

$$
T: D_i \rightarrow D_j
$$

is valid only if it preserves:

- invariant structure classes (𝕀)
- observer-consistent representation mappings (𝕄)
- dynamic admissibility constraints (𝔻)
- non-degeneracy of structure under composition

---

### X.2 Allowed interaction types

Only the following interaction types are admissible:

- **projection**: mapping structure into representational space
- **transformation**: computation within admissibility constraints
- **execution**: realization of admissible trajectories
- **observation**: bounded extraction of state representation

No other interaction types are permitted.

---

### X.3 Forbidden interaction types

The following are explicitly invalid:

- control (any domain influencing 𝔻)
- redefinition (any domain modifying core semantics)
- inversion (any downstream domain modifying upstream definitions)
- implicit feedback loops altering admissibility rules

---

### X.4 Composition rule

If:

$$
T_1: D_a \rightarrow D_b, \quad T_2: D_b \rightarrow D_c
$$

are admissible, then:

$$
T_2 \circ T_1
$$

must also preserve full AMAS admissibility.

If composition introduces hidden control structure, the system is invalid.

---

### X.5 Closure principle

All inter-domain operations must remain inside the AMAS admissibility space:

> no composition of valid domain operations may produce an invalid AMAS state or transition

---

### X.6 Interpretation constraint

Domains are not independent systems.

They are:

> coordinate subspaces of a single coupled admissibility system

---

## 6. Mesoscopic Computation Principle (AMAS)

AMAS defines a **mesoscopic regime of representation** as a structural level at which system behavior becomes representable in terms of stable, causally informative relations that are not present at either extreme of description.

---

## 6.1 Mesoscopic regime condition

A representation (x_t^{(m)}) is mesoscopic if:

* not reducible to microstate enumeration
* not equivalent to macroscopic aggregation
* preserves relational structure lost at both extremes

---

## 6.2 Mesoscope operator

[
\mathcal{M}: X_t \rightarrow x_t^{(m)}
]

* X_t: AMAS admissible state
* x_t^{(m)}: mesoscopic representation

---

## 6.3 Structural role

Mesoscopic regime is not a primitive layer.
It emerges from:

* measurement constraints (𝕄)
* computation limits (𝕔)
* observer bounds

---

## 6.4 Interpretation constraint

Mesoscopic representations are NOT:

* fundamental states
* macroscopic summaries
* arbitrary embeddings

They are:

> minimal stable representational regimes under AMAS constraints

---

## 6.5 Principle

Meaningful structure is defined as:

> what persists in the mesoscopic regime under bounded observation

---

## 6.6 Role in repository

* ontology → entities
* theory → relations
* measurement → projections
* computation → transformations
* invariants → constraints
* dynamics → evolution

Mesoscopic structure spans multiple domains but is not a layer.

---

## 7. Dynamics Authority Rule

All evolution is governed exclusively by:

```
amas-core/6-dynamics/
```

No external system may define temporal laws.

---

## 8. Cybernetic Interpretation

Cybernetics is generalized as:

* classical: feedback control
* AMAS: admissible trajectory selection under evolution constraints

No control layer exists.

---

## 9. System Validity Conditions

Repository is AMAS-valid iff:

* dynamics defined only in core
* no external control laws exist
* projections preserve invariants
* inference does not induce control
* systems are execution-only

---

## 10. Summary

AMAS defines:

* admissible structure
* admissible representation
* admissible evolution
* admissible transformation
* admissible execution

Nothing outside AMAS-core defines possibility.

Everything else is a constrained projection of it.
