# Repository Structure Specification

## 1. Purpose

This document defines the **complete, canonical repository structure**.

It is the single source of truth for:

* directory layout
* layer separation
* functional roles
* allowed composition

It does NOT define:

* theory
* measurement
* computation

---

## 2. Structural Principles

The repository is organized along three orthogonal axes:

### 2.1 Epistemic Axis (Truth)

```
invariant-structure-core/
```

Defines what is true independent of implementation.

---

### 2.2 Functional Axis (Process)

```
detection → deconvolution → intervention
```

Defines what the system does.

---

### 2.3 Instantiation Axis (Realization)

```
projections/ → systems/ → validation/
```

Defines how the system is instantiated and tested.

---

## 3. Complete Repository Structure

```
invariant-structure-core/
    1-ontology/
    2-theory/
    3-measurement/
    4-computation/
    5-invariants/

    meta/
        meta-spec.md
        stack-governance.md
        repository-structure.md
        audit-spec.md


# Functional definitions are embedded directly, NOT via a separate folder

cio-core/                  # L1: Detection
    observer-spec.md
    delta-spec.md
    alignment-spec.md
    validation-protocol.md


aid-core/                  # L2: Deconvolution
    bdm.md
    perturbation-calculus.md
    algorithmic-probability.md
    causal-deconvolution.md
    assembly-analysis.md


control-core/              # L3: Intervention
    control-primitives.md
    stability-conditions.md
    feedback-models.md


projections/
    cio/
        1-ontology/
        2-measurement/
        3-computation/
        4-system-model/

    structural-break/
        1-temporal-ontology/
        2-measurement/
        3-computation/
        4-detection-model/

    mesoscope/
        1-ontology/
        2-measurement/
        3-computation/
        4-control-model/

    llm-steering/
        1-ontology/
        2-measurement/
        3-computation/
        4-control-model/


systems/
    cio-cps/
        system-spec.md
        architecture.md
        dashboard/
        agents/
        observers/
        estimators/

    mesoscope-prototype/
        system-spec.md
        control-loop.md
        simulation/

    llm-steering-prototype/
        system-spec.md
        prompt-control/
        routing/
        experiments/

    adia-prototype/
        system-spec.md
        architecture.md
        demo-flow.md

        simulation/
        live-mode/

        observers/
        estimators/
        delta-engine/
        alignment-engine/

        dashboard/
        optional-control/


validation/
    experiments/
        cio/
            erdos-renyi-phase-transition.ipynb
            noise-robustness.ipynb

        aid/
            perturbation-analysis.ipynb
            bdm-validation.ipynb

        cross-mode/
            sim-vs-live-alignment.ipynb


inference/
    causal-deconvolution.md
    assembly-analysis.md
```

---

## 4. Layer Mapping (Critical)

| Layer       | Location                               | Role                   |
| ----------- | -------------------------------------- | ---------------------- |
| Ontology    | invariant-structure-core/1-ontology    | defines existence      |
| Theory      | invariant-structure-core/2-theory      | defines structure      |
| Measurement | invariant-structure-core/3-measurement | defines evaluation     |
| Computation | invariant-structure-core/4-computation | defines approximations |
| Invariants  | invariant-structure-core/5-invariants  | defines constraints    |

| Function           | Location      | Role                        |
| ------------------ | ------------- | --------------------------- |
| Detection (L1)     | cio-core/     | Δ-alignment                 |
| Deconvolution (L2) | aid-core/     | causal structure extraction |
| Intervention (L3)  | control-core/ | system steering             |

---

## 5. Structural Rules

### 5.1 No Implicit Layers

All functional stages must map explicitly to:

* cio-core (L1)
* aid-core (L2)
* control-core (L3)

No hidden pipelines are allowed.

---

### 5.2 No Folder-Level Ambiguity

Folders must have exactly one role:

* definition
* projection
* system
* validation
* inference

Mixed roles are invalid.

---

### 5.3 No Reverse Dependency

Forbidden:

* systems defining core
* projections modifying core
* validation redefining measurement

---

### 5.4 Projections Are Views Only

They may:

* reinterpret
* specialize

They may NOT:

* redefine invariants
* introduce new primitives

---

### 5.5 Systems Are Instantiations Only

They may:

* implement
* combine

They may NOT:

* define theory
* alter measurement

---

### 5.6 Validation Is External

Validation artifacts:

* test
* falsify

They do not define structure.

---

## 6. Mesoscopic Computation Principle (Critical)

### 6.1 Statement

All computation within the system occurs on observer-induced representations:

x_t = φ_O(X_t)

These representations are necessarily **mesoscopic** due to observer constraints.

---

### 6.2 Interpretation

* Micro-level (raw state X_t):

  * unbounded
  * not computationally accessible

* Macro-level (aggregates):

  * loses structural information
  * destroys relational detail

* Mesoscopic level (x_t):

  * preserves relational structure
  * remains computationally tractable
  * supports estimator operation

---

### 6.3 Consequence

The mesoscopic level is NOT:

* an ontological primitive
* a theoretical construct

It IS:

> the minimal level at which structure becomes computationally accessible under bounded observation

---

### 6.4 Repository Implication

The mesoscopic perspective is expressed ONLY as a projection:

```
projections/
    mesoscope/
```

It must NOT appear in:

* ontology
* theory
* invariant definitions

---

### 6.5 System-Wide Role

All functional stages operate implicitly at the mesoscopic level:

* Detection (cio-core) → Δ-signals over x_t
* Deconvolution (aid-core) → perturbations on x_t
* Intervention (control-core) → feedback via x_t

---

### 6.6 Governance Constraint

Any artifact that:

* treats mesoscopic level as fundamental
* embeds it into ontology or theory

is INVALID.

---

## 7. Summary

The repository is valid only if:

* epistemic layers remain isolated
* functional stages remain explicit
* projections remain non-authoritative
* systems remain downstream
* validation remains external
* mesoscopic computation remains an emergent property of observation, not a primitive

This document fully defines the structure of the repository.
