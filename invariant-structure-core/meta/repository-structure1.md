# Repository Structure Specification

## Invariant Structure Stack — Global Organization

---

## 1. Purpose

This document defines the canonical repository structure for the invariant-structure stack.

It specifies:

* top-level domains
* second-level organization
* structural roles of each directory

It does NOT define:

* ontology
* theory
* measurement
* computation

It enforces:

structural separation and governance-compliant organization of all artifacts.

---

## 2. Top-Level Domains

The repository is partitioned into the following domains:

invariant-structure-core/
projections/
inference/
systems/
validation/
meta/

---

## 3. Domain Definitions

### 3.1 invariant-structure-core/

Contains canonical definitions:

* ontology
* theory
* measurement
* computation
* invariants

Constraints:

* no implementation
* no estimators
* no system-specific logic

---

### 3.2 projections/

Contains domain-specific mappings of the core framework.

Each projection MUST mirror core layers:

1-ontology/
2-measurement/
3-computation/
4-system-model/ or 4-control-model/

Constraints:

* must not redefine ontology or theory
* must declare observer and transformation class
* must preserve structural invariants

---

### 3.3 inference/

Contains procedural strata operating on measurement outputs.

This domain defines:

* detection (L1)
* deconvolution (L2)
* intervention (L3)

It is NOT a structural layer.

Structure:

inference/
1-detection/
cio-core/

```
2-deconvolution/
    aid-core/

3-intervention/
    control-core/
```

---

#### 3.3.1 Detection (L1)

Implements:

* observer projections
* delta operators
* alignment detection

Constraint:

* operates strictly on representations x_t

---

#### 3.3.2 Deconvolution (L2)

Implements:

* algorithmic information dynamics (AID)
* perturbation calculus
* generative structure inference

Constraint:

* cannot define structure
* cannot redefine measurement

---

#### 3.3.3 Intervention (L3)

Implements:

* control primitives
* feedback models
* stability conditions

Constraint:

* must operate only on signals derived from L1/L2

---

### 3.4 systems/

Contains executable systems and prototypes.

Examples:

systems/
cio-cps/
mesoscope-prototype/
llm-steering-prototype/
adia-prototype/

Each system MUST include:

* system-spec.md
* architecture.md (or equivalent)

Constraints:

Systems:

* implement observers, estimators, delta, alignment
* may include real-time pipelines and dashboards

Systems MUST NOT:

* define ontology
* define theory
* define invariants

---

### 3.5 validation/

Contains experiments and empirical verification.

Structure:

validation/
experiments/
cio/
aid/
cross-mode/

Constraints:

* validates claims
* does not define structure

---

### 3.6 meta/

Contains governance and control specifications.

Includes:

* meta-spec.md
* stack-governance.md
* repo-structure.md
* audit-spec.md

---

## 4. Projection Definitions

Projections define interpretations of the same structure.

Examples:

projections/
cio/
structural-break/
mesoscope/
llm-steering/

Constraint:

same structure, different representation, invariants preserved

---

## 5. Structural Constraints

### 5.1 Layer vs Procedure Separation

Core layers:

ontology -> theory -> measurement -> computation -> invariants

Inference layers:

inference/
detection -> deconvolution -> intervention

Constraint:

* inference operates on measurement outputs
* inference does not define structure

---

### 5.2 Directionality

No reverse influence allowed:

core -> projections -> inference -> systems -> validation

Forbidden:

* systems redefining theory
* inference redefining measurement
* projections redefining ontology

---

### 5.3 Projection Discipline

Projections MUST:

* map core definitions
* preserve invariants
* declare observer dependence

---

### 5.4 System Discipline

Systems MUST:

* implement computation
* expose signals
* remain observer-relative

Systems MUST NOT:

* define structural primitives
* embed theoretical assumptions

---

### 5.5 Inference Discipline

Inference modules MUST:

* operate on representations x_t
* use estimator outputs C_i(x_t)

They MUST NOT:

* define structure
* privilege specific estimators

---

## 6. Non-Degeneracy Requirement

All domains must operate over:

* non-degenerate observers
* non-degenerate estimators

Degenerate artifacts are:

* allowed only for falsification
* not allowed for primary claims

---

## 7. Summary

This structure enforces:

* separation of concerns
* epistemic integrity
* invariance preservation

It ensures:

structure remains independent of representation, computation, and implementation.
