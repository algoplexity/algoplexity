<!-- Merged from: repository-structure1.md -->
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

<!-- End of repository-structure1.md -->



<!-- Merged from: repository-structure2.md -->
# Repository Structure Specification

## Invariant Structure Core — Full Structural Map

---

# 1. Purpose

This document defines the **complete repository structure** of the invariant-structure-core project.

It serves as:

* the canonical map of all folders
* the boundary definition for each domain
* the top-level governance reference

It does NOT:

* define theory
* define measurement
* define computation

---

# 2. Global Structure

```plaintext
invariant-structure-core/

    invariant-structure-core/
        1-ontology/
        2-theory/
        3-measurement/
        4-computation/
        5-invariants/
        meta/

    projections/
    systems/
    validation/
    inference/
```

---

# 3. Core Layer (Invariant Structure Core)

## invariant-structure-core/

Defines the **foundational stack**:

```plaintext
1-ontology/      # what exists
2-theory/        # what structure is
3-measurement/   # how structure is evaluated
4-computation/   # how measurement is approximated
5-invariants/    # what must remain unchanged
meta/            # governance + metadata rules
```

---

# 4. Projections (Domain Instantiations)

## projections/

Each projection is a **domain-specific mapping** of the core stack.

### Structure:

```plaintext
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
```

---

## Projection Semantics

Each projection MUST:

* map core definitions into a domain
* preserve invariants
* remain observer-relative

Each internal folder represents:

| Folder        | Role                                           |
| ------------- | ---------------------------------------------- |
| 1-ontology    | domain-specific objects                        |
| 2-measurement | domain measurement definitions                 |
| 3-computation | estimator approximations                       |
| 4-*           | domain-specific system/detection/control model |

---

# 5. Systems (Executable Instantiations)

## systems/

Systems are **real implementations** of projections.

### Structure:

```plaintext
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
```

---

## System Semantics

Systems MUST:

* implement computation only
* consume representations x_t
* produce estimator outputs C_i(x_t)

Systems MUST NOT:

* define structure
* redefine measurement
* embed theoretical assumptions

---

# 6. Validation (Empirical Layer)

## validation/

Validation contains **experiments and proofs**.

### Structure:

```plaintext
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
```

---

## Validation Semantics

Validation MUST:

* test invariance properties
* demonstrate Δ-alignment
* include falsification cases

Validation MUST NOT:

* define structure
* introduce new measurement rules

---

# 7. Inference (Procedure Layer)

## inference/

Inference defines **procedural analysis methods**.

### Structure:

```plaintext
inference/
    causal-deconvolution.md
    assembly-analysis.md
```

---

## Inference Semantics

Inference:

* operates on representations x_t
* uses estimator outputs C_i(x_t)

Inference MUST NOT:

* define structure
* redefine measurement
* privilege specific estimators

---

# 8. Structural Principles

## 8.1 Mesoscopic Principle

All computation occurs at the **mesoscopic level**:

* not micro (raw agents)
* not macro (aggregates)
* but interaction structure

---

## 8.2 Directionality

```plaintext
core → projections → inference → systems → validation
```

No reverse influence allowed.

---

# 9. Summary

This structure ensures:

* separation of concerns
* invariance preservation
* observer relativity
* implementability without theory corruption

---

# 10. Governance References

This document defines structure only.

All behavioral and validity constraints are defined in:

- structure-constraints.md
- non-degeneracy-spec.md
- audit-spec.md

These documents MUST be consulted for:

- allowed transformations
- validity of artifacts
- system compliance

<!-- End of repository-structure2.md -->



<!-- Merged from: repository-structure3.md -->
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

<!-- End of repository-structure3.md -->



<!-- Merged from: repository-structure4.md -->
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

<!-- End of repository-structure4.md -->
