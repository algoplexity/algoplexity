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
