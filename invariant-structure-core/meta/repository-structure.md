# Repository Structure Specification

## Invariant Structure Stack — Canonical Canvas

---

## 1. Purpose and Scope

This document is the canonical repository-structure specification for the invariant-structure stack. It is the single source of truth for:

- canonical directory layout
- layer separation and roles
- allowed composition and governance constraints

It DOES NOT define:

- ontology
- theory
- measurement
- computation

It enforces structural separation, epistemic integrity, and governance-compliant organization of all artifacts.

---

## 2. Top-Level Domains (Global View)

The repository is partitioned into these top-level domains:

- invariant-structure-core/
- projections/
- inference/
- systems/
- validation/
- meta/

These domains establish the directionality and boundaries of responsibility across the project.

---

## 3. Canonical Map (Global Structure)

Canonical tree (core-focused view):

```plaintext
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

Functional definitions (detection, deconvolution, intervention) are expressed in dedicated inference/procedure cores rather than mixed into the epistemic core.

---

## 4. Domain Definitions and Roles

### 4.1 invariant-structure-core/

Holds canonical, epistemic definitions:

- 1-ontology/      — what exists (ontology)
- 2-theory/        — what defines structure (theory)
- 3-measurement/   — how structure is evaluated (measurement)
- 4-computation/   — how measurement is approximated (computation)
- 5-invariants/    — structural constraints that must remain unchanged
- meta/            — governance, metadata, audit rules

Constraints:
- No implementation code, estimators, or system-specific logic may be placed here.
- Core defines structure only; it must remain implementation-agnostic.

---

### 4.2 projections/

Projections are domain-specific mappings of the core stack — views of core structure specialized for a domain. Each projection must mirror core layers and preserve invariants.

Typical projection layout:

```plaintext
projections/
    <projection-name>/
        1-ontology/
        2-measurement/
        3-computation/
        4-system-model/      # or 4-control-model/ / 4-detection-model/
```

Projection semantics and constraints:
- Map core definitions to a domain-specific representation.
- Preserve structural invariants; do NOT redefine ontology or theory.
- Declare observer and transformation classes where applicable.
- Projections are views only — non-authoritative with respect to core structure.

Examples:
- projections/cio/
- projections/structural-break/
- projections/mesoscope/
- projections/llm-steering/

---

### 4.3 inference/

Inference contains procedural analysis methods operating on measurement outputs. It is a procedural layer, not a structural layer.

Procedural cores (recommended structure):

```plaintext
inference/
    1-detection/
        cio-core/
    2-deconvolution/
        aid-core/
    3-intervention/
        control-core/
```

Inference responsibilities and constraints:
- Operates on representations x_t and estimator outputs C_i(x_t).
- Implements detection (L1), deconvolution (L2), and intervention (L3).
- Must not define structure or redefine measurement.
- Must not privilege specific estimators.

---

### 4.4 systems/

Systems are executable instantiations and prototypes built from projections and inference outputs. They implement computation and pipelines.

Typical systems layout:

```plaintext
systems/
    <system-name>/
        system-spec.md
        architecture.md
        dashboard/
        agents/
        observers/
        estimators/
        ...
```

System constraints:
- Systems implement computation only; they consume representations x_t and produce estimator outputs C_i(x_t).
- Systems must not define ontology, theory, or invariants.
- Each system SHOULD include system-spec and architecture documentation.

Examples:
- systems/cio-cps/
- systems/mesoscope-prototype/
- systems/llm-steering-prototype/
- systems/adia-prototype/

---

### 4.5 validation/

Validation contains experiments, empirical verification, and falsification cases.

Typical validation layout:

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

Validation constraints:
- Validation tests invariance properties and demonstrates Δ-alignment.
- Validation must not define structure or introduce new measurement rules.

---

### 4.6 meta/

Meta contains governance, control specifications, and auditing artifacts.

Examples:
- meta/meta-spec.md
- meta/stack-governance.md
- meta/repository-structure.md
- meta/audit-spec.md

Meta defines rules for allowed transformations, artifact validity, and compliance.

---

## 5. Layer Mapping (Reference)

| Layer       | Canonical Location                         | Role                           |
|-------------|--------------------------------------------|--------------------------------|
| Ontology    | invariant-structure-core/1-ontology        | defines existence              |
| Theory      | invariant-structure-core/2-theory          | defines structure              |
| Measurement | invariant-structure-core/3-measurement     | defines evaluation             |
| Computation | invariant-structure-core/4-computation     | defines approximations         |
| Invariants  | invariant-structure-core/5-invariants      | defines constraints            |

Functional procedures:
- Detection (L1)     — cio-core/      (Δ-alignment)
- Deconvolution (L2) — aid-core/      (causal structure extraction)
- Intervention (L3)  — control-core/  (system steering)

---

## 6. Structural Principles and Rules

### 6.1 Directionality

The repository enforces strict directionality:

```plaintext
core → projections → inference → systems → validation
```

Reverse influence is forbidden. Examples of prohibited actions:
- Systems redefining theory.
- Inference redefining measurement.
- Projections redefining ontology.

### 6.2 Layer vs Procedure Separation

- Core layers (ontology → theory → measurement → computation → invariants) are epistemic and must not be mixed with procedural or implementation artifacts.
- Inference layers (detection → deconvolution → intervention) are procedural and operate on measurement outputs.

### 6.3 No Implicit Layers / No Folder Ambiguity

- All functional stages must map explicitly to cio-core (L1), aid-core (L2), or control-core (L3).
- Folders must have exactly one role: definition (core), projection, system, validation, or inference. Mixed roles are invalid.

### 6.4 Projections Are Views Only

Projections may reinterpret or specialize core definitions, but they must not:
- Redefine invariants.
- Introduce new structural primitives.
- Alter core ontology or theory.

### 6.5 Systems Are Instantiations Only

Systems may implement and combine artifacts to produce executable behavior, but they must not:
- Define new theory.
- Alter measurement definitions.
- Claim authority over structure.

### 6.6 Inference Discipline

Inference modules must:
- Operate on representations x_t.
- Use estimator outputs C_i(x_t).
- Not define core structure or privilege particular estimators.

### 6.7 Non-Degeneracy Requirement

All domains must operate over non-degenerate observers and non-degenerate estimators. Degenerate artifacts are allowed only for falsification and not for primary claims.

---

## 7. Mesoscopic Computation Principle (Critical)

### 7.1 Statement

All computation within the system occurs on observer-induced representations:

x_t = φ_O(X_t)

These mesoscopic representations preserve relational structure while remaining computationally tractable.

### 7.2 Interpretation

- Micro-level (raw state X_t): unbounded, not computationally accessible.
- Macro-level (aggregates): loses structural information.
- Mesoscopic level (x_t): preserves relational structure and supports estimator operation.

### 7.3 Repository Implication

- The mesoscopic perspective is expressed ONLY as a projection (for example: projections/mesoscope/).
- It MUST NOT be embedded into ontology, theory, or invariants.
- It is NOT an ontological or theoretical primitive — it is an emergent level resulting from observation.

### 7.4 System-Wide Role

Functional stages operate implicitly over x_t:
- Detection (cio-core) → Δ-signals over x_t
- Deconvolution (aid-core) → perturbations on x_t
- Intervention (control-core) → feedback via x_t

Any artifact treating the mesoscopic level as fundamental (rather than emergent via observation) is invalid.

---

## 8. Examples and Canonical Artifacts (Illustrative)

- projections/cio/1-ontology/, projections/mesoscope/...
- inference/cio-core/observer-spec.md, inference/aid-core/causal-deconvolution.md
- systems/cio-cps/system-spec.md, systems/adia-prototype/architecture.md
- validation/experiments/... notebooks and protocols
- meta/stack-governance.md, meta/audit-spec.md

Each area should include clear documentation (system-spec, architecture, validation-protocols) appropriate to its role.

---

## 9. Governance References and Compliance

This document defines structure only. For rules on allowed transformations and artifact validity consult:
- meta/structure-constraints.md (or structure-constraints.md)
- meta/non-degeneracy-spec.md (or non-degeneracy-spec.md)
- meta/audit-spec.md (or audit-spec.md)

These govern:
- allowed transformations
- validity of artifacts
- system compliance and auditability

---

## 10. Summary

The repository is valid only if:

- epistemic layers remain isolated from implementation
- functional procedures are explicit and mapped to the proper inference cores
- projections remain view-only and non-authoritative
- systems remain downstream and do not alter structure
- validation remains external and falsification-focused
- the mesoscopic computation level remains emergent from observation, not a primitive

This specification enforces separation of concerns, invariance preservation, and governance-compliant organization across the repository.

---
