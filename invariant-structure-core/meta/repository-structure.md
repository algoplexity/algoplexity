Here is the **clean, corrected, final v1.0** of your `repository-structure.md`, with all fixes applied:

* constraints removed (properly delegated)
* non-degeneracy removed (delegated)
* mesoscopic section corrected (no ontology leakage)
* meta location unified
* scope tightened

---

# 📜 Repository Structure Specification

## Invariant Structure Stack — Canonical Canvas (v1.0)

---

## 1. Purpose and Scope

This document defines the **canonical repository structure** for the invariant-structure stack.

It is the single source of truth for:

* canonical directory layout
* layer separation and roles
* domain boundaries and relationships

It DOES NOT define:

* ontology
* theory
* measurement
* computation
* behavioral constraints
* validity conditions

This document specifies **structure only**.

---

## 2. Top-Level Domains (Global View)

The repository is partitioned into the following domains:

* `invariant-structure-core/`
* `projections/`
* `inference/`
* `systems/`
* `validation/`

These domains define **responsibility boundaries** and enforce **directionality** across the stack.

---

## 3. Canonical Map (Global Structure)

```plaintext
invariant-structure-core/
    1-ontology/
    2-theory/
    3-measurement/
    4-computation/
    5-invariants/
    meta/

projections/
inference/
systems/
validation/
```

Functional procedures (detection, deconvolution, intervention) are **not part of the core stack** and are defined in the inference domain.

---

## 4. Domain Definitions and Roles

---

### 4.1 invariant-structure-core/

This is the **epistemic core** of the system.

```plaintext
1-ontology/      # what exists
2-theory/        # what structure is
3-measurement/   # how structure is evaluated
4-computation/   # how measurement is approximated
5-invariants/    # what must remain unchanged
meta/            # governance, metadata, audit specifications
```

#### Structural Notes

* Must remain **implementation-agnostic**
* Must not contain:
  * estimators
  * system logic
  * domain-specific adaptations
 
The core defines **structure only**.

---

### 4.2 projections/

Projections are **domain-specific mappings** of the core stack.

Projections reference observer-induced representations (x_t = φ_O(X_t)) defined in the measurement layer and interpret them within domain-specific contexts.

```plaintext
projections/
    <projection-name>/
        1-ontology/
        2-measurement/
        3-computation/
        4-system-model/   # or detection/control variants
```

#### Semantics

Each projection:

* maps core definitions into a domain
* preserves invariants
* declares observer and transformation context

#### Constraints

Projections MUST NOT:

* redefine ontology or theory
* introduce new structural primitives
* alter invariants

They are **views**, not authorities.

---

### 4.3 inference/

Inference defines **procedural layers** operating on representations and measurements.

```plaintext
inference/
    1-detection/
        cio-core/
    2-deconvolution/
        aid-core/
    3-intervention/
        control-core/
```

#### Roles

* Detection (L1): structural transition detection
* Deconvolution (L2): causal structure extraction
* Intervention (L3): system steering

#### Constraints

Inference MUST:

* operate on representations ( x_t )
* use estimator outputs ( C_i(x_t) )

Inference MUST NOT:

* define structure
* redefine measurement
* privilege specific estimators

---

### 4.4 systems/

Systems are **executable instantiations**.

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

#### Roles

* implement pipelines
* process representations
* generate estimator outputs

#### Constraints

Systems MUST NOT:

* define ontology or theory
* redefine measurement
* alter invariants

Systems are **downstream implementations only**.

---

### 4.5 validation/

Validation contains **experiments and empirical verification**.

```plaintext
validation/
    experiments/
        cio/
        aid/
        cross-mode/
```

#### Roles

* test invariance
* demonstrate Δ-alignment
* include falsification

#### Constraints

Validation MUST NOT:

* define structure
* introduce measurement rules

---

### 4.6 Canonical Internal Structures (Normative)

This section defines **normative internal structures** for key domains.

These structures are **binding references**, but kept minimal to preserve document clarity.

---

### 4.6.1 Inference Modules

```plaintext
inference/
    1-detection/
        cio-core/
            observer-spec.md
            delta-spec.md
            alignment-spec.md
            validation-protocol.md

    2-deconvolution/
        aid-core/
            bdm.md
            perturbation-calculus.md
            algorithmic-probability.md
            causal-deconvolution.md
            assembly-analysis.md

    3-intervention/
        control-core/
            control-primitives.md
            stability-conditions.md
            feedback-models.md
```

---

### 4.6.2 Systems (Reference Layouts)

```plaintext
systems/
    cio-cps/
        dashboard/
        agents/
        observers/
        estimators/

    mesoscope-prototype/
        control-loop.md
        simulation/

    llm-steering-prototype/
        prompt-control/
        routing/
        experiments/

    adia-prototype/
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

### 4.6.3 Validation (Reference Layouts)

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

## 5. Layer Mapping (Reference)

| Layer       | Location                               | Role          |
| ----------- | -------------------------------------- | ------------- |
| Ontology    | invariant-structure-core/1-ontology    | existence     |
| Theory      | invariant-structure-core/2-theory      | structure     |
| Measurement | invariant-structure-core/3-measurement | evaluation    |
| Computation | invariant-structure-core/4-computation | approximation |
| Invariants  | invariant-structure-core/5-invariants  | constraints   |

Functional procedures:

* Detection (L1) → inference/1-detection/cio-core
* Deconvolution (L2) → inference/2-deconvolution/aid-core
* Intervention (L3) → inference/3-intervention/control-core

---

## 6. Structural Principles (Reference Only)

This document references, but does not define, system constraints.

### 6.1 Directionality

```plaintext
core → projections → inference → systems → validation
```

Reverse influence is not permitted.

Examples of invalid behavior:

* systems redefining theory
* inference redefining measurement
* projections redefining ontology

---

## 7. Mesoscopic Computation Principle

### 7.1 Statement

All operational computation occurs on **observer-induced representations**:

$$
x_t = \phi_O(X_t)
$$

### 7.2 Interpretation

* Micro (raw state): not directly computable
* Macro (aggregates): loses structure
* Mesoscopic (representation): preserves relational structure

### 7.3 Repository Role

* Mesoscopic computation is realized in:

  * projections
  * inference
  * systems

* It is **not a primitive of ontology or theory**

### 7.4 Constraint

The mesoscopic level:

* emerges from observation
* must not be treated as a foundational ontological construct

---

## 8. Examples (Illustrative)

* projections/cio/, projections/mesoscope/
* inference/1-detection/cio-core/
* systems/cio-cps/, systems/adia-prototype/
* validation/experiments/...

---

## 9. Governance References

This document defines structure only.

All behavioral and validity rules are defined in:

* invariant-structure-core/meta/structure-constraints.md
* invariant-structure-core/meta/non-degeneracy-spec.md
* invariant-structure-core/meta/audit-spec.md

These govern:

* allowed transformations
* validity conditions
* system compliance

---

## 10. Summary

The repository is valid if:

* epistemic layers remain isolated
* procedural layers remain downstream
* projections remain non-authoritative
* systems remain implementation-only
* validation remains external and falsifiable
* mesoscopic computation remains observer-induced, not primitive

---

# ✅ This version is now:

* structurally complete
* governance-compliant
* non-bloated
* scalable
* aligned with your meta-spec + stack-governance

---

