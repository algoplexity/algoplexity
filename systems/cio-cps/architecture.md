# 📄 CIO-CPS architecture.md (v1.0 — invariant-structure compliant)

## 1. Purpose

This document defines the **structural architecture of CIO-CPS**.

It specifies:

* component topology
* data flow structure
* interface contracts
* dependency structure

It does NOT define:

* meaning of signals
* structural interpretation
* inference rules
* measurement semantics

---

## 2. System Decomposition

CIO-CPS is decomposed into four interacting subsystems:

```
Observers → Representations → Estimators → Outputs
```

Each subsystem is strictly separated.

---

## 3. Core Data Object

### Representation Bundle

[
\mathcal{X}_t = { x_t^{(1)}, x_t^{(2)}, ..., x_t^{(n)} }
]

Where:

* (x_t^{(i)} = \phi_{O_i}(X_t))
* each element is independent per observer

This is the primary data structure in the system.

---

## 4. Component Architecture

---

### 4.1 Observers Layer

Location:

```
observers/
```

Responsibility:

* generate (x_t^{(i)})
* define encoding functions (\phi_{O_i})
* maintain local buffering

Interface:

```
X_t → x_t^{(i)}
```

Constraint:

* no access to other observers
* no knowledge of estimators

---

### 4.2 Estimators Layer

Location:

```
estimators/
```

Responsibility:

* compute (C_i(x_t^{(i)}))
* operate independently per observer stream

Interface:

```
x_t^{(i)} → C_i(x_t^{(i)})
```

Constraint:

* no cross-observer aggregation
* no interpretation of outputs
* no structural inference

---

### 4.3 Agent Layer (Orchestration Only)

Location:

```
agents/
```

Responsibility:

* schedule observer execution
* route data streams
* manage computational execution order

Interface:

```
Observer outputs → Estimator inputs → Output streams
```

Constraint:

* no transformation of data semantics
* no aggregation logic
* no cross-stream reasoning

---

### 4.4 Dashboard Layer

Location:

```
dashboard/
```

Responsibility:

* render system outputs
* visualize raw streams
* expose estimator outputs

Interface:

```
C_i(x_t) → visualization
```

Constraint:

* no labeling of structure
* no derived interpretation
* no regime or transition semantics

---

## 5. Data Flow Architecture

```
        X_t
         ↓
   ┌──────────────┐
   │ Observers O_i │
   └──────────────┘
         ↓
   x_t^{(i)} streams
         ↓
   ┌──────────────┐
   │ Estimators C │
   └──────────────┘
         ↓
   C_i(x_t^{(i)})
         ↓
   Output streams
         ↓
   Dashboard / Logging
```

No additional transformations exist.

---

## 6. Cross-Observer Structure

CIO-CPS architecture explicitly forbids:

* merging observer streams in system layer
* computing similarity between observers
* aligning representations
* defining joint structure

Each observer stream is architecturally independent.

Cross-observer structure is handled only in:

> inference layer (cio-core)

---

## 7. Dependency Constraints

### Allowed dependencies

* observers → estimators
* estimators → dashboard
* agents → all components (orchestration only)

### Forbidden dependencies

* estimators → observers
* dashboard → observers
* any component → inference layer
* any component → invariants or theory layers

---

## 8. Interface Contracts

---

### 8.1 Observer Output

```python
x_t = {
    "observer_id": str,
    "data": Any,
    "timestamp": float
}
```

---

### 8.2 Estimator Output

```python
C_i = {
    "estimator_id": str,
    "observer_id": str,
    "value": float,
    "timestamp": float
}
```

---

### 8.3 Stream Bundle

```python
Stream = List[C_i]
```

---

## 9. Physical Implementation Mapping (Optional)

If deployed physically (e.g. totem system):

Allowed mapping:

* observer → sensor module
* estimator → compute module
* dashboard → display module

Forbidden mapping:

* physical state → structural meaning
* visual encoding → regime semantics

---

## 10. System Invariants

Architecture must preserve:

* observer independence
* estimator independence
* non-aggregation in system layer
* representation-only processing

---

## 11. Role in Full Stack

CIO-CPS architecture is:

* downstream of projections
* upstream of inference
* neutral with respect to structure

It defines:

> how data flows, not what data means

---

## 12. Summary

This architecture defines:

* component separation
* data flow structure
* interface contracts
* strict independence constraints

It does NOT define:

* meaning
* structure
* inference
* measurement interpretation

---

