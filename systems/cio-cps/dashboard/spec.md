# 📄 CIO-CPS dashboard/spec.md (v1.0 — invariant-structure compliant)

## 1. Purpose

This document defines the **presentation layer of CIO-CPS**.

It specifies how system outputs are rendered.

It does NOT define:

* structure
* meaning
* inference
* measurement interpretation
* coordination logic

The dashboard is a **visualization interface only**.

---

## 2. Role of Dashboard

The dashboard is a **passive rendering layer** over:

* observer outputs (x_t^{(i)})
* estimator outputs (C_i(x_t^{(i)}))

It performs no transformation of semantics.

---

## 3. Input Data Streams

The dashboard consumes only:

### 3.1 Observer Streams

```python
x_t = {
    "observer_id": str,
    "data": Any,
    "timestamp": float
}
```

---

### 3.2 Estimator Streams

```python
C_i = {
    "estimator_id": str,
    "observer_id": str,
    "value": float,
    "timestamp": float
}
```

---

## 4. Rendering Constraints

The dashboard MAY:

* plot numeric values
* display time series
* visualize raw streams
* render per-observer separation
* show estimator outputs as independent channels

The dashboard MUST NOT:

* label “regimes”
* infer “transitions”
* compute alignment
* aggregate observers
* assign semantic categories
* suggest causal interpretation

---

## 5. Visualization Model

All visualization is a function:

[
V : {x_t^{(i)}, C_i(x_t^{(i)})} \rightarrow \text{display space}
]

Constraint:

> V is injective on data appearance, not on meaning.

Meaning is explicitly undefined at this layer.

---

## 6. View Types

### 6.1 Observer View

Displays:

* raw (x_t^{(i)})
* sampling rate
* timestamp stream

Constraint:

* no comparison across observers

---

### 6.2 Estimator View

Displays:

* (C_i(x_t^{(i)}))
* per-estimator time series

Constraint:

* no aggregation or ranking

---

### 6.3 System View (Structural Neutral)

Displays:

* multi-stream juxtaposition only
* no computed relations

Constraint:

* purely spatial alignment, not semantic alignment

---

## 7. Cross-Stream Constraint

The dashboard MUST NOT:

* compute similarity between streams
* overlay interpretations
* infer synchronization
* compute differences across observers

Any such operations belong to:

> inference layer (cio-core)

---

## 8. Temporal Rendering Constraint

Time alignment MAY be visualized only as:

* axis alignment
* timestamp ordering

It MUST NOT be interpreted as:

* causal ordering
* phase transitions
* regime structure

---

## 9. Interaction Constraint

If interactive:

Allowed:

* zoom
* filter by observer ID
* toggle estimator visibility

Forbidden:

* “highlight anomalies”
* “detect transitions”
* “suggest structure”
* “auto-cluster behavior”

---

## 10. Physical Implementation Mapping (Totem System)

If mapped to physical device:

Allowed:

* LED intensity = raw estimator value
* color channels = raw scalar mappings
* spatial separation = observer identity

Forbidden:

* color = regime
* brightness = coordination level
* pulse = structural transition

All physical encoding must remain **non-semantic**.

---

## 11. Role in CIO-CPS Stack

Dashboard is:

* downstream of all computation
* upstream of all inference interpretation (none exists here)
* strictly representational

It defines:

> how outputs are seen, not what they mean

---

## 12. System Invariants

Dashboard must preserve:

* observer separation
* estimator independence
* raw data fidelity
* non-aggregation principle

It must not introduce:

* structural labels
* interpretive overlays
* inferred metrics

---

## 13. Summary

The dashboard provides:

* visualization of observer streams
* visualization of estimator outputs
* temporal rendering of raw system data

It does NOT provide:

* interpretation
* inference
* structural meaning
* coordination signals

---

## 14. Final Boundary Statement

The dashboard is the **last non-inferential layer** in CIO-CPS.

Beyond this point:

> all structure, meaning, and hypothesis testing belongs exclusively to inference and validation layers.

---
