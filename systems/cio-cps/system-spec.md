# 📄 CIO-CPS system-spec.md (v2.0 — invariant-structure compliant)

## 1. Purpose

CIO-CPS is a **system-layer implementation** of the CIO projection.

It is responsible only for:

* instantiating observers
* producing representations (x_t)
* executing estimators (C_i(x_t))
* streaming outputs to downstream layers

It does NOT define structure, meaning, or interpretation.

---

## 2. System Boundary

CIO-CPS operates strictly within:

[
\mathcal{X}*t = {x_t^{(i)} = \phi*{O_i}(X_t)}
]

The system:

* does not access (X_t)
* does not define (\phi)
* does not define measurement functionals
* does not interpret estimator outputs

---

## 3. System Components

### 3.1 Observers

Location:

```
systems/cio-cps/observers/
```

Role:

* implement encoding functions (\phi_{O_i})
* produce representations (x_t^{(i)})
* maintain buffering and sampling state

Constraints:

* no estimator logic
* no aggregation logic
* no interpretation of outputs

---

### 3.2 Estimators

Location:

```
systems/cio-cps/estimators/
```

Role:

* compute (C_i(x_t^{(i)}))
* operate independently per observer stream
* produce scalar or vector outputs

Constraints:

* no cross-observer logic
* no structural interpretation
* no temporal region detection logic

---

### 3.3 Agents

Location:

```
systems/cio-cps/agents/
```

Role:

* orchestrate data flow between components
* schedule computation
* manage buffering and execution

Constraints:

* no inference logic
* no aggregation semantics
* no decision-making over structure

---

### 3.4 Dashboard

Location:

```
systems/cio-cps/dashboard/
```

Role:

* display raw observer outputs
* display estimator outputs
* visualize system state streams

Constraints:

* no labeling of structure
* no regime classification
* no interpretation of signals

---

## 4. Data Flow

```
X_t
 ↓
φ_{O_i}
 ↓
x_t^{(i)}
 ↓
C_i(x_t^{(i)})
 ↓
streamed outputs
```

No additional transformations are defined at system level.

---

## 5. System Constraints

The system MUST:

* operate only on representations (x_t)
* execute estimators as black-box functions
* preserve observer separation
* stream outputs without interpretation

The system MUST NOT:

* define structure
* define measurement meaning
* perform inference
* aggregate across observers semantically
* detect transitions or regimes

---

## 6. Cross-Observer Handling

If multiple observer streams exist:

* they are stored separately
* they are not merged semantically
* they are not aligned or compared in-system

Any cross-observer reasoning is reserved for:

> inference layer (cio-core)

---

## 7. Physical Implementation (Optional)

If embodied (e.g., totem, LEDs, sensors):

Allowed:

* encoding raw estimator values into visual channels
* mapping numeric outputs to display parameters

Forbidden:

* mapping visuals to structural meaning (e.g., “regime”, “coordination”, “transition”)

Physical layer is strictly a rendering substrate.

---

## 8. System Role in Stack

CIO-CPS is:

* downstream of projections
* upstream of inference
* neutral with respect to structure

It functions as:

> a representation + estimator execution substrate

---

## 9. Summary

CIO-CPS:

* generates representations
* executes estimators
* exposes raw outputs

It does NOT:

* interpret outputs
* define structure
* perform inference
* assign meaning to signals

---

## 10. Key Shift from v1

Removed:

* regime semantics
* coordination interpretation
* Δ-based meaning assignment
* alignment narratives
* structural visualization claims

Result:

> system is now purely operational, not interpretive

---
