# 📄 FINAL — CIO SIMULATION ARCHITECTURE (LOCKED)

---

## 1. Scope

This document defines the **canonical, invariant-preserving mapping** from:

* Paper A (Theory)
* Paper B (Computation)
* Paper C (Measurement)

to an **implementable simulation architecture**.

This document is **normative and locked**.

---

## 2. Core Invariants

---

### A1 — Observer Primacy

[
\text{System state is defined solely by } L^*
]

No other metric defines system state.

---

### B1 — Dual-Path MDL (Bounded Observer)

[
L^* = \min(L_{\text{sym}}, L_{\text{lat}}) \quad \text{subject to observer budget } \mathcal{O}
]

---

### B1.1 — Explicit Observer Budget (ENFORCED)

Both paths MUST operate under identical constraints:

```python
OBSERVER_CONFIG = {
    "max_model_params": K,
    "context_length": W,
    "compression_scheme": "lz77",
}
```

Violation invalidates MDL.

---

### B2 — Temporal Computation

All computations operate on **sequences**, never snapshots.

---

### C1 — Derived Observables Only

All observables derive strictly from:

* ( L^* )
* ( \Delta L )
* ( r_{\text{eff}} )

---

### S1 — Simulation Separation

```text
simulation → sequence → observer → metrics
```

Simulation cannot define state.

---

### O1 — Observer Closure

```text
window → encode → compress → compare
```

---

### O2 — Encoding Invariant (CRITICAL)

Encoding defines the observer and MUST be:

* deterministic
* information-preserving
* temporally consistent

---

## 3. Computational Definitions

---

### Primary Variables

[
\Delta L = |L_{\text{sym}} - L_{\text{lat}}|
]

---

### Smoothed Rate

[
r_{\text{eff}} = \text{EMA}\left(\Delta L^*\right)
]

---

### Derived Observables

```python
def E_O(L_star):
    return 1 - normalize(L_star)
```

```python
E_dir = delta_L  # diagnostic only
```

---

### Causal Contribution

[
I(G,e) = L^*(G) - L^*(G \setminus e)
]

Constraint:

* identical encoding
* identical observer
* identical window

---

## 4. Data Contract

---

### Interaction State

```python
S_t = {
    "adj": A_t,
    "motion": M_t  # optional
}
```

---

### Sequence Encoding (LOCKED)

```python
def encode_sequence(window):
    tokens = []
    for S_t in window:
        A = S_t["adj"]
        tokens.append(A.flatten(order="C"))
    return np.concatenate(tokens)
```

---

### Encoding Constraints

* fixed ordering
* no stochastic transforms
* no feature selection
* identical across runs

---

## 5. Observer Definition

```text
SlidingWindow
    ↓
encode_sequence
    ↓
L_sym
    ↓
L_lat
    ↓
L* = min under budget
    ↓
ΔL
```

---

## 6. Control Definition

Control acts only on compressibility:

```python
if L_star > L_high:
    increase_coupling()

elif L_star < L_low:
    increase_noise()
```

---

### Control Constraint

System must operate within a **target band**, not at minimum entropy.

---

## 7. Dynamic Rate (REQUIRED SMOOTHING)

```python
def compute_r_eff(L_star_series, alpha=0.2):
    diff = np.diff(L_star_series)
    return ema(diff, alpha)[-1]
```

---

## 8. Implementation Phases (LOCKED)

---

### Phase 0 — Architecture Lock

**Requirement:**

* this document committed

**Exit condition:**

* no ambiguity in invariants or encoding

---

### Phase 1 — Symbolic Observer

**Implement:**

* sliding window
* encoding (locked)
* LZ compression
* `L_sym`

**Must NOT include:**

* latent models
* control
* hardware

**Validation:**

| Scenario  | Expected |
| --------- | -------- |
| aligned   | low      |
| random    | high     |
| clustered | medium   |

**Exit condition:**

> L_sym ordering is stable and reproducible

---

### Phase 2 — Dual-Path Observer

**Add:**

* `L_lat`
* `ΔL`
* `L*`

**Constraint:**

* obey OBSERVER_CONFIG

**Exit condition:**

> ΔL distinguishes regimes

---

### Phase 3 — Dynamics & Control

**Add:**

* `r_eff`
* control loop

**Exit condition:**

> system can traverse compressibility regimes

---

### Phase 4 — Causal Measurement

**Add:**

* perturbation
* `I(G,e)`

**Exit condition:**

> removing causal edges increases L*

---

### Phase 5 — Hardware Integration

**Add:**

* Wokwi
* Tinkercad

**Constraint:**

> hardware must not alter observer logic

---

## 9. Traceability Table

| Paper | Concept         | Implementation        |
| ----- | --------------- | --------------------- |
| A     | Observer        | observer pipeline     |
| A     | Compressibility | L*                    |
| B     | L_sym           | symbolic compression  |
| B     | L_lat           | latent model          |
| B     | ΔL              | abs difference        |
| B     | r_eff           | smoothed derivative   |
| C     | E_O             | inverse normalized L* |
| C     | E_dir           | ΔL                    |
| C     | I(G,e)          | perturbation          |

---

# 🔒 FINAL LOCK

> This document defines a complete, invariant-preserving, and falsifiable implementation of the CIO framework.

No structural changes are permitted beyond this point.

---

This is the transition point from architecture → execution.
