# CIO System Constraints

## 0. Purpose

This document defines the **fully traceable mapping** from:

* Paper A (Theory)
* Paper B (Computation)
* Paper C (Measurement)

to a **concrete, falsifiable simulation and hardware system**.

The system implements a **bounded observer** and measures compressibility as an observer-relative quantity.

---

# 1. Core Invariants

## A1 — State Definition (Paper A)

System state is defined only by:

L* (minimum description length)

No additional state variables are permitted.

---

## B1 — Observer-Relative MDL (Paper B)

L* is defined under a fixed observer:

L* = min_O (L_sym, L_lat)

Constraint:

* Both symbolic and latent descriptions MUST operate under the **same observer budget**

---

## C1 — Observable Mapping (Paper C)

Primary observable:

E_O = 1 - normalize(L*)

Interpretation:

* High E_O → high compressibility → high coordination

---

## C2 — Representation Disagreement

ΔL = |L_sym - L_lat|

Interpretation:

* Measures disagreement between observers
* Indicates latent structure not captured symbolically

---

## C3 — Rate of Structural Change

r_eff = EMA(diff(L*), α)

* Smoothed derivative of L*
* Required for stability and phase detection

---

## C4 — Causal Contribution

I(G, e) = L*(G \ e) - L*(G)

Constraints:

* identical encoding
* identical observer
* identical window

---

# 2. Observer Definition

## 2.1 Observer Components

An observer O is defined as:

O = (φ, M, B)

Where:

* φ → encoding function
* M → compression / model class
* B → resource budget

---

## 2.2 Observer Budget (MANDATORY)

```python
OBSERVER_CONFIG = {
    "context_length": W,
    "max_model_params": K,
    "compression_scheme": "lz77"
}
```

All computation MUST respect this constraint.

---

# 3. Encoding Function (φ)

## 3.1 Input State

S_t = {
"adj": A_t,
"motion": M_t
}

---

## 3.2 Deterministic Encoding (MANDATORY)

```python
def encode_sequence(window):
    tokens = []
    for S_t in window:
        tokens.append(S_t["adj"].flatten(order="C"))
    return np.concatenate(tokens)
```

Constraints:

* deterministic
* fixed ordering
* no stochastic preprocessing
* identical across observers

---

## 3.3 Phase Constraint

* Phase 1–2: adjacency only
* Phase 3+: motion may be added

---

# 4. Dual Description (Paper B)

## 4.1 Symbolic Description

L_sym = compress(encoded_sequence)

Implementation:

* LZ77 / zlib
* entropy-based proxy

---

## 4.2 Latent Description

L_lat = model-based encoding under same budget

Constraint:

* bounded model capacity
* identical input sequence

---

## 4.3 Minimum Description

L* = min(L_sym, L_lat)

---

# 5. Observer Hierarchy (CRITICAL)

## 5.1 Real-Time Observer (Hub)

O_hub = (φ, M_zlib, B_tight)

Properties:

* real-time
* bounded
* approximate compressibility

Outputs:

* L*_hub
* ΔL_hub
* r_eff_hub

Used for:

* control
* live measurement

---

## 5.2 Reference Observer (Offline)

O_ref = (φ, M_advanced, B_relaxed)

Where:

* M_advanced may include neural BDM or CTM approximations

Properties:

* offline
* higher fidelity
* less encoding bias

Outputs:

* K_O_ref(X)

Used for:

* validation
* falsification
* calibration

---

## 5.3 Observer Consistency Constraint

φ MUST be identical across:

* O_hub
* O_ref

---

## 5.4 Observer Divergence (Mesoscope)

D(X) = K_O_hub(X) - K_O_ref(X)

Interpretation:

* measures observer bias
* reveals hidden structure

---

# 6. Simulation Structure

## 6.1 Data Flow

simulation → sequence → encoding → observers → metrics → control

---

## 6.2 Separation of Concerns

* simulation generates data
* observer computes metrics
* control modifies system

---

# 7. Control Law

Correct directionality:

```python
if L_star too high:
    increase coupling

if L_star too low:
    increase noise
```

---

# 8. Phases (LOCKED)

## Phase 0 — Architecture Validation

* invariants enforced
* encoding fixed
* observer defined

---

## Phase 1 — Synthetic Observer Validation

* adjacency-only sequences
* validate:

  * L*
  * ΔL
  * r_eff

---

## Phase 2 — Network Simulation

* generate interaction graphs
* validate scenarios:

  * aligned / random / clustered

---

## Phase 3 — Firmware Simulation (Wokwi)

* simulate node signals
* validate data pipeline

---

## Phase 4 — Hardware Mapping (TinkerCAD)

* validate physical constraints
* wiring and signal fidelity

---

## Phase 5 — Real-Time Control

* close feedback loop
* apply control laws

---

## Phase 6 — Experimental Validation

* perturb system
* measure:

  * L* response
  * causal effects

---

## Phase 7 — Observer Comparison

* run O_ref offline
* compare against O_hub
* validate observer-relative theory

---

# 9. Validation Requirements

System must demonstrate:

| Scenario  | L*     | ΔL     |
| --------- | ------ | ------ |
| Aligned   | Low    | Low    |
| Random    | High   | High   |
| Clustered | Medium | Medium |

---

# 10. Final Constraints

The system is valid only if:

1. L* remains the sole state variable
2. observer budget is enforced
3. encoding is deterministic and shared
4. observers are explicitly defined
5. causal metrics preserve encoding consistency

---

# 11. Interpretation

This system is:

* not a generic simulation
* not a machine learning pipeline

It is:

A physical instantiation of observer-relative compressibility

and

an experimental mesoscope over observer hierarchies
