# CIO Simulation Architecture (IMPLEMENTATION SPEC v1.2)

---

## 0. Purpose

This document defines the **concrete system architecture** for implementing the Collective Intelligence Observatory (CIO) across:

* Simulation (Python)
* Firmware (Wokwi / ESP32)
* Hardware (TinkerCAD)
* Real-time processing

It specifies:

* system components
* data flow
* interfaces
* execution model

This document is **implementation-only**.
All theoretical constraints are defined in `cio-constraints.md`.

---

# 1. System Overview

## 1.1 High-Level Pipeline

```
Nodes → Hub (Ingress) → Reconstruction Layer → Graph Builder → Encoder → Metric Engine → Control → Nodes
```

---

## 1.2 Core Components

| Component            | Location        | Role                                 |
| -------------------- | --------------- | ------------------------------------ |
| Node                 | Firmware        | Generates motion + proximity signals |
| Hub (Ingress)        | Firmware/Bridge | Receives packets (transport only)    |
| Reconstruction Layer | Realtime Engine | Builds temporally valid snapshots    |
| Graph Builder        | Realtime Engine | Constructs G(t)                      |
| Encoder              | Realtime Engine | Produces encoded sequences           |
| Metric Engine        | Realtime Engine | Computes L*, ΔL, r_eff               |
| Control Engine       | Actuation       | Applies feedback                     |
| Data Store           | Data Layer      | Persists streams                     |

---

# 2. Execution Model

## 2.1 Time

System operates in discrete ticks:

```
tick = 100ms (configurable)
```

At each tick:

1. Nodes emit state
2. Hub ingests packets
3. Reconstruction layer evaluates valid snapshot
4. Graph is built
5. Window updated
6. Metrics computed
7. Control signal emitted

---

## 2.2 Windowing

```
WINDOW_SIZE = W
```

* Sliding FIFO buffer
* Required for all metric computation

---

## 2.3 Network Size Constraint

* Number of nodes **N MUST remain constant per run**
* All matrices are fixed size **N × N**
* Dynamic resizing is forbidden

If nodes become inactive:

* They remain represented (zero rows/columns)
* Encoding dimensionality must not change

---

# 3. Node Specification

## 3.1 Responsibilities

Each node:

* reads IMU (or simulated motion)
* estimates proximity (RSSI or synthetic)
* emits packet per tick

---

## 3.2 Packet Format

```json
{
  "node_id": int,
  "timestamp": int,
  "neighbors": [id_1, id_2]
}
```

---

## 3.3 Output Constraints

* fixed rate aligned with tick
* no buffering
* timestamps MUST be monotonic

---

# 4. Hub (Ingress Layer)

## 4.1 Responsibilities

* receive packets from nodes
* forward packets to Reconstruction Layer

---

## 4.2 Constraints

* MUST NOT construct snapshots
* MUST NOT reorder packets intentionally
* MUST NOT interpret semantics

---

# 4.5 Reconstruction Layer (MANDATORY FOR PHASE 3+)

## 4.5.1 Purpose

Transforms asynchronous packets into **temporally consistent snapshots**.

Ensures:

```
S_t corresponds to a single time τ
```

---

## 4.5.2 Inputs

* unordered packets
* each packet includes timestamp

---

## 4.5.3 Outputs

```
S_t = { "adj": A_t }
```

---

## 4.5.4 Processing Pipeline

### 1. Jitter Buffer

* packets stored by timestamp τ

### 2. Temporal Gating

Evaluate only:

```
τ = t - Δ
```

Where:

```
Δ = maximum network delay
```

---

### 3. Coverage Validation

```
coverage = received_nodes / N
```

Constraint:

```
coverage ≥ θ
```

Where:

```
θ ∈ [0.6, 0.8]
```

---

### 4. Snapshot Validity Rule

A snapshot is valid iff:

```
all packets share timestamp τ
AND
coverage ≥ θ
```

Otherwise:

* snapshot MUST be discarded
* no interpolation allowed

---

## 4.5.5 Constraints

* no temporal mixing
* no interpolation
* bounded memory
* deterministic processing

---

# 5. Graph Builder

## 5.1 Input

* valid packet set at time τ

---

## 5.2 Output

```
A_t[i,j] = 1 if j ∈ neighbors_i else 0
```

---

## 5.3 Constraints

* symmetric
* binary
* diagonal = 0
* fixed N × N

---

# 6. Encoder

## 6.1 Input

```
window = [S_{t-W}, ..., S_t]
```

---

## 6.2 Implementation

```python
def encode_sequence(window):
    tokens = []
    for S_t in window:
        tokens.append(S_t["adj"].flatten(order="C"))
    return np.concatenate(tokens)
```

---

## 6.3 Constraints

* deterministic
* order-consistent
* structure-preserving

---

# 7. Metric Engine

## 7.1 Outputs

```
L_star, delta_L, r_eff
```

---

## 7.2 Computation

### Symbolic Length

```
L_sym = len(zlib.compress(encoded.tobytes()))
```

---

### Latent Length

Entropy-based estimator

---

### Minimum Description Length

```
L_star = min(L_sym, L_lat)
```

---

### Rate

```
r_eff = ema(diff(L_star), alpha=0.2)
```

---

### Normalization

```
E_O = 1 - (L_star / (W*N*N))
```

---

# 8. Control Engine

## 8.1 Policy

```
if L_star high → increase coupling
if L_star low → increase noise
```

---

# 9. Data Storage

JSONL streams

---

# 10. Interfaces

* Node → Hub: JSON
* Hub → Engine: queue/memory

---

# 11. Phase Mapping

| Phase   | Components               |
| ------- | ------------------------ |
| Phase 1 | Encoder + Metric         |
| Phase 2 | + Graph Builder          |
| Phase 3 | + Nodes + Reconstruction |
| Phase 4 | + Hardware               |
| Phase 5 | + Control                |

---

# 12. Failure Modes

| Failure        | Cause           |
| -------------- | --------------- |
| inflated L*    | temporal mixing |
| chaotic r_eff  | async aliasing  |
| observer drift | low coverage    |
| flat metrics   | no structure    |

---

# 13. Non-Goals

* no ML
* no adaptive encoding

---

# 14. Status

v1.2 — distribution-safe, reconstruction-complete

---

# CIO CONSTRAINTS (cio-constraints.md)

---

## 1. Dimensional Invariance

* N constant
* adjacency size fixed

---

## 2. Temporal Coherence

Snapshots MUST satisfy:

* single timestamp
* no mixing

---

## 3. Observability Constraint

```
coverage ≥ θ
```

---

## 4. Encoding Invariance

* flatten order fixed
* deterministic

---

## 5. Bounded Computation

* no unbounded memory
* no adaptive models

---

## 6. Observer Validity

Observer input MUST be:

* temporally coherent
* sufficiently observed

---

## 7. Forbidden Operations

* interpolation
* temporal averaging across timestamps
* dynamic resizing

---

## 8. CIO Validity Condition

CIO metrics are valid iff:

* reconstruction constraints satisfied
* encoding constraints satisfied

---

## 9. Failure Interpretation

If constraints violated:

* L* reflects reconstruction error
* r_eff becomes non-physical

---

## 10. Summary

CIO requires:

* fixed dimension
* temporal coherence
* bounded observation

---
