# CIO Simulation Architecture (IMPLEMENTATION SPEC v1.1)

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
Nodes → Hub → Graph Builder → Encoder → Metric Engine → Control → Nodes
```

---

## 1.2 Core Components

| Component      | Location         | Role                                 |
| -------------- | ---------------- | ------------------------------------ |
| Node           | Wokwi / Firmware | Generates motion + proximity signals |
| Hub            | Firmware         | Aggregates node data                 |
| Graph Builder  | Realtime Engine  | Constructs G(t)                      |
| Encoder        | Realtime Engine  | Produces encoded sequences           |
| Metric Engine  | Realtime Engine  | Computes L*, ΔL, r_eff               |
| Control Engine | Actuation        | Applies feedback                     |
| Data Store     | Data Layer       | Persists streams                     |

---

# 2. Execution Model

## 2.1 Time

System operates in discrete ticks:

```
tick = 100ms (configurable)
```

At each tick:

1. Nodes emit state
2. Hub aggregates
3. Graph is built
4. Window updated
5. Metrics computed
6. Control signal emitted

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
  "position": [x, y],
  "motion": [ax, ay, az],
  "neighbors": [id_1, id_2]
}
```

---

## 3.3 Output Rate

* fixed rate (aligned with tick)
* no buffering

---

# 4. Hub Specification

## 4.1 Responsibilities

* receive packets from all nodes
* synchronize by timestamp
* construct system snapshot S_t

---

## 4.2 Snapshot Format

```python
S_t = {
    "adj": A_t,
    "motion": M_t
}
```

---

## 4.3 Synchronization

* drop late packets
* no interpolation (Phase 1–3)

---

# 5. Graph Builder

## 5.1 Input

* node packets at time t

---

## 5.2 Output

Adjacency matrix:

```python
A_t[i, j] = 1 if j in neighbors_i else 0
```

---

## 5.3 Constraints

* symmetric (enforced)
* binary (Phase 1–2)
* diagonal MUST be zero: `A[i,i] = 0`
* size MUST remain N × N

---

# 6. Encoder

## 6.1 Input

```
window = [S_{t-W}, ..., S_t]
```

---

## 6.2 Output

```
encoded: np.ndarray
```

---

## 6.3 Implementation

```python
def encode_sequence(window):
    tokens = []
    for S_t in window:
        tokens.append(S_t["adj"].flatten(order="C"))
    return np.concatenate(tokens)
```

---

## 6.4 Encoder Constraints

* deterministic
* order-consistent
* structure-preserving
* flattening order MUST NOT change

Any modification invalidates comparability.

---

# 7. Metric Engine

## 7.1 Inputs

* encoded sequence
* previous L* values

---

## 7.2 Outputs

```python
{
  "L_star": float,
  "delta_L": float,
  "r_eff": float
}
```

---

## 7.3 Computation

### Symbolic Length

```python
L_sym = len(zlib.compress(encoded.tobytes()))
```

---

### Latent Length (Bounded Proxy)

```python
L_lat = latent_proxy_estimate(encoded)
```

#### Latent Estimator Constraints

The latent estimator MUST be:

* deterministic
* parameter-static
* bounded in compute and memory

Allowed:

* entropy estimators
* fixed predictive models

Forbidden:

* neural networks
* online learning
* adaptive models

---

### Minimum Description Length

```python
L_star = min(L_sym, L_lat)
```

---

### Disagreement

```python
delta_L = abs(L_sym - L_lat)
```

---

### Rate of Change

```python
r_eff = ema(diff(L_star_series), alpha=0.2)
```

---

## 7.4 Normalization

Let:

```
L_max = W × N × N
```

Then:

```
E_O = 1 - (L_star / L_max)
```

### Constraints

* must be fixed and theoretical
* no adaptive or rolling normalization
* must be invariant across runs

---

## 7.5 Causal Metrics (Offline Only)

Edge contribution:

```
I(G, e) = L*(G \ e) - L*(G)
```

Constraints:

* NOT computed in real-time
* restricted to Phase 6–7

Rationale:

* violates real-time compute bounds

---

# 8. Control Engine

## 8.1 Input

* L_star
* r_eff

---

## 8.2 Output

```python
{
  "noise_level": float,
  "coupling_strength": float
}
```

---

## 8.3 Policy

```python
if L_star > HIGH_THRESHOLD:
    increase coupling

if L_star < LOW_THRESHOLD:
    increase noise
```

---

# 9. Data Storage

## 9.1 Streams

Stored as JSONL:

* node_stream.jsonl
* interaction_frame.jsonl
* metrics.jsonl

---

## 9.2 Location

```
09_DATA/
    raw/
    processed/
```

---

# 10. Interfaces

## 10.1 Node → Hub

* transport: serial / websocket / BLE
* format: JSON

---

## 10.2 Hub → Engine

* in-memory or message queue

---

## 10.3 Engine → Control

* direct call (Phase 1–3)
* message-based (Phase 4+)

---

# 11. Phase Mapping

| Phase   | Active Components                        |
| ------- | ---------------------------------------- |
| Phase 1 | Encoder + Metric Engine (synthetic data) |
| Phase 2 | + Graph Builder                          |
| Phase 3 | + Node Simulation                        |
| Phase 4 | + Hardware                               |
| Phase 5 | + Control Engine                         |
| Phase 6 | Full system                              |
| Phase 7 | + Offline analysis                       |

---

# 12. Failure Modes

| Failure        | Cause                          |
| -------------- | ------------------------------ |
| unstable r_eff | no smoothing                   |
| noisy L*       | inconsistent encoding          |
| desync         | timestamp mismatch             |
| flat metrics   | no structure in data           |
| invalid L*     | adjacency constraint violation |

---

# 13. Non-Goals

* no ML training in runtime
* no adaptive encoding
* no dynamic topology resizing

---

# 14. Deliverables per Phase

## Phase 1

* working encoder
* L_sym computation
* latent proxy estimator
* synthetic adjacency matrices
* full pipeline validation

### Constraints

* encoder MUST be used
* direct encoded sequence generation is forbidden

---

## Phase 2

* graph builder
* structured scenarios

---

## Phase 3

* node simulation

---

## Phase 4

* hardware validation

---

## Phase 5+

* closed-loop control

---

# 15. Summary

This architecture defines:

* deterministic data pipeline
* bounded observer model
* fixed-dimensional encoding
* phase-aligned implementation

All correctness constraints are enforced externally in `cio-constraints.md`.

---

# ✅ Status

This version is now:

* implementation-complete
* constraint-aligned
* ambiguity-free
* safe for multi-developer execution

---

