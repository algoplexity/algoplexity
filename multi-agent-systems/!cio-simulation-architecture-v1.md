# CIO Simulation Architecture (IMPLEMENTATION SPEC v1.0)

## 0. Purpose

This document defines the **concrete system architecture** for implementing the Collective Intelligence Observatory (CIO) across:

* Simulation (Python)
* Firmware (Wokwi / ESP32)
* Hardware (TinkerCAD)
* Realtime processing

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

```text
Nodes → Hub → Realtime Engine → Metrics → Control → Nodes
```

---

## 1.2 Core Components

| Component      | Location               | Role                                 |
| -------------- | ---------------------- | ------------------------------------ |
| Node           | 02_WOKWI / 03_FIRMWARE | Generates motion + proximity signals |
| Hub            | 02_WOKWI / 03_FIRMWARE | Aggregates node data                 |
| Graph Builder  | 04_REALTIME_ENGINE     | Constructs G(t)                      |
| Encoder        | 04_REALTIME_ENGINE     | Produces encoded sequences           |
| Metric Engine  | 04_REALTIME_ENGINE     | Computes L*, ΔL, r_eff               |
| Control Engine | 05_ACTUATION           | Applies feedback                     |
| Data Store     | 09_DATA                | Persists streams                     |

---

# 2. Execution Model

## 2.1 Time

System operates in discrete ticks:

```text
tick = 100ms (configurable)
```

At each tick:

1. nodes emit state
2. hub aggregates
3. graph is built
4. window updated
5. metrics computed
6. control signal emitted

---

## 2.2 Windowing

Sliding window:

```python
WINDOW_SIZE = W  # e.g. 50 ticks
```

* FIFO buffer
* required for all metric computation

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
  "position": [x, y],          // optional (simulation)
  "motion": [ax, ay, az],
  "neighbors": [id_1, id_2]
}
```

---

## 3.3 Output Rate

* fixed rate (aligned with tick)
* no buffering on node

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
    "adj": A_t,      # NxN adjacency matrix
    "motion": M_t    # NxD motion matrix
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
A_t[i,j] = 1 if j in neighbors_i else 0
```

---

## 5.3 Constraints

* symmetric (enforced)
* binary (Phase 1–2)

---

# 6. Encoder

## 6.1 Input

Sliding window:

```python
window = [S_{t-W}, ..., S_t]
```

---

## 6.2 Output

Encoded sequence:

```python
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

### Latent Length (Phase-dependent)

```python
L_lat = model_estimate(encoded)
```

---

### Minimum Length

```python
L_star = min(L_sym, L_lat)
```

---

### Disagreement

```python
delta_L = abs(L_sym - L_lat)
```

---

### Rate

```python
r_eff = ema(diff(L_star_series), alpha=0.2)
```

---

# 8. Control Engine

## 8.1 Input

* L_star
* r_eff

---

## 8.2 Output

Control signals:

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

```text
09_DATA/
    raw/
    processed/
```

---

# 10. Interfaces

## 10.1 Node → Hub

* transport: serial / websocket / BLE (phase-dependent)
* format: JSON

---

## 10.2 Hub → Engine

* in-memory (Python) or message queue

---

## 10.3 Engine → Control

* direct function call (Phase 1–3)
* message-based (Phase 4+)

---

# 11. Phase Mapping

| Phase   | Active Components                        |
| ------- | ---------------------------------------- |
| Phase 1 | Encoder + Metric Engine (synthetic data) |
| Phase 2 | + Graph Builder                          |
| Phase 3 | + Node Simulation (Wokwi)                |
| Phase 4 | + Hardware (TinkerCAD)                   |
| Phase 5 | + Control Engine                         |
| Phase 6 | Full system                              |
| Phase 7 | + Offline analysis                       |

---

# 12. Failure Modes

| Failure        | Cause                 |
| -------------- | --------------------- |
| unstable r_eff | no smoothing          |
| noisy L*       | inconsistent encoding |
| desync         | timestamp mismatch    |
| flat metrics   | no structure in data  |

---

# 13. Non-Goals

* no ML training pipeline (runtime)
* no adaptive encoding
* no dynamic topology rules

---

# 14. Deliverables per Phase

## Phase 1

* working encoder
* L_sym computation
* synthetic sequences

---

## Phase 2

* graph builder
* scenario generator

---

## Phase 3

* Wokwi node simulation

---

## Phase 4

* TinkerCAD validation

---

## Phase 5+

* closed-loop control

---

# 15. Summary

This architecture defines:

* deterministic data pipeline
* modular components
* phase-aligned build order

All correctness constraints are enforced externally in `cio-constraints.md`.
