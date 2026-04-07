# CIO Simulation Architecture — Formal & Executable Version

---

## 1. Purpose

This document defines a **complete, end-to-end simulation architecture** for the Collective Intelligence Observatory (CIO). It includes:

* **High-level overview** of layers, data flows, and interface contracts
* **Formally specified metric computations** and sliding window aggregation
* **Control logic** for closed-loop experiments
* **Safeguards** for edge conditions

This specification is **formally executable** and reproducible.

---

## 2. End-to-End Architectural Overview (Context)

### 2.1 Layer Flow

```
[TinkerCAD: Physical Layer]
    ↓ hardware_config.json
[Wokwi: Firmware Layer]
    ↓ node_stream.jsonl
[Network Simulation: Interaction Field]
    ↓ interaction_frame.jsonl
[Realtime Engine: CIO Core]
    ↓ metrics_stream.jsonl
[Experiment Layer]
    ↓ validated_results
```

### 2.2 Layer Responsibilities & Outputs

**TinkerCAD (Physical Layer)**

* Defines circuits: ESP32 + IMU
* Output: `hardware_config.json`

**Wokwi (Firmware Layer)**

* Simulates node firmware, generates IMU readings and BLE packet emissions
* Output: `node_stream.jsonl` per tick

**Network Simulation (Interaction Field)**

* Converts node emissions → interaction graph G(t)
* Simulates RSSI, latency, packet loss
* Output: `interaction_frame.jsonl`

**Realtime Engine (CIO Core)**

* Aggregates motion and graph over sliding window
* Computes metrics `E_O`, `E_dir`, `E_O_total`
* Applies state classification
* Output: `metrics_stream.jsonl`

**Experiment Layer**

* Runs perturbations, validates system
* Output: structured `experiment_run` with trajectories and summary metrics

### 2.3 Interface Contracts (Original)

* `hardware_config.json` → ensures Wokwi uses correct pins, sampling rates
* `node_stream.jsonl` → timestamped IMU + packet emissions, unique node IDs
* `interaction_frame.jsonl` → symmetric edges, normalized timestamps
* `metrics_stream.jsonl` → metrics per tick: E_O, E_dir, E_O_total

---

## 3. Formal Metric Definitions (Finalized)

**3.1 Graph Entropy (Structural Coordination E_O)**

```text
p_i = degree(node_i) / sum_j degree(node_j)
E_O = - Σ_i p_i log(p_i) / log(N)
if sum_j degree(node_j) == 0: E_O = 0  # safeguard
```

**3.2 Directional Instability (E_dir)**

```text
E_dir = 1 - (2 / (N*(N-1))) Σ_{i<j} cos(θ_ij)
# exclude nodes with ||v_i|| < ε
```

**3.3 Fusion: Total Coordination**

```text
E_O_total = E_O - alpha * E_dir
```

**3.4 Sliding Window**

* Option A: aggregate motion & graph over last 50 ticks
* Metrics computed across window → reduces noise

**3.5 Control Signal Mapping**

```text
noise_strength = control_signal
coupling_strength = 1 - control_signal
control applied at t+1
```

**3.6 State Classification (5 states)**

```python
if E_dir > 0.7:
    state = 'chaos'
elif E_O > 0.7 and E_dir < 0.2:
    state = 'stable'
elif E_O < 0.3 and E_dir > 0.5:
    state = 'fragile'
elif E_O < 0.3 and E_dir < 0.2:
    state = 'false_coordination'
else:
    state = 'transitional'
```

---

## 4. Edge & Weight Rules

* Edge exists if **both nodes emit in same tick**
* Symmetric adjacency, no self-loops
* RSSI → weight:

```text
W_ij = clamp((rssi + 90)/60, 0, 1)
```

---

## 5. Safeguards

* `E_O = 0` if graph is fully disconnected
* Exclude near-zero vectors from `E_dir` to avoid NaN
* Tick index = authoritative clock; timestamp optional/logging

---

## 6. Execution Loop (Per Tick)

```text
for each tick t:
    update node_stream
    update sliding window buffers (motion + graph)
    compute M_features
    compute G_window(t)
    compute E_O, E_dir
    compute E_O_total
    classify state_t
    generate control_signal_t
    apply control at t+1
```

---

## 7. Minimal Simulation Scenarios (Canonical)

1. **Aligned motion** → high E_O, low E_dir
2. **Random motion** → low E_O, high E_dir
3. **Clustered groups** → medium E_O, medium E_dir

---

## 8. Summary

* Fully **formally specified and reproducible**
* Original high-level architecture **preserved for context**
* Metrics, control, and state spaces **internally consistent**
* Safe under **edge conditions**
* Ready for **first execution and experiment tracing**
