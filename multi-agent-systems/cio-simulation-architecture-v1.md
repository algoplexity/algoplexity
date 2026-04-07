# CIO End-to-End Simulation Architecture + Interface Contracts (Corrected v2)

## 1. Purpose

This document defines a complete, end-to-end simulation architecture for the Collective Intelligence Observatory (CIO), spanning:

* TinkerCAD (hardware design)
* Wokwi (firmware simulation)
* Network simulation (interaction field)
* Realtime engine (metric computation)
* Actuation / Control layer
* Experiment and validation layers

It also defines explicit **interface contracts** between layers to ensure:

* reproducibility
* traceability
* hardware–software equivalence
* cybernetic regulation (closed-loop control)

---

## 2. End-to-End CIO Simulation Pipeline

### 2.1 System Flow

```
[TinkerCAD]
    ↓ hardware_config
[Wokwi]
    ↓ node_stream
[Network Simulation]
    ↓ interaction_graph
[Realtime Engine]
    ↓ metrics_stream
[Actuation / Control Layer]
    ↓ control_signal
[Experiment Layer]
    ↓ validated_results
```

---

## 3. Layer Definitions

### 3.1 TinkerCAD — Physical Layer

Function:

* Define circuit design (ESP32 + IMU)
* Define sampling rate and pin mapping

Output:

```
hardware_config = {
  node_type,
  imu_type,
  sampling_rate_hz,
  pin_map,
  power_profile
}
```

---

### 3.2 Wokwi — Firmware & Signal Layer

Function:

* Simulate node firmware
* Generate motion signals and BLE packet emissions

Output:

```
node_stream_i(t) = {
  timestamp,
  sequence_id,
  node_id,
  imu: {ax, ay, az, gx, gy, gz},
  packet_emit: true
}
```

---

### 3.3 Network Simulation — Interaction Field

Function:

* Convert packet emissions into interaction graph
* Simulate RSSI, latency, packet loss

Output:

```
interaction_frame(t) = {
  timestamp,
  edges: [
    {i, j, rssi, latency, packet_loss}
  ]
}
```

Derived:

```
G(t) = adjacency matrix
W_ij(t) = normalized symmetric RSSI weights
```

---

### 3.4 Realtime Engine — CIO Core

Function:

* Construct G(t) and M(t)
* Compute CIO metrics

Input:

* node_stream
* interaction_frame

Output:

```
M(t) = {X_i(t) for all agents i}
X_i(t) = [ax, ay, az, gx, gy, gz]

s_t = {
  timestamp,
  G,
  M_features = M(t),
  E_O,
  E_dir = D(M(t)),
  I_edges,
  E_O_total
}
```

Where:

```
E_O_total = E_O - α·E_dir  # Corrected fusion: penalize instability
```

---

### 3.5 Actuation / Control Layer

Function:

* Classify coordination state
* Generate control signals
* Close cybernetic loop

Input:

* metrics_stream (s_t)

Output:

```
state_t = classify(E_O, E_dir)

control_signal = control_policy(state_t)
```

Example states and actions:

| State                | Condition           | Action               |
| -------------------- | ------------------- | -------------------- |
| True coordination    | E_O high, E_dir low | maintain interaction |
| Fragile coordination | E_O low, E_dir high | reduce coupling      |
| False coordination   | E_O low, E_dir low  | induce interaction   |
| Chaos                | E_dir very high     | dampen system        |

Control signal can drive LEDs, haptic cues, or experiment feedback.

---

### 3.6 Experiment Layer

Function:

* Run trials and perturbations
* Validate system behavior

Output:

```
experiment_run = {
  run_id,
  conditions,
  time_series: [s_t],
  state_series: [state_t],
  control_signals: [control_signal],
  summary_metrics
}
```

---

## 4. Interface Contracts

### 4.1 Contract A — TinkerCAD → Wokwi

File: `hardware_config.json`

Schema:

```
{
  "node_type": "ESP32",
  "imu_type": "MPU6050",
  "sampling_rate_hz": 50,
  "pin_map": {"sda": 21, "scl": 22},
  "power_profile": {"battery": "3.7V LiPo"}
}
```

Guarantees:

* Firmware uses correct pins
* Sampling rates consistent

---

### 4.2 Contract B — Wokwi → Network Simulation

File: `node_stream.jsonl`

Schema:

```
{
  "timestamp": 1710000000,
  "sequence_id": 1,
  "node_id": "node_1",
  "imu": {"ax":0.1,"ay":0.0,"az":9.8,"gx":0.01,"gy":0.02,"gz":0.00},
  "packet_emit": true
}
```

Guarantees:

* Consistent time base
* Unique node identifiers
* Sequence for packet ordering

---

### 4.3 Contract C — Network Simulation → Realtime Engine

File: `interaction_frame.jsonl`

Schema:

```
{
  "timestamp": 1710000000,
  "edges": [
    {"i":"node_1","j":"node_2","rssi":-60,"latency":10,"packet_loss":0}
  ]
}
```

Guarantees:

* Symmetric edge representation: W_ij = W_ji
* Normalized timestamps

---

### 4.4 Contract D — Realtime Engine → Experiment Layer

File: `metrics_stream.jsonl`

Schema:

```
{
  "timestamp": 1710000000,
  "E_O": 0.75,
  "E_dir": 0.12,
  "E_O_total": 0.73,
  "state_class": "True coordination"
}
```

Guarantees:

* Metrics aligned with theoretical expectations
* State classification included for cybernetic loop

---

## 5. CIO Metric Definitions (Corrected)

### 5.1 Structural Coordination (E_O)

Derived from G(t) using graph compressibility / entropy proxy.

### 5.2 Directional Instability (E_dir)

Derived from motion field M(t) using variance / divergence of motion vectors.

### 5.3 Local Causal Contribution (I(G,e))

Change in E_O when edge e is perturbed.

### 5.4 Total Coordination (E_O_total)

```
E_O_total = E_O - α·E_dir  # penalize instability
```

---

## 6. Validation Loop

1. Simulate signals (Wokwi + network)
2. Construct interaction graph G(t)
3. Compute metrics s_t
4. Classify coordination state state_t
5. Apply control / perturbations
6. Measure changes in metrics and system response
7. Compare with theoretical predictions from Paper A/B
8. Adjust parameters as needed for repeatability

---

## 7. Key Insight

The CIO simulation architecture is a **coordinated multi-tool system** with strict interface contracts, now including a **cybernetic control loop**. It ensures:

* Metrics reflect both motion and interaction structure
* Closed-loop regulation can be tested in simulation
* Reproducibility, traceability, and hardware–software equivalence are enforced

---

End of document.
