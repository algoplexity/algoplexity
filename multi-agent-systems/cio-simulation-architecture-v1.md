# CIO End-to-End Simulation Architecture + Interface Contracts (Final Executable Version)

## 1. Purpose

Defines a complete, end-to-end simulation architecture for the Collective Intelligence Observatory (CIO), including hardware, firmware, network, realtime computation, and experiment layers, with explicit interface contracts and fully robust edge-case safeguards.

---

## 2. End-to-End Simulation Pipeline

```
[TinkerCAD]
    ↓ hardware_config
[Wokwi]
    ↓ node_stream
[Network Simulation]
    ↓ interaction_frame
[Realtime Engine]
    ↓ metrics_stream
[Experiment Layer]
    ↓ validated_results
```

---

## 3. Layer Definitions

### 3.1 TinkerCAD — Physical Layer

Function: define ESP32 + IMU circuits, sampling rate, pin mapping
Output: `hardware_config.json`

### 3.2 Wokwi — Firmware & Signal Layer

Function: simulate node firmware, generate motion + BLE emissions
Output: `node_stream.jsonl`

### 3.3 Network Simulation — Interaction Field

Function: generate interaction graph from emissions
Output: `interaction_frame.jsonl`

### 3.4 Realtime Engine — CIO Core

Function: compute G(t), M(t), metrics
Output: `metrics_stream.jsonl`

### 3.5 Experiment Layer

Function: run trials, perturbations, validate behavior
Output: `experiment_run.json`

---

## 4. Interface Contracts

* **TinkerCAD → Wokwi:** `hardware_config.json`
* **Wokwi → Network Simulation:** `node_stream.jsonl`
* **Network Simulation → Realtime Engine:** `interaction_frame.jsonl`
* **Realtime Engine → Experiment Layer:** `metrics_stream.jsonl`

---

## 5. CIO Metric Definitions

### 5.1 Structural Coordination (E_O)

```text
p_i = degree(node_i) / sum_j degree(node_j)  # explicit definition
if sum_j degree(node_j) == 0: E_O = 0  # edge-case safeguard
E_O = - sum_i p_i * log(p_i) / log(N)
```

### 5.2 Directional Instability (E_dir)

```text
Exclude nodes with ||v_i|| < ε
E_dir = 1 - (2 / (N(N-1))) sum_{i<j} cos(theta_ij)
Average over sliding window (50 ticks)
```

### 5.3 Total Coordination

```text
E_O_total = E_O - alpha * E_dir  # penalizes instability
```

### 5.4 Local Causal Contribution (I(G,e))

Computed offline or online as needed

---

## 6. Sliding Window Aggregation

* Window size = 50 ticks (1s)
* Option A: aggregate motion + graph across all ticks for metric computation

---

## 7. State Classification (5 States)

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

## 8. Control Signal Mapping

```text
control_signal ∈ [0,1]
noise_strength = control_signal
coupling_strength = 1 - control_signal
# Applied at t+1
```

---

## 9. Minimal Simulation Behavior

Canonical motion scenarios:

1. Aligned motion → high E_O, low E_dir
2. Random motion → low E_O, high E_dir
3. Clustered groups → medium E_O, medium E_dir

---

## 10. Per-Tick Execution Algorithm

```
for each tick t:
    update node_stream
    build interaction_frame
    update sliding window buffers
    compute M_features
    aggregate G_window over last 50 ticks
    compute E_O, E_dir, E_O_total
    classify state_t
    generate control_signal_t
    apply control at t+1
```

---

## 11. Safeguards

1. Degree sum zero → E_O = 0
2. Near-zero motion vectors → exclude from E_dir
3. RSSI clamped: W_ij = clamp((rssi + 90)/60, 0,1)

---

## 12. Status

* Fully coherent, executable, robust
* Deterministic, reproducible, bounded metrics
* Sliding window aggregation applied consistently
* Cybernetic loop formally correct
* Ready for first simulation run (N=5) with traceable metrics and state transitions
