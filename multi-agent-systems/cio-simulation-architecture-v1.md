# CIO End-to-End Simulation Architecture — Final Canvas (v1.0)

---

## 1. Purpose

This canvas defines the fully aligned, executable end-to-end simulation architecture for the **Collective Intelligence Observatory (CIO)**, including sliding window aggregation, per-tick execution, and closed-loop cybernetic control. It spans:

* **TinkerCAD** — hardware design and configuration
* **Wokwi** — firmware simulation and motion generation
* **Network Simulation** — interaction field and RSSI modeling
* **Realtime Engine** — metric computation and state classification
* **Experiment Layer** — trial execution, perturbations, and validation

It also specifies all **interface contracts**, **metric normalization**, and **control semantics**.

---

## 2. System Flow

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

**Function:** Define circuits (ESP32 + IMU), sampling rate, pin mapping, and battery.

**Output:** `hardware_config`

```json
{
  "node_type": "ESP32",
  "imu_type": "MPU6050",
  "sampling_rate_hz": 50,
  "pin_map": {"sda": 21, "scl": 22},
  "power_profile": {"battery": "3.7V LiPo"}
}
```

### 3.2 Wokwi — Firmware & Signal Layer

**Function:** Simulate firmware, generate motion signals, BLE packet emissions.

**Output:** `node_stream_i(t)`

```json
{
  "timestamp": 1710000000,
  "node_id": "node_1",
  "imu": {"ax":0.1,"ay":0.0,"az":9.8,"gx":0.01,"gy":0.02,"gz":0.00},
  "packet_emit": true
}
```

### 3.3 Network Simulation — Interaction Field

**Function:** Convert packet emissions into interaction graph, simulate RSSI, latency, packet loss.

**Output:** `interaction_frame(t)`

```json
{
  "timestamp": 1710000000,
  "edges": [
    {"i":"node_1","j":"node_2","rssi":-60,"latency":10}
  ]
}
```

*Weights:* `W_ij = clamp((rssi + 90)/60, 0, 1)`
*Edge rule:* edge exists if both nodes emit in the same timestep (Option A)

### 3.4 Realtime Engine — CIO Core

**Function:** Construct G(t), compute motion features M(t), calculate metrics.

**Input:** node_stream, interaction_frame

**Per-Tick Execution:**

```python
for each tick t:
    update node_stream
    build interaction_frame
    update sliding window buffers (50 ticks)
    compute M_features
    compute G(t)
    compute E_O = normalized graph entropy
    compute E_dir = pairwise cosine divergence
    compute E_O_total = E_O - alpha * E_dir
    classify state_t
    generate control_signal_t
    apply control at t+1
```

**Output:** `metrics_stream`

```json
{
  "timestamp": 1710000000,
  "E_O": 0.75,
  "E_dir": 0.12,
  "E_O_total": 0.70,
  "state_t": "stable",
  "control_signal_t": 0.3
}
```

### 3.5 Experiment Layer

**Function:** Run trials, perturbations, and validate system behavior.

**Output:** `experiment_run`

```json
{
  "run_id": "exp_001",
  "conditions": {...},
  "time_series": [metrics_stream],
  "summary_metrics": {...}
}
```

---

## 4. CIO Metric Definitions

| Metric    | Definition                                    | Notes                               |
| --------- | --------------------------------------------- | ----------------------------------- |
| E_O       | normalized graph entropy of G(t): H(p)/log(N) | 0 ordered → 1 disordered            |
| E_dir     | 1 - (2 / (N(N-1))) Σ_{i<j} cos(θ_ij)          | captures pairwise motion divergence |
| E_O_total | E_O_total = E_O - alpha * E_dir               | penalizes instability               |
| I(G,e)    | change in E_O if edge e is perturbed          | offline computation feasible        |

**State Classification (5 states):**

```python
if E_dir > 0.7:
    state = "chaos"
elif E_O > 0.7 and E_dir < 0.2:
    state = "stable"
elif E_O < 0.3 and E_dir > 0.5:
    state = "fragile"
elif E_O < 0.3 and E_dir < 0.2:
    state = "false_coordination"
else:
    state = "transitional"
```

**Control Signal Mapping:**

```
0.0 → no intervention
0.5 → moderate noise injection
1.0 → strong damping / enforced sync
```

---

## 5. Sliding Window

*Window size:* 50 ticks (1s)
*Aggregation method:* Option A — compute metrics using all samples in the window directly

---

## 6. Initial Motion Scenarios

1. Aligned motion → expect high E_O, low E_dir
2. Random motion → expect low E_O, high E_dir
3. Clustered groups → medium E_O, medium E_dir

---

## 7. Interface Contracts

* Contract A: TinkerCAD → Wokwi (`hardware_config.json`)
* Contract B: Wokwi → Network Simulation (`node_stream.jsonl`)
* Contract C: Network Simulation → Realtime Engine (`interaction_frame.jsonl`)
* Contract D: Realtime Engine → Experiment Layer (`metrics_stream.jsonl`)

---

## 8. Validation & Expected Output

* Metric trajectories should reflect canonical scenarios.
* State transitions should follow control logic.
* Fusion formula and sliding window guarantee bounded, stable E_O_total.

---

## 9. Status

* **Fully executable, unambiguous**
* **Closed-loop dynamical system**
* **Ready for first end-to-end simulation run (N=5)**
