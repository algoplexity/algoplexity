# CIO Simulation Architecture — Original Component Overview

## 1. Overview

The CIO simulation system is designed to **span multiple platforms** to emulate both cyber-physical and agentic AI dynamics:

* **TinkerCAD:** low-fidelity, quick prototyping of embedded circuits and node interconnections
* **Wokwi:** high-fidelity simulation of microcontrollers (Arduino, ESP32) and sensor/actuator behavior
* **In-memory Python/JS simulation:** fast experimentation with motion fields, interaction graphs, and cybernetic control loops
* **Optional hybrid setup:** integrate hardware-in-the-loop (physical nodes) with simulated network

The system emphasizes **coupled motion and network dynamics**, with metrics (`E_O`, `E_dir`) and a **control loop** acting as a cybernetic regulator.

---

## 2. Node Layer

* **Node Types:** ESP32/Arduino clones in Wokwi, or virtual agents in Python
* **Sensors:** IMU (3-axis accelerometer + gyro), BLE radios for interaction
* **Outputs:** motion state, BLE packet emission, optional LEDs for debugging
* **Node Count:** configurable, scalable (3–10 nodes typical)

**Node responsibilities:**

1. Sense local motion (IMU) → `M(t)`
2. Emit packets (with RSSI) → interaction graph edges
3. Respond to control signals → motion perturbation or coupling

---

## 3. Motion Field Layer (M(t))

* **Function:** capture the kinematic state of all nodes

* **Platforms:**

  * TinkerCAD: simplified accelerometer simulation for initial proof-of-concept
  * Wokwi: accurate IMU emulation with realistic noise
  * Python/JS: can generate synthetic motion patterns (aligned, random, clustered)

* **Features computed:**

  * Mean motion vector
  * Variance across nodes
  * Pairwise alignment (cosine similarity)

* **Sliding Window:** optional, 50 ticks (1 s)

---

## 4. Interaction Graph Layer (G(t))

* **Nodes → Graph:** edges represent detectable interactions
* **RSSI → Weight Mapping:** -30 dBm → 1.0, -90 dBm → 0.0, clamped
* **Edge Formation:** deterministic, exists if **both nodes emit in the same timestep**
* **Graph Properties:** undirected, symmetric, no self-loops
* **Platforms:**

  * Wokwi: BLE packet simulation
  * Python/JS: synthetic RSSI-based adjacency

---

## 5. Metrics Layer (CIO Metrics)

* **Structural Coordination (E_O):** normalized graph entropy from node degree distribution

```text
p_i = degree(node_i) / sum_j degree(node_j)
E_O = - Σ p_i log(p_i) / log(N)
```

* **Directional Instability (E_dir):** pairwise cosine divergence of motion vectors

```text
E_dir = 1 - (2 / (N*(N-1))) Σ_{i<j} cos(theta_ij)
```

* **Fusion:** `E_O_total = E_O - alpha * E_dir`
* **Local Causal Contribution (I(G,e))**: edge removal analysis, optional online/offline

---

## 6. Control Layer

* **Signal:** `control_signal ∈ [0,1]`
* **Physical Effects:** maps to

```text
noise_strength = control_signal
coupling_strength = 1 - control_signal
```

* **Timing:** applied at t+1
* **Platforms:** Wokwi nodes, Python/JS simulation, optional hardware-in-the-loop

---

## 7. State Classification

* **5 states:** `stable`, `fragile`, `false_coordination`, `chaos`, `transitional`
* **Rule-based mapping:** E_O and E_dir thresholds
* **Function:** feeds control loop decisions

---

## 8. Experiment Layer

* **Perturbations supported:** motion noise, RSSI degradation, node dropout, forced synchronization pulses
* **Validation metrics:** recovery dynamics, metric trajectories
* **Initial motion patterns:** aligned, random, clustered

---

## 9. Data Flow Across Platforms

```
Node IMU + BLE -> Node Stream
         |                        \
         v                         \
     Motion Field (M(t))           \
         |                          \
         v                           v
Interaction Graph (G(t)) -> Metrics (E_O, E_dir, E_O_total) -> State Classification -> Control -> Nodes
```

* **TinkerCAD:** low-fidelity embedded tests
* **Wokwi:** high-fidelity node + network simulation
* **Python/JS:** offline simulation, graph & motion aggregation, control loop

---

## 10. Logging / Output

* `node_stream.jsonl` → per-timestep node data
* `interaction_frame.jsonl` → adjacency edges + weights
* `metrics_stream.jsonl` → E_O, E_dir, E_O_total, state, control signal

*Supports cross-platform reproducibility and validation.*


