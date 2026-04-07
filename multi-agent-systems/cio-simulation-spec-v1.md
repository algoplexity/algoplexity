# CIO Simulation Specification Sheet — v1

---

## 1. Time & Synchronization

* **Authoritative clock:** simulation time (Wokwi) with wall-clock reference for logging.
* **Time model:** fixed timestep `dt = 20 ms` (50 Hz sampling).
* **Packet ordering:** sorted by timestamp; out-of-order packets buffered.
* **Clock drift handling:** negligible within simulation; real deployment may require drift correction.

## 2. Nodes & Determinism

* **Initial node count:** 5 nodes (scalable).
* **Node count:** fixed per experiment.
* **Determinism:** seeded RNG for reproducibility; RSSI and motion noise optional.

## 3. Motion Field M(t)

* **IMU interpretation:** raw 6D vectors `[ax, ay, az, gx, gy, gz]`.
* **Derived features (M_features):**

  * Mean motion vector per timestep
  * Variance across nodes
  * Pairwise alignment (cosine similarity)
* **Sampling window:** per timestep (instantaneous); optional 1s sliding window.

## 4. Interaction Graph G(t)

* **RSSI → weight mapping:** linear normalization

  * -30 dBm → 1.0
  * -90 dBm → 0.0
* **Edge conditions:**

  * Only exists if `packet_emit == true`
  * Symmetric edges (undirected)
  * No self-loops
* **Network realism:** abstracted; optional collision and spatial modeling.

## 5. CIO Metrics

### 5.1 Structural Coordination (E_O)

```
E_O(t) = - sum_i p_i log(p_i)
```

* `p_i` = normalized degree distribution of G(t) at timestep t
* Graph entropy proxy for compressibility

### 5.2 Directional Instability (E_dir)

```
E_dir(t) = 1 - (1/N) sum_{i<j} cos(theta_ij)
```

* `theta_ij` = angle between motion vectors of node i and j
* Value 0 → perfectly aligned, 1 → maximally divergent

### 5.3 Fusion (Total Coordination)

```
E_O_total = E_O + alpha * E_dir
```

* `alpha = 0.1` default, tunable per experiment

### 5.4 Local Causal Contribution I(G,e)

* Offline: remove edge e, recompute E_O, difference = I(G,e)
* Online feasible for small graphs (<10 nodes) but computationally expensive

## 6. State Classification

* **Thresholds:**

  * High E_O > 0.7, low E_O < 0.3
  * High E_dir > 0.5, low E_dir < 0.2
* **Function:** rule-based `classify(E_O, E_dir)` → {stable, fragile, false coordination}

## 7. Control Layer

* **Signal:** scalar `control_signal ∈ [0,1]`
* **Targets:** nodes (motion perturbation) or experiment parameters (e.g., enforced coupling)
* **Timing:** applied at next timestep

## 8. Experiment Layer

* **Perturbations supported:**

  * Motion noise injection
  * Network degradation (RSSI attenuation, packet loss)
  * Node dropout
  * Forced synchronization pulses
* **Validation:**

  * Recovery to stable coordination
  * Metric comparison to theoretical expectation

## 9. Data Strategy

* **Execution mode:** streaming in-memory with optional logging to JSONL
* **Compression:** optional, downsample to 25 Hz if needed
* **Streams:**

  * `node_stream.jsonl`: IMU + packet_emit
  * `interaction_frame.jsonl`: adjacency edges with weights
  * `metrics_stream.jsonl`: E_O, E_dir, E_O_total per timestep

---

This specification provides a **deterministic blueprint** for the CIO simulation pipeline and ensures reproducible, traceable metrics aligned with the cybernetic regulator concept.
