# 🚀 CIO Simulation Architecture — Canonical Specification

## 1️⃣ Core Principle

> **System Purpose:** Estimate the **compressibility of multi-agent interaction dynamics** and control them via closed-loop cybernetics.

**Key objects:**

* `interaction_sequence` → sequence of agent interactions and motion
* `L_sym`, `L_latent` → dual representation description lengths
* `L*` → minimum MDL estimate
* `ΔL` → regime/change signal
* `r_eff` → effective rate of compressible structure
* `I(G,e)` → causal edge contribution
* Control loop → modulates motion & network coupling

---

## 2️⃣ Time & Synchronization

* **Authoritative clock:** tick index `t = 0,1,2,...`
* **Timestep:** `dt = 20 ms` (50 Hz)
* **Sliding window:** W = 50 ticks (1 s)
* **Causal loop:** control applied at `t+1` based on `L*` and `ΔL`

**Implication:** deterministic, reproducible, fully discrete-time system.

---

## 3️⃣ Node Definition

* **Nodes:** N agents, fixed per experiment
* **Inputs:** IMU signals `[ax, ay, az, gx, gy, gz]`, `packet_emit`
* **Determinism:** seeded RNG for motion/noise reproducibility
* **Optional perturbations:** node dropout, noise injection, enforced coupling

---

## 4️⃣ Interaction Graph

* **Graph G(t):** constructed from emissions & RSSI mapping
* **Edge rules:** symmetric, no self-loops, optional weight attenuation
* **Windowed graph:** `G_window(t) = aggregate(G_{t-W:t})`
* **Output:** adjacency list/weighted graph → feeds observer

---

## 5️⃣ Dual-Path Observer

### 5.1 Symbolic Representation (L_sym)

* Compress adjacency sequence using **Lempel-Ziv** or block entropy
* Output: `L_sym(t)` = symbolic MDL estimate

### 5.2 Latent Representation (L_latent)

* Predictive model (Markov / small autoregressive MLP) → log-likelihood of sequence
* Output: `L_latent(t)` = latent MDL estimate

### 5.3 MDL Selection

```text
L*(t) = min(L_sym(t), L_latent(t))
ΔL(t) = |L_sym(t) - L_latent(t)|
```

* `L*` → primary system observable
* `ΔL` → regime-change detector

---

## 6️⃣ Effective Rate

```text
r_eff(t) = (L*(t) - L*(t-W)) / (W * N)
```

* Measures **rate of new compressible structure**
* Normalization ensures **cross-experiment comparability**

---

## 7️⃣ Causal Contribution

```text
I(G,e) = L*(G_window) - L*(G_window \ e)
```

* Quantifies **edge importance** in maintaining compressibility
* Replaces heuristic local metrics

---

## 8️⃣ Control Layer

* **Signal:** `control_signal ∈ [0,1]`
* **Effect mapping:**

| control_signal | motion_noise | coupling_strength |
| -------------- | ------------ | ----------------- |
| 0.0            | 0            | 1                 |
| 0.5            | 0.5          | 0.5               |
| 1.0            | 1            | 0                 |

* **Goal:** move system along `L* / r_eff` landscape
* **Timing:** applied at `t+1`
* **Target:** motion field + network interactions

---

## 9️⃣ Simulation Pipeline

```
for t in ticks:
    update node_stream
    build interaction_frame
    aggregate sliding window buffers (motion + G_window)
    compute L_sym, L_latent → L*, ΔL
    compute r_eff
    classify regime (via ΔL / r_eff)
    compute I(G,e) if needed
    generate control_signal
    apply control at t+1
```

---

## 🔟 Edge-Case Handling

* **Disconnected graph:** sum_degrees = 0 → E_O = 0
* **Near-zero motion vectors:** exclude from E_dir / L computation
* **ΔL spikes:** validated against synthetic scenario generator

---

## 11️⃣ Synthetic Scenario Generator (for Validation)

* **Aligned motion:** low L*, low ΔL
* **Random motion:** high L*, low coordination
* **Clustered structures:** intermediate L*, observable ΔL

**Purpose:** verify observer correctness **before hardware simulation**

---

## 12️⃣ Integration Sequence

1. **Realtime Engine (Python module)** → canonical metric generator
2. **Synthetic scenario generator** → stress-test observer
3. **Network simulation** → edge construction, RSSI → interaction_frame
4. **Wokwi** → firmware signal generation, synchronized tick injection
5. **TinkerCAD** → hardware mapping, validation layer

> **Core idea:** start from **observer correctness**, then propagate constraints outward

---

## 13️⃣ Data Streams

* `node_stream.jsonl` → raw IMU + emit events
* `interaction_frame.jsonl` → windowed graph edges + weights
* `metrics_stream.jsonl` → L*, ΔL, r_eff, regime classification
* Optional: `I_stream.jsonl` → causal contributions

---

# ✅ Key Properties

* **Deterministic, reproducible**
* **Bounded metrics**
* **Causal control loop**
* **Theory-aligned:** compressibility → effective rate → causal contribution → control
* **Scalable & modular**: observer can swap latent model without breaking pipeline

---

This specification can serve as a **canonical reference** for all team members: theoretical alignment, simulation fidelity, and hardware mapping all documented in one place.

---
