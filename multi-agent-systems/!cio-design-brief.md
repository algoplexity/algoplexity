# 📄 CYBN8001 Design Brief

## **Project Title:**

**Cybernetic Intelligence Observatory (CIO): A Bounded Observer for Measuring Collective Intelligence**

---

## 1. Project Overview

The **Cybernetic Intelligence Observatory (CIO)** is a cyber-physical system (CPS) that measures and regulates collective behaviour using **minimum description length (MDL)** under a **bounded observer**.

Unlike conventional multi-agent systems that optimise predefined rewards, the CIO defines the system state **exclusively** as:

[
L^* = \min(L_{sym}, L_{lat})
]

Where:

* (L_{sym}): symbolic description length (compression-based)
* (L_{lat}): latent description length (predictive model-based)

All measurements are performed under an explicitly defined observer:

[
O = (\phi, M, B)
]

* ( \phi ): deterministic encoding of system state
* ( M ): bounded model class (dual description)
* ( B ): strict computational budget

Derived observables:

* **Observer-relative entropy**
  [
  E_O = 1 - \frac{L^*}{L_{max}}
  ]

* **Representation disagreement (mesoscopic signal)**
  [
  \Delta L = |L_{sym} - L_{lat}|
  ]

* **Structural change rate**
  [
  r_{eff} = EMA(\Delta L)
  ]

This system treats **collective intelligence as compressibility**, not performance.

---

## 2. Research Foundation (Traceability to Prior Work)

The CIO directly instantiates three prior research stages:

### Theory (Paper A)

* Collective intelligence = **observer-relative compressibility**
* System state defined only by (L^*)

### Computation (Paper B)

* Formal observer:
  [
  O = (\phi, M, B)
  ]
* Dual-description requirement:

  * symbolic and latent models share encoding and budget

### Measurement (Paper C)

Defines measurable quantities:

* (L^*), (E_O), (\Delta L), (r_{eff})

---

## 3. System Architecture (CPS Implementation)

The CIO integrates **sensing, computation, and actuation** through a strict data pipeline:

### 3.1 Physical Sensing Layer

Each agent (ESP32 node) emits:

```json
S_t = {
  "motion": [ax, ay, az]
}
```

This represents **raw physical movement**.

---

### 3.2 Geometry → Structure Mapping

Physical motion is transformed into a **binary interaction graph**:

[
S_t \rightarrow A_t \in {0,1}^{N \times N}
]

Implementation details:

* fixed dimension (N = 5)
* symmetric adjacency matrix
* edges defined by spatial proximity threshold

This step converts **continuous physics into discrete symbolic structure**, enabling MDL computation.

---

### 3.3 Encoding Function (φ)

The observer uses a deterministic encoding:

```python
encode_sequence(window):
    return concatenate(A_t.flatten(order="C"))
```

Constraints:

* fixed ordering
* no stochastic preprocessing
* identical across all observers

---

### 3.4 Formal Bounded Observer

The system explicitly implements:

[
O_{hub} = (\phi, M, B)
]

#### Budget (B)

```python
OBSERVER_BUDGET = {
    context_length_W = 50,
    max_dim_N = 5,
    L_MAX_bytes = 156.25,
    markov_model_cost_bytes = 16.0
}
```

Constraints:

* fixed temporal window
* fixed spatial dimension
* explicit model cost penalty

---

### 3.5 Dual Description Models (M)

#### Model 1 — Symbolic (Compression)

[
L_{sym} = \text{zlib}(\phi(S_{t:t+W}))
]

* dictionary-based compression
* bounded by (L_{max})

---

#### Model 2 — Latent (Predictive)

[
L_{lat} = L(\text{data}|\text{Markov}) + L(\text{model})
]

Implementation:

* first-order Markov transition model
* Laplace smoothing for stability
* explicit model cost (16 bytes)

This ensures:

* structured representation
* bounded model complexity
* fair comparison with (L_{sym})

---

### 3.6 State and Observables

[
L^* = \min(L_{sym}, L_{lat})
]

[
E_O = 1 - \frac{L^*}{L_{max}}
]

[
\Delta L = |L_{sym} - L_{lat}|
]

[
r_{eff} = EMA(\Delta L)
]

**Critical Constraint:**
(L^*) is the **only system state variable**

---

### 3.7 Control Layer (Cybernetic Feedback)

Control law:

```
if L* increases → increase coupling
if L* decreases → increase noise
```

Implementation:

* agents receive:

  * alignment signal (mean direction)
  * chaos signal = (1 - E_O)

This enables:

* self-organisation
* adaptive behaviour
* operation near phase boundary

---

### 3.8 Actuation Layer

Agents respond via:

* motion adjustments
* visual indicators (LEDs)
* dynamic coupling behaviour

---

## 4. The Mesoscope (Observer Hierarchy)

The CIO functions as a **mesoscopic measurement instrument** by comparing observers:

### Real-Time Observer

[
O_{hub} = (\phi, M_{bounded}, B_{tight})
]

### Offline Reference Observer

[
O_{ref} = (\phi, M_{advanced}, B_{relaxed})
]

### Observer Divergence

[
D(X) = K_{O_{hub}}(X) - K_{O_{ref}}(X)
]

This reveals:

* observer bias
* hidden structure
* limits of real-time inference

---

## 5. Experimental Plan

### Phase 1 — Validation

* synthetic adjacency patterns
* verify:

  * aligned → low (L^*)
  * random → high (L^*)
  * clustered → intermediate

---

### Phase 2 — Simulation

* multi-agent dynamics
* observe phase transition using (r_{eff})

---

### Phase 3 — Hardware Deployment

* ESP32 + IMU nodes
* real-time data streaming

---

### Phase 4 — Closed-Loop Control

* apply feedback via (E_O)
* demonstrate emergent coordination

---

### Phase 5 — Perturbation Testing

* inject noise / faults
* measure recovery dynamics

---

## 6. Offline Causal Analysis (Post-Demo)

The system produces structured datasets enabling:

### Causal Contribution

[
I(G, e) = L^*(G \setminus e) - L^*(G)
]

Identifies:

* critical nodes
* structural dependencies

---

### Observer Comparison

* evaluate (O_{hub}) vs (O_{ref})
* quantify epistemic limits

---

### Phase Transition Detection

* use (r_{eff}) to locate critical points

---

## 7. Materials and Budget

| Component           | Cost (AUD) |
| ------------------- | ---------- |
| ESP32 (×5)          | ~$75       |
| IMU sensors         | ~$50       |
| LEDs + resistors    | ~$20       |
| Breadboard + wiring | ~$30       |
| Power supply        | ~$25       |

**Total:** ~$200 (within $250 limit)

---

## 8. Risks and Mitigation

| Risk             | Mitigation                 |
| ---------------- | -------------------------- |
| Sensor noise     | handled via MDL robustness |
| Network latency  | ZOH buffering              |
| Model bias       | dual-description + ΔL      |
| Hardware failure | modular nodes              |

---

## 9. Cybernetic Principles

The CIO explicitly implements:

* feedback control
* observer-relative measurement
* self-organisation
* emergence

Cycle:

**Sensing → Encoding → Observation → Control → Actuation → Environment**

---

## 10. Responsible Design

* low-power components
* reusable hardware
* modular architecture
* defined disassembly plan

---

## 11. Expected Outcomes

The system will demonstrate:

1. A working CPS integrating sensing, computation, and actuation
2. Real-time measurement of collective intelligence via MDL
3. Emergent coordination without central control logic
4. A mesoscopic instrument for scientific analysis

---

## 12. Conclusion

The Cybernetic Intelligence Observatory is:

> A **bounded observer embedded in a cyber-physical system**, measuring and regulating collective behaviour through minimum description length.

By enforcing strict observer constraints and dual-description modelling, the CIO transforms collective intelligence into a **measurable, controllable, and experimentally testable phenomenon**.

---

