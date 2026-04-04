# CIO System Specification

The **Collective Intelligence Observatory (CIO)** is a cyber-physical system designed to provide real-time estimation and post hoc causal analysis of collective intelligence in multi-agent systems.

The CIO serves as a bounded observer that implements the theoretical constructs defined in [`theory/core-definitions.md`](../theory/core-definitions.md) and [`theory/observer-model.md`](../theory/observer-model.md), while explicitly separating real-time measurement from offline causal inference.

---

## 1. Observer Realization

The CIO instantiates the observer function:

$$
x_t = O(A_1(t), A_2(t), \dots, A_n(t))
$$

using sensor-derived interaction data. In the minimal configuration, agents are equipped with:

- **Inertial Measurement Units (IMU)** — motion dynamics
- **Proximity sensing (e.g., RSSI via BLE)** — interaction inference

These signals are transformed into a quantized, time-indexed representation:

$$
x_t \in \mathcal{X}
$$

where $\mathcal{X}$ is a finite encoding space determined by the observer resolution parameter $B(t)$.

---

## 2. Real-Time Complexity Estimation (L-Level)

Due to computational constraints, the CIO employs a bounded estimator:

$$
\hat{K}_{L}(x_t)
$$

implemented via lossless compression proxies (e.g., Lempel–Ziv). This enables real-time computation of:

- Coordination Energy $E_O(t)$
- Cost of Autonomy $C_{auto}(t)$

These quantities provide **instantaneous estimates of structural coordination**, enabling continuous monitoring of system dynamics.

---

## 3. Adaptive Observer Resolution

The observer dynamically adjusts its resolution $B(t)$ to satisfy constraints analogous to Ashby's Law of Requisite Variety (see [`theory/observer-model.md §4`](../theory/observer-model.md#4-ashy-s-law-of-requisite-variety)):

- High-variance regimes are captured with finer granularity
- Low-variance regimes are compressed efficiently

This maintains sensitivity to **phase transitions** without exceeding computational limits.

---

## 4. Cybernetic Feedback Loop

The CIO operates as a **second-order cybernetic system**, embedding measurement within a feedback loop:

1. **Sensing** — acquisition of interaction data from IMU and BLE sensors
2. **Encoding** — construction of $x_t$ via observer $O$
3. **Estimation** — computation of $\hat{K}_L(x_t)$, $E_O(t)$, $C_{auto}(t)$
4. **Evaluation** — projection into phase space (see [`theory/phase-space.md`](../theory/phase-space.md))
5. **Actuation** — visual or environmental feedback to agents

Rather than enforcing direct control, the system performs **information-theoretic steering**: modulating the informational environment perceived by agents to encourage self-organization toward regimes of lower coordination energy with preserved autonomy.

---

## 5. Symbolic Emission and Data Contract

To enable higher-fidelity offline analysis, the CIO emits an uncompressed symbolic sequence:

$$
s_t = \left[ t,\; B(t),\; O_{B(t)}(A_1(t), \dots, A_n(t)) \right]
$$

forming the stream:

$$
\Sigma_{CIO} = \{s_t\}
$$

This sequence is explicitly **decoupled from real-time approximations**, preserving the full informational structure required for H-level offline inference.

---

## 6. Causal Inference Layer (H-Level)

The emitted stream $\Sigma_{CIO}$ supports offline computation of higher-order quantities:

- Algorithmic complexity $\hat{K}_H(x_t)$ via Neural BDM
- Directional coordination $E_{dir}(t)$ (see [`theory/core-definitions.md §3`](../theory/core-definitions.md#3-directional-coordination-energy-e_dir))
- Information contribution $I(G,e)$ (see [`theory/core-definitions.md §4`](../theory/core-definitions.md#4-information-contribution-ig-e))

Using perturbation analysis, the system identifies:

- Causally significant interactions
- Generative substructures
- Independent coordination regimes

This establishes a **causal decomposition** of collective intelligence, complementing real-time observability.

---

## 7. Separation of Concerns: L-Level vs H-Level

A central design principle of the CIO is the strict separation between:

| Level | Mode | Estimator | Purpose |
| :--- | :--- | :--- | :--- |
| **L-Level** | Real-time, bounded | $\hat{K}_L$ (Lempel–Ziv) | Continuous monitoring, feedback |
| **H-Level** | Offline, unbounded | $\hat{K}_H$ (Neural BDM) | Causal inference, scientific validation |

This ensures that computational constraints do not limit the validity of scientific inference, while still enabling real-time interaction and feedback.

---

## 8. Hardware Summary

| Component | Role |
| :--- | :--- |
| IMU (per agent) | Motion dynamics — acceleration, orientation |
| BLE proximity sensor (per agent) | Pairwise RSSI-based proximity |
| Central hub (Raspberry Pi or equivalent) | Data aggregation, L-level computation |
| Offline compute node | H-level Neural BDM, causal analysis |

For the full experimental validation protocol built on this hardware, see [`cio-validation-protocol.md`](cio-validation-protocol.md).
