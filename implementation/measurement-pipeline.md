# Measurement Pipeline

The 6-step pipeline for measuring collective intelligence using the CIO. This document serves as the implementation reference; for the theoretical basis of each quantity, see [`theory/core-definitions.md`](../theory/core-definitions.md).

---

## Overview

Given a multi-agent system observed over time, the pipeline produces three complementary quantities:

| Quantity | Role |
| :--- | :--- |
| $E_O(t)$ | Structural coordination (see [`theory/core-definitions.md §2`](../theory/core-definitions.md#2-coordination-energy-e_o)) |
| $E_{dir}(t)$ | Temporal / causal structure (see [`theory/core-definitions.md §3`](../theory/core-definitions.md#3-directional-coordination-energy-e_dir)) |
| $I(G,e)$ | Local causal contribution (see [`theory/core-definitions.md §4`](../theory/core-definitions.md#4-information-contribution-ig-e)) |

---

## Step 1 — System Encoding

Construct an observer-dependent representation:

$$
x_t = O(A_1(t), A_2(t), \dots, A_n(t))
$$

where $x_t$ may be a string (symbolic trace), a graph (interaction network), or a tensor (spatiotemporal encoding). The observer resolution $B(t)$ is set according to Ashby's Law — see [`theory/observer-model.md`](../theory/observer-model.md).

**CIO hardware:** IMU motion vectors + BLE proximity → quantized time-indexed representation. See [`cio-system-spec.md §1`](cio-system-spec.md#1-observer-realization).

---

## Step 2 — Complexity Estimation

Estimate algorithmic complexity:

$$
\hat{K}(x_t) \approx K(x_t)
$$

| Mode | Estimator | Use case |
| :--- | :--- | :--- |
| L-Level (real-time) | Lempel–Ziv / compression proxy | Continuous monitoring |
| H-Level (offline) | Neural BDM | Causal decomposition, scientific validation |

See [`cio-system-spec.md §2–3`](cio-system-spec.md#2-real-time-complexity-estimation-l-level) for the L/H separation.

---

## Step 3 — Structural Measurement

Compute Coordination Energy:

$$
E_O(t) = \sum_{i=1}^{n} \hat{K}\!\left(x_t^{(i)}\right) - \hat{K}(x_t)
$$

Project the current state into the phase space $(E_O, K_{joint}, C_{auto})$. See [`theory/phase-space.md`](../theory/phase-space.md) for regime interpretation.

---

## Step 4 — Temporal Measurement

Estimate Directional Coordination:

$$
E_{dir}(t) = \hat{K}(x_t) - \hat{K}(x_t \mid x_{t-1})
$$

High $E_{dir}$ indicates strong causal dependence on the prior state; low $E_{dir}$ indicates near-random temporal evolution.

---

## Step 5 — Causal Perturbation

For interaction-graph representations $G(t) = (V, E)$, compute the Information Contribution of each edge:

$$
I(G,e) = \hat{C}(G) - \hat{C}(G \setminus e)
$$

Apply the Generative Separation Criterion $I(G,e) > \log(2)$ to identify:

- Causally significant edges
- Independent generative components

---

## Step 6 — Generative Segmentation and Feedback

Apply the threshold to partition the interaction graph into causally independent substructures. Then:

1. Compute Composite Interaction Intelligence: $\mathcal{I}_{interaction} = \hat{E} \cdot \Phi \cdot F$
2. Evaluate encoding adequacy (low $\Phi$ despite apparent coordination → update observer $O_{t+1}$)
3. Emit symbolic sequence $s_t$ to H-level store (see [`cio-system-spec.md §5`](cio-system-spec.md#5-symbolic-emission-and-data-contract))
4. Actuate feedback to agents if operating in closed-loop mode

---

## Computational Considerations

- Exact perturbation analysis scales as $O(|E|)$ per timestep
- For large systems, approximate perturbation strategies are required
- L-level (Steps 1–4) must complete within one sensing interval; H-level (Step 5–6 full) runs offline

For the full experimental validation built on this pipeline, see [`cio-validation-protocol.md`](cio-validation-protocol.md).
