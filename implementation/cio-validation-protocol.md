# CIO Experimental Validation Protocol

Empirical validation protocol for the Collective Intelligence Observatory (CIO). This document merges and supersedes `archive/CIO-validation.md` (v1) and `archive/CIO-Validation-v2.md` (v2).

---

## 1. Objective

Empirically validate that **Embodied Interaction Intelligence** can be captured from motion and proximity data using three core metrics grounded in algorithmic information theory:

1. **Normalised Coordination Index** ($\hat{E}$)
2. **Causal Density** ($\Phi$)
3. **Functional Alignment** ($F$)

with **adaptive input encoding** guided by second-order cybernetic principles and Neural BDM.

---

## 2. Participants and Hardware

- **Participants:** 2–4 individuals per session
- **Hardware:**
  - Wearable motion sensors (IMU)
  - BLE-based proximity sensors
  - Central hub (Raspberry Pi or equivalent) for data aggregation
- **Data Collected:**
  - Time-series motion vectors
  - Pairwise proximity signals
  - Optional: affective state, environmental context

Hardware specification is detailed in [`cio-system-spec.md`](cio-system-spec.md).

---

## 3. Core Metrics

| Metric | Definition | Interpretation |
| :--- | :--- | :--- |
| $\hat{E}(G_t)$ | Normalised coordination index | 0–1, where 1 = fully interdependent |
| $\Phi(G_t)$ | Fraction of causally meaningful edges | Proportion of interactions contributing $> \log(2)$ in Neural BDM |
| $F(G_t)$ | Functional alignment | Task performance or stability/recovery under perturbation |

**Normalised Coordination Index:**

$$
\hat{E}(G_t) = \max\!\left(0,\;\frac{L_\theta(G_{joint}) - \sum_i L_\theta(G_{individual}^{(i)})}{L_\theta(G_{joint})}\right)
$$

**Causal Density:**

$$
\Phi(G_t) = \frac{|\{e : I(G_t, e) > \log(2)\}|}{|E|}
$$

where $I(G,e)$ is the Information Contribution defined in [`theory/core-definitions.md §4`](../theory/core-definitions.md#4-information-contribution-ig-e).

**Composite Interaction Intelligence Score:**

$$
\mathcal{I}_{interaction}(G_t) = \hat{E}(G_t) \cdot \Phi(G_t) \cdot F(G_t)
$$

---

## 4. Research Hypotheses

| ID | Hypothesis |
| :--- | :--- |
| H1 | Aligned interactions → higher $\hat{E}$ |
| H2 | Structured interaction → higher $\Phi$ |
| H3 | Effective/stable interaction → higher $F$ |
| H4 | $\mathcal{I}_{interaction}^{Aligned} > \mathcal{I}_{interaction}^{Misaligned}$ |
| H5 | Non-obvious coordination yields moderate/high $\hat{E}$, high $\Phi$, high $F$ |

---

## 5. Adaptive Input Encoding (2nd-Order Cybernetic)

### 5.1 Deterministic Transduction

Analog signals → binary tapes via fixed lenses: first derivative, second derivative, or level bisection.

### 5.2 Algorithmic Causal Deconvolution (ACD)

Perturbation analysis separates independent interaction components:

$$
\Delta K(e) > \log(2) + \varepsilon \;\Rightarrow\; \text{edge is causally significant}
$$

Partitions interaction window $W_t$ into components $\{W_t^{(i)}\}$.

### 5.3 Minimal Generator Approximation

Each component is described by a minimal algorithmic generator $g_i^*$, reducing total description length.

### 5.4 Algorithmic Causal Graph Inference

Conditional Neural BDM loss determines causal edges:

$$
K_\theta(Y \mid X) < K_\theta(Y) \;\Rightarrow\; X \to Y
$$

### 5.5 Global Description Geometry

Normalised interaction complexity:

$$
\rho_t = \frac{K^*(W_t)}{|W_t| - 2}
$$

---

## 6. Experimental Conditions

| Condition | Description | Expected Metrics |
| :--- | :--- | :--- |
| A: Aligned | Shared goal | High $\hat{E}, \Phi, F$ |
| B: Misaligned | Conflicting/independent goals | Low metrics |
| C: Perturbed | Mid-task disruption | Drop then recovery |
| D: Edge Cases | Non-obvious coordination (e.g., bounding overwatch) | Moderate/high $\hat{E}$, high $\Phi, F$ |

---

## 7. Validation Phases

1. **Perturbation Consistency Test** — confirm separation of independent components
2. **Generator Recovery** — validate minimal generator per component
3. **Causal Discovery** — inferred DAG matches true causal dependencies
4. **Structural Break Detection** — detect regime shifts via normalised $\rho_t$, margin collapse, depth escalation
5. **Noise Robustness** — introduce bounded sensor noise (1–10%) and verify metric stability
6. **False Causality Rejection** — ensure correlation is not misidentified as causation
7. **Algorithmic Intervention Test** — replace parent signals with noise to verify edge collapse
8. **Scaling and Boundedness** — increase participant/component count; ensure bounded search and interpretability

---

## 8. Expected Outcomes

| Condition | Expected Result |
| :--- | :--- |
| Aligned | High interaction intelligence $\mathcal{I}_{interaction}$ |
| Misaligned | Low interaction intelligence |
| Perturbed | Dynamic drop/recovery observable in time series |
| Edge Cases | Correctly identified as structured (high $\Phi$) despite non-synchronous appearance |

---

## 9. Convergent and Discriminant Validity

- **Convergent:** Metrics correlate with independent performance measures
- **Discriminant:** Conditions produce distinct metric profiles
- **Perturbation Sensitivity:** Detects drop/recovery in Condition C
- **Edge Case Sensitivity:** Correctly identifies structured but non-synchronous interactions
- **Causal Decomposition:** Perturbation identifies meaningful generative substructures

---

## 10. Neural Calibration Phase

Prior to experimental trials:

- Random motion → high Neural BDM loss (establishes baseline)
- Structured patterns → low loss (validates sensitivity)
- Edge-case scenarios → verify $\hat{E}, \Phi, F$ robustness

---

## 11. Limitations

- Single-modal: motion and proximity only
- Neural estimator is model-dependent
- Computational cost of perturbation analysis scales as $O(|E|)$ per timestep
- Approximation strategies may be required for large systems

---

## 12. Notes

- Core metrics are **algorithmic information-based**; no statistical inference is used
- Input encoding adapts dynamically to observed interaction structure (2nd-order cybernetic)
- Provides a foundation for downstream domain-specific validation

---

*Supersedes: `archive/CIO-validation.md` (v1) and `archive/CIO-Validation-v2.md` (v2).*
