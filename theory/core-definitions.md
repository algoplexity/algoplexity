# Core Definitions

Canonical single-source definitions for all fundamental quantities used across the Algoplexity research programme. Cross-references to these definitions replace repeated inline formulations in `paper/` and `implementation/`.

---

## 1. Collective Intelligence (CI)

Let $\{A_1, A_2, \dots, A_n\}$ be interacting agents and $O$ a bounded observer (see [observer-model.md](observer-model.md)).

Define the observed individual and joint representations:

$$
x_t^{(i)} = O(A_i(t)), \qquad x_t = O(A_1(t), \dots, A_n(t))
$$

A system exhibits **Collective Intelligence** at time $t$ if and only if:

$$
K(x_t) < \sum_{i=1}^{n} K\!\left(x_t^{(i)}\right) \qquad \text{and} \qquad K(x_t) > \varepsilon
$$

where $K(\cdot)$ is Kolmogorov complexity (estimated via $\hat{K}$), and $\varepsilon > 0$ excludes trivial or degenerate systems.

**Interpretation:** the joint behavior admits a shorter generative description than the sum of its parts, while remaining non-trivial.

---

## 2. Coordination Energy (E_O)

$$
E_O(x_t) = \sum_{i=1}^{n} K\!\left(x_t^{(i)}\right) - K(x_t)
$$

| Value | Meaning |
| :--- | :--- |
| $E_O > 0$ | Coordinated intelligence — joint behavior more compressible than parts |
| $E_O \approx 0$ | Independent agents — no shared generative structure |
| $E_O < 0$ | Fragmentation — interference, adversarial dynamics, or observer mis-specification |

> **Note:** sign convention follows the v3+ formulation where positive $E_O$ indicates coordination. The earlier v1–v2 convention used the opposite sign; see `archive/` for historical versions.

---

## 3. Directional Coordination Energy (E_dir)

To capture temporal causality between successive observations:

$$
E_{dir}(t) = K(x_t) - K(x_t \mid x_{t-1})
$$

| Value | Meaning |
| :--- | :--- |
| High $E_{dir}$ | Strong causal / predictive dependence on prior state |
| Low $E_{dir}$ | Weak or random temporal evolution |

This quantity measures the **predictive structure** of the system — how much information the previous state provides about the current state.

---

## 4. Information Contribution (I(G, e))

Represent the system as an interaction graph $G(t) = (V, E)$ where $V$ is the set of agents and $E$ the set of interactions. The **information contribution** of edge $e \in E$ is:

$$
I(G, e) = C(G) - C(G \setminus e)
$$

where $C(G)$ denotes the algorithmic complexity of the graph.

| Value | Meaning |
| :--- | :--- |
| $I(G,e) > 0$ | Edge contributes causal structure |
| $I(G,e) \approx 0$ | Redundant edge |
| $I(G,e) < 0$ | Edge introduces noise |

**Generative Separation Criterion:** $I(G,e) > \log(2)$ indicates a causal boundary between independent generative mechanisms.

**Causal Decomposition Theorem:** Coordination energy decomposes as:

$$
E_O(G) = \sum_{e \in E} I(G, e)
$$

Collective intelligence is therefore not a global property but arises from **localized causal contributions** of individual interactions.

---

## 5. Summary Table

| Symbol | Name | Formula | Reference |
| :--- | :--- | :--- | :--- |
| CI | Collective Intelligence | $K(x_t) < \sum K(x_t^{(i)})$ and $K(x_t) > \varepsilon$ | §1 above |
| $E_O$ | Coordination Energy | $\sum K(x_t^{(i)}) - K(x_t)$ | §2 above |
| $E_{dir}$ | Directional Coordination | $K(x_t) - K(x_t \mid x_{t-1})$ | §3 above |
| $I(G,e)$ | Information Contribution | $C(G) - C(G \setminus e)$ | §4 above |

All quantities depend on the choice of observer $O$. See [observer-model.md](observer-model.md) for the formal treatment of observer-dependence and bounded approximation.
