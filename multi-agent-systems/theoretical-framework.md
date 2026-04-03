# 📄 SECTION 2 — THEORETICAL FRAMEWORK (Draft)

Below is a **clean, publication-style draft** you can directly use and refine.

---

## **2. Theoretical Framework**

### **2.1 Observer-Relative Representation**

Consider a multi-agent system consisting of agents
[
{A_1, A_2, \dots, A_n}
]
evolving over time.

We define an **observer** $O$ as a mapping from the joint agent state to a finite representation:

[
x_t = O(A_1(t), A_2(t), \dots, A_n(t))
]

where $x_t$ is a structured object (e.g., string, graph, tensor) encoding the system at time $t$.

This formulation is **observer-relative**, acknowledging that all measurements of collective behavior are mediated by a bounded encoding.

---

### **2.2 Algorithmic Complexity**

Let $K(x)$ denote the Kolmogorov complexity of object $x$, i.e., the length of the shortest program that generates $x$ on a universal Turing machine.

Since $K(x)$ is uncomputable, we assume access to an estimator:

[
\hat{K}(x) \approx K(x)
]

such as Neural Block Decomposition Methods (Neural BDM) or compression-based proxies.

---

### **2.3 Definition of Collective Intelligence**

We define **collective intelligence (CI)** as a property of the observed system $x_t$ satisfying:

[
K(x_t) < \sum_{i=1}^{n} K(x_t^{(i)})
]

and

[
K(x_t) > \epsilon
]

where $x_t^{(i)} = O(A_i(t))$ is the individual projection of agent $i$, and $\epsilon > 0$ excludes trivial solutions.

---

### **Interpretation**

* The first condition captures **compressibility due to coordination**
* The second excludes degenerate or trivial systems

Thus, CI corresponds to:

> **non-trivial compressible structure emerging from multi-agent interaction**

---

### **2.4 Coordination Energy**

We define the **Coordination Energy**:

[
E_O(x_t) = \sum_{i=1}^{n} K(x_t^{(i)}) - K(x_t)
]

---

### **Interpretation**

* $E_O > 0$: shared structure / coordination
* $E_O \approx 0$: independence
* $E_O < 0$: interference or antagonism

---

### **2.5 Directional Coordination**

To capture temporal causality, consider successive observations:

[
x_{t-1} \rightarrow x_t
]

Define **Directional Coordination Energy**:

[
E_{dir}(t) = K(x_t) - K(x_t \mid x_{t-1})
]

---

### **Interpretation**

* High $E_{dir}$: strong causal dependence
* Low $E_{dir}$: weak or random evolution

This quantity captures the **predictive structure** of the system.

---

### **2.6 Interaction Graph Representation**

We represent the system at time $t$ as a graph:

[
G(t) = (V, E)
]

where:

* $V$ = agents
* $E$ = interactions

The observer $O$ induces a mapping:

[
x_t \equiv O(G(t))
]

---

### **2.7 Causal Perturbation and Information Contribution**

For each element $e \in E$, define its **information contribution**:

[
I(G, e) = C(G) - C(G \setminus e)
]

where $C(G)$ is the algorithmic complexity of the graph.

---

### **Interpretation**

* $I(G,e) > 0$: element contributes to structure
* $I(G,e) < 0$: element acts as noise
* $I(G,e) \approx 0$: redundant

---

### **2.8 Causal Decomposition of Coordination**

We propose that Coordination Energy admits a decomposition:

[
E_O(G) = \sum_{e \in E} I(G,e)
]

---

### **Theorem 1 (Causal Decomposition of Collective Intelligence)**

> The coordination energy of a multi-agent system can be decomposed into the sum of algorithmic information contributions of its interactions.

---

### **Implication**

Collective intelligence is **not global**, but arises from **localized causal contributions**.

---

### **2.9 Generative Separation Criterion**

We define a threshold:

[
I(G,e) > \log(2)
]

as indicating a **causal boundary** between independent generative mechanisms.

---

### **Interpretation**

Edges exceeding this threshold:

* connect distinct algorithmic sources
* define separable substructures

---

### **2.10 Summary**

The framework defines three complementary quantities:

| Quantity  | Role                      |
| --------- | ------------------------- |
| $E_O$     | structural coordination   |
| $E_{dir}$ | temporal/causal structure |
| $I(G,e)$  | local causal contribution |

Together, these provide a **multi-scale, causally grounded characterization of collective intelligence**.

---

