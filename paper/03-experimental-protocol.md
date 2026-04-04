# 📄 **3. Experimental Protocol**

---

## **3.1 Overview**

The proposed framework defines a set of measurable quantities for collective intelligence based on algorithmic complexity and causal perturbation. This section outlines a **general experimental protocol** for validating these quantities across synthetic and real-world multi-agent systems.

The protocol is designed to test three core properties:

1. **Detectability** — whether collective intelligence can be identified via compressibility
2. **Causality** — whether temporal structure is captured via directional complexity
3. **Decomposability** — whether coordination can be attributed to localized causal interactions

---

## **3.2 General Measurement Pipeline**

Given a multi-agent system observed over time, the experimental pipeline proceeds as follows:

---

### **Step 1 — System Encoding**

Construct an observer-dependent representation:

[
x_t = O(A_1(t), A_2(t), \dots, A_n(t))
]

where $x_t$ may take the form of:

* a string (symbolic trace),
* a graph (interaction network),
* or a tensor (spatiotemporal encoding).

---

### **Step 2 — Complexity Estimation**

Estimate algorithmic complexity:

[
\hat{K}(x_t)
]

using:

* Neural BDM (preferred), or
* compression-based approximations (e.g., Lempel–Ziv)

---

### **Step 3 — Structural Measurement**

Compute coordination energy:

[
E_O(t) = \sum_i \hat{K}(x_t^{(i)}) - \hat{K}(x_t)
]

---

### **Step 4 — Temporal Measurement**

Estimate directional coordination:

[
E_{dir}(t) = \hat{K}(x_t) - \hat{K}(x_t \mid x_{t-1})
]

---

### **Step 5 — Causal Perturbation**

For interaction-based representations $G(t) = (V, E)$:

* For each $e \in E$:

  * construct perturbed system $G \setminus e$
  * compute:

[
I(G,e) = \hat{C}(G) - \hat{C}(G \setminus e)
]

---

### **Step 6 — Generative Segmentation**

Apply the threshold:

[
I(G,e) > \log(2)
]

to identify:

* causal boundaries
* independent generative components

---

## **3.3 Experiment Class I: Synthetic Systems**

---

### **Objective**

Validate that the proposed measures correctly distinguish between:

* randomness
* trivial order
* structured complexity

---

### **Setup**

Simulated systems include:

1. **Random processes** (e.g., i.i.d. bit strings)
2. **Periodic systems** (low-complexity generators)
3. **Cellular automata** (e.g., Rule 30, Rule 110)
4. **Composite systems** formed by combining multiple generators

---

### **Procedure**

* Generate time series $x_t$ from each system
* Compute:

  * $\hat{K}(x_t)$
  * $E_O(t)$
  * $E_{dir}(t)$
* Apply perturbation analysis for composite systems

---

### **Hypotheses**

* **H1:** Random systems exhibit high $K(x)$ and low $E_{dir}$
* **H2:** Trivial systems exhibit low $K(x)$ and low $E_O$
* **H3:** Structured systems exhibit intermediate $K(x)$ and high $E_{dir}$
* **H4:** Composite systems can be decomposed via $I(G,e)$

---

### **Expected Outcome**

The framework recovers known distinctions between:

* randomness
* order
* computational structure

and correctly identifies **multiple generative sources**.

---

## **3.4 Experiment Class II: Interaction Graph Systems**

---

### **Objective**

Evaluate collective intelligence in systems defined by **agent interactions**, independent of physical embodiment.

---

### **Setup**

* Agents: $V = {A_1, \dots, A_n}$
* Interactions: edges $E(t)$ constructed from:

  * proximity
  * communication
  * co-attention
  * or shared events

---

### **Representation**

[
x_t = O(G(t))
]

where $G(t)$ is the interaction graph.

---

### **Procedure**

For each timestep:

1. Construct $G(t)$
2. Estimate $\hat{K}(G(t))$
3. Compute:

   * $E_O(t)$
   * $E_{dir}(t)$
4. Perform perturbation analysis:

   * compute ${I(G,e)}$

---

### **Hypotheses**

* **H1:** Coordinated interaction reduces global complexity
* **H2:** Independent behavior yields low $E_O$
* **H3:** Over-constrained systems reduce $E_{dir}$ (loss of autonomy)
* **H4:** High-performing systems lie in an intermediate regime

---

### **Key Claim**

> Collective intelligence manifests as a **phase transition** between disorder and over-constrained order.

---

### **Outputs**

* Time series of $E_O(t)$
* Time series of $E_{dir}(t)$
* Distribution of ${I(G,e)}$
* Identified causal subgraphs

---

## **3.5 Experiment Class III: Cross-Domain Validation**

---

### **Objective**

Test whether the framework generalizes across **different substrates**.

---

### **Systems**

1. Human interaction graphs
2. Artificial multi-agent systems
3. Language-based agent traces (e.g., sequential token exchanges)

---

### **Procedure**

* Encode each system into $x_t$
* Apply identical measurement pipeline

---

### **Hypotheses**

* **H1:** Systems with higher task performance exhibit:
  [
  \hat{K}(x) \downarrow, \quad E_{dir} \uparrow
  ]

* **H2:** Causal decomposition reveals similar structural patterns across domains

---

### **Expected Outcome**

Demonstration of:

> **substrate-independent signatures of collective intelligence**

---

## **3.6 Evaluation Metrics**

---

### **Structural Metrics**

* $\hat{K}(x_t)$
* $E_O(t)$

---

### **Temporal Metrics**

* $E_{dir}(t)$

---

### **Causal Metrics**

* Distribution of $I(G,e)$
* Number of components after segmentation

---

### **Emergent Metrics**

* Stability of causal structures over time
* Sensitivity to perturbations

---

## **3.7 Computational Considerations**

---

### **Complexity Estimation**

* Neural BDM provides higher fidelity but is computationally intensive
* Compression-based methods provide real-time approximations

---

### **Perturbation Cost**

* Exact perturbation scales as $O(|E|)$ per timestep
* Approximation strategies may be required for large systems

---

## **3.8 Summary**

The proposed protocol provides a **fully specified, reproducible methodology** for:

* detecting collective intelligence,
* measuring its magnitude, and
* decomposing its causal structure.

Crucially, the framework is **model-free**, **unsupervised**, and **substrate-independent**, enabling application across a wide class of multi-agent systems.

---

