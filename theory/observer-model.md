# Observer Model

All quantities defined in [core-definitions.md](core-definitions.md) are **observer-relative**. This document formalises the bounded observer, its adaptive behaviour, and the implications of observer-dependence for the theory.

---

## 1. Formal Definition

An **observer** $O$ is a mapping from the joint agent state to a finite, structured representation:

$$
x_t = O(A_1(t), A_2(t), \dots, A_n(t))
$$

where $x_t \in \mathcal{X}$ is a structured object (string, graph, tensor) encoding the system at time $t$.

**Observer projection** to a single agent:

$$
x_t^{(i)} = O(A_i(t))
$$

This formulation is explicitly **observer-relative**: all measurements of collective behavior are mediated by a bounded encoding. There is no assumption of a privileged or universal observer.

---

## 2. Observer-Dependence

Because all quantities depend on $O$, two different observers $O$ and $O'$ may measure different coordination energies for the same system:

$$
E_O \;\neq\; E_{O'}
$$

**Consequence:** collective intelligence is not an intrinsic property of a system, but a relational property between the system and an observer. Apparent fragmentation ($E_O < 0$) may indicate a genuine lack of coordination, or simply that the observer's encoding is mismatched to the system's generative structure.

---

## 3. Adaptive Observer Resolution

A bounded observer must tune its resolution parameter $B(t)$ to balance sensitivity and computational cost:

$$
B(t) : \mathcal{X} \to \mathcal{X}_{B(t)}
$$

The adaptive loop:

1. Compute metrics $E_O(t)$, $K_{joint}(t)$, $C_{auto}(t)$
2. Detect representation mismatch (e.g. low $\Phi$ despite apparent coordination)
3. Update encoding: sampling rate, derivative features, edge definitions
4. Iterate: $O_{t+1} = O_t + \Delta O_t$

**High-variance regimes** are captured with finer granularity; **low-variance regimes** are compressed efficiently. This maintains sensitivity to phase transitions without exceeding computational limits.

---

## 4. Ashby's Law of Requisite Variety

The adaptive observer embodies **Ashby's Law of Requisite Variety** (Ashby, 1956):

> *Only variety can absorb variety.*

For the observer to detect all relevant coordination structures, its representational variety must be at least as large as the variety of the system it observes. Formally, the observer resolution $B(t)$ must satisfy:

$$
V(O_{B(t)}) \geq V(\text{system state at time } t)
$$

where $V(\cdot)$ denotes the variety (number of distinguishable states) of the argument.

When this condition is violated, the observer is **undersampling** the system, and apparent coordination may be an artifact of representational compression rather than genuine joint structure.

---

## 5. L-Level and H-Level Separation

The CIO hardware architecture (see [../implementation/cio-system-spec.md](../implementation/cio-system-spec.md)) enforces a strict separation between two observer modes:

| Level | Mode | Complexity Estimator | Purpose |
| :--- | :--- | :--- | :--- |
| **L-Level** | Real-time, bounded | $\hat{K}_L$ (Lempel–Ziv proxy) | Instantaneous phase-space monitoring |
| **H-Level** | Offline, unbounded | $\hat{K}_H$ (Neural BDM) | Causal decomposition, scientific inference |

This ensures that computational constraints do not limit the validity of scientific inference.

---

## 6. References

- Ashby, W. R. (1956). *An Introduction to Cybernetics*. Chapman & Hall.
- Behrouz, A., et al. (2025). *Nested Learning*. NeurIPS 2025.
- Zenil, H., & Adams, A. (2022). Algorithmic Information Dynamics of Cellular Automata. *arXiv:2112.13177*.
