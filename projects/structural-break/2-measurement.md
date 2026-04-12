# projections/structural-break / 2-measurement  
## Temporal Structural Functionals

---

## 1. Purpose

This layer defines how **structure is evaluated over time-indexed representation trajectories**.

It does not define:

- regimes  
- breakpoints  
- detection rules  
- statistical models  

It defines:

> functionals that evaluate structural properties over temporal representation sequences.

---

## 2. Representation Basis

From temporal ontology:

\[
\mathcal{X}_{1:T} = \{ x_1, x_2, ..., x_T \}
\]

Where:

\[
x_t = \phi_O(X_t)
\]

All measurement operates on this ordered sequence.

---

## 3. Measurement Functional Family

Define a family of temporal functionals:

\[
\mathcal{F}^{SB} : \mathcal{X}_{1:T} \rightarrow \mathcal{Y}
\]

Where:

- \(\mathcal{Y}\): abstract evaluation space  
- \(\mathcal{F}^{SB}\): structural evaluation functionals over trajectories  

Each functional evaluates structure **as it evolves over time**.

---

## 4. Structural Complexity Functional (Temporal Form)

Define:

\[
\mathcal{K}^{SB} : \mathcal{X}_{1:T} \rightarrow \mathbb{R}^+
\]

Interpretation:

- evaluates structural organization of the trajectory  
- reflects compressibility or regularity over time  
- remains abstract (no computational definition)

Constraint:

> \(\mathcal{K}^{SB}\) is not an estimator.

---

## 5. Local Structural Evaluation

Define local window:

\[
W_t = \{ x_{t-k}, ..., x_t, ..., x_{t+k} \}
\]

Define local functional:

\[
\mathcal{K}^{loc}(W_t)
\]

This evaluates structure within a neighborhood around time \(t\).

---

## 6. Structural Variation Functional

Define temporal variation:

\[
\Delta \mathcal{K}_t = \mathcal{K}^{loc}(W_{t+1}) - \mathcal{K}^{loc}(W_t)
\]

This captures **change in structural evaluation over time**.

No interpretation (e.g., “break”) is assigned here.

---

## 7. Temporal Consistency Principle

Structure over time is evaluated through:

- persistence: stability of \(\mathcal{K}^{loc}(W_t)\)  
- variation: changes in \(\mathcal{K}^{loc}(W_t)\)  

Both are abstract properties of the trajectory.

---

## 8. Projection Principle (Temporal Form)

Measurement is defined as:

\[
\text{measurement} = \Pi(\mathcal{X}_{1:T})
\]

Where:

- \(\Pi\): temporal projection operator  
- outputs are structural evaluations over time  

Measurement is not segmentation.

---

## 9. Separation Constraints

This layer must not depend on:

- change-point algorithms  
- statistical tests  
- segmentation procedures  
- thresholding rules  

It depends only on:

- representation trajectory  
- temporal ordering  
- structural relations  

---

## 10. Non-Identity Principle

For any estimator:

\[
\hat{\mathcal{K}}^{SB} \neq \mathcal{K}^{SB}
\]

Computed values approximate structural functionals but do not define them.

---

## 11. Role in Structural-Break Projection

This layer connects:

- ontology → defines trajectory  
- computation → approximates temporal functionals  

It defines:

> how structure is evaluated over time without introducing segmentation

---

## 12. Summary

This layer defines:

- temporal structural functionals  
- local and global evaluation over trajectories  
- structural variation over time  
- abstraction independent of detection or segmentation  

It does NOT define:

- regimes  
- structural breaks  
- detection logic  
- statistical inference  

---
