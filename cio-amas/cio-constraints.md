# **AMAS-Admissible CIO Constraints**

---

## **0. Purpose**

This document defines the **admissible construction of artifacts and projections** for the CIO system under AMAS.

It specifies only:

* the structure of the artifact ( r )
* the projection function ( \phi(r) )
* constraints required for deterministic, reproducible evaluation

It does **not** define:

* predicates ( {C_i} )
* interpretations
* system objectives
* control logic
* semantics of any computed quantity

---

# **1. Artifact Definition**

## 1.1 Artifact Structure

The system produces an artifact:

[
r = {S_t}_{t=1}^T
]

Where each timestep:

[
S_t = {A_t}
]

* ( A_t \in \mathbb{R}^{N \times N} ) (adjacency or proximity matrix)
* ( T ) = fixed observation window length

---

## 1.2 Artifact Constraints

* Finite and fully specified
* No hidden variables
* All information used in evaluation must be contained in ( r )

---

## 1.3 Determinism of Generation (Optional)

If simulation is used:

* randomness must be seeded
* generation must be reproducible

---

# **2. Projection Function ( \phi )**

## 2.1 Definition

[
\phi(r) = \text{concatenate}\big(\text{vec}(A_1), \dots, \text{vec}(A_T)\big)
]

Where:

* ( \text{vec} ) = deterministic flattening with fixed ordering

---

## 2.2 Constraints

The projection must be:

* deterministic
* total (defined for all valid ( r ))
* invariant across all runs
* identical across all evaluators

---

## 2.3 Encoding Rules

* fixed ordering (e.g., row-major)
* no stochastic preprocessing
* no adaptive encoding
* no phase-dependent structure

---

# **3. Derived Functionals (Admissible)**

The following are **allowed as functions over ( \phi(r) )**:

---

## 3.1 Compression-Based Length

[
L_{sym}(r) = \text{compress}(\phi(r))
]

Constraints:

* fixed compression algorithm
* fixed parameters
* deterministic execution

---

## 3.2 Model-Based Length

[
L_{lat}(r) = \text{model_encode}(\phi(r))
]

Constraints:

* fixed model class
* bounded resources
* deterministic inference

---

## 3.3 Minimum Description

[
L^*(r) = \min(L_{sym}(r), L_{lat}(r))
]

---

## 3.4 Description Difference

[
\Delta L(r) = |L_{sym}(r) - L_{lat}(r)|
]

---

## 3.5 Temporal Derivative

[
r_{eff}(r) = \text{EMA}(\text{diff}(L^*(r)))
]

---

## 3.6 General Constraint

All derived quantities must:

* depend only on ( \phi(r) )
* be deterministic
* be reproducible

---

# **4. Observer Definition**

## 4.1 Observer Structure

An observer is defined as:

[
O = (\phi, M, B)
]

Where:

* ( \phi ) = projection function
* ( M ) = computation or encoding method
* ( B ) = bounded resource constraint

---

## 4.2 Observer Constraints

* ( \phi ) must be identical across all observers
* all computations must respect defined resource bounds
* observers must operate on the same input ( \phi(r) )

---

## 4.3 Multiple Observers

If multiple observers are used:

* each must satisfy the same projection constraint
* differences must arise only from ( M ) and ( B )

---

# **5. Domain Consistency**

All functions and future predicates must operate over the same domain:

[
X = \text{range}(\phi)
]

Constraints:

* no mixing of representations
* no phase-dependent domain changes
* no hidden feature injection

---

# **6. Separation Constraints**

The following must remain strictly separate from this document:

---

## 6.1 Predicate Definitions

* No definition of ( C_i )
* No thresholds
* No classification rules

---

## 6.2 Interpretation

* No semantic labels (e.g., coordination, alignment, chaos)
* No mapping from values to meaning

---

## 6.3 Control or Feedback

* No modification of ( r ) based on derived quantities
* No closed-loop dependence involving evaluation outputs

---

## 6.4 Hypothesis Encoding

* No assumptions about relationships between derived quantities and system properties

---

# **7. Admissibility Guarantees**

If all constraints are satisfied:

---

## 7.1 Evaluability Support

All future predicates ( C_i ) can be:

[
C_i: X \rightarrow {0,1}
]

and are:

* well-defined
* computable

---

## 7.2 Domain Alignment

All predicates share the same domain ( X )

---

## 7.3 Independence

No structure in this document:

* enforces
* implies
* or biases

any specific predicate outcome

---

# **8. Final Statement**

This document defines only:

> a deterministic mapping from system observations to a fixed representation space ( X )

and a set of **derived functionals over that space**

It does not define:

* what constitutes structure
* what constitutes coordination
* what constitutes validity

All such determinations are deferred to:

[
{C_i} \quad \text{and} \quad f(r)
]

under the AMAS falsifiability framework.

---

# ✅ Result

This version is now:

* ✔ **Audit-admissible** ( A({C_i}) ) compatible
* ✔ **Falsifiability-safe**
* ✔ Free of:

  * semantic leakage
  * circularity
  * hidden predicates

---
