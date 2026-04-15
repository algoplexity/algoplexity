Perfect — this is the decisive step.
Below is a **fully revised, AMAS-admissible Simulation Architecture**.

This version is:

* ✔ Cleanly separated from theory
* ✔ Free of semantic leakage
* ✔ Free of predicate pre-definition
* ✔ Compatible with audit ( A({C_i}) )
* ✔ Ready for falsifiable testing

---

# **CIO Simulation Architecture (AMAS-Admissible v2.0)**

---

## **0. Purpose**

This document defines the **artifact generation and transformation pipeline** for the CIO system.

It specifies:

* how data is generated
* how artifacts ( r ) are constructed
* how projection ( \phi(r) ) is computed

It does **not** define:

* predicates ( {C_i} )
* interpretations
* evaluation criteria
* system objectives

---

# **1. System Role**

The system is a **generator of artifacts**:

[
G(\theta, \text{seed}) \rightarrow r
]

Where:

* ( \theta ) = parameter configuration
* `seed` = deterministic random seed
* ( r ) = artifact used for evaluation

---

# **2. Artifact Definition**

## **2.1 Structure**

[
r = {A_t}_{t=1}^T
]

Where:

* ( A_t \in {0,1}^{N \times N} )
* ( T ) = window length
* ( N ) = fixed number of nodes

---

## **2.2 Constraints**

* ( N ) constant per run
* matrix dimension fixed
* diagonal = 0
* representation complete (no hidden variables)

---

## **2.3 Admissible Domain**

An artifact ( r ) is **admissible** iff:

* all ( A_t ) are defined
* all snapshots satisfy temporal coherence (Section 4)

Artifacts outside this domain are excluded from evaluation.

---

# **3. Execution Model**

## **3.1 Time**

Discrete ticks:

```
tick = Δt
```

---

## **3.2 State Evolution**

Internal system state ( S_t ) evolves via:

[
S_{t+1} = F(S_t, \theta, \xi_t)
]

Where:

* ( \xi_t ) = random input (seeded)

---

## **3.3 Constraint**

* ( F ) must be independent of any computed metric
* no feedback from evaluation pipeline

---

# **4. Reconstruction Layer**

## **4.1 Purpose**

Construct temporally coherent snapshots from node outputs.

---

## **4.2 Output**

[
S_t \rightarrow A_t
]

---

## **4.3 Validity Conditions**

A snapshot is valid iff:

* all data corresponds to a single timestamp
* no temporal mixing occurs

---

## **4.4 Coverage Handling**

Let:

[
\text{coverage} = \frac{\text{observed nodes}}{N}
]

Constraint:

[
\text{coverage} \ge \theta
]

---

## **4.5 Constraint Handling**

If coverage condition fails:

* snapshot is **discarded**
* discard events MUST be logged

---

## **4.6 Determinism**

* identical inputs → identical ( A_t )
* no interpolation
* no stochastic reconstruction

---

# **5. Graph Builder**

## **5.1 Mapping**

[
A_t[i,j] =
\begin{cases}
1 & \text{if interaction observed} \
0 & \text{otherwise}
\end{cases}
]

---

## **5.2 Constraints**

* symmetric
* binary
* fixed dimension

---

# **6. Windowing**

## **6.1 Definition**

[
W_t = [A_{t-W}, \dots, A_t]
]

---

## **6.2 Constraints**

* fixed size ( W )
* FIFO update
* no dynamic resizing

---

# **7. Projection Function ( \phi(r) )**

## **7.1 Definition**

[
\phi(r) = \text{concat}(\text{vec}(A_1), \dots, \text{vec}(A_T))
]

---

## **7.2 Implementation**

```python
def encode_sequence(window):
    tokens = []
    for A_t in window:
        tokens.append(A_t.flatten(order="C"))
    return np.concatenate(tokens)
```

---

## **7.3 Constraints**

* deterministic
* fixed ordering
* invariant across runs

---

# **8. Derived Functionals**

The following functions are computed over ( \phi(r) ):

---

## **8.1 Compression Length**

[
L_{sym}(r) = \text{compress}(\phi(r))
]

---

## **8.2 Model-Based Length**

[
L_{lat}(r) = \text{model}(\phi(r))
]

---

## **8.3 Minimum Description**

[
L^*(r) = \min(L_{sym}(r), L_{lat}(r))
]

---

## **8.4 Difference**

[
\Delta L(r) = |L_{sym}(r) - L_{lat}(r)|
]

---

## **8.5 Temporal Variation**

[
r_{eff}(r) = \text{EMA}(\text{diff}(L^*(r)))
]

---

## **8.6 Constraints**

All functionals must:

* depend only on ( \phi(r) )
* be deterministic
* be reproducible

---

## **8.7 Status**

These quantities are:

> **uninterpreted numerical functionals**

They carry **no predefined meaning**.

---

# **9. Control Engine (Isolated)**

## **9.1 Definition**

Control modifies system state:

[
S_{t+1} = F(S_t, \theta, u_t)
]

---

## **9.2 Constraint (CRITICAL)**

Control MUST satisfy:

* no dependence on predicates ( {C_i} )
* no use of evaluation outcomes ( f(r) )

---

## **9.3 Experimental Separation**

Two classes of runs:

1. **Open-loop runs** (used for evaluation)
2. **Closed-loop runs** (control active)

---

## **9.4 Restriction**

Only **open-loop runs** are admissible for falsifiability testing.

---

# **10. Data Output**

## **10.1 Stored Artifact**

Each run produces:

```
{
  "theta": ...,
  "seed": ...,
  "artifact": r,
  "coverage_log": [...],
}
```

---

## **10.2 Constraints**

* no labels
* no scenario identifiers
* no interpretation fields

---

# **11. Forbidden Operations**

The system MUST NOT:

* inject semantic labels
* classify system behavior
* adapt encoding dynamically
* modify generation based on metrics
* expose internal state beyond ( r )

---

# **12. Admissibility Guarantees**

If all constraints are satisfied:

---

## **12.1 Artifact Integrity**

* ( r ) is complete and reproducible

---

## **12.2 Projection Validity**

* ( \phi(r) ) is well-defined for all ( r )

---

## **12.3 Predicate Compatibility**

Future predicates ( C_i ) can be defined as:

[
C_i: \phi(r) \rightarrow {0,1}
]

without ambiguity or hidden dependencies

---

## **12.4 Independence**

No component of this system:

* defines
* implies
* or biases

any predicate outcome

---

# **13. Final Statement**

This system is:

> a deterministic generator of artifacts and their projections

It does not define:

* structure
* coordination
* correctness

All such determinations are delegated to:

[
{C_i}, \quad f(r), \quad A({C_i})
]

under the AMAS framework.

---

# ✅ Outcome

You now have a **clean experimental instrument**:

* ✔ CIO = generator of ( r )
* ✔ constraints = ensure validity of ( r )
* ✔ no embedded hypothesis

---

## ▶️ Next Step

Now we do the final piece:

👉 **Define explicit predicates ( {C_i} )**

Then:

1. Run **audit** ( A({C_i}) )
2. Run **falsifiability test**

This is where your hypothesis is *finally exposed to failure*.
