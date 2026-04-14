# CIO / 3-computation.md
## Bounded Trajectory Transformation Semantics under AMAS Constraints

---

## 1. Purpose

This document defines the **computation layer of the CIO projection under AMAS admissibility constraints**.

Computation is strictly:

> transformation of observer-induced representations of AMAS admissible trajectories

It does NOT define:
- system evolution rules
- causal inference mechanisms
- optimization procedures
- control policies
- modification of AMAS dynamics (𝔻)

All dynamics are defined exclusively in `amas-core/6-dynamics/`.

---

## 2. Computational Primitive

The primitive input to computation is:

> observer-projected trajectory representation

\[
x_t^{(O)} = \phi_O(\tau)
\]

Where:
- τ ∈ 𝔻 (AMAS admissible trajectory)
- φ_O = bounded observer mapping

Computation NEVER operates on 𝔻 directly.

---

## 3. Computation Operator

Computation is defined as:

\[
\mathcal{C} : x_t^{(O)} \rightarrow y_t^{(O)}
\]

Where:
- x_t^(O): observer representation
- y_t^(O): transformed representation

This transformation is:
- representation-preserving
- structure-manipulating only within admissible bounds

---

## 4. Fundamental Constraint: No Causal Authority

Computation MUST NOT:

- influence trajectories τ ∈ 𝔻
- modify dynamics (𝔻)
- introduce feedback into AMAS evolution
- encode control logic
- optimize system behavior

Computation is strictly:

> syntactic transformation over representations, not semantic intervention on dynamics

---

## 5. Non-Equivalence to Inference

Computation is NOT inference.

Distinction:

- computation → transforms representations
- inference → interprets or classifies representations

Computation MUST NOT:
- assign meaning to trajectories
- reconstruct hidden causal structure
- infer system “true state”

---

## 6. Estimator Constraint (Critical)

Any estimator used in computation is defined as:

> bounded functional over transformed representations

\[
E : y_t^{(O)} \rightarrow \mathbb{R}^n
\]

Estimators:
- compress
- summarize
- project structure

Estimators MUST NOT:
- determine actions
- influence inference logic
- act as decision components

---

## 7. Valid Computational Classes

Only the following are admissible:

### 7.1 Structural transformation
- filtering
- encoding
- compression
- reparameterization

### 7.2 Representation mapping
- feature extraction
- projection change
- basis transformation

### 7.3 Bounded aggregation
- statistical summaries
- invariant-preserving reductions

---

## 8. Forbidden Computational Classes

Explicitly invalid:

- reinforcement learning loops
- optimization over system outcomes
- policy learning
- control synthesis
- gradient-based system steering
- adaptive dynamics modification

Any such structure is classified as:

> implicit control and is invalid under AMAS

---

## 9. Mesoscopic Consistency Constraint

Computation must preserve mesoscopic structure:

- must not collapse relational structure
- must not over-resolve microstates
- must not over-aggregate macrostates

Computation operates entirely within:

> the mesoscopic representational regime defined by AMAS measurement constraints

---

## 10. Observer Consistency Rule

For all observers O₁, O₂:

If:

\[
x_t^{(O_1)} \neq x_t^{(O_2)}
\]

then computation must satisfy:

> transformations remain consistent across observer projections

Computation MAY vary across observers, but:

> cannot break structural equivalence classes defined in AMAS-core/5-invariants

---

## 11. Non-Control Closure Condition

A computation is valid only if:

- it does not introduce feedback into AMAS dynamics
- it does not approximate control policies
- it does not optimize system evolution
- it remains representationally closed

Computation is:

> closed over representation space, not open over dynamics space

---

## 12. Relationship to Other Layers

- measurement → produces x_t^(O)
- computation → transforms x_t^(O)
- inference → interprets y_t^(O)
- systems → execute representations as simulations
- validation → tests structural consistency

No upward causation is permitted.

---

## 13. Summary

CIO computation defines:

> bounded, non-causal transformation of observer-projected AMAS trajectories within mesoscopic representational space

It does NOT define:

- learning systems
- control mechanisms
- optimization processes
- causal intervention
- system evolution logic

The only primitive reality remains:

> AMAS admissible trajectories (𝔻)
