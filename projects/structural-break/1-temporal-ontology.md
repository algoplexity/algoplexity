# projections/structural-break / 1-temporal-ontology  
## Time-Indexed Representation Structure

---

## 1. Purpose

This ontology defines the **representation structure for time-indexed systems**.

It does not define:

- regimes  
- breaks  
- change points  
- statistical models  

It defines:

> how representations evolve over time as a sequence in observer space.

---

## 2. Underlying System

Assume a latent system state:

\[
X_t
\]

This state is not directly observable.

All access is through an observer:

\[
x_t = \phi_O(X_t)
\]

---

## 3. Representation Trajectory

The fundamental object is a time-indexed sequence:

\[
\mathcal{X}_{1:T} = \{ x_1, x_2, ..., x_T \}
\]

Where:

- \(x_t \in \mathcal{X}_O\)
- \(\mathcal{X}_O\): observer-induced representation space

This trajectory is the only object available for analysis.

---

## 4. Temporal Ordering

Representations are ordered:

\[
t_1 < t_2 < ... < t_T
\]

This ordering is intrinsic and must be preserved under admissible transformations.

---

## 5. Local Structural Context

Define a local temporal window:

\[
W_t = \{ x_{t-k}, ..., x_t, ..., x_{t+k} \}
\]

Structure is evaluated over such local neighborhoods.

No assumption is made about stationarity or distribution.

---

## 6. No Regime Assumption

This ontology does NOT assume:

- existence of regimes  
- segmentation of the trajectory  
- discrete state transitions  

The trajectory is continuous in representation space.

Any segmentation is introduced downstream.

---

## 7. Structural Continuity Constraint

Successive representations may be related by:

- smooth transformations  
- abrupt changes  
- noise-induced variation  

No constraint is imposed at ontology level.

---

## 8. Representation Independence

The ontology does not assume:

- specific encoding type  
- dimensionality  
- noise characteristics  
- sampling frequency consistency  

Only requirement:

> representations are elements of a valid observer-induced space.

---

## 9. Admissibility Constraint

All representations must satisfy constraints inherited from invariant-structure-core:

- invariance under admissible transformations  
- preservation of structural equivalence relations  
- non-degeneracy  

---

## 10. Primary Object

The primary object of this projection is:

\[
\mathcal{X}_{1:T}
\]

Not:

- individual points  
- inferred segments  
- statistical summaries  

Only the trajectory.

---

## 11. Summary

This ontology defines:

- time-indexed representation trajectories  
- ordering over representations  
- local temporal context windows  
- absence of segmentation assumptions  

It does NOT define:

- regimes  
- structural breaks  
- detection rules  
- statistical properties  

---
