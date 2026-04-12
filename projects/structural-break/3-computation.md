# projections/structural-break / 3-computation  
## Temporal Approximation of Structural Functionals

---

## 1. Purpose

This layer defines how **temporal structural functionals are approximated over representation trajectories**.

It does not define:

- structure  
- measurement functionals  
- regimes or breaks  

It defines:

> admissible computational procedures that approximate structural evaluations over time-indexed representations.

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

Computation operates only on this sequence.

---

## 3. Target of Computation

Computation approximates:

\[
\mathcal{F}^{SB}(\mathcal{X}_{1:T}), \quad \mathcal{K}^{SB}(\mathcal{X}_{1:T}), \quad \mathcal{K}^{loc}(W_t)
\]

Key constraint:

> these functionals are not computed directly; they are approximated.

---

## 4. Estimator Family

Define:

\[
\mathcal{C}^{SB} = \{ C_i \}
\]

Each estimator:

\[
C_i : \mathcal{X}_{1:T} \rightarrow \hat{\mathcal{Y}}
\]

Admissible estimator classes include:

- sliding-window estimators  
- segmentation-aware scoring models  
- change-sensitive statistical estimators  
- neural sequence models  
- perturbation-based temporal estimators  

Each provides a different approximation regime.

---

## 5. Temporal Locality Constraint

Estimators must respect temporal structure:

- operate on local windows \(W_t\) or sequences  
- preserve ordering  
- avoid permutation-invariant transformations  

Time ordering is not optional.

---

## 6. Structural Variation Approximation

Estimators may approximate:

\[
\Delta \mathcal{K}_t
\]

as:

\[
\widehat{\Delta \mathcal{K}}_t = C_i(W_{t+1}) - C_i(W_t)
\]

This produces a **temporal signal of structural variation**.

Interpretation is deferred to the next layer.

---

## 7. Non-Identity Principle

For any estimator:

\[
C_i(\mathcal{X}_{1:T}) \neq \mathcal{F}^{SB}(\mathcal{X}_{1:T}), \quad C_i \neq \mathcal{K}^{SB}
\]

Estimators approximate but do not define structure.

---

## 8. Representation-Only Constraint

Computation must:

- operate only on \(x_t\)  
- not access latent state \(X_t\)  
- not redefine measurement functionals  

---

## 9. Boundedness Constraint

All estimators are:

- resource-bounded  
- sample-limited  
- window-constrained  

No estimator is assumed exact or complete.

---

## 10. Independence Constraint

Computation must not define:

- segmentation  
- regimes  
- structural breaks  
- invariants  

It only produces signals from which such interpretations may be inferred.

---

## 11. Role in Structural-Break Projection

This layer connects:

- measurement → defines temporal functionals  
- detection-model → interprets computed signals  

It defines:

> how structural variation over time is approximated computationally

---

## 12. Summary

This layer defines:

- admissible temporal estimators  
- approximation of structural functionals over sequences  
- generation of structural variation signals  
- strict separation from segmentation or detection logic  

It does NOT define:

- breakpoints  
- regimes  
- decision rules  
- statistical significance  

---
