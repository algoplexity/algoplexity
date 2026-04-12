# projections/structural-break / 4-detection-model  
## Structural Discontinuity Identification

---

## 1. Purpose

This layer defines how **structural variation signals are interpreted to identify discontinuities in temporal representation trajectories**.

It does not define:

- structure  
- measurement functionals  
- computation  

It defines:

> admissible decision mappings from computed structural signals to identified discontinuities.

---

## 2. Input Signal

From computation layer, obtain:

\[
\widehat{\Delta \mathcal{K}}_t
\]

or more generally:

\[
S_t = C_i(\mathcal{X}_{1:T})
\]

Where:

- \(S_t\): temporal structural signal  
- derived from estimator outputs  

---

## 3. Detection Mapping

Define a detection operator:

\[
D : \{S_t\}_{t=1}^T \rightarrow \mathcal{B}
\]

Where:

- \(\mathcal{B} = \{t_1, t_2, ..., t_m\}\)  
- set of identified discontinuity indices  

---

## 4. Discontinuity Interpretation

A detected point \(t^*\) satisfies:

- significant deviation in structural signal  
- relative to local temporal context  

Formally, defined through:

\[
D(S_t) = \{ t \mid \Phi(S_{t-k:t+k}) \}
\]

Where:

- \(\Phi\): admissible decision functional  
- operates on local signal neighborhoods  

---

## 5. No Ontological Commitment

This layer does NOT assert:

- existence of “true regimes”  
- ground-truth breakpoints  
- underlying causal transitions  

It defines only:

> identified discontinuities in representation-level structure

---

## 6. Local Decision Principle

Detection is inherently local:

- operates on windows of \(S_t\)  
- compares relative variation  
- does not require global segmentation  

---

## 7. Admissible Detection Classes

Detection mappings may include:

- threshold-based detectors  
- peak detection operators  
- statistical hypothesis tests  
- model-based segmentation rules  
- learned decision functions  

All are admissible if they operate on \(S_t\).

---

## 8. Non-Equivalence Principle

Detected discontinuities are not identical to:

- true system transitions  
- measurement-level structure  
- latent regime changes  

They are:

> outputs of a decision mapping over estimated signals

---

## 9. Consistency Constraint

Detection outputs must satisfy:

- temporal ordering preservation  
- reproducibility under same estimator and parameters  
- stability under admissible perturbations  

---

## 10. Separation Constraints

This layer must not:

- redefine structure  
- access raw representations directly  
- modify estimator outputs  
- depend on latent system state  

It operates only on computed signals.

---

## 11. Role in Structural-Break Projection

This layer connects:

- computation → provides structural signals  
- downstream applications → use detected discontinuities  

It defines:

> how structural variation signals are converted into identified breakpoints

---

## 12. Summary

This layer defines:

- detection mappings from temporal signals to discontinuities  
- admissible classes of decision functions  
- local interpretation of structural variation  
- strict separation from ontology, measurement, and computation  

It does NOT define:

- true regimes  
- causal transitions  
- statistical ground truth  
- system-level explanations  

---
