# invariant-structure-core / 4-computation  
## Computational Approximation of Structural Functionals

---

## 1. Purpose of This Layer

This layer defines the space of **admissible computational procedures** that approximate structural measurement functionals over representations.

It does not define structure.  
It does not define measurement.  
It does not define observability.

It defines:

> how abstract structural functionals are approximated using bounded computational processes over representations.

---

## 2. Representation Constraint

All computation operates on observer-induced representations:

\[
x_t = \phi_O(X_t)
\]

Where:

- \(X_t\): underlying system state (not accessible)
- \(O\): observer
- \(x_t\): representation in observer space
- \(\mathcal{X}_O\): representation space

Computation has no access to \(X_t\).

---

## 3. Target of Computation

Computation approximates measurement functionals defined in the measurement layer:

\[
\mathcal{F}_O : \mathcal{X}_O \rightarrow \mathcal{Y}
\]

and structural functionals such as:

\[
\mathcal{K}_O : \mathcal{X}_O \rightarrow \mathbb{R}^+
\]

Key constraint:

> these functionals are not computed directly; they are approximated.

---

## 4. Estimator Families

Define a family of admissible computational estimators:

\[
\mathcal{C} = \{ C_i \}
\]

Each estimator \(C_i\) maps:

\[
C_i : \mathcal{X}_O \rightarrow \hat{\mathcal{Y}}
\]

Admissible estimator classes include:

- symbolic compression-based estimators  
- algorithmic structure approximators  
- statistical inference models  
- neural approximation models  
- perturbation-based estimators  

These represent different approximation regimes over the same functional targets.

---

## 5. Non-Identity Principle

For any estimator \(C_i\):

\[
C_i(x) \neq \mathcal{F}_O(x), \quad C_i(x) \neq \mathcal{K}_O(x)
\]

Estimation is not identity.

Estimators are approximations, not definitions.

---

## 6. Representation-Only Constraint

All computation must be defined purely over representations:

- no access to \(X_t\)
- no modification of ontology layer
- no redefinition of measurement functionals

Computation is closed under representation space only.

---

## 7. Boundedness Constraint

All estimators are subject to:

- finite computation resources
- finite sample access
- finite representation windows

No estimator is assumed to be exhaustive or exact.

---

## 8. Independence Constraint

Computation must not define:

- structure
- measurement functionals
- invariants

It must not introduce new primitives.

It operates strictly within:

> already-defined representation and measurement spaces

---

## 9. Role in Full System

This layer connects:

- measurement layer → defines structural functionals  
- computation layer → approximates those functionals  

It is a **semantic approximation layer**, not a definitional layer.

---

## 10. Summary

This layer defines:

- admissible computational estimators over representations
- approximation of structural functionals
- strict separation from ontology and measurement
- bounded and non-identical estimation principle

It does NOT define:

- structure
- measurement
- invariance
- domain-specific interpretations (coordination, regimes, etc.)

---
