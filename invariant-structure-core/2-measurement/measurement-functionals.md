# invariant-structure-core / 2-measurement  
## Measurement Functionals Over Observer-Induced Structure

---

## 1. Purpose of This Layer

This layer defines how **structure is mapped into abstract evaluative quantities**.

It does not define structure itself.  
It does not define computation or estimation methods.

It defines:

> observer-relative functionals over representations that probe structural properties.

---

## 2. Representation Basis

All measurement is defined over observer-induced representations:

\[
x_t = \phi_O(X_t)
\]

Where:

- \(X_t\): underlying system state (not accessible)
- \(O\): observer
- \(x_t\): representation in observer space
- \(\mathcal{X}_O = \{x_t\}\): representation space

All measurement operates on \(\mathcal{X}_O\), not on \(X_t\).

---

## 3. Measurement Functional Space

Define a family of observer-relative functionals:

\[
\mathcal{F}_O : \mathcal{X}_O \rightarrow \mathcal{Y}
\]

Where:

- \(\mathcal{F}_O\): measurement functional space
- \(\mathcal{Y}\): abstract codomain of structural evaluations

Each functional \(f \in \mathcal{F}_O\) maps a representation to a structural evaluation value.

No restriction is placed on \(\mathcal{Y}\) at this layer.

---

## 4. Structural Complexity Functional

Define a central abstract functional:

\[
\mathcal{K}_O : \mathcal{X}_O \rightarrow \mathbb{R}^+
\]

Interpretation:

- \(\mathcal{K}_O(x)\): observer-relative evaluation of structural complexity
- lower values correspond to more compressible structure under admissible descriptions
- higher values correspond to less compressible or more disordered representations

Important constraint:

> \(\mathcal{K}_O\) is not an estimator and has no computational specification at this layer.

---

## 5. Projection Principle

All measurable quantities are projections of structure into functional space:

\[
\text{measurement} = \Pi_O(\text{structure})
\]

Where:

- structure is defined in invariant-structure-core / theory layer
- \(\Pi_O\) is an observer-relative projection operator
- measurement outputs are not structure itself

This enforces a strict separation between:

- structure (ontological object)
- measurement (functional projection)

---

## 6. Observer Relativity

All functionals are observer-dependent:

\[
\mathcal{F}_O \neq \mathcal{F}_{O'}
\]

However, valid structural theory requires:

> invariance of structural relationships under admissible observer transformations

Thus:

- values may differ across observers
- structural ordering and equivalence must remain consistent

---

## 7. Separation Constraints

This layer enforces strict non-dependence rules:

### Measurement must NOT depend on:
- computational estimators
- learning algorithms
- compression implementations
- statistical inference procedures

### Measurement MAY depend on:
- representation structure \(x_t\)
- observer encoding \(\phi_O\)
- admissible structural decompositions (from theory layer)

---

## 8. Non-Identity Principle

Any computational approximation \(\hat{\mathcal{K}}_O\) is not equivalent to \(\mathcal{K}_O\):

\[
\hat{\mathcal{K}}_O \neq \mathcal{K}_O
\]

Computational systems approximate measurement functionals but do not define them.

This preserves ontological separation between:

- definition (measurement layer)
- approximation (computation layer)

---

## 9. Role in the Full System

This layer connects:

- **Theory layer:** defines what structure is  
- **Computation layer:** approximates measurement functionals  

Measurement acts as an intermediate semantic interface:

> structure → projection → evaluative quantities

---

## 10. Summary

This layer defines:

- observer-relative functional mapping of representations
- abstract structural complexity functional \(\mathcal{K}_O\)
- projection-based interpretation of measurement
- strict separation from computational implementation
- invariance constraints across observers

It does NOT define:

- algorithms
- estimators
- detection methods
- coordination or system-specific observables

---
