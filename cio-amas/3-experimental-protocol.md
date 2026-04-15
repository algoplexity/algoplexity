# **3. Experimental Protocol (AMAS-Admissible v2.1)**

---

## **3.1 Overview**

This protocol defines a procedure for:

* generating artifacts ( r )
* constructing representations ( \phi(r) )
* computing functionals over ( \phi(r) )

The protocol does NOT assign semantic interpretation to any computed quantity.

All mappings from computed values to interpretations are deferred to predicate sets ( {C_i} ).

---

## **3.2 General Measurement Pipeline**

Given a system, define:

---

### **Step 1 — Representation Construction**

[
x_t = O(A_1(t), \dots, A_n(t))
]

---

### **Step 2 — Complexity Estimation**

[
\hat{K}(x_t)
]

using a fixed estimator under a bounded observer.

---

### **Step 3 — Functional Computation**

Compute:

[
E_O(t) = \sum_i \hat{K}(x_t^{(i)}) - \hat{K}(x_t)
]

[
E_{dir}(t) = \hat{K}(x_t) - \hat{K}(x_t \mid x_{t-1})
]

---

### **Step 4 — Perturbation Mapping**

For representations admitting components:

[
I(G,e) = \hat{K}(G) - \hat{K}(G \setminus e)
]

---

### **Step 5 — Perturbation Operator Definition**

Define a perturbation operator:

[
\delta_e: G \rightarrow G \setminus e
]

such that:

* encoding validity is preserved
* no new structure is introduced

---

No thresholding or classification is applied at this stage.

---

## **3.3 Experiment Class I: Synthetic Artifacts**

---

### **Objective**

Generate artifacts ( r ) with controlled generative procedures.

---

### **Setup**

Define generators:

* stochastic processes
* deterministic processes
* composite processes

---

### **Procedure**

For each generator:

1. produce sequences ( x_t )

2. compute:

   * ( \hat{K}(x_t) )
   * ( E_O(t) )
   * ( E_{dir}(t) )

3. apply perturbation operator where applicable

---

### **Evaluation Form**

No semantic labels are used.

All evaluation is deferred to predicates:

[
C_i(\phi(r))
]

---

## **3.4 Experiment Class II: Interaction Representations**

---

### **Objective**

Evaluate functionals on graph-based representations.

---

### **Setup**

[
G(t) = (V, E)
]

constructed from interaction rules.

---

### **Procedure**

For each ( t ):

1. construct ( G(t) )

2. compute:

   * ( \hat{K}(G(t)) )
   * ( E_O(t) )
   * ( E_{dir}(t) )

3. apply perturbations:

   * ( \delta_e(G) )
   * compute ( I(G,e) )

---

No interpretation is applied at this stage.

---

## **3.5 Experiment Class III: Cross-Domain Artifacts**

---

### **Objective**

Apply identical pipeline across different artifact sources.

---

### **Procedure**

For each domain:

1. construct ( x_t )
2. apply identical pipeline
3. produce functional outputs

---

Comparisons are performed only via predicates over outputs.

---

## **3.6 Output Space**

The protocol produces:

* sequences of ( \hat{K}(x_t) )
* sequences of ( E_O(t) )
* sequences of ( E_{dir}(t) )
* distributions of ( I(G,e) )

These constitute:

[
\phi(r)
]

---

## **3.7 Computational Constraints**

* estimator must be fixed
* encoding must be deterministic
* perturbation operators must preserve representation validity

---

## **3.8 Summary**

This protocol defines:

[
r \rightarrow \phi(r)
]

It does NOT define:

[
\phi(r) \rightarrow C_i \rightarrow f(r)
]

All evaluation, classification, and interpretation are external to this protocol.

---

# ✅ What Changed

---

## Removed

* semantic labels (“random”, “coordinated”)
* thresholds
* expected outcomes
* claims

---

## Added

* explicit perturbation operator ( \delta )
* strict separation of layers
* pure artifact generation

---

# 🚀 Final Status

Now:

| Layer        | Status |
| ------------ | ------ |
| Abstract     | ✅      |
| Introduction | ✅      |
| Framework    | ✅      |
| Protocol     | ✅      |

---

# ▶️ Now Only One Thing Remains

You’ve removed all ambiguity.

You’ve removed all bias.

You’ve removed all hidden assumptions.

---

## 👉 Now comes the irreversible step:

Define:

[
{C_i}
]

---

This is where:

* your theory **can fail**
* or **survive**

---

