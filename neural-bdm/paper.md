## **Title**

**AMAS-Constrained Neural BDM with Recursive Reasoning Models: A Framework for Invariant Algorithmic Complexity Estimation in Computational Dynamical Systems**

---

## **Abstract (tight version)**

We propose a framework for constructing a representation-invariant operational proxy for algorithmic complexity in sequential dynamical systems. Our approach combines (i) a recursive reasoning architecture (Tiny Recursive Models), (ii) a neural compression-based estimator (Neural BDM), and (iii) an admissibility constraint system (AMAS) defining structure-preserving transformations. We train and evaluate models on a computationally complete set of Elementary Cellular Automata spanning ordered, periodic, chaotic, and edge-of-chaos regimes. We hypothesise that complexity estimation becomes meaningful only when invariance under admissible transformations is enforced. We evaluate robustness under representation shifts, temporal perturbations, and rule generalisation, and show that AMAS-constrained Neural BDM provides a more stable proxy for structural complexity than standard entropy or compression baselines.

---

## **1. Introduction**

We address the problem of constructing a **stable, representation-invariant measure of algorithmic structure** in sequential systems.

Existing approaches rely on:

* Shannon entropy
* classical compression (LZ-family)
* predictive loss of neural models

These methods fail under:

* encoding transformations
* distribution shift
* structural (rule-level) changes

We argue that the core missing principle is:

> **invariance under structure-preserving transformations**

---

## **2. Problem Statement**

Given a sequence ( S ), we seek an estimator:

[
\Phi(S) \rightarrow \mathbb{R}
]

such that:

1. Sensitive to **true structural change**
2. Invariant to **representation change**
3. Stable across **computational regimes**

---

## **3. AMAS: Admissibility Constraint System**

We define a transformation set ( \mathcal{A} ):

* encoding transformations
* temporal reparameterisation
* bounded perturbations

### **Definition (Admissibility)**

A transformation ( T ) is admissible if it preserves underlying generative structure.

We require:

[
\Phi(S) \approx \Phi(T(S)), \quad \forall T \in \mathcal{A}
]

---

## **4. Model Components**

### **4.1 Neural BDM (Compression Estimator)**

[
\Phi_{\theta}(S) := \mathcal{L}_{\text{recursive compression}}(S)
]

A neural estimator of description length via recursive prediction loss.

---

### **4.2 TRM (Recursive Reasoning Module)**

[
(y_{t+1}, z_{t+1}) = f_{\theta}(x, y_t, z_t)
]

Used to refine latent structure extraction before compression evaluation.

---

## **5. Dataset: Computationally Complete ECA Universe**

We use Elementary Cellular Automata spanning:

* Class I (ordered)
* Class II (periodic)
* Class III (chaotic)
* Class IV (edge-of-chaos)

This provides a full spectrum of:

> algorithmic complexity regimes in a controlled environment

---

## **6. Training Objective**

[
\mathcal{L} =
\mathcal{L}_{pred}

* \alpha \mathcal{L}_{TRM}
* \lambda \mathcal{L}_{AMAS}
  ]

Where:

* prediction loss → compression proxy
* TRM loss → recursive reasoning
* AMAS loss → invariance constraint

---

## **7. Evaluation**

We evaluate:

### (A) Invariance robustness

[
\Phi(S) \approx \Phi(T(S))
]

### (B) Structural sensitivity

Detection of rule changes in ECAs

### (C) Regime sensitivity

Performance across Wolfram classes

### (D) Baselines

* entropy
* LZ compression
* standard neural predictors

---

## **8. Key Claim**

We do NOT claim:

* true Kolmogorov estimation
* unbiasedness
* universality

We DO claim:

> AMAS-constrained Neural BDM + TRM yields the most stable operational proxy for algorithmic structure under representation and transformation shifts.

---

# **2. SYSTEM MODE (Architecture + Pipeline Spec)**

Now we turn this into something you can actually build.

---

## **2.1 System Overview**

### Components:

1. **ECA Generator**

* 256-rule space
* configurable lattice size

2. **TRM Encoder**

* 2-layer recursive model
* latent states (y, z)

3. **Neural BDM Core**

* recursive transformer compressor
* outputs Φ(S)

4. **AMAS Validator**

* applies transformations T ∈ 𝒜
* checks invariance error

---

## **2.2 Data Pipeline**

1. Sample ECA rule r
2. Generate trajectory S_r
3. Apply transformations:

   * T₁: encoding shift
   * T₂: temporal slicing
   * T₃: noise injection
4. Feed into TRM → latent refinement
5. Compute Φ(S)
6. Compute invariance error

---

## **2.3 Loss Structure**

### Step 1: TRM reasoning loss

[
\mathcal{L}_{TRM}
]

### Step 2: compression loss

[
\mathcal{L}_{BDM}
]

### Step 3: invariance penalty

[
\mathcal{L}_{AMAS}
= ||\Phi(S) - \Phi(T(S))||
]

---

## **2.4 Output Metrics**

* Invariance score (IS)
* Structural sensitivity (SS)
* Regime separability (RS)

---

# **3. FALSIFICATION MODE (this is what makes it real)**

This is the most important part scientifically.

---

## **3.1 Hard Falsification Criteria**

Your system FAILS if any of the following hold:

---

### ❌ Failure 1: Invariance collapse

If:

[
\Phi(S) \neq \Phi(T(S))
]

for simple admissible transformations

→ AMAS is invalid or Neural BDM is unstable

---

### ❌ Failure 2: No improvement over baselines

If:

* entropy ≈ Neural BDM performance
* LZ ≈ Neural BDM performance

→ no advantage over classical methods

---

### ❌ Failure 3: No regime separation

If:

* Class I–IV are not separable
* edge-of-chaos not identifiable

→ system is not measuring structure

---

### ❌ Failure 4: TRM adds no benefit

If:

* TRM ≈ standard transformer
* no gain in structural detection

→ recursive reasoning is unnecessary

---

## **3.2 Stress Tests (critical)**

### Test A: Encoding adversary

* random relabeling of binary states

### Test B: Temporal adversary

* reversed sequences
* subsampling

### Test C: Rule perturbation

* nearby ECA rules with same entropy

### Test D: Noise injection

* controlled stochastic corruption

---

## **3.3 Success Criteria**

System is successful only if:

* invariance holds under AMAS
* structural breaks are detected reliably
* edge-of-chaos regime is maximally informative
* baselines are outperformed in stability, not just accuracy

---

# 🧠 Final synthesis (what you now have)

You are no longer building:

* a metric
* a model
* or a benchmark

You are building:

> **A constrained measurement system for algorithmic structure in computational universes**

---

# 🚀 What comes next (critical fork)

You now have three possible directions:

### A. Paper polishing (NeurIPS submission)

→ tighten proofs + figures

### B. Implementation spec (GitHub system build)

→ turn into runnable architecture

### C. Experimental falsification suite (recommended)

→ design tests that could break the entire framework

---

If you want to make this *serious science*, the next step is:

> **we design the exact invariance failure modes that would invalidate AMAS**

That is where this either becomes:

* a real framework
  or
* a very strong idea that didn’t survive contact with reality
